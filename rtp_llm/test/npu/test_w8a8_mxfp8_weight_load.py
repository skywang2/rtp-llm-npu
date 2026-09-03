"""ModelSlim (W8A8_MXFP8) weight loading path verification on Ascend NPU.

Runs INSIDE the NPU container (needs torch_npu):
  docker exec rtp-llm-cann-9.2.0-weekly bash -c \
    "cd /mnt/docker/w30060538/rtp-llm-npu && \
     PYTHONPATH=/mnt/docker/w30060538/rtp-llm-npu \
     /root/miniconda3/envs/rtp-env/bin/python \
     rtp_llm/test/npu/test_w8a8_mxfp8_weight_load.py"

Validates the full static-quantization loading chain:
  AscendW8A8MXFP8Weight._load_raw_tensor (orientation normalization)
    -> _split (no-op at tp=1)
    -> PerBlockFp8Weight._postprocess (NPU branch: real transpose + swizzle)
plus numeric round-trip (E8M0 dequant vs original bf16 weights).
"""

import functools
import sys
from typing import Dict, List
from unittest.mock import MagicMock

import torch
import torch_npu  # noqa: F401

from rtp_llm.config.quant_config import AscendW8A8MXFP8Config
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.linear_attn_weight import LinearAttnAtomicWeight, LinearAttnConfig
from rtp_llm.model_loader.w8a8_mxfp8_weight import AscendW8A8MXFP8Weight
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.model_weight import (
    W,
    merge_qkv_hf,
    transpose,
)
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    merge_qkvz_transpose_reorder,
    transpose_stack_moe_w1,
)
from rtp_llm.utils.model_weight import stack_, stack_moe_w1

torch.npu.set_device(0)
torch.manual_seed(123)

HIDDEN = 256  # % 32 == 0
QKV_OUT = 192  # (q 4 heads*32 + k 2*32 + v 2*32) * ... use plain sizes
INTER = 128
N_EXPERTS = 4
GROUP = 32
PREFIX = "model.language_model."


class FakeTensorSource(TensorSource):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors

    def load_tensor(self, name: str, data_type=torch.float16) -> List[torch.Tensor]:
        if name not in self._tensors:
            raise KeyError(f"Tensor {name!r} not found")
        return [self._tensors[name].to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def get_database(self):
        return None


class StubDevice:
    """No-op stand-in for load_config.exported_device (AscendImpl hooks are
    identity implementations)."""

    def maybe_rewrite_weight_by_key(self, key, weight):
        return weight

    def shuffle_moe_weight(self, x, datatype, name):
        return x


def make_load_config():
    lc = MagicMock()
    lc.tp_size = 1
    lc.tp_rank = 0
    lc.dp_size = 1
    lc.dp_rank = 0
    lc.ep_size = 1
    lc.ep_rank = 0
    lc.ffn_tp_size = 1
    lc.ffn_tp_rank = 0
    lc.lm_head_tp_size = 1
    lc.lm_head_tp_rank = 0
    lc.compute_dtype = torch.bfloat16
    lc.merge_lora = False
    lc.moe_pure_tp_mode = False
    lc.bit = 8
    lc.hidden_size = HIDDEN
    lc.head_num = 4
    lc.head_num_kv = 2
    lc.size_per_head = 32
    lc.get_selected_experts.return_value = list(range(N_EXPERTS))
    lc.exported_device = StubDevice()
    return lc


def quantize_module(w_nk: torch.Tensor):
    """Quantize an [N, K] module weight the ModelSlim way.

    Returns (kernel [N, K] fp8, scale [N, K//32] uint8).
    """
    kernel, scale_3d = torch_npu.npu_dynamic_mx_quant(
        w_nk.to("npu"), dst_type=torch.float8_e4m3fn
    )
    # flatten the pair-split [N, kp//2, 2] output back to the ckpt layout [N, kp]
    scale = scale_3d.reshape(w_nk.shape[0], w_nk.shape[1] // GROUP).cpu()
    return kernel.cpu(), scale


def e8m0_dequant(kernel: torch.Tensor, scale_swizzled: torch.Tensor) -> torch.Tensor:
    """Dequantize [K, N] fp8 kernel with swizzled [kp//2, N, 2] E8M0 scale."""
    kp2, n, two = scale_swizzled.shape
    assert two == 2
    scale_2d = scale_swizzled.permute(1, 0, 2).reshape(n, kp2 * 2)  # [N, kp]
    exp = torch.pow(2.0, scale_2d.float() - 127.0)  # [N, kp]
    exp_full = exp.repeat_interleave(GROUP, dim=1)  # [N, K]
    w_nk = kernel.float().T * exp_full  # [N, K]
    return w_nk


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def load_and_check(weight: WeightModule, tensors: Dict[str, torch.Tensor],
                   qc: AscendW8A8MXFP8Config, expected: Dict[str, tuple],
                   reference: Dict[str, torch.Tensor] = None):
    qw = WeightModule.create(weight, qc)
    assert isinstance(qw, AscendW8A8MXFP8Weight), type(qw)
    res = qw.load(FakeTensorSource(tensors), layer_id=0, device="npu",
                  load_config=make_load_config())
    for name, (shape, dtype) in expected.items():
        assert name in res, f"{name} missing from {list(res.keys())}"
        got = res[name]
        assert tuple(got.shape) == shape, f"{name}: {tuple(got.shape)} != {shape}"
        assert got.dtype == dtype, f"{name}: {got.dtype} != {dtype}"
        assert got.device.type == "npu", f"{name} on {got.device}"
    if reference:
        for name, ref in reference.items():
            w_nk = e8m0_dequant(res[name], res[name.replace(".kernel", ".weight_only_quant_scale")
                                            if ".kernel" in name else name + "_scale_missing"])
            # caller passes explicit pairs instead
    return res


def main():
    qc = AscendW8A8MXFP8Config(bits=8, group_size=32, is_quanted=True)
    attn_config = AttnConfig(hidden_size=HIDDEN, size_per_head=32, head_num=4, head_num_kv=2)
    ffn_config = FfnConfig(is_gated_activation=True, align_size=0)
    moe_config = MoeConfig(expert_num=N_EXPERTS, align_size=0)
    lin_config = LinearAttnConfig.__new__(LinearAttnConfig)
    lin_config.linear_num_key_heads = 4
    lin_config.linear_num_value_heads = 4
    lin_config.linear_key_head_dim = 32
    lin_config.linear_value_head_dim = 32

    results = {}

    # ---------------- dense: attn_o_w (process_fun=transpose) ----------------
    N_o, K_o = 128, HIDDEN
    w_bf16 = torch.randn(N_o, K_o, dtype=torch.bfloat16)
    kernel, scale = quantize_module(w_bf16)
    tensors = {
        PREFIX + "layers.0.self_attn.o_proj.weight": kernel,
        PREFIX + "layers.0.self_attn.o_proj.weight_scale": scale,
    }
    weight = AttnAtomicWeight(
        W.attn_o_w,
        [__import__("rtp_llm.utils.model_weight", fromlist=["CkptWeightInfo"]).CkptWeightInfo(
            PREFIX + "layers.{i}.self_attn.o_proj.weight")],
        process_fun=transpose,
        config=attn_config,
    )
    res = load_and_check(weight, tensors, qc, {
        W.attn_o_w: ((K_o, N_o), torch.float8_e4m3fn),
        W.attn_o_s: ((K_o // GROUP // 2, N_o, 2), torch.uint8),
    })
    w_rec = e8m0_dequant(res[W.attn_o_w], res[W.attn_o_s])
    cs = cos_sim(w_rec, w_bf16)
    print(f"[dense attn_o_w] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99
    results["attn_o_w"] = cs

    # ---------------- dense: attn_qkv_w (merge_qkv_hf) ----------------
    N_q, N_k, N_v = 128, 64, 64
    q = torch.randn(N_q, HIDDEN, dtype=torch.bfloat16)
    k = torch.randn(N_k, HIDDEN, dtype=torch.bfloat16)
    v = torch.randn(N_v, HIDDEN, dtype=torch.bfloat16)
    qk, qs = quantize_module(q)
    kk, ks = quantize_module(k)
    vk, vs = quantize_module(v)
    CK = __import__("rtp_llm.utils.model_weight", fromlist=["CkptWeightInfo"]).CkptWeightInfo
    tensors = {
        PREFIX + "layers.0.self_attn.q_proj.weight": qk,
        PREFIX + "layers.0.self_attn.q_proj.weight_scale": qs,
        PREFIX + "layers.0.self_attn.k_proj.weight": kk,
        PREFIX + "layers.0.self_attn.k_proj.weight_scale": ks,
        PREFIX + "layers.0.self_attn.v_proj.weight": vk,
        PREFIX + "layers.0.self_attn.v_proj.weight_scale": vs,
    }
    weight = AttnAtomicWeight(
        W.attn_qkv_w,
        [
            CK(PREFIX + "layers.{i}.self_attn.q_proj.weight"),
            CK(PREFIX + "layers.{i}.self_attn.k_proj.weight"),
            CK(PREFIX + "layers.{i}.self_attn.v_proj.weight"),
        ],
        process_fun=merge_qkv_hf,
        config=attn_config,
    )
    N_total = N_q + N_k + N_v
    res = load_and_check(weight, tensors, qc, {
        W.attn_qkv_w: ((HIDDEN, N_total), torch.float8_e4m3fn),
        W.attn_qkv_s: ((HIDDEN // GROUP // 2, N_total, 2), torch.uint8),
    })
    w_rec = e8m0_dequant(res[W.attn_qkv_w], res[W.attn_qkv_s])  # [N_total, K]
    ref = torch.cat([q, k, v], dim=0)  # rows: q|k|v
    cs = cos_sim(w_rec, ref)
    print(f"[dense attn_qkv_w] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99
    results["attn_qkv_w"] = cs

    # ---------------- linear attn: in_proj_qkvz (cat dim0 + .T) ----------------
    N_qkv, N_z, K_qkvz = 128, 64, HIDDEN
    qkv = torch.randn(N_qkv, K_qkvz, dtype=torch.bfloat16)
    z_t = torch.randn(N_z, K_qkvz, dtype=torch.bfloat16)
    qkvv, qkvs = quantize_module(qkv)
    zk, zs = quantize_module(z_t)
    tensors = {
        PREFIX + "layers.0.linear_attn.in_proj_qkv.weight": qkvv,
        PREFIX + "layers.0.linear_attn.in_proj_qkv.weight_scale": qkvs,
        PREFIX + "layers.0.linear_attn.in_proj_z.weight": zk,
        PREFIX + "layers.0.linear_attn.in_proj_z.weight_scale": zs,
    }
    weight = LinearAttnAtomicWeight(
        W.linear_attn_qkvz_w,
        [
            CK(PREFIX + "layers.{i}.linear_attn.in_proj_qkv.weight"),
            CK(PREFIX + "layers.{i}.linear_attn.in_proj_z.weight"),
        ],
        functools.partial(merge_qkvz_transpose_reorder, linear_attention_config=None),
        lin_config,
    )
    N_qkvz = N_qkv + N_z
    res = load_and_check(weight, tensors, qc, {
        W.linear_attn_qkvz_w: ((K_qkvz, N_qkvz), torch.float8_e4m3fn),
        W.linear_attn_qkvz_s: ((K_qkvz // GROUP // 2, N_qkvz, 2), torch.uint8),
    })
    w_rec = e8m0_dequant(res[W.linear_attn_qkvz_w], res[W.linear_attn_qkvz_s])
    cs = cos_sim(w_rec, torch.cat([qkv, z_t], dim=0))
    print(f"[linear_attn_qkvz] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99
    results["linear_attn_qkvz"] = cs

    # ---------------- MoE stacked format ----------------
    # gate_up_proj: [E, 2N, K] (gate first half in ckpt), down_proj: [E, N, K]
    moe_w1_ckpt = torch.randn(N_EXPERTS, 2 * INTER, HIDDEN, dtype=torch.bfloat16)
    moe_w2_ckpt = torch.randn(N_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16)
    # quantize each expert slice along K (last dim)
    def quantize_3d(t):
        e, n, kk = t.shape
        kern = torch.empty(e, n, kk, dtype=torch.float8_e4m3fn)
        sc = torch.empty(e, n, kk // GROUP, dtype=torch.uint8)
        for i in range(e):
            kern[i], sc[i] = quantize_module(t[i])
        return kern, sc

    w1_k, w1_s = quantize_3d(moe_w1_ckpt)
    w2_k, w2_s = quantize_3d(moe_w2_ckpt)
    tensors = {
        PREFIX + "layers.0.mlp.experts.gate_up_proj.weight": w1_k,
        PREFIX + "layers.0.mlp.experts.gate_up_proj.weight_scale": w1_s,
        PREFIX + "layers.0.mlp.experts.down_proj.weight": w2_k,
        PREFIX + "layers.0.mlp.experts.down_proj.weight_scale": w2_s,
    }
    w1_weight = MoeAtomicWeight(
        W.moe_w1,
        [CK(PREFIX + "layers.{i}.mlp.experts.gate_up_proj.weight")],
        process_fun=transpose_stack_moe_w1,
        config=moe_config,
        stacked_ckpt_keys=True,
    )
    res = load_and_check(w1_weight, tensors, qc, {
        W.moe_w1: ((N_EXPERTS, HIDDEN, 2 * INTER), torch.float8_e4m3fn),
        W.moe_s1: ((N_EXPERTS, HIDDEN // GROUP // 2, 2 * INTER, 2), torch.uint8),
    })
    # numeric check: dequant w1 and compare with up|gate-swapped ckpt
    w1_rec = torch.empty(N_EXPERTS, 2 * INTER, HIDDEN)
    for i in range(N_EXPERTS):
        w1_rec[i] = e8m0_dequant(res[W.moe_w1][i], res[W.moe_s1][i])
    # transpose_stack_moe_w1 swaps to up|gate; build the same reference
    ref_w1 = torch.cat([moe_w1_ckpt[:, INTER:, :], moe_w1_ckpt[:, :INTER, :]], dim=1)
    cs = cos_sim(w1_rec, ref_w1)
    print(f"[moe_w1 stacked] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99
    results["moe_w1"] = cs

    w2_weight = MoeAtomicWeight(
        W.moe_w2,
        [CK(PREFIX + "layers.{i}.mlp.experts.down_proj.weight")],
        process_fun=stack_,
        config=moe_config,
        stacked_ckpt_keys=True,
    )
    res = load_and_check(w2_weight, tensors, qc, {
        W.moe_w2: ((N_EXPERTS, INTER, HIDDEN), torch.float8_e4m3fn),
        W.moe_s2: ((N_EXPERTS, INTER // GROUP // 2, HIDDEN, 2), torch.uint8),
    })
    w2_rec = torch.empty(N_EXPERTS, HIDDEN, INTER)
    for i in range(N_EXPERTS):
        w2_rec[i] = e8m0_dequant(res[W.moe_w2][i], res[W.moe_s2][i])
    cs = cos_sim(w2_rec, moe_w2_ckpt)
    print(f"[moe_w2 stacked] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99
    results["moe_w2"] = cs

    # ---------------- MoE split format ----------------
    tensors = {}
    ref_w1_split = []
    for e in range(N_EXPERTS):
        for part, nm in ((0, "gate"), (1, "up")):
            t = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16)
            kern, sc = quantize_module(t)
            tensors[f"{PREFIX}layers.0.mlp.experts.{e}.{nm}_proj.weight"] = kern
            tensors[f"{PREFIX}layers.0.mlp.experts.{e}.{nm}_proj.weight_scale"] = sc
            ref_w1_split.append((part, e, t))
        t2 = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16)
        k2, s2 = quantize_module(t2)
        tensors[f"{PREFIX}layers.0.mlp.experts.{e}.down_proj.weight"] = k2
        tensors[f"{PREFIX}layers.0.mlp.experts.{e}.down_proj.weight_scale"] = s2
    w1_weight = MoeAtomicWeight(
        W.moe_w1,
        [CK(PREFIX + "layers.{i}.mlp.experts.{expert_id}.up_proj.weight")]
        + [CK(PREFIX + "layers.{i}.mlp.experts.{expert_id}.gate_proj.weight")],
        process_fun=stack_moe_w1,
        config=moe_config,
    )
    res = load_and_check(w1_weight, tensors, qc, {
        W.moe_w1: ((N_EXPERTS, HIDDEN, 2 * INTER), torch.float8_e4m3fn),
        W.moe_s1: ((N_EXPERTS, HIDDEN // GROUP // 2, 2 * INTER, 2), torch.uint8),
    })
    print("[moe_w1 split] shapes OK")

    print("\nALL W8A8_MXFP8 WEIGHT LOADING TESTS PASSED")
    print("cos_sims:", {k: round(v, 6) for k, v in results.items()})


if __name__ == "__main__":
    sys.exit(main())
