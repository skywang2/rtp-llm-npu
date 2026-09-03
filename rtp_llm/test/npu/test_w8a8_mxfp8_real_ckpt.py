"""Real ModelSlim W8A8_MXFP8 ckpt validation on Ascend NPU (TODO 4.1), pytest version.

Uses the REAL ModelSlim-quantized Qwen3-0.6B ckpt instead of synthetic tensors:
  CKPT: /workspace/weights/Qwen3-0.6B-W8A8-MXFP8   (fp8 kernel + uint8 E8M0 scale)
  BF16: /workspace/weights/Qwen3-0.6B              (original, numeric reference)

Validates:
  1. quant_model_description.json detection -> AscendW8A8MXFP8Config + exclude_modules
  2. Full loading chain on real tensors: _load_raw_tensor -> _split ->
     _postprocess (NPU real transpose + E8M0 swizzle)
  3. Numeric round-trip vs the original bf16 model.

NOTE on ModelSlim anti_outlier smoothing (verified against this ckpt):
  q/k/v/gate/up weights are stored SMOOTHED: W' = W * diag(s), with the inverse
  factor fused into the preceding RMSNorm (ln' = ln / s); o_proj / down_proj are
  NOT smoothed. Hence dequant(W') != W for q/k/v/gate/up; the correct equivalence
  is (x * ln') @ W'^T == (x * ln) @ W^T — exactly what inference computes.
  attn_qkv_w is therefore validated with this end-to-end math check instead of a
  raw weight comparison (raw comparison gives cos_sim ~0.946 by design).

Runs INSIDE the rtp-llm container (needs torch_npu + the two ckpt dirs):
  docker exec rtp-llm bash -c "cd /mnt/docker/w30060538/rtp-llm-npu && \
    PYTHONPATH=/mnt/docker/w30060538/rtp-llm-npu \
    /root/miniconda3/envs/rtp-env/bin/python -m pytest \
    rtp_llm/test/npu/test_w8a8_mxfp8_real_ckpt.py -v -s"
"""

import functools
import json
import os

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from rtp_llm.config.quant_config import AscendW8A8MXFP8Config, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.w8a8_mxfp8_weight import AscendW8A8MXFP8Weight
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
    transpose_pad,
)

torch.npu.set_device(0)

CKPT = "/workspace/weights/Qwen3-0.6B-W8A8-MXFP8"
BF16_DIR = "/workspace/weights/Qwen3-0.6B"
GROUP = 32

HIDDEN = 1024
INTER = 3072
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
PREFIX = "model.layers.{i}."


class RealTensorSource(TensorSource):
    """Loads tensors from the sharded safetensors ckpt via the index."""

    def __init__(self, ckpt_dir: str):
        index_path = os.path.join(ckpt_dir, "quant_model_weights.safetensors.index.json")
        with open(index_path) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._dir = ckpt_dir

    def load_tensor(self, name: str, data_type=torch.float16):
        fname = self._weight_map[name]
        with safe_open(os.path.join(self._dir, fname), framework="pt") as fh:
            t = fh.get_tensor(name)
        return [t.to(data_type)]

    def has_tensor(self, name: str) -> bool:
        return name in self._weight_map

    def get_database(self):
        return None


class StubDevice:
    def maybe_rewrite_weight_by_key(self, key, weight):
        return weight

    def shuffle_moe_weight(self, x, datatype, name):
        return x


def make_load_config():
    from unittest.mock import MagicMock

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
    lc.head_num = HEADS
    lc.head_num_kv = KV_HEADS
    lc.size_per_head = HEAD_DIM
    lc.exported_device = StubDevice()
    return lc


def e8m0_dequant(kernel: torch.Tensor, scale_swizzled: torch.Tensor) -> torch.Tensor:
    """Dequantize [K, N] fp8 kernel with swizzled [kp//2, N, 2] E8M0 scale."""
    kp2, n, two = scale_swizzled.shape
    assert two == 2
    scale_2d = scale_swizzled.permute(1, 0, 2).reshape(n, kp2 * 2)  # [N, kp]
    exp = torch.pow(2.0, scale_2d.float() - 127.0)  # [N, kp]
    exp_full = exp.repeat_interleave(GROUP, dim=1)  # [N, K]
    return kernel.float().T * exp_full  # [N, K]


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


_BF16_HANDLE = None


def bf16_ref(name: str) -> torch.Tensor:
    """Load a tensor from the original bf16 model (CPU, bf16)."""
    global _BF16_HANDLE
    if _BF16_HANDLE is None:
        _BF16_HANDLE = safe_open(os.path.join(BF16_DIR, "model.safetensors"), framework="pt")
    return _BF16_HANDLE.get_tensor(name)


def load_and_check(weight, qc, tensor_source, layer_id, expected):
    """Create the quant wrapper, load layer `layer_id`, assert shape/dtype/device."""
    qw = WeightModule.create(weight, qc)
    assert isinstance(qw, AscendW8A8MXFP8Weight), type(qw)
    res = qw.load(tensor_source, layer_id=layer_id, device="npu",
                  load_config=make_load_config())
    for name, (shape, dtype) in expected.items():
        assert name in res, f"{name} missing from {list(res.keys())}"
        got = res[name]
        assert tuple(got.shape) == tuple(shape), f"{name}: {tuple(got.shape)} != {tuple(shape)}"
        assert got.dtype == dtype, f"{name}: {got.dtype} != {dtype}"
        assert got.device.type == "npu"
    return res


# ---------------- fixtures ----------------


@pytest.fixture(scope="module")
def qc():
    return QuantizationConfig.load_from_ckpt(CKPT)


@pytest.fixture(scope="module")
def tensor_source():
    return RealTensorSource(CKPT)


@pytest.fixture(scope="module")
def attn_config():
    return AttnConfig(hidden_size=HIDDEN, size_per_head=HEAD_DIM,
                      head_num=HEADS, head_num_kv=KV_HEADS)


@pytest.fixture(scope="module")
def ffn_config():
    return FfnConfig(is_gated_activation=True, align_size=0)


# ---------------- tests ----------------


def test_config_detection(qc):
    assert isinstance(qc, AscendW8A8MXFP8Config), f"got {type(qc)}"
    assert qc.is_quanted() and qc.group_size() == 32
    # FLOAT entries must be collected as concrete excludes
    for probe in [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.2.mlp.down_proj.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
    ]:
        assert probe in qc.exclude_modules, f"{probe} missing from excludes"
    # quantized layers must NOT be excluded
    assert "model.layers.5.mlp.down_proj.weight" not in qc.exclude_modules
    assert "model.layers.0.self_attn.q_proj.weight" not in qc.exclude_modules
    print(f"[config] load_from_ckpt OK (excludes={len(qc.exclude_modules)}, "
          f"partial-quant down_proj L0-2 detected)")


def test_attn_o_w(qc, tensor_source, attn_config):
    """Single-tensor dense path (process_fun=transpose), raw weight comparison."""
    weight = AttnAtomicWeight(
        W.attn_o_w,
        [CkptWeightInfo(PREFIX + "self_attn.o_proj.weight")],
        process_fun=transpose,
        config=attn_config,
    )
    res = load_and_check(weight, qc, tensor_source, layer_id=0, expected={
        W.attn_o_w: ((2048, HIDDEN), torch.float8_e4m3fn),      # [K, N]
        W.attn_o_s: ((2048 // GROUP // 2, HIDDEN, 2), torch.uint8),
    })
    ref = bf16_ref("model.layers.0.self_attn.o_proj.weight")     # [N=1024, K=2048]
    cs = cos_sim(e8m0_dequant(res[W.attn_o_w], res[W.attn_o_s]), ref)
    print(f"[real attn_o_w] dequant cos_sim vs bf16 ckpt = {cs:.6f}")
    assert cs > 0.99, cs


def test_attn_qkv_w(qc, tensor_source, attn_config):
    """q/k/v merge path; smoothed weights -> end-to-end math comparison."""
    weight = AttnAtomicWeight(
        W.attn_qkv_w,
        [CkptWeightInfo(PREFIX + "self_attn.q_proj.weight"),
         CkptWeightInfo(PREFIX + "self_attn.k_proj.weight"),
         CkptWeightInfo(PREFIX + "self_attn.v_proj.weight")],
        process_fun=merge_qkv_hf,
        config=attn_config,
    )
    n_q, n_kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    res = load_and_check(weight, qc, tensor_source, layer_id=0, expected={
        W.attn_qkv_w: ((HIDDEN, n_q + 2 * n_kv), torch.float8_e4m3fn),  # [K, N_total]
        W.attn_qkv_s: ((HIDDEN // GROUP // 2, n_q + 2 * n_kv, 2), torch.uint8),
    })
    ref = torch.cat([bf16_ref(f"model.layers.0.self_attn.{p}_proj.weight")
                     for p in ("q", "k", "v")], dim=0)  # [N_total, K]
    # q/k/v are anti_outlier-smoothed (W' = W*diag(s), ln' = ln/s): validate via
    # the end-to-end math that inference computes, not raw weight comparison.
    w_rec = e8m0_dequant(res[W.attn_qkv_w], res[W.attn_qkv_s]).cpu()   # [N_total, K]
    ln_ck = tensor_source.load_tensor(
        "model.layers.0.input_layernorm.weight", torch.float32)[0].float()
    ln_hf = bf16_ref("model.layers.0.input_layernorm.weight").float()
    torch.manual_seed(0)
    x = torch.randn(8, HIDDEN)
    cs = cos_sim((x * ln_ck.unsqueeze(0)) @ w_rec.T.float(),
                 (x * ln_hf.unsqueeze(0)) @ ref.T.float())
    print(f"[real attn_qkv_w] e2e math cos_sim (x*ln')@W'^T vs (x*ln)@W^T = {cs:.6f}")
    assert cs > 0.99, cs


def test_ffn_w2_quantized(qc, tensor_source, ffn_config):
    """down_proj of a quantized layer (L5: W8A8_MXFP8), raw weight comparison."""
    weight = FfnAtomicWeight(
        W.ffn_w2,
        [CkptWeightInfo(PREFIX + "mlp.down_proj.weight", identity)],
        process_fun=functools.partial(transpose_pad, align_size=0, dim=1),
        config=ffn_config,
    )
    res = load_and_check(weight, qc, tensor_source, layer_id=5, expected={
        W.ffn_w2: ((INTER, HIDDEN), torch.float8_e4m3fn),   # [K=inter, N=hidden]
        W.ffn_s2: ((INTER // GROUP // 2, HIDDEN, 2), torch.uint8),
    })
    # down_proj is NOT anti_outlier-smoothed (factor fused through silu is
    # impossible), so direct raw weight comparison is valid here.
    ref = bf16_ref("model.layers.5.mlp.down_proj.weight")   # [N=hidden, K=inter]
    cs = cos_sim(e8m0_dequant(res[W.ffn_w2], res[W.ffn_s2]), ref)
    print(f"[real ffn_w2 L5] dequant cos_sim vs bf16 ckpt = {cs:.6f}")
    assert cs > 0.99, cs


def test_ffn_w2_float_fallback(qc, tensor_source, ffn_config):
    """down_proj of a FLOAT layer (L0: excluded) must fall back to bf16 loading."""
    weight = FfnAtomicWeight(
        W.ffn_w2,
        [CkptWeightInfo(PREFIX + "mlp.down_proj.weight", identity)],
        process_fun=functools.partial(transpose_pad, align_size=0, dim=1),
        config=ffn_config,
    )
    qw = WeightModule.create(weight, qc)
    # wrapping happens for all layers; the FLOAT fallback is decided at load time
    assert isinstance(qw, AscendW8A8MXFP8Weight), type(qw)
    res = qw.load(tensor_source, layer_id=0, device="npu",
                  load_config=make_load_config())
    assert W.ffn_w2 in res, list(res.keys())
    assert W.ffn_s2 not in res, "excluded layer must not produce a scale tensor"
    got = res[W.ffn_w2]
    assert got.dtype == torch.bfloat16, got.dtype
    assert tuple(got.shape) == (INTER, HIDDEN), tuple(got.shape)
    ref = bf16_ref("model.layers.0.mlp.down_proj.weight").float()   # [N, K]
    cs = cos_sim(got.cpu(), ref.T)                                  # loaded [K, N]
    print(f"[real ffn_w2 L0 fallback] bf16, no scale, cos_sim vs bf16 ckpt = {cs:.6f}")
    assert cs > 0.999, cs


def test_ffn_composite_float_fallback(qc, tensor_source, ffn_config):
    """E2E-shape regression: FfnWeight composite calls sub-weight loading stages
    (_load_raw_tensor/_split/_postprocess) directly, bypassing load(). The L0
    FLOAT down_proj must fall back to bf16 through that nested path too."""
    from rtp_llm.model_loader.ffn_weight import FfnWeight

    w1 = FfnAtomicWeight(
        W.ffn_w1,
        [CkptWeightInfo(PREFIX + "mlp.up_proj.weight", identity)],
        process_fun=functools.partial(transpose_pad, align_size=0, dim=0),
        config=ffn_config,
    )
    w3 = FfnAtomicWeight(
        W.ffn_w3,
        [CkptWeightInfo(PREFIX + "mlp.gate_proj.weight", identity)],
        process_fun=functools.partial(transpose_pad, align_size=0, dim=0),
        config=ffn_config,
    )
    w2 = FfnAtomicWeight(
        W.ffn_w2,
        [CkptWeightInfo(PREFIX + "mlp.down_proj.weight", identity)],
        process_fun=functools.partial(transpose_pad, align_size=0, dim=1),
        config=ffn_config,
    )
    ffn = WeightModule.create(
        FfnWeight([w1, w3, w2], config=ffn_config), qc
    )
    res = ffn.load(tensor_source, layer_id=0, device="npu",
                   load_config=make_load_config())
    # w13 (up/gate) is quantized in every layer; w2 must be bf16 at L0
    assert res[W.ffn_w13].dtype == torch.float8_e4m3fn, res[W.ffn_w13].dtype
    assert W.ffn_s13 in res and W.ffn_s2 not in res, sorted(res.keys())
    assert res[W.ffn_w2].dtype == torch.bfloat16, res[W.ffn_w2].dtype
    assert tuple(res[W.ffn_w2].shape) == (INTER, HIDDEN), tuple(res[W.ffn_w2].shape)
    ref = bf16_ref("model.layers.0.mlp.down_proj.weight").float().T   # [K, N]
    cs = cos_sim(res[W.ffn_w2].cpu(), ref)
    print(f"[real ffn composite L0] w13 fp8 + w2 bf16 fallback, "
          f"w2 cos_sim vs bf16 ckpt = {cs:.6f}")
    assert cs > 0.999, cs
