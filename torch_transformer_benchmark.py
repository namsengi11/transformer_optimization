#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as _tl_libdevice
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when Triton is absent/broken
    triton = None  # type: ignore
    tl = None  # type: ignore
    _tl_libdevice = None  # type: ignore
    _TRITON_AVAILABLE = False


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


# Step 21 capacity protocol, calibrated by
# agent/experiments/21-long-sequence-streaming/capacity-b64-s128.json,
# capacity-b8-s1024.json, capacity-b1-s8192.json, capacity-b1-s16384.json,
# and capacity-row6-l1.json. Equal
# B*S=8192 shapes peaked at 76 MB (S=128), 312 MB (S=1024), and 2.26 GB
# (S=8192), proving that a token-count-only gate misses quadratic attention.
# B=1/S=16384 peaked at 8.93 GB allocated / 12.97 GB reserved, while the
# row-6-like B=10000/S=128 case peaked at 10.51 GB allocated and remains
# dense on this 16 GiB card. 2*score + 6*input + 2*ffn tracks those peaks;
# the original safe admission boundary was 65% of installed VRAM. Step 23
# raises the measured-memory ceiling to 85% and derives both streaming axes
# from the shape and the remaining capacity. Never use current free memory:
# routing and tiling must not depend on neighbouring processes.
_STREAMING_MEMORY_BUDGET_FRAC = 0.85
# Isolated step-23 sweeps found that fixed score/probability workspaces reserve
# up to about 1.5x their tensor bytes once allocator block rounding and GEMM workspace
# are included. This calibrated multiplier is shape-independent; the remaining
# terms below still scale from the requested shape and dtype.
_STREAMING_REFERENCE_WORKSPACE_RESERVE = 1.50
# Dense matrix rows reach their stable throughput region around 8192 tokens
# per launch (for example B=64,S=128). More batch above that point needlessly
# raises residency for already-long sequences and can reduce per-token latency.
_STREAMING_OPTIMIZED_TOKEN_TARGET = 8192


def _capacity_terms_bytes(
    config: TransformerConfig, element_bytes: int
) -> Tuple[int, int, int]:
    rows = config.batch_size * config.seq_len
    linear_bytes = rows * config.d_model * element_bytes
    score_bytes = (
        config.batch_size
        * config.num_heads
        * config.seq_len
        * config.seq_len
        * element_bytes
    )
    ffn_bytes = rows * config.ffn_dim * element_bytes
    estimated_dense_bytes = 6 * linear_bytes + 2 * score_bytes + 2 * ffn_bytes
    return linear_bytes, score_bytes, estimated_dense_bytes


def _installed_cuda_memory_bytes(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        return None


def _capacity_streaming_required(
    config: TransformerConfig, dtype: torch.dtype, device: torch.device
) -> bool:
    total_bytes = _installed_cuda_memory_bytes(device)
    if total_bytes is None:
        return False
    element_bytes = torch.empty((), dtype=dtype).element_size()
    _, _, estimated_dense_bytes = _capacity_terms_bytes(config, element_bytes)
    return estimated_dense_bytes > int(total_bytes * _STREAMING_MEMORY_BUDGET_FRAC)


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class ChunkedBaselineTransformerBlock(BaselineTransformerBlock):
    """BaselineTransformerBlock with the FFN sub-step computed in row-chunks
    along the flattened (batch*seq) dimension, to bound peak memory for very
    large ``batch*seq*ffn_dim`` intermediates. Canonical matrix row 14
    (batch=32, seq=100000, ffn_dim=1024) would materialize a 12.2 GiB GELU
    intermediate in one shot, although its much larger dense-attention score
    tensor causes the benchmark driver to preflight-block that row first.

    In exact (real-number) arithmetic this changes nothing: LayerNorm/GELU/
    the two FFN Linears are all strictly row-wise (no cross-token mixing,
    unlike attention), so slicing the input into row-chunks, running each
    through ffn_in -> gelu -> ffn_out, and concatenating the results is the
    same computation as running it in one shot. In FLOATING-POINT arithmetic
    it is NOT automatically bit-exact, though: cuBLAS can select a different
    GEMM kernel/reduction order depending on the row (M) count, exactly the
    class of bug this codebase already hit once for softmax (see
    _fused_attn_probs's docstring on the dtype=torch.float32 divergence at
    long sequences). Measured directly: at d_model=512/ffn_dim=2048, chunk
    sizes 256/512 are bit-exact but 64/128 are not (max_abs ~6e-4); at
    d_model=1024/ffn_dim=100000 (the pre-2026-08-31 misdecoded row-14 shape),
    128 through 4096 are all bit-exact but 64 is not (max_abs ~2.6e-4).
    The relationship is NOT a simple "big enough is safe" threshold -- a
    d_model=1024/ffn_dim=8192 sweep found 128 exact, then 256/512/1024/
    2048/4096 all INexact (~3-4e-4), then 8192+ exact again. So the default
    below (4096) is an empirically-checked-for-these-shapes choice, not a
    proven-safe-in-general one: verify bit-exactness against
    BaselineTransformer at a feasible reduced-size shape (same d_model and
    ffn_dim, smaller batch/seq) before trusting a new (d_model, ffn_dim,
    chunk_size) combination, the same way this one was checked. In
    particular, that legacy measurement is not correctness evidence for the
    canonical row-14 ffn_dim=1024 shape.

    Reuses norm1/attention/norm2/ffn_in/ffn_out from BaselineTransformerBlock
    unchanged (no new parameters), so state_dict() keys and copy_model_weights
    are unaffected.
    """

    def __init__(
        self, d_model: int, num_heads: int, ffn_dim: int, ffn_chunk_size: int = 4096
    ) -> None:
        super().__init__(d_model, num_heads, ffn_dim)
        if ffn_chunk_size <= 0:
            raise ValueError("ffn_chunk_size must be positive")
        self.ffn_chunk_size = ffn_chunk_size

    def _chunked_ffn(self, h: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = h.shape
        flat = h.reshape(batch * seq_len, d_model)
        chunks = []
        for start in range(0, flat.shape[0], self.ffn_chunk_size):
            piece = flat[start : start + self.ffn_chunk_size]
            chunks.append(
                self.ffn_out(F.gelu(self.ffn_in(piece), approximate="none"))
            )
        return torch.cat(chunks, dim=0).reshape(batch, seq_len, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self._chunked_ffn(self.norm2(x))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class ChunkedBaselineTransformer(BaselineTransformer):
    """BaselineTransformer with every block's FFN computed in row-chunks.
    Identical parameters/state_dict keys to BaselineTransformer; only used
    when explicitly requested (--chunk-baseline-ffn), never by default, so
    every existing measurement against the plain BaselineTransformer is
    unaffected by this class's existence.
    """

    def __init__(self, config: TransformerConfig, ffn_chunk_size: int = 4096) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                ChunkedBaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim, ffn_chunk_size
                )
                for _ in range(config.num_layers)
            ]
        )


class QueryTiledBaselineSelfAttention(BaselineSelfAttention):
    """Dense reference attention with bounded score/probability residency.

    K and V remain complete while only query rows are tiled. Every query is
    still compared with every permitted key, so this preserves full causal
    or padded attention. Parameters and state-dict keys are unchanged.
    """

    def __init__(self, d_model: int, num_heads: int, query_tile_size: int) -> None:
        super().__init__(d_model, num_heads)
        if query_tile_size <= 0:
            raise ValueError("query_tile_size must be positive")
        self.query_tile_size = query_tile_size

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        # Fixed-capacity workspaces prevent the CUDA allocator from retaining
        # hundreds of progressively larger causal score/probability blocks.
        # Only the active [:queries, :keys] view participates in each tile.
        score_workspace = torch.empty(
            batch,
            self.num_heads,
            self.query_tile_size,
            seq_len,
            device=x.device,
            dtype=x.dtype,
        )
        context = torch.empty_like(q)

        for start in range(0, seq_len, self.query_tile_size):
            end = min(start + self.query_tile_size, seq_len)
            # Keep the key extent fixed across tiles. This intentionally does
            # extra causal-upper-triangle work: progressively changing key
            # extents fragment the CUDA allocator, while a preallocated
            # softmax output adds a third live score-sized buffer because
            # ATen's out= path still needs an internal result. The stable
            # two-buffer form below reuses one fixed score allocation and one
            # fixed-shape softmax result through the caching allocator.
            key_end = seq_len
            query_count = end - start
            scores = score_workspace[:, :, :query_count, :key_end]
            torch.matmul(
                q[:, :, start:end],
                k[:, :, :key_end].transpose(-2, -1),
                out=scores,
            )
            scores.mul_(self.scale)
            if causal:
                query_positions = torch.arange(start, end, device=x.device)[:, None]
                key_positions = torch.arange(key_end, device=x.device)[None, :]
                scores.masked_fill_(key_positions > query_positions, float("-inf"))
            if valid_token_mask is not None:
                scores.masked_fill_(
                    ~valid_token_mask[:, None, None, :key_end], float("-inf")
                )
            probs_float = torch.softmax(scores.float(), dim=-1)
            if x.dtype == torch.float32:
                probs = probs_float
            else:
                # Scores are dead after softmax, so reuse their fixed storage
                # for the model-dtype probabilities instead of allocating a
                # differently sized cast result for every causal tile.
                scores.copy_(probs_float)
                probs = scores
            context[:, :, start:end].copy_(
                torch.matmul(probs, v[:, :, :key_end])
            )

        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class QueryTiledBaselineTransformer(ChunkedBaselineTransformer):
    """Capacity-bounded eager reference with an unchanged state dict."""

    def __init__(
        self,
        config: TransformerConfig,
        query_tile_size: int,
        ffn_chunk_size: int = 4096,
    ) -> None:
        super().__init__(config, ffn_chunk_size=ffn_chunk_size)
        for layer in self.layers:
            layer.attention = QueryTiledBaselineSelfAttention(
                config.d_model, config.num_heads, query_tile_size
            )


def _get_fused_qkv_weights(attn: "BaselineSelfAttention") -> Tuple[torch.Tensor, torch.Tensor]:
    """Lazily fuse an attention module's q/k/v projection weight+bias into a
    single packed [3*d_model, d_model] weight and [3*d_model] bias, so that
    one GEMM (F.linear) can replace three separate ones.

    The packed tensors are cached as a plain (non-parameter, non-buffer)
    attribute on the attention module and rebuilt only if the source
    parameters' storage/dtype/device change (e.g. after copy_model_weights
    followed by .to(device, dtype), which happens once before the first
    forward call). This keeps state_dict()/load_state_dict() untouched:
    no new nn.Parameter or persistent buffer is registered.
    """
    q_w, k_w, v_w = attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
    q_b, k_b, v_b = attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias

    cache_key = (
        q_w.data_ptr(), k_w.data_ptr(), v_w.data_ptr(),
        q_b.data_ptr(), k_b.data_ptr(), v_b.data_ptr(),
        q_w.dtype, q_w.device,
    )
    cached = getattr(attn, "_fused_qkv_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    fused_weight = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
    fused_bias = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
    attn._fused_qkv_cache = (cache_key, fused_weight, fused_bias)
    return fused_weight, fused_bias


def _get_linear_fp16_weights(linear: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lazily cache fp16 copies of an nn.Linear's weight+bias, used only by
    UserOptimizedTransformer's fp32-model fp16-GEMM path (see
    _forward_core's use_fp16_gemm branch). Cached as a plain (non-parameter,
    non-buffer) attribute on the Linear module itself -- the same strategy
    _get_fused_qkv_weights uses -- so weights are cast to fp16 once and
    rebuilt only if the source weight/bias identity, dtype, or device
    changes, never every forward call."""
    w, b = linear.weight, linear.bias
    cache_key = (w.data_ptr(), w.dtype, w.device, b.data_ptr(), b.dtype, b.device)
    cached = getattr(linear, "_fp16_gemm_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]
    w16 = w.to(torch.float16).contiguous()
    b16 = b.to(torch.float16).contiguous()
    linear._fp16_gemm_cache = (cache_key, w16, b16)
    return w16, b16


def _get_fused_qkv_weights_fp16(attn: "BaselineSelfAttention") -> Tuple[torch.Tensor, torch.Tensor]:
    """fp16 copy of the fused QKV weight/bias produced by
    _get_fused_qkv_weights, used only by the fp32 model's fp16-GEMM path.
    Cached separately from (but keyed off the identity of) the fp32 fused
    tensors, so it is rebuilt exactly when those are rebuilt -- and, like
    _get_fused_qkv_weights itself, costs nothing beyond the first forward
    call for a given set of source weights."""
    fused_weight, fused_bias = _get_fused_qkv_weights(attn)
    cache_key = (fused_weight.data_ptr(), fused_bias.data_ptr())
    cached = getattr(attn, "_fused_qkv_fp16_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]
    w16 = fused_weight.to(torch.float16).contiguous()
    b16 = fused_bias.to(torch.float16).contiguous()
    attn._fused_qkv_fp16_cache = (cache_key, w16, b16)
    return w16, b16


def _split_heads_3(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B, S, D] x3 -> contiguous [B, H, S, D/H] x3.

    Pure data movement, so it is bit-exact by construction no matter who
    generates the kernel -- which is what makes it safe to hand to
    torch.compile (see _get_compiled_layout_helpers). The explicit
    .contiguous() is not an extra copy: without it torch.matmul materializes
    exactly the same contiguous tensors internally (the head transpose is
    not expressible as a batched-GEMM stride, since the batch and head
    strides are not uniform over a single flattened batch dimension)."""
    batch, seq_len, d_model = q.shape
    head_dim = d_model // num_heads
    return (
        q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous(),
        k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous(),
        v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous(),
    )


def _merge_heads(context: torch.Tensor) -> torch.Tensor:
    """[B, H, S, D/H] -> [B, S, D]. Pure data movement, see _split_heads_3."""
    batch, num_heads, seq_len, head_dim = context.shape
    return context.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)


# Lazily-built process-wide torch.compile'd versions of the two layout
# helpers above. dict rather than module globals so the lazy build can be
# reset in one place; "disabled" latches True forever after any failure.
_COMPILED_LAYOUT: Dict[str, object] = {"split": None, "merge": None, "disabled": False}


def _get_compiled_layout_helpers():
    """Return (compiled_split_heads_3, compiled_merge_heads), or None.

    Why torch.compile is admissible here when it is banned everywhere else
    in this file: the two functions it compiles contain NO arithmetic at
    all, only view/transpose/contiguous/reshape. Inductor cannot change a
    rounding decision that is never made. Verified directly across fp16,
    bf16 and fp32 -- `torch.equal`, max_abs exactly 0 -- alongside the ops
    that do NOT survive the same test and are therefore deliberately left
    on ATen: layer_norm (fp16 max_abs 9.8e-4, bf16 3.9e-3), the fused
    add+layer_norm inductor would prefer to generate (fp16 7.8e-3 -- a full
    ulp class, exactly the drift that makes the compiled baseline
    non-compliant), and gelu (small but nonzero, 3.8e-6 fp16). Each callsite
    is still probed at runtime (_resolve_compiled_layout) rather than
    trusted.

    Why bother: ATen's `direct_copy_kernel` is an elementwise kernel with
    4-D index math and no vectorization on the strided side, and at the
    shapes this model runs it is launch/occupancy-bound rather than
    bandwidth-bound on a 36-SM card. Inductor emits a tiled copy instead.
    Measured under CUDA-graph replay (which is how this actually runs -- in
    eager wall-clock the comparison is swamped by dynamo's per-call guard
    overhead, which graph replay removes entirely), for the three q/k/v
    copies of one layer:

        shape              ATen     inductor   hand-written Triton
        B8  S128  D512    15.8us     8.2us          9.3us
        B32 S512  D512   293.9us   269.9us        515.7us
        B4  S2048 D512   154.4us   130.3us        252.6us
        B16 S256  D1024  150.1us   132.4us        253.3us

    i.e. inductor wins at every shape, and beats a hand-tuned Triton
    kernel written specifically for this GPU everywhere except the
    smallest shape -- the same lesson as the GEMM investigation in
    OPTIMIZATION_LOG.md step 7, in the opposite direction.

    The `use_static_cuda_launcher=False` patch is the Windows/torch-2.8 bug
    already recorded in that step: the static launcher overflows a C long
    with the 64-bit CUDA stream handle. It is NOT limited to max-autotune
    GEMM templates as previously assumed -- it fires on an ordinary
    pointwise clone too. The patch only has to be active while inductor
    first compiles in this process (verified: later compiles of new shapes,
    and all steady-state launches, work fine outside it), so it is scoped to
    the one-time probe and never costs anything per call."""
    if _COMPILED_LAYOUT["disabled"]:
        return None
    if _COMPILED_LAYOUT["split"] is None:
        try:
            _COMPILED_LAYOUT["split"] = torch.compile(
                _split_heads_3, dynamic=False, fullgraph=True
            )
            _COMPILED_LAYOUT["merge"] = torch.compile(
                _merge_heads, dynamic=False, fullgraph=True
            )
        except Exception:
            _COMPILED_LAYOUT["disabled"] = True
            return None
    return _COMPILED_LAYOUT["split"], _COMPILED_LAYOUT["merge"]


# Below this sequence length the block-triangular causal Q@K^T (see
# _blocked_causal_qk) costs more in extra kernel launches than it saves in
# skipped work -- measured at seq_len=128, where the blocked form runs at
# 0.26-0.56x the speed of one full GEMM.
_TRI_QK_MIN_SEQ = 256


def _causal_qk_block_size(seq_len: int) -> Optional[int]:
    """Query-block size for _blocked_causal_qk, or None to keep a single
    full GEMM.

    Chosen by measurement on this GPU (fp16 and bf16, head_dim=64), timing
    the blocked form against one torch.matmul at the shapes this codebase
    runs -- speedup of blocked vs full:

        seq_len   BR=64   BR=128   BR=256   BR=512
          128     0.26x    0.55x      --       --
          256     0.70x    1.28x    1.00x      --
          512     1.32x    1.31x    1.23x    0.96x
         2048     0.87x    1.74x    1.72x    1.52x   (fp16)
         2048     1.01x    1.82x    1.98x    1.59x   (bf16)

    Hence: nothing below 256; 128 in the middle, where the win comes from
    skipping work; 256 once rows are long enough that each block's own GEMM
    still wants to be big. Deliberately a coarse two-step function of
    seq_len rather than a fitted curve -- the surrounding numbers move with
    driver and cuBLAS versions, and _resolve_tri_qk_enabled probes the
    result for bit-exactness anyway.
    """
    if seq_len < _TRI_QK_MIN_SEQ:
        return None
    return 256 if seq_len >= 1024 else 128


def _blocked_causal_qk(
    q: torch.Tensor, kt: torch.Tensor, block: int, out: torch.Tensor
) -> torch.Tensor:
    """Causal Q@K^T that computes only the block-lower-triangle.

    For the query block [start, end), every key j >= end is strictly above
    the diagonal for EVERY query in that block, so
    _fused_scale_mask_kernel overwrites those columns with -inf regardless
    of what Q@K^T left there. They are therefore never computed, and
    `out`'s upper-triangular blocks keep whatever the caching allocator
    handed us.

    That is safe rather than merely untested: the kernel does load those
    positions, but discards the loaded value in the same `tl.where` that
    substitutes -inf, so not even a NaN can propagate out of them.
    Verified directly by filling `out` with NaN before the call and
    requiring torch.equal on both `probs` and the attention output.

    Bit-exactness turns on WHICH dimension gets shortened. Each block
    truncates the GEMM's N (the key axis); the reduction axis K stays at
    head_dim, untouched. Shortening a non-reduction dimension cannot
    reassociate anything, so every computed element is the identical dot
    product it was before -- verified torch.equal over the lower triangle
    at every shape and dtype tried.

    Shortening the REDUCTION axis is a different matter and is NOT done
    here: truncating probs@V's K breaks bit-exactness at seq_len=2048
    (fp16 7.8e-3, bf16 2.5e-1) while passing at 512 -- the same
    non-monotonic cuBLAS K-tiling sensitivity as the softmax kernel-
    selection bug in step 5 and the fused-QKV probe in step 8. So only
    Q@K^T is blocked; probs@V is left whole.
    """
    seq_len = q.shape[-2]
    for start in range(0, seq_len, block):
        end = min(start + block, seq_len)
        torch.matmul(
            q[:, :, start:end, :], kt[:, :, :, :end],
            out=out[:, :, start:end, :end],
        )
    return out


def _fast_path_gate_thresholds() -> Tuple[float, float]:
    """Base tolerances used to verify optional fused fast paths.

    Overridable via TJ_ATOL / TJ_RTOL; defaults match the harness. These
    thresholds are not used to select FP16-GEMM precision.
    """
    atol = float(os.environ.get("TJ_ATOL", "0.002"))
    rtol = float(os.environ.get("TJ_RTOL", "0.02"))
    return atol, rtol


# Optional fused kernels must clear a stricter bar than the public accuracy
# threshold before they are admitted. FP16-GEMM itself is selected directly
# and does not use this margin.
_FAST_PATH_GATE_MARGIN = 0.9


if _TRITON_AVAILABLE:
    # Block-size / warp-count search space for the fused scale+mask kernel
    # below, autotuned per distinct Sk. Deliberately wide (small blocks for
    # short rows like seq_len=64, large blocks to amortize the per-block
    # loop overhead for seq_len=2048) since this GPU has only 36 SMs --
    # autotuning picks whichever config actually keeps them fed for a given
    # shape rather than hardcoding one choice tuned for seq_len=128 only.
    _SCALE_MASK_CONFIGS = [
        triton.Config({"BLOCK_N": bn}, num_warps=nw)
        for bn in (64, 128, 256, 512, 1024, 2048)
        for nw in (2, 4, 8)
    ]

    @triton.autotune(configs=_SCALE_MASK_CONFIGS, key=["SK"])
    @triton.jit
    def _fused_scale_mask_kernel(
        scores_ptr, out_ptr, mask_ptr,
        H, SQ, SK,
        stride_sb, stride_sh, stride_sq, stride_sk,
        stride_ob, stride_oh, stride_oq, stride_ok,
        stride_mb, stride_mk,
        scale,
        CAUSAL: tl.constexpr,
        MASK_ACTIVE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fuses two of the attention epilogue's five ATen touches of the
        raw [B, H, Sq, Sk] QK^T score tensor -- the scale multiply and the
        (up to two) masked_fills -- into a single kernel that reads the raw
        scores once and writes the scaled+masked scores once, in the model
        dtype. The softmax itself is deliberately left to ATen's native
        `torch.softmax(scaled, dim=-1)` (called by _fused_attn_probs right
        after this kernel, directly on the half/bfloat16 output of this
        kernel -- no dtype= kwarg, no explicit .float() upcast; ATen's CUDA
        softmax already accumulates in fp32 internally and is bit-identical
        to an explicit-upcast reference, see _fused_attn_probs's docstring)
        rather than also being fused in here.

        That split is load-bearing, not an oversight: an earlier version of
        this kernel *did* fuse the softmax reduction too (single Triton
        kernel, online two-pass max/sum), and its output was only ever
        ~1e-4 to 1e-5 max_abs away from `torch.softmax(scores.float(),
        dim=-1)` per-call -- reduction-order and exp-evaluation noise
        inherent to it being a different implementation of the same
        reduction, not a logic bug. But at fp16/bf16 there is no headroom
        for that (see the class docstring's numerical-fidelity discussion:
        one fp16 ulp near magnitude 1 is ~4x the atol): compounded across
        even a single BaselineTransformerBlock, that per-call noise already
        consumed ~98% of the atol=0.002 budget (measured max_abs=0.00195312
        after just one layer), and across the default 6-layer stack it blew
        through it entirely (measured max_abs=0.0078125 -- a full fp16 ulp
        -- with real accuracy-check FAILures). A plain elementwise
        scale+mask fusion has no reduction in it at all, so it is exactly
        as bit-exact as the ops it replaces (verified: `torch.equal` against
        `(scores*scale).masked_fill(mask, -inf)`, not just close) while
        still cutting kernel launches (2-3 ATen ops -> 1) and traffic (2-3
        read+write passes over the [B,H,S,S] tensor -> 1) ahead of the
        softmax.

        Numerical fidelity with BaselineSelfAttention's arithmetic:
          - the raw score is upcast to fp32, multiplied by `scale`, then
            rounded back down to the model dtype -- reproducing the exact
            rounding of the eager chain's `scores * self.scale` (a
            Half/BFloat16 tensor times a Python float), not a
            higher-precision approximation of it.
          - masked (causal-disallowed or padding-invalid) positions are set
            to exactly -inf in the model dtype, matching
            `masked_fill(mask, -inf)`; both masks are combined into one
            `tl.where` chain per element instead of two successive
            masked_fills, verified bit-exact against the two-call form.
        """
        row_id = tl.program_id(0)
        hq = H * SQ
        b = row_id // hq
        rem = row_id % hq
        h = rem // SQ
        i = rem % SQ

        row_base_s = b * stride_sb + h * stride_sh + i * stride_sq
        row_base_o = b * stride_ob + h * stride_oh + i * stride_oq
        mask_row_base = b * stride_mb

        out_dtype = out_ptr.dtype.element_ty

        for start in range(0, SK, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            col_ok = cols < SK
            s = tl.load(scores_ptr + row_base_s + cols * stride_sk, mask=col_ok, other=0.0)
            v = (s.to(tl.float32) * scale).to(out_dtype)
            if CAUSAL:
                v = tl.where(cols <= i, v, -float("inf"))
            if MASK_ACTIVE:
                mvals = tl.load(mask_ptr + mask_row_base + cols * stride_mk, mask=col_ok, other=0)
                v = tl.where(mvals != 0, v, -float("inf"))
            tl.store(out_ptr + row_base_o + cols * stride_ok, v, mask=col_ok)


def _triton_fused_scale_mask(
    scores_raw: torch.Tensor,
    scale: float,
    causal: bool,
    mask_active: bool,
    valid_token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Launches _fused_scale_mask_kernel over every (batch, head, query) row
    of `scores_raw` ([B, H, Sq, Sk], the *unscaled* Q@K^T output in the
    model dtype) and returns the scaled+masked scores in that same dtype
    (softmax is still to come -- see _fused_attn_probs). Raises on any
    failure (unsupported shape, no CUDA, compile error) -- callers must
    catch and fall back to the ATen chain; this function itself never falls
    back silently so a real bug here can't masquerade as "shape
    unsupported"."""
    B, H, SQ, SK = scores_raw.shape
    out = torch.empty_like(scores_raw)
    if mask_active:
        assert valid_token_mask is not None
        stride_mb, stride_mk = valid_token_mask.stride()
        mask_arg = valid_token_mask
    else:
        # Unused inside the kernel (MASK_ACTIVE=False dead-code-eliminates
        # the load), but the launch still needs a real CUDA tensor argument
        # with a defined pointer/dtype -- reuse scores_raw to avoid an
        # allocation.
        stride_mb, stride_mk = 0, 0
        mask_arg = scores_raw

    grid = (B * H * SQ,)
    _fused_scale_mask_kernel[grid](
        scores_raw, out, mask_arg,
        H, SQ, SK,
        *scores_raw.stride(),
        *out.stride(),
        stride_mb, stride_mk,
        scale,
        CAUSAL=causal,
        MASK_ACTIVE=mask_active,
    )
    return out


# ---------------------------------------------------------------------------
# Shape-specialized fused residual-add + LayerNorm
# ---------------------------------------------------------------------------
#
# WHY A SHAPE-SPECIALIZED KERNEL IS THE RIGHT MOVE HERE, when steps 7 and 9 of
# OPTIMIZATION_LOG.md both concluded that hand-writing Triton was NOT:
#
# Those two steps tried to beat cuBLAS at GEMM and ATen/inductor at a strided
# copy on the *default* shape (d_model=512, ffn_dim=2048), where the workload
# is 85% GEMM and therefore compute-bound. The graded configurations are a
# completely different regime. Rows 1-13 use d_model and ffn_dim in
# {32, 128, 1024}, with most rows at 128; row 14 uses the same width set but
# an infeasible seq_len=100000. At (batch=64, seq=128, d_model=128,
# ffn=128, layers=4) the dense matmul cost is ~0.35 ms of a ~0.9 ms forward,
# and MEASURED profiles put LayerNorm alone at 35-42% of total GPU time
# (per-shape numbers in OPTIMIZATION_LOG.md step 11). ATen's layer_norm
# kernel is written for wide rows; at d_model=32 it moves 512 KB in 9.4 us,
# about 55 GB/s on a card that sustains several hundred. There is a large,
# measurable gap here that simply does not exist at d_model=512.
#
# What the given shapes buy us specifically:
#   * d_model <= 1024 and a power of two for every graded configuration, so
#     ONE Triton block holds an entire residual row. The LayerNorm reduction
#     becomes a single in-register pass -- no loop over the feature axis, no
#     two-pass "compute moments, then re-read the row" structure, no
#     cross-block communication.
#   * d_model that small also means a block can hold SEVERAL rows at once
#     (BLOCK_M), which is what actually fixes the d_model=32 case: one row
#     per block would launch 65536 blocks of 32 elements each.
#
# and what the fusion buys us, independent of shape: at every residual site
# the model currently runs
#       (optional fp16->fp32 cast) -> masked_fill -> add -> layer_norm
#         -> (optional fp32->fp16 cast)
# as four or five separate ATen kernels, each streaming the whole [B, S, D]
# activation. This kernel does all of it in one pass, reading x and delta
# once and writing x_new and the normalized output once.
#
# NUMERICS. Everything here except the LayerNorm reduction itself is exactly
# as bit-exact as the ops it replaces: an fp16->fp32 upcast is lossless, the
# residual add happens in fp32 in the same order, and masked_fill is
# elementwise. The mean/variance reduction is NOT bit-identical to ATen's
# Welford kernel -- a different reduction tree gives different last bits --
# so, exactly like every other non-bit-exact idea in this project, its use is
# decided by a runtime probe (_resolve_fused_ln_enabled) rather than
# asserted: bit-exactness is required outright on the fp16/bf16 paths, and
# the fp32 path is gated on the same margin-scaled harness criterion the
# fp16-GEMM gate uses, measured against the exact fp32 reference so the
# combined error of both approximations is what gets judged.
if _TRITON_AVAILABLE:

    @triton.jit
    def _add_ln_kernel(
        x_ptr, delta_ptr, xout_ptr, nout_ptr, w_ptr, b_ptr, valid_ptr,
        M, D, eps,
        HAS_DELTA: tl.constexpr,
        MASK_DELTA: tl.constexpr,
        MASK_X: tl.constexpr,
        WRITE_X: tl.constexpr,
        MASK_OUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One residual site: x_new = mask(x + mask(delta)); out = LN(x_new).

        Operates on the activation flattened to [M, D] with M = batch*seq_len
        (both inputs are contiguous [B, S, D], so the flatten is a view and
        the row stride is exactly D). Handles a BLOCK_M x BLOCK_D tile of
        rows per program; BLOCK_D >= D so the reduction axis is never tiled.

        The masking flags are separate rather than one "apply the padding
        mask" switch because BaselineTransformerBlock applies the padding
        mask at two different points with different meanings, and collapsing
        them would silently change the arithmetic:
          * MASK_DELTA zeroes invalid rows of `delta` BEFORE the add,
            reproducing BaselineSelfAttention's own trailing
            `output.masked_fill(~valid_token_mask[..., None], 0)`. It must
            not touch `x`, which is NOT masked at that point (in layer 0 `x`
            is the raw, unmasked model input).
          * MASK_X zeroes the residual AFTER the add, reproducing the block's
            trailing `x = x.masked_fill(...)`. The LayerNorm then reads the
            masked value, matching the baseline's op order.
          * MASK_OUT zeroes the normalized output, for the final_norm site
            where the baseline masks after normalizing.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        rm = rows < M
        cm = cols < D
        tile = rm[:, None] & cm[None, :]
        offs = rows[:, None] * D + cols[None, :]

        x = tl.load(x_ptr + offs, mask=tile, other=0.0).to(tl.float32)
        if HAS_DELTA:
            d = tl.load(delta_ptr + offs, mask=tile, other=0.0).to(tl.float32)
            if MASK_DELTA:
                v = tl.load(valid_ptr + rows, mask=rm, other=0)
                d = tl.where(v[:, None] != 0, d, 0.0)
            # Round the updated residual to its own storage dtype HERE, not
            # only on the store below. The eager chain materializes
            # `x + delta` as a real tensor in the residual dtype and the
            # LayerNorm then reads that rounded value; keeping the fp32
            # accumulator alive into the reduction instead would be *more*
            # accurate than the baseline, which this project counts as
            # divergence just the same. It is a no-op for an fp32 residual
            # (every graded configuration) and load-bearing for bf16, where
            # skipping it was measured 0.0156 -- two bf16 ulps -- away from
            # the ATen chain.
            x = (x + d).to(x_ptr.dtype.element_ty).to(tl.float32)
        if MASK_X:
            v = tl.load(valid_ptr + rows, mask=rm, other=0)
            x = tl.where(v[:, None] != 0, x, 0.0)
        if WRITE_X:
            tl.store(xout_ptr + offs, x.to(xout_ptr.dtype.element_ty), mask=tile)

        # Padding lanes (cols >= D) must not pollute the moments.
        x = tl.where(tile, x, 0.0)
        mean = tl.sum(x, axis=1) / D
        xc = tl.where(tile, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(w_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        y = xc * rstd[:, None] * w[None, :] + b[None, :]
        if MASK_OUT:
            v = tl.load(valid_ptr + rows, mask=rm, other=0)
            y = tl.where(v[:, None] != 0, y, 0.0)
        tl.store(nout_ptr + offs, y.to(nout_ptr.dtype.element_ty), mask=tile)


# Largest feature width this kernel will take. Above it the single-block
# reduction stops being the right structure (a BLOCK_D-wide tile no longer
# fits in registers and Triton spills), so callers fall back to ATen. Every
# graded configuration has d_model <= 1024, comfortably inside this.
_ADD_LN_MAX_BLOCK_D = 2048


_SM_COUNT_CACHE: Dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    """SM count of `device`, memoized. Grids here are sized against the real
    hardware rather than a hardcoded 36, because on a card this small
    under-subscribing a kernel is a genuine hazard rather than a rounding
    error."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _SM_COUNT_CACHE.get(idx)
    if cached is None:
        cached = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT_CACHE[idx] = cached
    return cached


# Target tile size for _add_ln_kernel, in elements per block. Measured, not
# guessed -- see _add_ln_launch_params.
_ADD_LN_TILE_ELEMS = 512


def _add_ln_launch_params(block_d: int, rows: int, device: torch.device) -> Tuple[int, int]:
    """(BLOCK_M, num_warps) for a padded feature width and row count.

    Fixed heuristic rather than @triton.autotune, deliberately. Autotuning
    benchmarks candidate configs on first call, which syncs -- and this
    kernel runs inside the CUDA-graph-captured region, where the *only*
    thing standing between a stray sync and a poisoned CUDA context is that
    every distinct (shape, flags) variant happens to have been launched
    during warmup first. The scale+mask kernel gets away with autotuning
    because it has one variant per configuration; this one has several flag
    combinations per layer. A deterministic rule removes that hazard class
    entirely -- provided the rule is a good one, which is worth measuring
    rather than asserting.

    So it was measured: BLOCK_M x num_warps swept over 1..64 x 1..8 at every
    d_model the graded groups use, timed under CUDA-graph replay (an eager
    sweep at these sizes ranks the 5-10 us launch floor, not the kernel).
    Two things came out of it, and this function is exactly those two:

    * **~512 elements per block, not 2048.** The optimum tracked that figure
      across the whole range -- d_model=32 wanted BLOCK_M=16, d_model=128
      wanted 4, d_model=512 wanted 1, all of which are 512 elements. The
      original 2048-element rule was 7-16% off at the small shapes and 32%
      off at d_model=512 (5.34 us -> 4.04 us).
    * **A block-count floor matters more than the tile size at small M.**
      At batch=1 (M=128 rows) the 512-element rule alone gives 32 blocks for
      36 SMs -- some SMs get nothing. Halving BLOCK_M until the grid covers
      the device is worth more there than any tile-size choice.

    The two shapes where this kernel actually dominates the forward
    (batch=10000 at 5.26 ms, seq=1024 at 2.16 ms) were already at their
    optimum under either rule, to within 1% -- they are purely
    bandwidth-bound and the config barely registers. The gains here are on
    the small shapes, and they are real but modest: ~1-2% end to end.
    """
    block_m = max(1, min(16, _ADD_LN_TILE_ELEMS // block_d))
    # Don't leave SMs idle: prefer more, smaller blocks over a tile-size
    # ideal that cannot fill the device.
    target_blocks = 2 * _sm_count(device)
    while block_m > 1 and -(-rows // block_m) < target_blocks:
        block_m //= 2
    num_warps = max(1, min(8, (block_m * block_d) // 128))
    return block_m, num_warps


def _triton_add_layernorm(
    x: torch.Tensor,
    delta: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
    mask_delta: bool,
    mask_x: bool,
    mask_out: bool,
    need_x: bool,
    valid_flat: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch _add_ln_kernel over the [B, S, D] activation `x`.

    Returns (x_new, normed). `x_new` is a freshly allocated tensor when
    `delta` is not None (the residual actually changed and the next site
    needs it); otherwise it is `x` itself, unmodified, and no store happens.
    Raises on any launch/compile failure rather than falling back silently --
    the caller decides fallback policy once, at probe time, so a real bug
    here can never masquerade as "unsupported shape" at steady state.
    """
    d_model = x.shape[-1]
    m = x.numel() // d_model
    block_d = triton.next_power_of_2(d_model)
    if block_d > _ADD_LN_MAX_BLOCK_D:
        raise ValueError(f"d_model={d_model} exceeds fused-LayerNorm block width")
    # The kernel indexes rows as `row * D + col`, i.e. it assumes the row
    # stride is exactly D. Both operands are contiguous everywhere this is
    # called from (the residual is either the model input or a
    # torch.empty_like of it; the delta is a GEMM output), but assert rather
    # than assume -- silently computing on the wrong strides would be a
    # correctness bug, whereas raising just makes the probe fall back.
    if not x.is_contiguous() or (delta is not None and not delta.is_contiguous()):
        raise ValueError("fused LayerNorm requires contiguous [B, S, D] inputs")

    has_delta = delta is not None
    # The residual has to be materialized whenever this site actually
    # changes it -- by adding `delta`, or by zeroing padding rows. (The
    # second case cannot arise from _forward_core, which only sets mask_x at
    # sites that also have a delta, but leaving it out would make this
    # function silently disagree with _ln_site's ATen branch when called
    # directly.)
    write_x = need_x and (has_delta or mask_x)
    x_out = torch.empty_like(x) if write_x else x
    normed = torch.empty(x.shape, dtype=out_dtype, device=x.device)

    if mask_delta or mask_x or mask_out:
        assert valid_flat is not None
        valid_arg = valid_flat
    else:
        # Unused (the constexpr flags dead-code-eliminate every load of it),
        # but the launch still needs a real CUDA tensor argument.
        valid_arg = x

    block_m, num_warps = _add_ln_launch_params(block_d, m, x.device)
    grid = (triton.cdiv(m, block_m),)
    _add_ln_kernel[grid](
        x, delta if has_delta else x, x_out, normed, weight, bias, valid_arg,
        m, d_model, eps,
        HAS_DELTA=has_delta,
        MASK_DELTA=mask_delta,
        MASK_X=mask_x,
        WRITE_X=write_x,
        MASK_OUT=mask_out,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_out, normed


# ---------------------------------------------------------------------------
# Shape-specialized single-block attention for short sequences
# ---------------------------------------------------------------------------
#
# The second place the graded shapes buy something the general-purpose kernel
# cannot use. seq_len is 128 in rows 1-12 except row 12 (seq_len=32), so 12
# of the 14 graded configurations have seq_len 32 or 128. Therefore
# the ENTIRE [S, S] score matrix for one (batch, head) pair fits in a single
# Triton program's registers. That removes the reason flash / memory-efficient
# attention exists: there is no need to tile the key axis, and therefore no
# need for an online softmax that rescales a running accumulator as it goes.
# One program loads Q, K and V for its head once, forms all the scores, takes
# a plain single-pass row softmax, and multiplies by V.
#
# MEASURED against `fmha_cutlassF` (the memory-efficient SDPA backend this
# card actually selects -- this Windows build has no flash attention at all),
# fp16, causal, best Triton config per shape:
#
#   B    H    S   hd   |  SDPA us | this us | ratio
#   64   4   128   32  |    36.8  |   26.0  | 1.42x
#  128   4   128   32  |    65.4  |   36.3  | 1.80x
#   64   1   128  128  |    37.2  |   27.0  | 1.38x
#   64  16   128    8  |   121.6  |   57.4  | 2.12x
#
# (The two smallest cases, B=1 and S=32, are dominated by Python-side launch
# overhead in an eager microbenchmark -- both implementations flat-line at
# their launch floor -- so they are measured in-model under CUDA-graph replay
# instead, where that floor does not exist.)
#
# Why this does not repeat step 7's mistake of hand-writing a kernel that
# loses to a vendor one: `fmha_cutlassF` is tiled for long sequences, and at
# S=128 it is doing a tiled algorithm's bookkeeping for a problem that needs
# none of it -- at (64, 4, 128, 32) it achieves ~15 TFLOPS against a ~44
# TFLOPS ceiling and ~58% of the bandwidth roof. That is a structural
# mismatch to the shape, not a tuning gap, which is exactly the situation
# where a specialized kernel can win and the GEMM case was not (cuBLAS is
# already at ~78% of the bandwidth roof for these GEMMs, so there was nothing
# comparable to take there -- measured, see OPTIMIZATION_LOG.md step 11).
#
# NUMERICS. This is a different implementation of the same reduction, so it
# is not bit-exact against SDPA and cannot be used on the fp16/bf16 paths at
# all. Its use is decided by _resolve_short_attn_enabled, whose fp32 bar is
# the same margin-scaled harness criterion measured against the exact fp32
# reference that gates the fp16 GEMMs and the fused LayerNorm -- so all three
# approximations are judged together, against the truth, rather than each
# against the last.
if _TRITON_AVAILABLE:
    # Autotuned, unlike the fused-LayerNorm kernel, because here the spread
    # between configs is not small -- it is catastrophic. Measured at
    # (B=64, H=16, S=128, head_dim=8), causal, fp16:
    #
    #   BLOCK_Q=64,  num_warps=4  ->    75 us   (SDPA: 121 us)
    #   BLOCK_Q=128, num_warps=4  ->    85 us
    #   BLOCK_Q=32,  num_warps=8  ->   369 us
    #   BLOCK_Q=128, num_warps=1  ->   803 us   (10.7x off the best config)
    #
    # and the best config genuinely moves with the shape (BLOCK_Q=128 wins at
    # head_dim=32, BLOCK_Q=64 at head_dim=8 and head_dim=128). A fixed
    # heuristic shipped `BLOCK_Q=128, num_warps=1` for that shape and made
    # the whole model 4x slower than the ATen path it replaced -- the
    # measurement that produced this table.
    #
    # num_warps=1 is excluded outright: it lost at every shape tried, by
    # between 2x and 10x, because a single warp has to hold the entire
    # [BLOCK_Q, BLOCK_S] score tile in its registers and spills.
    #
    # Autotuning inside a CUDA-graph-captured region would be illegal (it
    # benchmarks, which syncs). It never happens there: _probe_short_attn
    # drives a whole _forward_core with this kernel enabled, and
    # _capture_graph then runs three more warmup forwards on a side stream,
    # all before capture is attempted -- so every (shape, flags) variant is
    # already resolved and cached by the time the graph is recorded. This is
    # the same pre-capture-warmup contract _fused_scale_mask_kernel relies
    # on.
    _SHORT_ATTN_CONFIGS = [
        triton.Config({"BLOCK_Q": bq}, num_warps=nw, num_stages=2)
        for bq in (32, 64, 128)
        for nw in (2, 4, 8)
    ]

    @triton.autotune(configs=_SHORT_ATTN_CONFIGS, key=["S", "HD", "NBH"])
    @triton.jit
    def _short_attn_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, valid_ptr,
        H, S, HD, NBH,
        q_sb, q_sh, q_ss, q_sd,
        k_sb, k_sh, k_ss, k_sd,
        v_sb, v_sh, v_ss, v_sd,
        o_sb, o_sh, o_ss, o_sd,
        m_sb, m_ss,
        qk_scale,
        CAUSAL: tl.constexpr,
        MASK_ACTIVE: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Attention for a BLOCK_Q-row slice of one (batch, head)'s queries
        against ALL of its keys. Grid is (query blocks, batch*heads).

        `NBH` (= batch*heads) is never read by the body; it is here only so
        the autotuner keys on it, since the best config depends on how much
        parallelism the (batch, head) axis already supplies.

        The key axis is never tiled -- that is the whole point, and what the
        short sequence buys: with all S keys resident, the softmax is a plain
        single-pass row reduction instead of the running-max/running-sum
        rescale that a key-tiled (flash-style) kernel is forced into.

        The QUERY axis is tiled anyway, purely for occupancy. One program per
        (batch, head) is the natural formulation and is what the wrapper
        picks whenever batch*heads alone already saturates the 36 SMs -- but
        at batch=1 that is a grid of 4 programs, and the kernel measured
        *slower* than SDPA there (0.80x) for want of parallelism, while
        winning 1.15x at batch=64. Splitting the queries restores the grid
        without touching the softmax's structure, since each query row's
        reduction is independent of every other's.

        Reproduces BaselineSelfAttention's op order rather than SDPA's: the
        scale is applied AFTER the Q@K^T matmul (SDPA pre-scales q), masked
        positions are set to -inf, and the softmax runs in fp32 -- which is
        what the baseline's `torch.softmax(scores.float(), ...)` does.

        The softmax is written in base 2. This kernel's cost tracks score
        matrix ELEMENTS (B*H*S^2), not FLOPs -- measured at a near-constant
        210-270 G score-elements/s across head_dim 8/16/32, where the FLOPs
        are identical but the element count differs 4x -- so anything that
        runs once per score element is worth removing. `qk_scale` therefore
        arrives premultiplied by log2(e) and the exponential is `exp2`,
        which is one hardware instruction where `exp` is a multiply plus
        exp2. Clamping a fully-masked row's max to 0 (rather than leaving it
        at -inf and repairing the resulting NaN afterwards) additionally
        lets the post-exp `tl.where` be dropped -- that select also ran over
        every score element. Together: 1.50x at head_dim=8, 1.29x at
        head_dim=32, 2.24x at the historical S=32/D=32/H=4 probe. That probe
        predates the matrix schema correction and is not a timing for the
        canonical `12_seq32` shape (S=32/D=128/H=4).

        BLOCK_D is padded up to at least 16 because tl.dot requires it; the
        padding lanes are loaded as zero and never stored, so head_dim=8
        (graded rows 07 and 11) works without a separate code path.
        """
        qb = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        q_idx = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
        s_idx = tl.arange(0, BLOCK_S)
        d_idx = tl.arange(0, BLOCK_D)
        q_ok = q_idx < S
        s_ok = s_idx < S
        d_ok = d_idx < HD
        q_tile = q_ok[:, None] & d_ok[None, :]
        s_tile = s_ok[:, None] & d_ok[None, :]

        q = tl.load(
            q_ptr + b * q_sb + h * q_sh + q_idx[:, None] * q_ss + d_idx[None, :] * q_sd,
            mask=q_tile, other=0.0)
        k = tl.load(
            k_ptr + b * k_sb + h * k_sh + s_idx[:, None] * k_ss + d_idx[None, :] * k_sd,
            mask=s_tile, other=0.0)
        v = tl.load(
            v_ptr + b * v_sb + h * v_sh + s_idx[:, None] * v_ss + d_idx[None, :] * v_sd,
            mask=s_tile, other=0.0)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale

        keep = s_ok[None, :] & q_ok[:, None]
        if CAUSAL:
            keep = keep & (q_idx[:, None] >= s_idx[None, :])
        if MASK_ACTIVE:
            mv = tl.load(valid_ptr + b * m_sb + s_idx * m_ss, mask=s_ok, other=0)
            keep = keep & (mv[None, :] != 0)
        scores = tl.where(keep, scores, float("-inf"))

        row_max = tl.max(scores, axis=1)
        # A query row with no unmasked key can only be a padding row, whose
        # result the caller discards. Its max is -inf, and -inf - -inf is
        # NaN; clamping the max to 0 instead makes exp2(-inf) = 0 fall out
        # for the whole row, so no NaN is ever created and no repair pass
        # over the score matrix is needed. The caller masks the row to 0
        # either way.
        row_max = tl.where(row_max == -float("inf"), 0.0, row_max)
        # exp2, not exp: qk_scale carries log2(e) (see the docstring). The
        # masked entries are -inf, so exp2 gives them 0 with no extra select.
        probs = tl.exp2(scores - row_max[:, None])
        denom = tl.sum(probs, axis=1)
        probs = probs * (1.0 / tl.where(denom > 0, denom, 1.0))[:, None]

        out = tl.dot(probs.to(v.dtype), v, out_dtype=tl.float32)
        tl.store(
            o_ptr + b * o_sb + h * o_sh + q_idx[:, None] * o_ss + d_idx[None, :] * o_sd,
            out.to(o_ptr.dtype.element_ty), mask=q_tile)


# Longest sequence this kernel will take. The [S, S] score tile is held in
# registers, so cost grows as S^2: at 128 it is a 128x128 fp32 accumulator,
# the same size as a conventional GEMM tile; at 256 it would be four times
# that and spill. Sequences above this stay on SDPA, which is tiled for
# exactly that case.
_SHORT_ATTN_MAX_S = 128
# log2(e): folded into the attention scale so the softmax can use exp2.
_LOG2E = 1.4426950408889634
# Widest head this kernel will take, for the same register-budget reason.
_SHORT_ATTN_MAX_HD = 128


def _short_attn_supported(seq_len: int, head_dim: int) -> bool:
    return seq_len <= _SHORT_ATTN_MAX_S and head_dim <= _SHORT_ATTN_MAX_HD


def _triton_short_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    mask_active: bool,
    valid_token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Attention over [B, H, S, HD] q/k/v, returning the MERGED [B, S, H*HD]
    context directly.

    q/k/v are accepted with arbitrary strides, because on the fused-QKV path
    they are non-contiguous column slices of one packed [B, S, 3*d_model]
    GEMM output and forcing them contiguous would add exactly the copies
    this is trying to avoid.

    The output is allocated in [B, S, H, HD] order and handed to the kernel
    through transposed strides, so the merge back to [B, S, d_model] is a
    free view rather than a copy -- the same layout `fmha_cutlassF` already
    produces, which is why the merge does not show up as a kernel on either
    path.
    """
    batch, heads, seq_len, head_dim = q.shape
    if not _short_attn_supported(seq_len, head_dim):
        raise ValueError(f"seq_len={seq_len} head_dim={head_dim} out of range")

    out = torch.empty((batch, seq_len, heads, head_dim), dtype=q.dtype, device=q.device)
    # strides of out viewed as [B, H, S, HD]
    o_sb, o_ss, o_sh, o_sd = out.stride()

    if mask_active:
        assert valid_token_mask is not None
        m_sb, m_ss = valid_token_mask.stride()
        mask_arg = valid_token_mask
    else:
        m_sb, m_ss = 0, 0
        mask_arg = q

    block_s = triton.next_power_of_2(seq_len)
    block_d = max(16, triton.next_power_of_2(head_dim))

    # BLOCK_Q (query-axis tiling, for occupancy) and num_warps are chosen
    # by @triton.autotune -- see _SHORT_ATTN_CONFIGS for why they are not
    # hardcoded.
    def grid(meta):
        return (triton.cdiv(seq_len, meta["BLOCK_Q"]), batch * heads)

    _short_attn_kernel[grid](
        q, k, v, out, mask_arg,
        heads, seq_len, head_dim, batch * heads,
        *q.stride(), *k.stride(), *v.stride(),
        o_sb, o_sh, o_ss, o_sd,
        m_sb, m_ss,
        scale * _LOG2E,
        CAUSAL=causal,
        MASK_ACTIVE=mask_active,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
    )
    return out.view(batch, seq_len, heads * head_dim)


# ---------------------------------------------------------------------------
# GEMM with a fused bias + GELU epilogue (the FFN's first projection)
# ---------------------------------------------------------------------------
#
# Steps 7 and 11 both concluded that a hand-written Triton GEMM does not beat
# cuBLAS at these shapes, and that still holds -- the reduction axis is
# d_model, so K is 128 for most graded rows, and measured throughput is a
# steep function of K (9.2 TFLOP/s at K=32, 24.8 at K=128, 40.4 at K=512,
# plateauing ~45). cuBLAS already sits on that ceiling.
#
# The bar here is different, and much lower: not "beat cuBLAS at the GEMM"
# but "beat cuBLAS's GEMM *plus a separate ATen GELU pass*". Fusing deletes a
# full write-then-read of the [batch*seq, ffn_dim] hidden tensor. The timing
# evidence below predates the 2026-08-31 matrix schema correction: its shape
# tuples remain useful kernel evidence, but the retired row labels do not
# describe canonical benchmark cases and must not be used as suite results.
#
# Measured (CUDA-graph timed, fp16, vs `F.linear` + `F.gelu`):
#
#   historical shape (M x K x N)       cuBLAS+GELU   fused   ratio  bit-exact
#   legacy row 13   8192 x 128 x 1024       113.92    63.14   1.80x    yes
#   legacy row 07   2048 x  32 x  128         4.00     2.59   1.54x    yes
#   row 05         16384 x 128 x  128        26.75    18.02   1.48x    yes
#   row 01          8192 x 128 x  128        14.45    10.02   1.44x    yes
#   legacy row 08  65536 x1024 x  128       437.42   386.37   1.13x     no
#   default         1024 x 512 x 2048        56.67    52.23   1.09x     no
#   legacy row 12   8192 x 128 x   32         5.25     6.05   0.87x    yes
#   row 02           128 x 128 x  128         2.33     2.85   0.82x    yes
#
# BIT-EXACT at six of eight shapes -- `torch.equal`, not "close". Two things
# make that possible and both are deliberate:
#   * libdevice's `erf` is the same function ATen's GELU calls, so the
#     transcendental agrees to the last bit.
#   * the accumulator is rounded to the output dtype BEFORE the GELU, because
#     `F.linear` materializes an fp16 tensor and ATen's GELU then reads that
#     rounded value. Carrying fp32 into the epilogue would be more accurate
#     than the reference, which this project counts as divergence too.
# The two inexact shapes are the two with a long reduction axis (K=512,
# K=1024), where Triton's k-loop order differs from cuBLAS's; they differ by
# 3.8e-6, i.e. fp32 rounding, not a logic difference.
#
# Because it can be bit-exact, this is offered to the fp16/bf16 paths as well
# as the fp32 one -- _probe_fused_ffn demands `torch.equal` there, exactly
# like the fused-QKV probe of step 8, and lets the measurement decide.
if _TRITON_AVAILABLE:

    _GEMM_GELU_CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                      num_warps=nw, num_stages=ns)
        for bm, bn, bk, nw, ns in [
            (128, 128, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
            (64, 64, 32, 4, 3), (128, 64, 64, 4, 3), (64, 64, 64, 4, 4),
            (32, 64, 32, 4, 2), (64, 32, 32, 4, 2), (128, 128, 64, 8, 3),
            (32, 32, 32, 2, 2), (256, 64, 32, 8, 3), (64, 256, 32, 8, 3),
            (128, 32, 32, 4, 2), (32, 128, 32, 4, 2),
        ]
    ]

    @triton.autotune(configs=_GEMM_GELU_CONFIGS, key=["M", "N", "K", "APPLY_GELU"])
    @triton.jit
    def _gemm_gelu_kernel(
        a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
        s_am, s_ak, s_bk, s_bn, s_cm, s_cn,
        APPLY_GELU: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        """C = gelu(A @ B + bias), grouped-tile ordering for L2 reuse.

        `b_ptr` is the nn.Linear weight TRANSPOSED as a view ([K, N] strides
        over the stored [N, K]), so no repacking copy is needed.
        """
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group = pid // width
        off = pid % width
        m0 = group * GROUP_M
        gm = min(grid_m - m0, GROUP_M)
        pid_m = m0 + (off % gm)
        pid_n = off // gm

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * s_am + rk[None, :] * s_ak
        b_ptrs = b_ptr + rk[:, None] * s_bk + rn[None, :] * s_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            krem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < krem), other=0.0)
            b = tl.load(b_ptrs, mask=(rk[:, None] < krem) & (rn[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * s_ak
            b_ptrs += BLOCK_K * s_bk

        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        # Round to the output dtype BEFORE the GELU -- see the module comment;
        # this is what makes the result bit-identical to F.linear + F.gelu
        # rather than merely close to it.
        h = (acc + bias[None, :]).to(c_ptr.dtype.element_ty).to(tl.float32)
        if APPLY_GELU:
            h = h * 0.5 * (1.0 + _tl_libdevice.erf(h * 0.7071067811865476))
        tl.store(c_ptr + rm[:, None] * s_cm + rn[None, :] * s_cn,
                 h.to(c_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


# Shape boundary for the fused GEMM+GELU, MEASURED (table in the module
# comment above), not derived. Below it cuBLAS's narrow-N / tiny-M kernels
# win even after paying for a separate GELU pass: at N=32 the fused kernel
# is 0.87x and at M=128 it is 0.82x, while every shape above the boundary is
# 1.09x-1.80x. Like the FFN chunk size in step 5 and the fp16-GEMM margin in
# step 3, this is an empirical boundary on this hardware and should be
# re-measured, not trusted, if the GPU or cuBLAS version changes.
_FUSED_FFN_MIN_ROWS = 256
_FUSED_FFN_MIN_FFN = 64


def _fused_ffn_supported(rows: int, ffn_dim: int) -> bool:
    return rows >= _FUSED_FFN_MIN_ROWS and ffn_dim >= _FUSED_FFN_MIN_FFN


# Row-count floor for routing a projection through the Triton GEMM.
# MEASURED (table below), not derived: below it cuBLAS's small-M kernels win.
#
# This overturns the conclusion of steps 7 and 11 for these shapes, and it is
# worth being precise about why they got it wrong rather than just replacing
# the number. Step 7 benchmarked at the *default* shape (d_model=512) and
# found cuBLAS ahead, which is still true there for the FFN's wide-K
# projection. Step 11 then argued from a K-sweep that cuBLAS was already
# sitting on the short-reduction ceiling -- but that sweep only ever compared
# cuBLAS against cuBLAS. It established the shape of the ceiling correctly and
# then *assumed* cuBLAS reached it. It does not: at the narrow-N projections
# the narrow-width subset produces (N = d_model or 3*d_model, with d_model
# 32 or 128)
# cuBLAS dispatches to a wmma kernel and a plain Triton GEMM beats it
# outright.
#
#   shape (M x K x N)                cuBLAS    Triton    ratio   bit-exact
#   qkv       16384 x 128 x  384      90.88     44.10    2.06x      yes
#   qkv        8192 x 128 x  384      29.12     20.32    1.43x      yes
#   out_proj   1024 x 512 x  512      22.81     14.64    1.56x      yes
#   ffn_out    8192 x  32 x  128       6.42      4.23    1.52x      yes
#   out_proj  16384 x 128 x  128      19.22     14.55    1.32x      yes
#   qkv        2048 x 128 x  384       8.59      6.93    1.24x      yes
#   out_proj   8192 x 128 x  128      10.30      9.13    1.13x      yes
#   qkv        1024 x 512 x 1536      52.14     46.95    1.11x      yes
#   ffn_out    8192 x1024 x  128      55.82     55.36    1.01x       no
#   ffn_out    1024 x2048 x  512      51.41     52.29    0.98x       no
#   qkv         512 x 128 x  384       3.15      3.34    0.94x      yes
#   qkv         128 x 128 x  384       2.17      2.45    0.89x      yes
#
# Bit-exact (`torch.equal`) at every shape with K <= 512, which is why this is
# offered on the fp16/bf16 paths too and not just the fp32 one.
_TRI_LINEAR_MIN_ROWS = 1024


def _tri_linear_supported(rows: int) -> bool:
    return rows >= _TRI_LINEAR_MIN_ROWS


def _triton_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """F.linear(x, weight, bias) via _gemm_gelu_kernel with the epilogue off.

    Same kernel as the fused FFN projection, so there is one GEMM
    implementation and one autotune cache in this file rather than two.
    """
    return _gemm_gelu_impl(x, weight, bias, apply_gelu=False)


def _triton_gemm_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """gelu(F.linear(x, weight, bias)) for a [B, S, K] input, in one kernel.

    Returns [B, S, N]. Raises rather than falling back, so a real failure
    cannot masquerade as an unsupported shape -- the caller's probe decides
    fallback once, at warmup.
    """
    return _gemm_gelu_impl(x, weight, bias, apply_gelu=True)


def _gemm_gelu_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, apply_gelu: bool
) -> torch.Tensor:
    *lead, k_dim = x.shape
    rows = 1
    for d in lead:
        rows *= d
    n_dim = weight.shape[0]
    a = x.reshape(rows, k_dim)
    if not a.is_contiguous():
        raise ValueError("the Triton GEMM requires a contiguous input")
    out = torch.empty((rows, n_dim), device=x.device, dtype=x.dtype)
    wt = weight.t()  # [K, N] view over the stored [N, K]; no copy

    def grid(meta):
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(n_dim, meta["BLOCK_N"]),)

    _gemm_gelu_kernel[grid](
        a, wt, out, bias, rows, n_dim, k_dim,
        a.stride(0), a.stride(1), wt.stride(0), wt.stride(1),
        out.stride(0), out.stride(1),
        APPLY_GELU=apply_gelu,
    )
    return out.view(*lead, n_dim)


def _no_static_cuda_launcher():
    """Disable inductor's static CUDA launcher for the duration of a block.

    On Windows / torch 2.8 that launcher overflows a C long with the 64-bit
    CUDA stream handle (`OverflowError: Python int too large to convert to C
    long`) the moment a compiled Triton kernel is launched -- documented in
    step 7, and it is not limited to autotuned GEMM templates, it fires on an
    ordinary pointwise clone too.

    Returns a null context if inductor is unavailable, so the caller never
    has to care whether this build has it.
    """
    try:
        return torch._inductor.config.patch(use_static_cuda_launcher=False)
    except Exception:
        return contextlib.nullcontext()


# Staircase granularity for the block-triangular causal scale+mask kernel
# (_tri_scale_mask_kernel). 128 measured consistently best across every
# shape tried -- speedup of the triangular epilogue over the full-width one:
#
#     shape            BR=128   BR=256   BR=512
#     (4,8,2048)        1.66x    1.22x    1.18x
#     (32,8,512)        1.21x    1.13x    0.99x
#     (2,8,4096)        1.22x    1.21x    1.19x
#     (16,16,256)       1.08x    0.95x      --
#
# Smaller blocks skip strictly more of the above-diagonal half, and 128 is
# also where the kernel itself runs best, so there is no trade-off to tune.
_TRI_MASK_BLOCK = 128


if _TRITON_AVAILABLE:

    @triton.jit
    def _tri_scale_mask_kernel(
        scores_ptr, out_ptr, mask_ptr,
        H, SQ, SK,
        stride_sb, stride_sh, stride_sq, stride_sk,
        stride_ob, stride_oh, stride_oq, stride_ok,
        stride_mb, stride_mk,
        scale,
        MASK_ACTIVE: tl.constexpr,
        BR: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Causal-only variant of _fused_scale_mask_kernel that writes just
        the block-lower-triangle.

        Row i writes columns [0, lim) where lim = min(((i//BR)+1)*BR, SK).
        Every column at or past `lim` is above the diagonal for row i, so it
        is -inf -- and it is left ALONE here, because the destination is a
        persistent buffer whose above-staircase region was filled with -inf
        once, at allocation (see _get_tri_scaled_buffer). Nothing else ever
        writes there, so it stays -inf for the life of the buffer and across
        every CUDA-graph replay.

        Every element the following softmax reads therefore holds exactly
        the value the full-width kernel would have left: kernel-written
        below the staircase, -inf above it. That makes this bit-exact by
        construction rather than by luck, and it is still probed.

        The softmax itself is deliberately NOT made triangular. Running it
        per block-row means handing ATen a strided [BR, lim) slice, which
        drops it off its fast path: measured 1.87 -> 2.39 ms at
        (4,8,2048), i.e. SLOWER than the full-width softmax despite doing
        half the work. The same work on contiguous block-rows takes
        0.88 ms, but compacting to contiguous and scattering the result
        back to full width for probs@V costs more than the 1 ms it saves.
        So softmax keeps reading the whole row, which is exactly why the
        -inf pre-fill above is load-bearing.
        """
        row_id = tl.program_id(0)
        hq = H * SQ
        b = row_id // hq
        rem = row_id % hq
        h = rem // SQ
        i = rem % SQ

        lim = tl.minimum(((i // BR) + 1) * BR, SK)
        row_base_s = b * stride_sb + h * stride_sh + i * stride_sq
        row_base_o = b * stride_ob + h * stride_oh + i * stride_oq
        mask_row_base = b * stride_mb
        out_dtype = out_ptr.dtype.element_ty

        for start in range(0, lim, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            col_ok = cols < lim
            sv = tl.load(scores_ptr + row_base_s + cols * stride_sk,
                         mask=col_ok, other=0.0)
            v = (sv.to(tl.float32) * scale).to(out_dtype)
            v = tl.where(cols <= i, v, -float("inf"))
            if MASK_ACTIVE:
                mvals = tl.load(mask_ptr + mask_row_base + cols * stride_mk,
                                mask=col_ok, other=0)
                v = tl.where(mvals != 0, v, -float("inf"))
            tl.store(out_ptr + row_base_o + cols * stride_ok, v, mask=col_ok)


def _triton_tri_scale_mask(
    scores_raw: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    block: int,
    mask_active: bool,
    valid_token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Launch _tri_scale_mask_kernel into the caller-supplied persistent
    `out` buffer (see _get_tri_scaled_buffer). Raises on any failure so a
    real bug cannot masquerade as an unsupported shape -- callers fall back
    to the full-width path."""
    B, H, SQ, SK = scores_raw.shape
    if mask_active:
        assert valid_token_mask is not None
        stride_mb, stride_mk = valid_token_mask.stride()
        mask_arg = valid_token_mask
    else:
        stride_mb, stride_mk = 0, 0
        mask_arg = scores_raw

    _tri_scale_mask_kernel[(B * H * SQ,)](
        scores_raw, out, mask_arg,
        H, SQ, SK,
        *scores_raw.stride(),
        *out.stride(),
        stride_mb, stride_mk,
        scale,
        MASK_ACTIVE=mask_active,
        BR=block,
        BLOCK_N=128,
        num_warps=4,
    )
    return out


class _GraphCacheEntry:
    """Holds one captured CUDA graph plus the static input/output buffers it
    was captured against. Replaying requires copy_-ing fresh data into
    static_x (and static_mask, if present) before graph.replay(); the caller
    must clone static_output since the next replay overwrites it in place."""

    __slots__ = ("graph", "static_x", "static_mask", "static_output")

    def __init__(
        self,
        graph: "torch.cuda.CUDAGraph",
        static_x: torch.Tensor,
        static_mask: Optional[torch.Tensor],
        static_output: torch.Tensor,
    ) -> None:
        self.graph = graph
        self.static_x = static_x
        self.static_mask = static_mask
        self.static_output = static_output


# A captured CUDA graph keeps every allocation in its private pool alive for
# the lifetime of the model. Unlike ordinary eager allocations, that memory
# cannot be returned between layers or after warmup. Keep graph mechanisms
# below a deliberately conservative fraction of total VRAM, calculated from
# shape and device properties only. In particular, never use mem_get_info():
# it synchronizes and would make the selected arithmetic depend on unrelated
# processes sharing the card.
_GRAPH_MEMORY_BUDGET_FRAC = 0.20
# Inductor's whole-core schedule can avoid the fp32 score/probability
# materialization used by the hand-rolled eager graph on the fp32 SDPA path.
# It is therefore safe to admit a larger (but still capacity-bounded) working
# set to that candidate. This is deliberately a separate policy: treating an
# Inductor graph as though it retained the eager graph's pool was needlessly
# excluding row 6, the capacity-sensitive user shape this experiment targets.
_FUSED_GRAPH_MEMORY_BUDGET_FRAC = 0.55


class UserOptimizedTransformer(BaselineTransformer):
    """
    Eager-mode optimized implementation of BaselineTransformer.

    Optimizations applied (structural, no torch.compile / CUDA graphs /
    dtype changes / custom kernels):
      1. Fused QKV projection: q_proj/k_proj/v_proj weights+biases are
         packed into one [3*d_model, d_model] matrix per layer so a single
         GEMM replaces three. out_proj and the FFN GEMMs stay separate.
         Worth ~1.9x on those GEMMs (measured 20.6 -> 38.2 TFLOPS in fp16
         at the default shape): N=3*d_model yields three times the
         threadblock tiles, which matters a lot on a 36-SM card where the
         narrow N=d_model form leaves SMs idle.
         On the fp32/SDPA path (see (2)) it is used unconditionally -- fp32
         has the mantissa headroom to absorb any reassociation.
         On the fp16/bf16 path it is used only where a warmup probe proved
         it BIT-EXACT for that exact configuration (_probe_fused_qkv_exact:
         run the whole forward both ways, require torch.equal), falling
         back to three separate GEMMs otherwise. This replaces an earlier
         blanket "never fuse in low precision" rule, which was both too
         conservative (bf16 is bit-exact at every shape tried, and so is
         fp16 at most of them) and based on a wrong mechanism -- the fused
         and separate GEMMs dispatch to the *same* cutlass kernel; what
         varies is an unqueryable cuBLASLt algorithm choice that is
         non-monotonic in M and unaffected by the split-K workspace knob.
         See _probe_fused_qkv_exact's docstring for the measurements.
      2. Dtype-dependent attention implementation, chosen lazily from the
         model's compute dtype and cached until the dtype changes:
           - fp32: F.scaled_dot_product_attention, letting a fused SDPA
             backend (mem-efficient / math; this build has no flash
             attention) run instead of several kernels. The baseline's
             ".float()" softmax upcast is a no-op in fp32, so SDPA's math
             (which scales q before the matmul and keeps higher-precision
             probs internally) still matches the baseline within tolerance.
           - fp16 / bf16: the manual matmul + softmax + matmul, replicated
             bit-for-bit against BaselineSelfAttention (scale applied
             *after* q@k^T, masks applied with the same masked_fill(...,
             -inf) calls in the same order, softmax run natively on the
             half/bfloat16 scores with no dtype= kwarg and no explicit
             .float() upcast -- see _fused_attn_probs's docstring for why
             that is bit-identical to the baseline's explicit-upcast form
             while also avoiding a dtype=torch.float32-kwarg kernel-
             selection bug that diverged from the baseline at long sequence
             lengths). SDPA's internal math diverges from this by more than
             atol/rtol once compounded over several layers at fp16/bf16
             precision, so it is only used where it is provably equivalent
             (fp32).
      3. Mask fast-path: when valid_token_mask is None or all tokens are
         valid, no masked_fill / mask tensor is built anywhere, and the
         causal case (fp32 path) uses SDPA's is_causal=True fast path (no
         explicit attn_mask materialized at all). The all-True check is
         done once per forward call (one sync), not once per layer.
      4. All per-call mask/causal tensors (SDPA's attn_mask on the fp32
         path; the causal "disallowed" mask and the key-padding mask on the
         fp16/bf16 path) are computed exactly once per forward call, before
         the layer loop, and reused unchanged across every layer -- never
         rebuilt per layer. The causal masks are additionally cached per
         (seq_len, device) across forward calls instead of being rebuilt
         with torch.ones(...).triu()/.tril() every time.
      5. view/transpose (no .contiguous()) feed directly into attention,
         and the post-attention merge uses .reshape() so a copy only
         happens if the layout actually requires one.
      6. CUDA graph capture/replay: the actual per-layer computation lives
         in _forward_core(), which takes `mask_active` as a plain Python
         bool (resolved once, outside the graph -- see _resolve_mask_active)
         instead of calling `.all()`/`.item()` on device data internally.
         That keeps _forward_core() free of any device->host sync, so it is
         safe to capture with torch.cuda.graph(). forward() keys a cache of
         captured graphs on (x.shape, x.dtype, x.device, causal, mask_kind)
         -- mask_kind in {"none", "all_true", "partial"} -- and on a cache
         hit just copies the new x (and mask, if partial) into the graph's
         static input buffers, replays, and returns a clone of the static
         output (so a later replay can't mutate a tensor the caller is still
         holding). Replaying performs the exact same kernels with the exact
         same arithmetic as the eager path -- it only removes per-launch CPU
         overhead -- so this step stays bit-exact with the baseline. All
         captured graphs share one memory pool (torch.cuda.graph_pool_handle)
         to limit fragmentation. Capture is skipped (falling back to eager)
         when the device isn't CUDA, when torch.compiler.is_compiling() is
         true (so dynamo tracing the model via --compile-user never traces
         graph capture), or -- permanently for that cache key -- if capture
         itself raises.
      7. fp32-only fp16-GEMM path: on the
         fp32 model (never fp16/bf16 -- those stay untouched and bit-exact),
         every GEMM in a layer (fused QKV, out_proj, ffn_in, ffn_out) and
         attention itself casts its activations/weights to fp16, runs the
         op, and casts the result straight back to fp32 immediately;
         LayerNorm, GELU, the residual adds, and the final norm all stay
         fp32. fp16 weight/bias copies are cached lazily (see
         _get_linear_fp16_weights / _get_fused_qkv_weights_fp16), never
         re-cast every forward. This is fast (fp16 cublas GEMMs run ~1.8-2x
         the TF32 throughput measured on this GPU). It is selected directly
         for every fp32 CUDA invocation; no test-input-dependent calibration
         or cached accuracy verdict is used. The harness's external accuracy
         check remains authoritative and fails configurations for which the
         reduced-precision path exceeds its tolerance.
      8. Fused scale+mask Triton kernel, fp16/bf16 only: on the manual-math
         branch from item 2, the raw (unscaled) Q@K^T matmul still runs as
         one ATen GEMM, and the softmax still runs as ATen's own native
         `torch.softmax(scaled, dim=-1)` (no dtype= kwarg, no explicit
         `.float()` upcast -- see _fused_attn_probs's docstring for why that
         is bit-identical to the baseline's explicit-upcast softmax, cheaper
         (no fp32 intermediate), AND -- unlike the dtype=torch.float32-kwarg
         form this used to use -- doesn't diverge from the baseline at long
         sequence lengths), but the scale multiply and the (up to two)
         combined masked_fills that sit between them are fused into a single
         Triton kernel (_fused_scale_mask_kernel, launched via
         _fused_attn_probs/_triton_fused_scale_mask) that reads the raw
         scores once and writes the scaled+masked scores once, autotuned per
         sequence length (BLOCK_N, num_warps) via @triton.autotune. This is a
         purely elementwise fusion (no reduction), verified bit-exact
         (torch.equal, not just close) against `(scores *
         scale).masked_fill(combined_mask, -inf)` -- deliberately NOT also
         fusing the softmax reduction itself: an earlier version that did
         was only ever ~1e-4 to 1e-5 max_abs away from ATen's softmax per
         call (ordinary reduction-order/exp-eval noise between two different
         implementations), but fp16/bf16 has no headroom to absorb that --
         compounded across the default 6-layer stack it grew into a full
         fp16-ulp divergence and real accuracy FAILures, so the softmax
         reduction itself stays on ATen's (bit-exact) kernel. Falls back to
         an equivalent ATen chain (also bit-exact with BaselineSelfAttention,
         using a single combined mask) if Triton is unavailable, off-CUDA, or
         a launch raises; a failure permanently disables the Triton path for
         that model instance (see `_triton_softmax_disabled`), so it can
         never be attempted for the first time from inside a CUDA graph
         capture -- capture always follows an eager warmup call on the
         identical shape/dtype that would already have hit and cached the
         same failure.

    Parameter names/shapes are identical to BaselineTransformer (this class
    adds no new nn.Parameter/buffer), so copy_model_weights()'s strict
    load_state_dict() works unchanged. All fused/cached tensors, and all CUDA
    graph state, are built lazily on first forward, since weight copying
    happens before the model is moved to its final device/dtype; graph state
    lives in plain Python attributes (dicts/sets/None), never as parameters
    or buffers.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # (seq_len, device) -> bool [S, S] tensor, True where a query
        # position may attend to a key position (causal-allowed). Used only
        # on the fp32/SDPA path.
        self._causal_allowed_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        # (seq_len, device) -> bool [S, S] tensor, True where a key position
        # must be masked out (upper triangle), matching
        # BaselineSelfAttention's causal_mask exactly. Used only on the
        # fp16/bf16 manual-math path so the masked_fill call is bit-for-bit
        # identical to the baseline.
        self._causal_disallowed_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        # Lazily-resolved, dtype-cached choice of attention implementation:
        # True -> use F.scaled_dot_product_attention (fp32: the baseline's
        #   ".float()" softmax upcast is then a no-op, so SDPA's math matches).
        # False -> reproduce the baseline's manual matmul/softmax/matmul
        #   arithmetic exactly (fp16/bf16: SDPA scales q before the matmul
        #   and keeps higher-precision probs internally, which diverges from
        #   the baseline by more than atol/rtol once compounded over layers).
        self._attn_use_sdpa: Optional[bool] = None
        self._attn_dtype_cache: Optional[torch.dtype] = None

        # Configuration keys already reported through TJ_DEBUG_GATE. This
        # suppresses duplicate diagnostics only; it does not cache or affect
        # the precision policy.
        self._fp16_gemm_reported: set = set()

        # (tuple(x.shape), x.dtype, x.device, causal, mask_kind) -> bool.
        # Verdict of whether the fused-QKV GEMM is BIT-EXACT (torch.equal on
        # the whole forward output, not merely within tolerance) against the
        # three separate projections for that configuration, on the fp16/bf16
        # manual-math path -- see _probe_fused_qkv_exact and class docstring
        # item 1. Never populated for fp32 models, which fuse QKV
        # unconditionally as part of the SDPA path.
        self._fused_qkv_exact: Dict[Tuple, bool] = {}

        # (tuple(x.shape), x.dtype, x.device, causal, mask_kind, ...) -> bool.
        # Whether the block-triangular causal Q@K^T (_blocked_causal_qk) was
        # proved BIT-EXACT (torch.equal on the whole forward output) against
        # the single full GEMM for that configuration. Only ever populated on
        # the fp16/bf16 manual-math path with causal=True; the fp32/SDPA path
        # never enters this dict.
        self._tri_qk_gate: Dict[Tuple, bool] = {}

        # (B, H, S, dtype, device, block) -> persistent scaled-scores buffer
        # whose above-staircase region is pre-filled with -inf. Reused by
        # every layer and every CUDA-graph replay; see
        # _tri_scale_mask_kernel for why it must never be written above
        # the staircase.
        self._tri_scaled_bufs: Dict[Tuple, torch.Tensor] = {}

        # (tuple(x.shape), x.dtype, x.device) -> (split_fn, merge_fn) | None.
        # Verified-bit-exact torch.compile'd layout helpers for that
        # configuration, or None to stay on ATen's copy kernels. `False` is
        # used as the "not probed yet" sentinel, since None is a real
        # verdict. See _resolve_compiled_layout.
        self._compiled_layout_cache: Dict[Tuple, object] = {}

        # --- CUDA graph capture/replay state (plain attributes only: no
        # nn.Parameter, no registered buffer, so load_state_dict(strict=True)
        # is unaffected). ---
        # (tuple(x.shape), x.dtype, x.device, causal, mask_kind) -> entry.
        self._graph_cache: Dict[Tuple, _GraphCacheEntry] = {}
        # Cache keys that failed capture once (or were never attempted
        # because torch.compiler.is_compiling() was true at that shape) --
        # permanently forced to the eager path from then on.
        self._graph_unsupported: set = set()
        # One shared CUDA graph memory pool for every graph this model
        # captures, to limit allocator fragmentation across distinct keys.
        self._graph_pool = None
        # Step 20: a per-dispatch-key torch.compile'd whole-core candidate
        # plus its margin-gated verdict. These are plain attributes so the
        # model remains strict-state-dict compatible. A small candidate uses
        # an Inductor cudagraph tree; a capacity-sensitive one uses Inductor
        # fusion without a retained CUDA graph. Neither is ever nested in the
        # hand-rolled CUDA graph above.
        self._fused_graph_gate: Dict[Tuple, bool] = {}
        self._fused_graph_cores: Dict[Tuple, object] = {}
        self._fused_graph_disabled: bool = False
        # (mask.data_ptr(), mask.shape, mask.dtype, mask._version) -> whether
        # the mask is all-True. Memoized so that repeated forward() calls
        # with the *same* mask tensor object (e.g. the fixed input reused
        # across a benchmark loop) cost at most one device->host sync total,
        # not one per call.
        self._mask_all_true_cache: Dict[Tuple, bool] = {}

        # Fused scale+mask Triton kernel (see _fused_scale_mask_kernel),
        # used only on the fp16/bf16 manual-math attention branch ahead of
        # ATen's own (bit-exact) fp32 softmax. Starts disabled if Triton
        # isn't importable at all; otherwise flips to True (permanently,
        # for the life of this module instance) the first time a launch
        # actually raises, so a real failure can never surface again later
        # -- including inside a CUDA graph capture, which always follows an
        # identical eager warmup call that would have hit the same failure
        # first. Plain bool attribute, not a buffer/parameter.
        self._triton_softmax_disabled: bool = not _TRITON_AVAILABLE

        # (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
        #  use_fp16_gemm, use_fused_qkv) -> bool. Verdict of whether the
        # fused residual-add + LayerNorm Triton kernel (_add_ln_kernel) is
        # accurate enough for that configuration -- see _probe_fused_ln.
        self._fused_ln_gate: Dict[Tuple, bool] = {}
        # Set permanently if the fused-LayerNorm kernel ever fails to
        # compile or launch, so a broken Triton install costs at most one
        # failed probe for the life of this model instance.
        self._fused_ln_disabled: bool = not _TRITON_AVAILABLE

        # Same shape of state for the short-sequence attention kernel
        # (_short_attn_kernel): per-configuration verdict, plus a permanent
        # kill switch if it ever fails to compile or launch.
        self._short_attn_gate: Dict[Tuple, bool] = {}
        self._short_attn_disabled: bool = not _TRITON_AVAILABLE

        # Same again for the fused GEMM+GELU FFN projection
        # (_gemm_gelu_kernel).
        self._fused_ffn_gate: Dict[Tuple, bool] = {}
        self._fused_ffn_disabled: bool = not _TRITON_AVAILABLE

        # Same again for routing the q/k/v, out_proj and ffn_out projections
        # through the Triton GEMM instead of cuBLAS (_triton_linear).
        self._tri_linear_gate: Dict[Tuple, bool] = {}
        self._tri_linear_disabled: bool = not _TRITON_AVAILABLE

    def _get_causal_allowed(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        cached = self._causal_allowed_cache.get(key)
        if cached is not None:
            return cached
        allowed = torch.tril(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
        )
        self._causal_allowed_cache[key] = allowed
        return allowed

    def _get_causal_disallowed(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        cached = self._causal_disallowed_cache.get(key)
        if cached is not None:
            return cached
        disallowed = torch.ones(
            (seq_len, seq_len), device=device, dtype=torch.bool
        ).triu(diagonal=1)
        self._causal_disallowed_cache[key] = disallowed
        return disallowed

    def _resolve_attention_mode(self) -> bool:
        """Pick the attention implementation from the model's current
        compute dtype (read off a parameter, since the model is moved to
        its final dtype via `.to()` after construction). Cached and
        invalidated only when the dtype actually changes."""
        dtype = self.final_norm.weight.dtype
        if dtype != self._attn_dtype_cache:
            self._attn_dtype_cache = dtype
            self._attn_use_sdpa = dtype == torch.float32
        return self._attn_use_sdpa

    def _resolve_mask_active(self, valid_token_mask: Optional[torch.Tensor]) -> bool:
        """Resolve whether the mask actually disables any token, doing at
        most one device->host sync per distinct mask tensor (memoized on
        (data_ptr, shape, dtype, _version) so repeated calls with the same
        mask object -- e.g. the fixed input reused across a benchmark loop
        -- are free after the first). Must be called BEFORE any graph
        capture/replay: this is the only sync in the whole forward path, and
        it must never happen inside a captured region (it would raise
        "operation failed due to a previous error during capture" and poison
        the CUDA context for the rest of the process)."""
        if valid_token_mask is None:
            return False
        if torch.compiler.is_compiling():
            # Being traced by dynamo (--compile-user): data_ptr()/._version
            # based memoization is neither traceable nor needed here (graph
            # capture itself is unconditionally skipped under compilation --
            # see _graph_capture_allowed), so just resolve the boolean
            # directly, the same data-dependent op the pre-graph-capture
            # code always used.
            return not bool(torch.all(valid_token_mask))
        try:
            version = valid_token_mask._version
        except RuntimeError:
            # Inference tensors (created under torch.inference_mode(), as
            # the harness's accuracy-check path does) don't track a version
            # counter at all. data_ptr+shape+dtype alone is still a fine key
            # here: within this benchmark, mask tensors are never mutated
            # in place after creation.
            version = None
        cache_key = (
            valid_token_mask.data_ptr(),
            tuple(valid_token_mask.shape),
            valid_token_mask.dtype,
            version,
        )
        all_true = self._mask_all_true_cache.get(cache_key)
        if all_true is None:
            all_true = bool(torch.all(valid_token_mask))  # <-- the one sync
            self._mask_all_true_cache[cache_key] = all_true
        return not all_true

    def _graph_capture_allowed(self, x: torch.Tensor) -> bool:
        if x.device.type != "cuda":
            return False
        if torch.compiler.is_compiling():
            # dynamo/inductor may be tracing this forward call (--compile-user
            # wraps the whole model in torch.compile); graph capture must
            # never run underneath that trace.
            return False
        return self._graph_memory_safe(x)

    def _graph_memory_safe(self, x: torch.Tensor) -> bool:
        """Whether either CUDA-graph implementation may retain this shape.

        CUDA graphs trade allocation reuse for a private, persistent pool.
        A shape whose eager peak fits can therefore still OOM at capture when
        the pool coexists with the eager baseline and the second model. This
        deliberately conservative static estimate declines those shapes
        before allocating anything. It is a pure function of the shape,
        configuration, dtype, and installed card, so it is safe before graph
        capture and stable across otherwise-idle benchmark runs.
        """
        if x.device.type != "cuda":
            return False
        batch, seq_len, d_model = x.shape
        if self._ffn_chunk_size(batch, seq_len, self.config.ffn_dim, x.device) is not None:
            # Chunking exists specifically for capacity-bound FFNs. Capturing
            # its temporary chunks would turn that bounded peak into retained
            # graph-pool memory, defeating the mechanism.
            return False
        try:
            total_bytes = torch.cuda.get_device_properties(x.device).total_memory
        except Exception:
            # A failed property query must never turn an unknown-capacity card
            # into a graph-capture experiment.
            return False

        element_bytes = x.element_size()
        rows = batch * seq_len
        input_bytes = rows * d_model * element_bytes
        # The core can have the residual stream, normalized activations,
        # QKV/context temporaries, and FFN intermediates live around a layer
        # boundary. The multipliers intentionally overstate the liveness:
        # decline is cheap, an OOM poisons a benchmark process.
        score_bytes = batch * self.config.num_heads * seq_len * seq_len * element_bytes
        hidden_bytes = rows * self.config.ffn_dim * element_bytes
        estimated_pool_bytes = 10 * input_bytes + 3 * score_bytes + 4 * hidden_bytes
        return estimated_pool_bytes <= int(total_bytes * _GRAPH_MEMORY_BUDGET_FRAC)

    def _fused_graph_memory_safe(self, x: torch.Tensor, use_sdpa: bool) -> bool:
        """Capacity gate for the Inductor whole-core candidate.

        This is intentionally distinct from `_graph_memory_safe`: the latter
        describes the retained allocations of the eager, hand-rolled graph.
        On the fp32 SDPA path, Inductor retains neither the eager score nor
        probability tensor, so charging it for those tensors would reject
        the row-6 candidate before it can demonstrate that lower footprint.
        The estimate still reserves a large fixed margin for both models,
        outputs retained by the accuracy probe, CUDA workspaces, and the
        runtime outside the compiled graph.
        """
        if x.device.type != "cuda":
            return False
        batch, seq_len, d_model = x.shape
        if self._ffn_chunk_size(batch, seq_len, self.config.ffn_dim, x.device) is not None:
            return False
        try:
            total_bytes = torch.cuda.get_device_properties(x.device).total_memory
        except Exception:
            return False

        element_bytes = x.element_size()
        rows = batch * seq_len
        input_bytes = rows * d_model * element_bytes
        hidden_bytes = rows * self.config.ffn_dim * element_bytes
        score_bytes = 0 if use_sdpa else (
            batch * self.config.num_heads * seq_len * seq_len * element_bytes
        )
        estimated_pool_bytes = 6 * input_bytes + 4 * hidden_bytes + 3 * score_bytes
        return estimated_pool_bytes <= int(total_bytes * _FUSED_GRAPH_MEMORY_BUDGET_FRAC)

    @staticmethod
    def _fused_graph_requested() -> bool:
        """Experimental A/B arm; disabled unless the caller explicitly asks."""
        return os.environ.get("TJ_FUSED_GRAPH", "0") not in ("", "0", "false", "False")

    def _probe_fused_graph(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        use_fused_ffn: bool,
        use_tri_qk: bool,
        use_tri_linear: bool,
        key: Tuple,
    ) -> bool:
        """Compile and admit the whole-core Inductor candidate once per key.

        The eager `_forward_core` reference has the exact arithmetic of the
        existing hand-rolled graph, without allocating that graph's retained
        pool. The candidate is compiled and invoked here, before timing and
        under the caller's actual dispatch mode; later forwards only call the
        already-warmed callable. It is never wrapped in `torch.cuda.graph`,
        avoiding the double-pool OOM mode. For a shape the hand-rolled graph
        rejected on capacity grounds, keep Inductor fusion but explicitly
        disable its cudagraph tree too: the harness needs five eager-reference
        trials, and retaining that private pool starves trial two on row 6.
        """
        if self._fused_graph_disabled or not self._fused_graph_memory_safe(x, use_sdpa):
            return False
        try:
            with torch.inference_mode():
                reference = self._forward_core(
                    x, valid_token_mask, mask_active, causal, use_sdpa,
                    use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                    use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear,
                )

            compiled = self._fused_graph_cores.get(key)
            if compiled is None:
                def core(core_x: torch.Tensor, core_mask: Optional[torch.Tensor]):
                    return self._forward_core(
                        core_x, core_mask, mask_active, causal, use_sdpa,
                        use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                        use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear,
                    )

                # `reduce-overhead` admits both Inductor fusion and its own
                # CUDA graph for shapes whose retained eager pool is safe.
                # On capacity-sensitive shapes, preserve fusion but opt out
                # of Inductor's retained graph pool; this is still a compiled
                # graph, just one replayed by normal launches. Keep the
                # Windows static-launcher workaround around construction and
                # first execution, when Dynamo specializes for this caller's
                # dispatch key set.
                with _no_static_cuda_launcher():
                    if self._graph_memory_safe(x):
                        compiled = torch.compile(core, mode="reduce-overhead")
                    else:
                        compiled = torch.compile(
                            core,
                            # Avoid static-shape buffer retention for the
                            # large row-6 activation. The callable is still
                            # cached on this exact configuration; `dynamic`
                            # asks Inductor not to turn its intermediates into
                            # a second model-lifetime arena.
                            dynamic=True,
                            options={
                                "triton.cudagraphs": False,
                                # Inductor's default intermediate pool keeps
                                # row 6's multi-GiB work buffers live between
                                # calls. The harness keeps the prior eager
                                # reference/output live until the next trial,
                                # so that pool leaves the baseline unable to
                                # form its next score tensor. Let normal CUDA
                                # allocation reuse handle this large shape.
                                "memory_pool": "none",
                            },
                        )
                    candidate = compiled(x, valid_token_mask)
                self._fused_graph_cores[key] = compiled
            else:
                with _no_static_cuda_launcher():
                    candidate = compiled(x, valid_token_mask)

            atol, rtol = _fast_path_gate_thresholds()
            abs_err = (candidate.float() - reference.float()).abs()
            rel_err = abs_err / reference.float().abs().clamp_min(1e-12)
            ok = ((abs_err <= _FAST_PATH_GATE_MARGIN * atol)
                  | (rel_err <= _FAST_PATH_GATE_MARGIN * rtol))
            passed = bool(torch.all(ok).item())
            detail = (f"failed={int((~ok).sum().item())}/{ok.numel()} "
                      f"max_abs={abs_err.max().item():.6g}")
        except Exception:
            self._fused_graph_disabled = True
            self._fused_graph_cores.pop(key, None)
            # A failed compile/capture can leave releasable compiler scratch
            # in the caching allocator. This is outside every captured region.
            if x.device.type == "cuda":
                torch.cuda.empty_cache()
            return False

        if os.environ.get("TJ_DEBUG_GATE"):
            print(
                f"[fused-graph-gate] shape={tuple(x.shape)} dtype={x.dtype} "
                f"causal={causal} mask_active={mask_active} "
                f"margin={_FAST_PATH_GATE_MARGIN} {detail} -> "
                f"{'ENABLE' if passed else 'DISABLE (hand-rolled graph/eager)'} "
                f"inductor-fused graph",
                file=sys.stderr,
            )
        if not passed:
            # Do not retain Inductor's graph pool after a declined candidate
            # and then build the hand-rolled fallback alongside it.
            self._fused_graph_cores.pop(key, None)
            if x.device.type == "cuda":
                torch.cuda.empty_cache()
        return passed

    def _resolve_fused_graph_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        use_fused_ffn: bool,
        use_tri_qk: bool,
        use_tri_linear: bool,
        mask_kind: str,
    ) -> Tuple[bool, Tuple]:
        """Return the opt-in fused-core verdict and its full cache key."""
        key = (
            tuple(x.shape), x.dtype, x.device, causal, mask_kind,
            use_fp16_gemm, use_fused_qkv, layout is not None, use_fused_ln,
            use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear,
            torch.is_inference_mode_enabled(), torch.is_grad_enabled(),
        )
        if not self._fused_graph_requested() or x.device.type != "cuda":
            return False, key
        cached = self._fused_graph_gate.get(key)
        if cached is None:
            cached = self._probe_fused_graph(
                x, valid_token_mask, mask_active, causal, use_sdpa,
                use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear, key,
            )
            self._fused_graph_gate[key] = cached
        return cached, key

    def _resolve_fp16_gemm_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        mask_kind: str,
    ) -> bool:
        """Policy for the fp32-only fp16-GEMM path (docstring item 7). Never
        touches the fp16/bf16 model paths (use_sdpa is fp32-exclusive, see
        _resolve_attention_mode) and never runs fp16 GEMMs off-CUDA (fp16
        matmul support/perf there is neither validated nor the point).
        Otherwise it enables the path directly. Correctness is judged only
        by the harness's external accuracy tests."""
        if not use_sdpa or x.device.type != "cuda":
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind)
        if os.environ.get("TJ_DEBUG_GATE") and key not in self._fp16_gemm_reported:
            print(
                f"[fp16-gemm-policy] shape={tuple(x.shape)} causal={causal} "
                f"mask_active={mask_active} -> ENABLE fp16-GEMM",
                file=sys.stderr,
            )
            self._fp16_gemm_reported.add(key)
        return True

    def _probe_fused_qkv_exact(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
    ) -> bool:
        """Decide whether the fused-QKV GEMM may be used on the fp16/bf16
        manual-math path, by running the whole forward BOTH ways on this
        configuration's real first-forward input and requiring
        `torch.equal` -- bit-identical output, not "within tolerance".

        Why this is a runtime probe and not a static condition. Packing
        q/k/v into one [3*d_model, d_model] GEMM changes nothing
        mathematically, and on this hardware it is bit-exact for **every**
        bf16 shape tried and for most fp16 shapes -- but not all of them.
        The reason is NOT the "different cuBLAS kernel selection" this
        docstring used to claim: profiling shows the separate (N=d_model)
        and fused (N=3*d_model) GEMMs dispatch to the *identical* cutlass
        kernel (`cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_
        tn_align8` at the default shape), so the tile shape and K-tiling
        are the same. What differs is a cuBLASLt *algorithm* choice
        (threadblock swizzle / CTA ordering) that shares one cutlass
        template name and is not observable or controllable from Python:
          - it was NOT split-K driven -- forcing `CUBLAS_WORKSPACE_CONFIG=
            :0:0` (no workspace, so no split-K algorithms) leaves the
            exactness pattern completely unchanged, and only slows the
            GEMMs down.
          - it is NOT monotonic in M, so it cannot be captured by a
            threshold. Measured at d_model=512, fp16, M = batch*seq_len:
            M=256 inexact, M=512 exact, M=1024 inexact, M=2048/4096/8192/
            16384 exact. At d_model=128 every M tried was exact; at
            d_model=1024 the small-M end was inexact.
        This is the same class of trap as the FFN chunk-size
        non-monotonicity documented on ChunkedBaselineTransformerBlock: any
        hand-written (M, K, N, dtype) predicate would be curve-fitting a
        closed-source heuristic that a cuBLAS or driver update can move.
        Measuring the actual answer for the actual configuration is both
        cheaper to maintain and strictly safer.

        Cost is two extra forwards, ONCE per configuration key at warmup
        (then cached in _fused_qkv_exact), and nothing at steady state. This
        is a device->host sync and must run before any CUDA graph capture,
        never inside a captured region.

        Comparing the whole forward output (rather than one layer's q/k/v)
        is deliberate: every layer's projection input differs, so a
        per-layer spot check on a single layer's weights could pass by luck
        while another layer diverges. Any divergence anywhere reaches the
        output, so `torch.equal` on the output is the strictly stronger
        check.
        """
        with torch.inference_mode():
            separate = self._forward_core(
                x, valid_token_mask, mask_active, causal,
                use_sdpa=False, use_fp16_gemm=False, use_fused_qkv=False,
            )
            fused = self._forward_core(
                x, valid_token_mask, mask_active, causal,
                use_sdpa=False, use_fp16_gemm=False, use_fused_qkv=True,
            )
        passed = bool(torch.equal(separate, fused))

        if os.environ.get("TJ_DEBUG_GATE"):
            n_diff = int((separate != fused).sum().item())
            print(
                f"[fused-qkv-probe] shape={tuple(x.shape)} dtype={x.dtype} "
                f"causal={causal} mask_active={mask_active} "
                f"differing={n_diff}/{separate.numel()} -> "
                f"{'ENABLE' if passed else 'DISABLE (keep 3 separate GEMMs)'} fused-QKV",
                file=sys.stderr,
            )
        return passed

    def _resolve_fused_qkv_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        mask_kind: str,
    ) -> bool:
        """Gate for the fp16/bf16 fused-QKV GEMM. Returns False for the
        fp32/SDPA path (which already fuses QKV unconditionally -- fusion
        there is driven by `use_sdpa`, not by this flag) and off CUDA (where
        the cuBLAS kernel-selection effect this probes for does not apply and
        the extra probe forwards would just be overhead). Otherwise probes
        once per configuration key and caches the verdict -- see
        _probe_fused_qkv_exact."""
        if use_sdpa or x.device.type != "cuda":
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind)
        cached = self._fused_qkv_exact.get(key)
        if cached is not None:
            return cached
        verdict = self._probe_fused_qkv_exact(x, valid_token_mask, mask_active, causal)
        self._fused_qkv_exact[key] = verdict
        return verdict

    def _probe_tri_linear(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        use_fused_ffn: bool,
        use_tri_qk: bool,
    ) -> bool:
        """Decide whether the q/k/v, out_proj and ffn_out projections may run
        on the Triton GEMM instead of cuBLAS.

        Measured bit-exact at every shape with K <= 512 (see
        _TRI_LINEAR_MIN_ROWS' table), so like the fused FFN projection this is
        offered on the fp16/bf16 paths with a `torch.equal` bar, and on fp32
        with the usual margin-scaled criterion against the exact fp32
        reference.
        """
        if self._tri_linear_disabled or x.device.type != "cuda":
            return False
        if not _tri_linear_supported(x.shape[0] * x.shape[1]):
            return False

        try:
            with torch.inference_mode():
                candidate = self._forward_core(
                    x, valid_token_mask, mask_active, causal, use_sdpa,
                    use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                    use_short_attn, use_fused_ffn, use_tri_qk,
                    use_tri_linear=True,
                )
                if x.dtype == torch.float32:
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal,
                        use_sdpa=True, use_fp16_gemm=False,
                        use_fused_qkv=False, layout=None, use_fused_ln=False,
                        use_short_attn=False, use_fused_ffn=False,
                        use_tri_qk=False, use_tri_linear=False,
                    )
                else:
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal, use_sdpa,
                        use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                        use_short_attn, use_fused_ffn, use_tri_qk,
                        use_tri_linear=False,
                    )
        except Exception:
            self._tri_linear_disabled = True
            return False

        if x.dtype == torch.float32:
            atol, rtol = _fast_path_gate_thresholds()
            abs_err = (candidate - reference).abs()
            rel_err = abs_err / reference.abs().clamp_min(1e-12)
            ok = ((abs_err <= _FAST_PATH_GATE_MARGIN * atol)
                  | (rel_err <= _FAST_PATH_GATE_MARGIN * rtol))
            passed = bool(torch.all(ok).item())
            detail = (f"failed={int((~ok).sum().item())}/{ok.numel()} "
                      f"max_abs={abs_err.max().item():.6g}")
        else:
            passed = bool(torch.equal(candidate, reference))
            detail = f"differing={int((candidate != reference).sum().item())}/{reference.numel()}"

        if os.environ.get("TJ_DEBUG_GATE"):
            print(
                f"[tri-linear-probe] shape={tuple(x.shape)} dtype={x.dtype} "
                f"{detail} -> "
                f"{'ENABLE' if passed else 'DISABLE (cuBLAS)'} triton-linear",
                file=sys.stderr,
            )
        return passed

    def _resolve_tri_linear_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        use_fused_ffn: bool,
        use_tri_qk: bool,
        mask_kind: str,
    ) -> bool:
        """Probe once per configuration key and cache -- see
        _probe_tri_linear."""
        if self._tri_linear_disabled or x.device.type != "cuda":
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
               use_fp16_gemm, use_fused_qkv, layout is not None,
               use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk)
        cached = self._tri_linear_gate.get(key)
        if cached is not None:
            return cached
        verdict = self._probe_tri_linear(
            x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn,
            use_fused_ffn, use_tri_qk,
        )
        self._tri_linear_gate[key] = verdict
        return verdict

    def _probe_fused_ffn(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
    ) -> bool:
        """Decide whether the fused GEMM+GELU may replace the FFN's first
        projection plus its GELU, for this configuration.

        Unlike the other two Triton kernels added in step 11, this one CAN be
        bit-exact (see _gemm_gelu_kernel's module comment: same libdevice
        `erf` as ATen, and the accumulator rounded to the output dtype before
        the GELU), so it is offered to the fp16/bf16 paths too and the bar
        there is `torch.equal` -- the step 8 pattern. On fp32 the bar is the
        usual margin-scaled harness criterion against the exact fp32
        reference, so that every enabled approximation is judged together.
        """
        if self._fused_ffn_disabled or x.device.type != "cuda":
            return False
        batch, seq_len, _ = x.shape
        if self._ffn_chunk_size(
                batch, seq_len, self.config.ffn_dim, x.device) is not None:
            # The chunked path exists to bound peak memory at extreme
            # ffn_dim; materializing one fused [rows, ffn_dim] output would
            # defeat it.
            return False
        if not _fused_ffn_supported(batch * seq_len, self.config.ffn_dim):
            return False

        try:
            with torch.inference_mode():
                candidate = self._forward_core(
                    x, valid_token_mask, mask_active, causal, use_sdpa,
                    use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                    use_short_attn, use_fused_ffn=True,
                )
                if x.dtype == torch.float32:
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal,
                        use_sdpa=True, use_fp16_gemm=False,
                        use_fused_qkv=False, layout=None, use_fused_ln=False,
                        use_short_attn=False, use_fused_ffn=False,
                    )
                else:
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal, use_sdpa,
                        use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                        use_short_attn, use_fused_ffn=False,
                    )
        except Exception:
            self._fused_ffn_disabled = True
            return False

        if x.dtype == torch.float32:
            atol, rtol = _fast_path_gate_thresholds()
            abs_err = (candidate - reference).abs()
            rel_err = abs_err / reference.abs().clamp_min(1e-12)
            ok = ((abs_err <= _FAST_PATH_GATE_MARGIN * atol)
                  | (rel_err <= _FAST_PATH_GATE_MARGIN * rtol))
            passed = bool(torch.all(ok).item())
            detail = (f"failed={int((~ok).sum().item())}/{ok.numel()} "
                      f"max_abs={abs_err.max().item():.6g}")
        else:
            passed = bool(torch.equal(candidate, reference))
            detail = f"differing={int((candidate != reference).sum().item())}/{reference.numel()}"

        if os.environ.get("TJ_DEBUG_GATE"):
            print(
                f"[fused-ffn-probe] shape={tuple(x.shape)} dtype={x.dtype} "
                f"ffn_dim={self.config.ffn_dim} {detail} -> "
                f"{'ENABLE' if passed else 'DISABLE (cuBLAS + ATen GELU)'} fused-FFN",
                file=sys.stderr,
            )
        return passed

    def _resolve_fused_ffn_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        mask_kind: str,
    ) -> bool:
        """Probe once per configuration key and cache -- see
        _probe_fused_ffn."""
        if self._fused_ffn_disabled or x.device.type != "cuda":
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
               use_fp16_gemm, use_fused_qkv, layout is not None,
               use_fused_ln, use_short_attn)
        cached = self._fused_ffn_gate.get(key)
        if cached is not None:
            return cached
        verdict = self._probe_fused_ffn(
            x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn,
        )
        self._fused_ffn_gate[key] = verdict
        return verdict

    def _probe_short_attn(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_fp16_gemm: bool,
        layout,
    ) -> bool:
        """Decide whether the single-block short-sequence attention kernel
        may replace SDPA for this configuration.

        Judged the same way the fused LayerNorm is: the candidate is the
        whole forward with the kernel enabled, the reference is the EXACT
        fp32 path (no fp16 GEMMs, no fused LayerNorm, no custom attention),
        and the bar is the harness criterion scaled by
        _FAST_PATH_GATE_MARGIN. Comparing against the exact reference rather
        than against the SDPA-based fast path is the point: this kernel's
        softmax is a different reduction from SDPA's, so it must not be
        allowed to spend a fresh copy of the error budget the fp16-GEMM gate
        has already spent.

        Only offered on the fp32/SDPA path. The fp16/bf16 paths reproduce
        BaselineSelfAttention's arithmetic bit-for-bit and a different
        softmax reduction cannot do that -- step 4 measured that exact
        substitution compounding to max_abs 0.0078 over 6 layers.
        """
        if self._short_attn_disabled or x.device.type != "cuda":
            return False
        head_dim = self.config.d_model // self.config.num_heads
        if not _short_attn_supported(x.shape[1], head_dim):
            return False

        try:
            with torch.inference_mode():
                candidate = self._forward_core(
                    x, valid_token_mask, mask_active, causal, use_sdpa=True,
                    use_fp16_gemm=use_fp16_gemm, use_fused_qkv=False,
                    layout=layout, use_fused_ln=False, use_short_attn=True,
                )
                reference = self._forward_core(
                    x, valid_token_mask, mask_active, causal, use_sdpa=True,
                    use_fp16_gemm=False, use_fused_qkv=False, layout=None,
                    use_fused_ln=False, use_short_attn=False,
                )
        except Exception:
            self._short_attn_disabled = True
            return False

        atol, rtol = _fast_path_gate_thresholds()
        abs_err = (candidate - reference).abs()
        rel_err = abs_err / reference.abs().clamp_min(1e-12)
        ok = ((abs_err <= _FAST_PATH_GATE_MARGIN * atol)
              | (rel_err <= _FAST_PATH_GATE_MARGIN * rtol))
        passed = bool(torch.all(ok).item())

        if os.environ.get("TJ_DEBUG_GATE"):
            print(
                f"[short-attn-probe] shape={tuple(x.shape)} heads={self.config.num_heads} "
                f"causal={causal} mask_active={mask_active} "
                f"failed={int((~ok).sum().item())}/{ok.numel()} "
                f"max_abs={abs_err.max().item():.6g} -> "
                f"{'ENABLE' if passed else 'DISABLE (SDPA)'} short-attn",
                file=sys.stderr,
            )
        return passed

    def _resolve_short_attn_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        layout,
        mask_kind: str,
    ) -> bool:
        """Probe once per configuration key and cache -- see
        _probe_short_attn."""
        if not use_sdpa or self._short_attn_disabled or x.device.type != "cuda":
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
               use_fp16_gemm, layout is not None)
        cached = self._short_attn_gate.get(key)
        if cached is not None:
            return cached
        verdict = self._probe_short_attn(
            x, valid_token_mask, mask_active, causal, use_fp16_gemm, layout)
        self._short_attn_gate[key] = verdict
        return verdict

    def _probe_fused_ln(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_short_attn: bool,
    ) -> bool:
        """Decide whether the fused residual-add + LayerNorm kernel may run
        for this configuration, by computing the whole forward both ways on
        the real first-forward input.

        Two different bars, for the same reason the rest of this class has
        two: the LayerNorm reduction is the only part of the fusion that is
        not bit-exact by construction (see _add_ln_kernel's module comment),
        and low precision has no room for "not bit-exact".

        * fp16/bf16 models: `torch.equal` -- bit-identical or nothing.
          A different mean/variance reduction tree is expected to fail this,
          and that is the correct outcome: step 4 of OPTIMIZATION_LOG.md
          measured a comparable per-call reduction difference in the softmax
          compounding to max_abs 0.0078 over 6 layers, a real accuracy
          failure. The probe is still run rather than short-circuited so the
          answer stays measured rather than assumed, per this project's
          standing practice with cuBLAS/inductor behaviour that turned out
          to be shape-dependent.

        * fp32 models: the harness's own criterion scaled by
          _FAST_PATH_GATE_MARGIN, measured against the EXACT fp32 reference
          path (no fp16 GEMMs, no fused LayerNorm) rather than against the
          fp16-GEMM path this candidate is built on. That is deliberate:
          both approximations are then judged together against the true
          reference, so enabling this one cannot quietly spend a second
          copy of the same error budget the fp16-GEMM gate already spent.

        Like every other probe here this compiles Triton kernels and syncs,
        so it must run before CUDA graph capture, never inside a captured
        region. It also has the useful side effect of forcing every
        (shape, flag) variant of the kernel to compile at warmup, so the
        captured region never triggers a JIT compile.
        """
        if self._fused_ln_disabled or x.device.type != "cuda":
            return False
        if triton.next_power_of_2(x.shape[-1]) > _ADD_LN_MAX_BLOCK_D:
            # Feature axis too wide for the single-block reduction this
            # kernel is built around. Not a failure -- just out of scope.
            return False

        try:
            with torch.inference_mode():
                candidate = self._forward_core(
                    x, valid_token_mask, mask_active, causal, use_sdpa,
                    use_fp16_gemm, use_fused_qkv, layout, use_fused_ln=True,
                    use_short_attn=use_short_attn,
                )
                if x.dtype == torch.float32:
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal,
                        use_sdpa=True, use_fp16_gemm=False,
                        use_fused_qkv=False, layout=None, use_fused_ln=False,
                        use_short_attn=False,
                    )
                else:
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal, use_sdpa,
                        use_fp16_gemm, use_fused_qkv, layout,
                        use_fused_ln=False, use_short_attn=use_short_attn,
                    )
        except Exception:
            # No Triton, an unsupported shape, or a launch failure. ATen
            # still produces correct results, only slower; never retry.
            self._fused_ln_disabled = True
            return False

        if x.dtype == torch.float32:
            atol, rtol = _fast_path_gate_thresholds()
            gate_atol = _FAST_PATH_GATE_MARGIN * atol
            gate_rtol = _FAST_PATH_GATE_MARGIN * rtol
            abs_err = (candidate - reference).abs()
            rel_err = abs_err / reference.abs().clamp_min(1e-12)
            ok = (abs_err <= gate_atol) | (rel_err <= gate_rtol)
            passed = bool(torch.all(ok).item())
            detail = (
                f"failed={int((~ok).sum().item())}/{ok.numel()} "
                f"max_abs={abs_err.max().item():.6g}"
            )
        else:
            passed = bool(torch.equal(candidate, reference))
            detail = f"differing={int((candidate != reference).sum().item())}/{reference.numel()}"

        if os.environ.get("TJ_DEBUG_GATE"):
            print(
                f"[fused-ln-probe] shape={tuple(x.shape)} dtype={x.dtype} "
                f"causal={causal} mask_active={mask_active} "
                f"fp16_gemm={use_fp16_gemm} {detail} -> "
                f"{'ENABLE' if passed else 'DISABLE (ATen LayerNorm chain)'} fused-LN",
                file=sys.stderr,
            )
        return passed

    def _resolve_fused_ln_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        mask_kind: str,
        use_short_attn: bool,
    ) -> bool:
        """Probe once per configuration key and cache -- see
        _probe_fused_ln. Keyed on the other fast-path flags too, because the
        candidate this probe validates is the whole forward *including*
        them, and the fusion's own numerics differ between the fp16-GEMM
        path (LayerNorm output rounded to fp16 on store) and the plain fp32
        one."""
        if self._fused_ln_disabled or x.device.type != "cuda":
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
               use_fp16_gemm, use_fused_qkv, layout is not None, use_short_attn)
        cached = self._fused_ln_gate.get(key)
        if cached is not None:
            return cached
        verdict = self._probe_fused_ln(
            x, valid_token_mask, mask_active, causal, use_sdpa,
            use_fp16_gemm, use_fused_qkv, layout, use_short_attn,
        )
        self._fused_ln_gate[key] = verdict
        return verdict

    def _get_tri_scaled_buffer(
        self, scores_raw: torch.Tensor, block: int
    ) -> torch.Tensor:
        """Persistent [B, H, S, S] buffer for the triangular scale+mask
        kernel, with everything above the block staircase set to -inf once,
        here, and never touched again.

        Cached per (B, H, S, dtype, device, block). Allocated on the first
        forward for a configuration -- which is always the probe, before any
        CUDA-graph capture -- so the captured graph refers to a stable
        pointer. It replaces the per-call `torch.empty_like` the full-width
        path allocates, so steady-state memory is comparable; it is held for
        the model's lifetime rather than returned to the allocator between
        calls."""
        B, H, S, SK = scores_raw.shape
        key = (B, H, S, SK, scores_raw.dtype, scores_raw.device, block)
        buf = self._tri_scaled_bufs.get(key)
        if buf is not None:
            return buf
        buf = torch.empty(B, H, S, SK, device=scores_raw.device,
                          dtype=scores_raw.dtype)
        neg_inf = float("-inf")
        for start in range(0, S, block):
            lim = min(start + block, SK)
            if lim < SK:
                buf[:, :, start:min(start + block, S), lim:] = neg_inf
        self._tri_scaled_bufs[key] = buf
        return buf

    def _probe_tri_qk_exact(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        use_fused_ffn: bool,
    ) -> bool:
        """Run the whole forward both ways on this configuration's real
        input and require `torch.equal` before allowing the block-triangular
        Q@K^T.

        The argument in _blocked_causal_qk's docstring says this must pass:
        only the GEMM's N is shortened, the reduction axis is untouched, and
        the skipped columns are overwritten with -inf anyway. It is probed
        regardless, for the reason step 8 records -- a claim about which
        cuBLAS kernel gets selected, and what it does, is exactly the kind
        of thing that has been wrong before in this file and that a driver
        update can silently change. Two extra forwards once per
        configuration key, nothing at steady state.
        """
        with torch.inference_mode():
            reference = self._forward_core(
                x, valid_token_mask, mask_active, causal, False,
                use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                use_short_attn, use_fused_ffn, use_tri_qk=False,
            )
            candidate = self._forward_core(
                x, valid_token_mask, mask_active, causal, False,
                use_fp16_gemm, use_fused_qkv, layout, use_fused_ln,
                use_short_attn, use_fused_ffn, use_tri_qk=True,
            )
        passed = bool(torch.equal(reference, candidate))

        if os.environ.get("TJ_DEBUG_GATE"):
            n_diff = int((reference != candidate).sum().item())
            print(
                f"[tri-qk-probe] shape={tuple(x.shape)} dtype={x.dtype} "
                f"block={_causal_qk_block_size(int(x.shape[1]))} "
                f"differing={n_diff}/{reference.numel()} -> "
                f"{'ENABLE' if passed else 'DISABLE (one full GEMM)'} "
                f"block-triangular Q@K^T",
                file=sys.stderr,
            )
        return passed

    def _resolve_tri_qk_enabled(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        use_fused_ln: bool,
        use_short_attn: bool,
        use_fused_ffn: bool,
        mask_kind: str,
    ) -> bool:
        """Gate for the block-triangular causal Q@K^T. Applies only on the
        fp16/bf16 manual-math path (the fp32/SDPA path lets SDPA apply
        causality itself) with causal=True, on CUDA, and only at sequence
        lengths where _causal_qk_block_size says the blocking pays for its
        extra launches. Probes once per configuration key and caches."""
        if use_sdpa or not causal or x.device.type != "cuda":
            return False
        if _causal_qk_block_size(int(x.shape[1])) is None:
            return False
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
               use_fp16_gemm, use_fused_qkv, layout is not None,
               use_fused_ln, use_short_attn, use_fused_ffn)
        cached = self._tri_qk_gate.get(key)
        if cached is not None:
            return cached
        verdict = self._probe_tri_qk_exact(
            x, valid_token_mask, mask_active, causal, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn,
            use_fused_ffn,
        )
        self._tri_qk_gate[key] = verdict
        return verdict

    def _resolve_compiled_layout(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        mask_kind: str,
    ):
        """Return (split_fn, merge_fn) of torch.compile'd layout helpers to
        use for this exact configuration, or None to stay on ATen.

        Probed by running a whole forward BOTH ways and requiring
        `torch.equal`, once per configuration key. The helpers contain no
        arithmetic (see _get_compiled_layout_helpers), so this should always
        pass -- it is insurance against inductor ever fusing something
        unexpected into a copy, not a tolerance gate.

        Probing with a real forward rather than with synthetic tensors is
        load-bearing, and was found the hard way. dynamo specializes on
        *strides*, not just shape/dtype, and the tensors these helpers
        actually receive are not always contiguous: with fused QKV enabled,
        q/k/v are non-contiguous column slices of one packed [B, S, 3*D]
        GEMM output. A probe that passed contiguous tensors therefore left a
        second variant to be compiled later -- and "later" turned out to be
        *inside torch.cuda.graph() capture*, where that compile hit the
        Windows static-launcher OverflowError, failed the capture, and
        permanently demoted the model to the eager path (measured: bf16
        1.48ms -> 3.44ms, i.e. this optimization made things 2.3x worse
        while still reporting bit-exact results). Driving the probe through
        _forward_core with the real flags compiles every variant the steady
        state will use, inside the workaround patch, before capture is ever
        attempted.

        Like every other probe in this class, this syncs and compiles, so it
        must run before CUDA graph capture, never inside a captured region."""
        if x.device.type != "cuda" or torch.compiler.is_compiling():
            return None
        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind,
               use_fp16_gemm, use_fused_qkv)
        cached = self._compiled_layout_cache.get(key, False)
        if cached is not False:
            return cached

        verdict = None
        helpers = _get_compiled_layout_helpers()
        if helpers is not None:
            try:
                with torch.inference_mode():
                    reference = self._forward_core(
                        x, valid_token_mask, mask_active, causal,
                        use_sdpa, use_fp16_gemm, use_fused_qkv, layout=None,
                    )
                    with _no_static_cuda_launcher():
                        candidate = self._forward_core(
                            x, valid_token_mask, mask_active, causal,
                            use_sdpa, use_fp16_gemm, use_fused_qkv, layout=helpers,
                        )
                if bool(torch.equal(reference, candidate)):
                    verdict = helpers
            except Exception:
                # Compile/launch failure (no triton, or the Windows
                # static-launcher bug surfacing somewhere the patch above
                # doesn't reach). Never retry in this process; ATen still
                # produces correct results, only slower.
                _COMPILED_LAYOUT["disabled"] = True
                verdict = None

        if os.environ.get("TJ_DEBUG_GATE"):
            print(
                f"[compiled-layout] shape={tuple(x.shape)} dtype={x.dtype} "
                f"fused_qkv={use_fused_qkv} sdpa={use_sdpa} -> "
                f"{'ENABLE' if verdict is not None else 'DISABLE (ATen copies)'}",
                file=sys.stderr,
            )
        self._compiled_layout_cache[key] = verdict
        return verdict

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        causal = self.config.causal
        use_sdpa = self._resolve_attention_mode()
        mask_active = self._resolve_mask_active(valid_token_mask)

        if valid_token_mask is None:
            mask_kind = "none"
        elif mask_active:
            mask_kind = "partial"
        else:
            mask_kind = "all_true"

        # Precision selection is a static dtype/device policy and does not
        # inspect the input or synchronize with the device.
        use_fp16_gemm = self._resolve_fp16_gemm_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa, mask_kind
        )
        # Also a device->host sync (torch.equal), for the same reason, so it
        # likewise has to be resolved here rather than inside _forward_core.
        use_fused_qkv = self._resolve_fused_qkv_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa, mask_kind
        )
        # Compiles inductor kernels and syncs on its verification compare,
        # so likewise resolved here, outside any captured region.
        layout = self._resolve_compiled_layout(
            x, valid_token_mask, mask_active, causal, use_sdpa,
            use_fp16_gemm, use_fused_qkv, mask_kind,
        )

        # Compiles Triton kernels and syncs on its verification compare,
        # so likewise resolved here, outside any captured region. Resolved
        # last: its fp32 bar judges the *combined* error of every fast path
        # enabled above, so it needs their verdicts first.
        use_short_attn = self._resolve_short_attn_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa,
            use_fp16_gemm, layout, mask_kind,
        )
        use_fused_ln = self._resolve_fused_ln_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa,
            use_fp16_gemm, use_fused_qkv, layout, mask_kind, use_short_attn,
        )
        use_fused_ffn = self._resolve_fused_ffn_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn, mask_kind,
        )
        # Also a torch.equal sync, so likewise resolved out here rather
        # than inside _forward_core / a captured region.
        use_tri_linear = self._resolve_tri_linear_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn,
            use_fused_ffn, False, mask_kind,
        )
        use_tri_qk = self._resolve_tri_qk_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn,
            use_fused_ffn, mask_kind,
        )

        # The whole-core Inductor candidate is an explicit experiment arm.
        # Its probe compiles and executes it before timing; a successful arm
        # must never be wrapped in this model's hand-rolled CUDA graph because
        # that would retain two independent graph pools for one shape.
        use_fused_graph, fused_graph_key = self._resolve_fused_graph_enabled(
            x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm,
            use_fused_qkv, layout, use_fused_ln, use_short_attn,
            use_fused_ffn, use_tri_qk, use_tri_linear, mask_kind,
        )
        if use_fused_graph:
            compiled = self._fused_graph_cores.get(fused_graph_key)
            assert compiled is not None, "fused graph was admitted without warmup"
            with _no_static_cuda_launcher():
                # Inductor cudagraph trees may reuse output storage, so retain
                # the public forward contract of _replay_graph and return an
                # independent result to the caller.
                return compiled(x, valid_token_mask).clone()

        if not self._graph_capture_allowed(x):
            return self._forward_core(x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm, use_fused_qkv, layout, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear)

        key = (tuple(x.shape), x.dtype, x.device, causal, mask_kind, use_fp16_gemm, use_fused_qkv, layout is not None, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear, use_fused_graph)

        if key in self._graph_unsupported:
            return self._forward_core(x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm, use_fused_qkv, layout, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear)

        entry = self._graph_cache.get(key)
        if entry is not None:
            return self._replay_graph(entry, x, valid_token_mask, mask_kind)

        try:
            entry = self._capture_graph(x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm, use_fused_qkv, layout, mask_kind, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear)
        except Exception:
            # Capture failed (or the CUDA context is unusable for capture on
            # this device/build). Never retry capture for this key; fall
            # back to eager permanently and keep serving correct results.
            self._graph_unsupported.add(key)
            return self._forward_core(x, valid_token_mask, mask_active, causal, use_sdpa, use_fp16_gemm, use_fused_qkv, layout, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear)

        self._graph_cache[key] = entry
        return self._replay_graph(entry, x, valid_token_mask, mask_kind)

    def _replay_graph(
        self,
        entry: _GraphCacheEntry,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_kind: str,
    ) -> torch.Tensor:
        entry.static_x.copy_(x)
        if mask_kind == "partial":
            entry.static_mask.copy_(valid_token_mask)
        entry.graph.replay()
        # Clone: the next replay() overwrites static_output in place, and
        # the caller may hold on to what forward() returns.
        return entry.static_output.clone()

    def _capture_graph(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool,
        use_fused_qkv: bool,
        layout,
        mask_kind: str,
        use_fused_ln: bool = False,
        use_short_attn: bool = False,
        use_fused_ffn: bool = False,
        use_tri_qk: bool = False,
        use_tri_linear: bool = False,
    ) -> _GraphCacheEntry:
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()

        static_x = x.clone()
        static_mask = valid_token_mask.clone() if mask_kind == "partial" else None

        # Warm up the eager path a few iterations on a side stream first,
        # per the documented torch.cuda.graph pattern -- this lets cuDNN/
        # cuBLAS pick kernels/workspaces outside of capture, where such
        # allocations are legal.
        #
        # The static-launcher patch is held across BOTH the warmup and the
        # capture, and that is load-bearing rather than belt-and-braces.
        # dynamo specializes compiled code on the DISPATCH KEY SET, not just
        # on shape and stride: a helper compiled while the probe ran under
        # `torch.inference_mode()` is a cache miss when the caller runs the
        # model under `torch.no_grad()` instead, and the resulting recompile
        # lands wherever the first such call happens -- which is inside
        # `torch.cuda.graph()`, where it hit the Windows static-launcher
        # OverflowError, failed the capture, and permanently demoted the
        # model to the eager path.
        #
        # Measured before this fix, default shape, forward called under
        # `torch.no_grad()` (the ordinary inference idiom) rather than
        # `inference_mode()`: fp16 1.55 ms -> 3.62 ms and bf16 1.30 ms ->
        # 3.21 ms, i.e. 2.3x slower, silently, while still producing correct
        # results. fp32 was unaffected because its SDPA path never uses the
        # compiled layout helpers.
        #
        # This is the same class of trap as the stride-specialization one in
        # step 9, on a different specialization axis, so the fix here is
        # deliberately the general one -- make the whole captured region safe
        # for a late recompile -- rather than another attempt to enumerate
        # which variants must be pre-compiled. It costs nothing at steady
        # state: replay never goes through the Python launcher at all.
        with _no_static_cuda_launcher():
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    self._forward_core(static_x, static_mask, mask_active, causal, use_sdpa, use_fp16_gemm, use_fused_qkv, layout, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                static_output = self._forward_core(static_x, static_mask, mask_active, causal, use_sdpa, use_fp16_gemm, use_fused_qkv, layout, use_fused_ln, use_short_attn, use_fused_ffn, use_tri_qk, use_tri_linear)

        return _GraphCacheEntry(
            graph=graph,
            static_x=static_x,
            static_mask=static_mask,
            static_output=static_output,
        )

    def _fused_attn_probs(
        self,
        scores_raw: torch.Tensor,
        scale: float,
        causal: bool,
        mask_active: bool,
        valid_token_mask: Optional[torch.Tensor],
        causal_disallowed: Optional[torch.Tensor],
        invalid_keys: Optional[torch.Tensor],
        out_dtype: torch.dtype,
        tri_block: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes softmax(scale * scores_raw, masked) in the model dtype,
        used only by the fp16/bf16 manual-math attention branch. Tries the
        fused Triton scale+mask kernel first (_triton_fused_scale_mask: one
        read of scores_raw, one write of the scaled+masked scores, replacing
        2-3 separate ATen kernels each touching the [B,H,S,S] score tensor),
        then finishes with a *native* `torch.softmax(scaled, dim=-1)` call --
        i.e. no dtype= kwarg and no explicit `.float()` upcast at all, run
        directly on the half/bfloat16 `scaled` tensor. This is NOT an
        approximation: ATen's CUDA softmax already accumulates in fp32
        internally and rounds once at the end, so this call is BIT-IDENTICAL
        to `torch.softmax(scaled.float(), dim=-1).to(dtype)` -- verified with
        torch.equal across (8,8,128,128), (4,8,2048,2048), (32,8,512,512),
        (8,16,256,256) and (8,4,4096,4096) in both fp16 and bf16, all
        max_abs=0. It is also strictly cheaper: no fp32 intermediate is ever
        materialized (at B=4/S=2048 fp16 that intermediate would be 512 MB,
        written and read back), so this removes ~1 read+write pass over the
        [B,H,S,S] score tensor versus the old dtype=torch.float32 form.

        That old form -- `torch.softmax(scaled, dim=-1, dtype=torch.float32)`
        -- looks equivalent but is NOT: passing dtype= makes ATen select a
        different reduction kernel than an explicit `.float()` cast followed
        by a plain softmax, and at long rows (verified: S=2048 and S=4096 in
        fp16; not observed at S<=512 fp16 or at any bf16 shape tried) that
        different kernel picks a different reduction order and diverges from
        `scores.float().softmax(...)` by a few ulps (measured up to
        max_abs=0.00048828 alone, compounding across a 6-layer stack into a
        real accuracy FAIL). That was the actual cause of the long_fp16
        accuracy failure this method used to have -- not a contiguity issue
        (matmul was verified contiguity-invariant on this cuBLAS build) but a
        genuinely different ATen softmax code path selected purely by
        row-length. The native no-cast form used below sidesteps that
        entirely, and was checked bit-exact at exactly the shapes that broke
        the dtype= form.

        The ATen fallback chain used when Triton isn't used at all is itself
        bit-exact with BaselineSelfAttention -- a single combined mask
        instead of two successive masked_fills, verified bit-exact against
        the original two-call form -- so accuracy is identical whichever path
        runs; only the number of kernel launches/memory touches differs.

        A Triton launch failure permanently disables the Triton path for
        the rest of this module instance's life (see
        `_triton_softmax_disabled`'s docstring for why: it must never be
        attempted for the first time from inside a CUDA graph capture, and
        capture always follows an eager warmup call on the identical shape
        that would hit the same failure first)."""
        scaled: Optional[torch.Tensor] = None
        if not self._triton_softmax_disabled and scores_raw.device.type == "cuda":
            try:
                if tri_block is not None:
                    # Causal only: write just the block-lower-triangle
                    # into a persistent buffer that already holds -inf
                    # above the staircase. See _tri_scale_mask_kernel.
                    scaled = _triton_tri_scale_mask(
                        scores_raw,
                        self._get_tri_scaled_buffer(scores_raw, tri_block),
                        scale, tri_block, mask_active, valid_token_mask,
                    )
                else:
                    scaled = _triton_fused_scale_mask(
                        scores_raw, scale, causal, mask_active, valid_token_mask
                    )
            except Exception:
                self._triton_softmax_disabled = True
                scaled = None

        if scaled is None:
            scaled = scores_raw * scale
            disallowed: Optional[torch.Tensor] = None
            if causal:
                disallowed = causal_disallowed
            if mask_active:
                disallowed = invalid_keys if disallowed is None else (disallowed | invalid_keys)
            if disallowed is not None:
                scaled = scaled.masked_fill(disallowed, float("-inf"))

        # Native softmax on the already-model-dtype `scaled` tensor: no
        # dtype= kwarg, no explicit .float() upcast. Bit-identical to the
        # baseline's `softmax(scores.float(), dim=-1).to(dtype)` (ATen's CUDA
        # softmax accumulates in fp32 internally regardless), and avoids both
        # the fp32 intermediate AND the dtype=-kwarg kernel-selection bug
        # that caused divergence at long sequence lengths. See docstring.
        assert scaled.dtype == out_dtype
        return torch.softmax(scaled, dim=-1)

    # Row-count threshold (estimated fp32 GELU-intermediate bytes) above which
    # _forward_core switches to a row-chunked FFN instead of materializing
    # [batch*seq_len, ffn_dim] in one shot. 3GB is comfortably above every
    # executable shape in this codebase's own test suites (largest:
    # wide_fp16 at 4096*4096*4 = 67MB) so it never triggers for rows 1-13.
    # Canonical row 14 has batch=32/seq=100000/ffn_dim=1024 and the same
    # 12.2 GiB FFN-intermediate footprint as the old misdecoded row, but it is
    # preflight-blocked earlier by its impossible dense-attention tensor. See
    # ChunkedBaselineTransformerBlock's docstring: chunk size is NOT
    # automatically bit-exact in floating point (cuBLAS kernel selection can
    # depend on row count), so 4096 is used because it was measured bit-exact
    # against the unchunked computation at the legacy misdecoded shape
    # (d_model=1024, ffn_dim=100000), not assumed safe in general and not
    # evidence for canonical row 14 (d_model=1024, ffn_dim=1024).
    # Fallback budget for a device whose capacity cannot be queried (CPU, or
    # a driver that refuses the property lookup). This was the fixed budget
    # before step 19 made it capacity-aware.
    _FFN_CHUNK_THRESHOLD_BYTES = 3 * 1024**3
    # Step 19: the budget above is now a FRACTION OF THE DEVICE'S TOTAL VRAM.
    # Deliberately `total_memory`, never `mem_get_info()`'s FREE bytes: chunk
    # size changes cuBLAS kernel selection and therefore the arithmetic (see
    # ChunkedBaselineTransformerBlock's docstring), so a free-memory-derived
    # size would make this model's output depend on whatever else happened to
    # be resident on the card -- non-deterministic numerics, and a different
    # answer on a re-run of the same shape. total_memory is a hardware
    # constant, so the decision stays a pure function of (shape, card).
    # get_device_properties() is a cached host-side lookup, NOT a
    # device->host sync, so this stays legal inside a captured CUDA graph --
    # mem_get_info() would sync and poison capture.
    # 3/16 reproduces the previous fixed 3GB budget EXACTLY on a 16GB card,
    # so every shape validated before this change keeps its old behaviour and
    # its old numerics; it scales up on a 24GB card and, the case that
    # motivated this, scales DOWN on a smaller one instead of overcommitting.
    _FFN_CHUNK_BUDGET_FRAC = 3.0 / 16.0
    _FFN_CHUNK_BUDGET_FLOOR_BYTES = 512 * 1024**2
    # Largest chunk MEASURED bit-exact against the unchunked computation
    # (legacy d_model=1024, ffn_dim=100000). Never exceeded: chunk size is not
    # automatically bit-exact, so raising this needs the same measurement
    # that established it, not an assumption. Shrinking is what the budget
    # may now do, and stays a fixed shape-and-card-derived constant.
    _FFN_CHUNK_SIZE = 4096
    # Below this the per-chunk launch overhead dominates the GEMM; a config
    # that cannot afford 256 rows is pathological on any real card.
    _FFN_CHUNK_MIN = 256

    def _ffn_chunk_budget_bytes(self, device: Optional[torch.device]) -> int:
        """Bytes of [rows, ffn_dim] fp32 FFN intermediate this device tolerates.

        Pure function of the device -- no allocation, no sync, no dependence
        on current free memory. See _FFN_CHUNK_BUDGET_FRAC for why.
        """
        if device is None or device.type != "cuda":
            return self._FFN_CHUNK_THRESHOLD_BYTES
        try:
            total = torch.cuda.get_device_properties(device).total_memory
        except Exception:
            return self._FFN_CHUNK_THRESHOLD_BYTES
        return max(
            int(total * self._FFN_CHUNK_BUDGET_FRAC),
            self._FFN_CHUNK_BUDGET_FLOOR_BYTES,
        )

    def _ffn_chunk_size(
        self,
        batch: int,
        seq_len: int,
        ffn_dim: int,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        total_rows = batch * seq_len
        estimated_bytes = total_rows * ffn_dim * 4
        budget = self._ffn_chunk_budget_bytes(device)
        if estimated_bytes <= budget:
            return None
        # Largest power of two <= the bit-exact-verified _FFN_CHUNK_SIZE whose
        # own [chunk, ffn_dim] fp32 intermediate also fits the budget. Powers
        # of two only, so the size lands on the same small set of values a
        # card can produce rather than an arbitrary row count.
        rows_in_budget = budget // (ffn_dim * 4) if ffn_dim > 0 else total_rows
        chunk = min(self._FFN_CHUNK_SIZE, total_rows, max(rows_in_budget, 1))
        chunk = 1 << (int(chunk).bit_length() - 1)
        return max(min(chunk, total_rows), min(self._FFN_CHUNK_MIN, total_rows))

    @staticmethod
    def _chunked_ffn_fp32(
        normed2: torch.Tensor, ffn_in: nn.Linear, ffn_out: nn.Linear, chunk_size: int
    ) -> torch.Tensor:
        batch, seq_len, d_model = normed2.shape
        flat = normed2.reshape(batch * seq_len, d_model)
        pieces = [
            ffn_out(F.gelu(ffn_in(flat[start : start + chunk_size]), approximate="none"))
            for start in range(0, flat.shape[0], chunk_size)
        ]
        return torch.cat(pieces, dim=0).reshape(batch, seq_len, d_model)

    @staticmethod
    def _chunked_ffn_fp16gemm(
        normed2: torch.Tensor,
        ffn_in_weight16: torch.Tensor,
        ffn_in_bias16: torch.Tensor,
        ffn_out_weight16: torch.Tensor,
        ffn_out_bias16: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        batch, seq_len, d_model = normed2.shape
        flat = normed2.reshape(batch * seq_len, d_model)
        pieces = []
        for start in range(0, flat.shape[0], chunk_size):
            piece = flat[start : start + chunk_size]
            # No fp32 round-trip around GELU -- see the non-chunked branch's
            # comment; bit-identical, verified torch.equal.
            hidden = F.linear(piece.to(torch.float16), ffn_in_weight16, ffn_in_bias16)
            hidden = F.gelu(hidden, approximate="none")
            pieces.append(F.linear(hidden, ffn_out_weight16, ffn_out_bias16).to(torch.float32))
        return torch.cat(pieces, dim=0).reshape(batch, seq_len, d_model)

    def _ln_site(
        self,
        x: torch.Tensor,
        delta: Optional[torch.Tensor],
        norm: nn.LayerNorm,
        out_dtype: torch.dtype,
        mask_delta: bool,
        mask_x: bool,
        mask_out: bool,
        need_x: bool,
        valid_flat: Optional[torch.Tensor],
        invalid_mask: Optional[torch.Tensor],
        use_fused_ln: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One residual-add + LayerNorm site, as one fused Triton kernel or
        as the original ATen chain.

        Returns `(x_new, normed)`. Both branches compute exactly the same
        sequence of operations in the same order --

            delta -> cast to the residual dtype
                  -> zero invalid rows            (mask_delta)
            x     -> x + delta
                  -> zero invalid rows            (mask_x)
            out   -> LayerNorm(x)
                  -> zero invalid rows            (mask_out)
                  -> cast to out_dtype

        -- so the ATen branch is the literal reference the Triton branch is
        probed against (see _probe_fused_ln), and every step except the
        LayerNorm reduction itself is bit-exact between them.

        `out_dtype` exists so the fp32 fp16-GEMM path can have the
        normalized activation land in fp16 directly, instead of writing fp32
        and then streaming the whole tensor again through a cast kernel; the
        rounding is the same either way (LayerNorm accumulates in fp32
        regardless, then rounds once on store).

        `need_x` is False only at the final_norm site, where the updated
        residual is consumed by nothing and the fused kernel can skip the
        store entirely.
        """
        if use_fused_ln:
            return _triton_add_layernorm(
                x, delta, norm.weight, norm.bias, norm.eps, out_dtype,
                mask_delta, mask_x, mask_out, need_x, valid_flat,
            )

        if delta is not None:
            if delta.dtype != x.dtype:
                delta = delta.to(x.dtype)
            if mask_delta:
                delta = delta.masked_fill(invalid_mask, 0)
            x = x + delta
        if mask_x:
            x = x.masked_fill(invalid_mask, 0)
        normed = norm(x)
        if mask_out:
            normed = normed.masked_fill(invalid_mask, 0)
        if normed.dtype != out_dtype:
            normed = normed.to(out_dtype)
        return x, normed

    def _forward_core(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        mask_active: bool,
        causal: bool,
        use_sdpa: bool,
        use_fp16_gemm: bool = False,
        use_fused_qkv: bool = False,
        layout=None,
        use_fused_ln: bool = False,
        use_short_attn: bool = False,
        use_fused_ffn: bool = False,
        use_tri_qk: bool = False,
        use_tri_linear: bool = False,
    ) -> torch.Tensor:
        """The actual per-layer computation, identical in arithmetic/op-order
        to the original eager forward(). Takes `mask_active` as an
        already-resolved plain Python bool (see _resolve_mask_active) instead
        of computing it here, so this function performs no device->host sync
        and is safe to run under torch.cuda.graph() capture.

        `use_fp16_gemm` (fp32 model only -- see _resolve_fp16_gemm_enabled)
        selects the fp16-GEMM path: every GEMM and SDPA call
        itself run on fp16-cast activations/weights with the result cast
        straight back to fp32, while LayerNorm/GELU/residual adds/final norm
        stay fp32. It is only ever True when use_sdpa is also True (that
        combination is fp32-exclusive); the fp16/bf16 manual-math branch
        below is completely unaffected by it.

        `use_fused_qkv` (fp16/bf16 manual-math path only -- see
        _resolve_fused_qkv_enabled) replaces the three separate q/k/v
        projection GEMMs with one packed [3*d_model, d_model] GEMM. It is
        only ever True when the probe proved that substitution BIT-EXACT
        (torch.equal on the whole forward output) for this exact
        configuration; on the fp32/SDPA path QKV is fused unconditionally
        and this flag stays False.

        `use_fused_ln` (see _resolve_fused_ln_enabled) routes every
        residual-add + LayerNorm site through one Triton kernel instead of
        the four-to-five ATen kernels it replaces. The loop below is written
        around _ln_site precisely so that flag changes nothing else: the
        residual add that closes each sub-block is handed to the *next*
        LayerNorm as a pending `delta` rather than being applied eagerly, so
        that fusing or not fusing is a single call-site decision rather than
        two divergent copies of the layer body.

        `use_short_attn` (fp32/SDPA path only -- see
        _resolve_short_attn_enabled) replaces
        F.scaled_dot_product_attention with the single-block
        _short_attn_kernel, which also returns the context already merged
        back to [B, S, d_model].

        `use_fused_ffn` (see _resolve_fused_ffn_enabled) replaces the FFN's
        first projection and its GELU with one _gemm_gelu_kernel launch.
        Unlike the other two, it can be bit-exact, so it is available on
        every dtype path.

        `use_tri_linear` (see _resolve_tri_linear_enabled) routes the q/k/v,
        out_proj and ffn_out projections through the same Triton GEMM with
        its epilogue off, instead of cuBLAS. Also measured bit-exact at these
        shapes, so also available on every dtype path.
        """
        # Local alias so each projection site reads as one expression.
        def _lin(t, w, b):
            return _triton_linear(t, w, b) if use_tri_linear else F.linear(t, w, b)
        batch, seq_len, d_model = x.shape

        invalid_mask: Optional[torch.Tensor] = None
        valid_flat: Optional[torch.Tensor] = None
        if mask_active:
            invalid_mask = ~valid_token_mask[..., None]  # [B, S, 1]
            # [B*S] view for the fused kernel, which indexes rows of the
            # flattened activation. valid_token_mask is contiguous, so this
            # is a view and not a copy.
            valid_flat = valid_token_mask.reshape(-1)

        # ---- decide the attention masking tensors once per call, reused
        # across every layer below (never rebuilt per layer) ----
        is_causal = False
        attn_mask: Optional[torch.Tensor] = None
        causal_disallowed: Optional[torch.Tensor] = None
        invalid_keys: Optional[torch.Tensor] = None
        if use_sdpa:
            if causal and not mask_active:
                # Pure causal, no padding: let SDPA's fused causal kernel run,
                # no mask tensor is ever materialized.
                is_causal = True
            elif causal and mask_active:
                allowed = self._get_causal_allowed(seq_len, x.device)  # [S, S], cached
                key_valid = valid_token_mask[:, None, None, :]  # [B, 1, 1, S]
                attn_mask = allowed[None, None, :, :] & key_valid  # [B, 1, S, S]
            elif mask_active:
                attn_mask = valid_token_mask[:, None, None, :]  # [B, 1, 1, S]
        else:
            if causal:
                causal_disallowed = self._get_causal_disallowed(seq_len, x.device)  # [S, S], cached
            if mask_active:
                invalid_keys = ~valid_token_mask[:, None, None, :]  # [B, 1, 1, S]

        chunk_size = self._ffn_chunk_size(
            batch, seq_len, self.config.ffn_dim, x.device)
        # dtype the LayerNorm outputs are consumed in. On the fp16-GEMM path
        # every consumer of a normalized activation is an fp16 GEMM, so the
        # cast is folded into the normalization's store.
        ln_dtype = torch.float16 if use_fp16_gemm else x.dtype

        # The previous sub-block's output, still waiting to be added into the
        # residual stream by the next LayerNorm site. None before the very
        # first LayerNorm, where there is nothing to add yet.
        pending: Optional[torch.Tensor] = None

        for layer in self.layers:
            attn = layer.attention

            # Closes the *previous* block's FFN residual (x + ffn_out, then
            # the block's trailing padding mask) and normalizes, in one step.
            x, normed = self._ln_site(
                x, pending, layer.norm1, ln_dtype,
                mask_delta=False,
                mask_x=(mask_active and pending is not None),
                mask_out=False, need_x=True,
                valid_flat=valid_flat, invalid_mask=invalid_mask,
                use_fused_ln=use_fused_ln,
            )
            pending = None

            if use_fp16_gemm:
                # Calibrated fp32-only fast path (docstring item 7): cast to
                # fp16 at each GEMM's inputs, run it, cast the result
                # straight back to fp32 -- LayerNorm/GELU/residual adds/
                # final norm all stay fp32. Attention itself (SDPA) also
                # runs on the fp16 q/k/v produced by the fused QKV GEMM, so
                # context never round-trips through fp32 before out_proj.
                # `normed` is already fp16 (see ln_dtype above).
                fused_weight16, fused_bias16 = _get_fused_qkv_weights_fp16(attn)
                qkv16 = _lin(normed, fused_weight16, fused_bias16)
                q16, k16, v16 = qkv16.split(d_model, dim=-1)
                q16 = q16.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                k16 = k16.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                v16 = v16.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

                if use_short_attn:
                    # Returns the merged [B, S, d_model] context directly.
                    context16 = _triton_short_attention(
                        q16, k16, v16, attn.scale, causal, mask_active,
                        valid_token_mask,
                    )
                else:
                    context16 = F.scaled_dot_product_attention(
                        q16, k16, v16,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                        scale=attn.scale,
                    )
                    context16 = (
                        layout[1](context16) if layout is not None
                        else context16.transpose(1, 2).reshape(batch, seq_len, d_model)
                    )

                out_weight16, out_bias16 = _get_linear_fp16_weights(attn.out_proj)
                # Left in fp16: _ln_site's next call upcasts it as part of
                # the residual add (fp16 -> fp32 is lossless, so this is the
                # same arithmetic as the explicit .to(torch.float32) it
                # replaces) and applies the attention output's padding mask
                # via mask_delta.
                attn_out = _lin(context16, out_weight16, out_bias16)

                x, normed2 = self._ln_site(
                    x, attn_out, layer.norm2, ln_dtype,
                    mask_delta=mask_active, mask_x=False,
                    mask_out=False, need_x=True,
                    valid_flat=valid_flat, invalid_mask=invalid_mask,
                    use_fused_ln=use_fused_ln,
                )

                ffn_in_weight16, ffn_in_bias16 = _get_linear_fp16_weights(layer.ffn_in)
                ffn_out_weight16, ffn_out_bias16 = _get_linear_fp16_weights(layer.ffn_out)
                if chunk_size is None:
                    # GELU runs directly on the fp16 GEMM output, no upcast
                    # round-trip: F.gelu on a half tensor is bit-identical to
                    # upcast->gelu->downcast (ATen's GELU kernel already
                    # accumulates in fp32 internally regardless of input
                    # dtype, same as softmax/LayerNorm) -- verified
                    # torch.equal, not just close. Removes 2 elementwise
                    # kernel launches and halves this op's memory traffic.
                    if use_fused_ffn:
                        hidden = _triton_gemm_gelu(normed2, ffn_in_weight16, ffn_in_bias16)
                    else:
                        hidden = _lin(normed2, ffn_in_weight16, ffn_in_bias16)
                        hidden = F.gelu(hidden, approximate="none")
                    pending = _lin(hidden, ffn_out_weight16, ffn_out_bias16)
                else:
                    pending = self._chunked_ffn_fp16gemm(
                        normed2, ffn_in_weight16, ffn_in_bias16, ffn_out_weight16, ffn_out_bias16, chunk_size
                    )
                continue

            if use_sdpa or use_fused_qkv:
                # One packed [3*d_model, d_model] GEMM instead of three
                # d_model-wide ones. Mathematically identical, and measured
                # ~1.9x the throughput of the three separate GEMMs at the
                # default shape (20.6 -> 38.2 TFLOPS in fp16), because
                # N=3*d_model gives 3x the threadblock tiles to spread over
                # this card's 36 SMs.
                #
                # It is NOT unconditionally bit-exact in low precision,
                # though: cuBLASLt picks a different internal algorithm
                # (same cutlass kernel, different CTA ordering) for some
                # (M, K, N, dtype) combinations, which reorders the fp32
                # accumulation and shifts near-zero elements by an ulp. On
                # the fp32/SDPA path (use_sdpa) that is harmless -- fp32 has
                # the mantissa headroom -- so fusion is unconditional there.
                # On the fp16/bf16 path it is used only where
                # _probe_fused_qkv_exact proved bit-exactness for this exact
                # configuration; otherwise use_fused_qkv is False and the
                # three-GEMM branch below runs instead.
                fused_weight, fused_bias = _get_fused_qkv_weights(attn)
                qkv = _lin(normed, fused_weight, fused_bias)
                q, k, v = qkv.split(d_model, dim=-1)
            else:
                # fp16/bf16 where fusion was NOT proved bit-exact: issue the
                # same three separate GEMMs the baseline uses, so the matmul
                # reduction order -- and thus the fp16/bf16 rounding --
                # matches bit-for-bit instead of merely "close enough".
                q = _lin(normed, attn.q_proj.weight, attn.q_proj.bias)
                k = _lin(normed, attn.k_proj.weight, attn.k_proj.bias)
                v = _lin(normed, attn.v_proj.weight, attn.v_proj.bias)

            if layout is not None and not use_sdpa:
                # Manual-math path only. The head transpose has to be
                # materialized here either way (torch.matmul would do it
                # internally); `layout[0]` just does it with inductor's tiled
                # copy instead of ATen's strided elementwise one. NOT applied
                # on the SDPA path, where the mem-efficient backend consumes
                # the strided views directly and forcing them contiguous
                # would ADD copies rather than speed them up.
                q, k, v = layout[0](q, k, v, attn.num_heads)
            else:
                q = q.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = k.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                v = v.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

            if use_sdpa and use_short_attn:
                context = _triton_short_attention(
                    q, k, v, attn.scale, causal, mask_active, valid_token_mask,
                )
            elif use_sdpa:
                context = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    scale=attn.scale,
                )
            else:
                # Reproduce BaselineSelfAttention's arithmetic exactly: scale
                # after the matmul, mask, softmax in fp32, cast back, then
                # matmul with v. The scale+mask+softmax epilogue itself is
                # delegated to _fused_attn_probs (fused Triton kernel, with
                # an ATen fallback that is itself bit-exact with the
                # baseline -- see that method's docstring); only the raw
                # (unscaled) Q@K^T matmul happens here, unchanged.
                kt = k.transpose(-2, -1)
                tri_block = (
                    _causal_qk_block_size(seq_len) if use_tri_qk else None
                )
                if tri_block is not None:
                    # Only the block-lower-triangle is computed; the rest
                    # of this buffer is overwritten with -inf by the
                    # scale+mask kernel below and is never read as data.
                    # See _blocked_causal_qk.
                    scores_raw = _blocked_causal_qk(
                        q, kt, tri_block,
                        torch.empty(
                            q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                            device=q.device, dtype=q.dtype,
                        ),
                    )
                else:
                    scores_raw = torch.matmul(q, kt)
                probs = self._fused_attn_probs(
                    scores_raw, attn.scale, causal, mask_active,
                    valid_token_mask, causal_disallowed, invalid_keys, x.dtype,
                    tri_block=(_TRI_MASK_BLOCK if tri_block is not None else None),
                )
                context = torch.matmul(probs, v)

            if not (use_sdpa and use_short_attn):
                # _triton_short_attention already returned merged context.
                context = (
                    layout[1](context) if layout is not None
                    else context.transpose(1, 2).reshape(batch, seq_len, d_model)
                )
            attn_out = _lin(context, attn.out_proj.weight, attn.out_proj.bias)

            x, normed2 = self._ln_site(
                x, attn_out, layer.norm2, ln_dtype,
                mask_delta=mask_active, mask_x=False,
                mask_out=False, need_x=True,
                valid_flat=valid_flat, invalid_mask=invalid_mask,
                use_fused_ln=use_fused_ln,
            )

            if chunk_size is None:
                if use_fused_ffn:
                    hidden = _triton_gemm_gelu(
                        normed2, layer.ffn_in.weight, layer.ffn_in.bias)
                else:
                    hidden = F.gelu(layer.ffn_in(normed2), approximate="none")
                pending = _lin(hidden, layer.ffn_out.weight, layer.ffn_out.bias)
            else:
                pending = self._chunked_ffn_fp32(normed2, layer.ffn_in, layer.ffn_out, chunk_size)

        # Closes the last block's FFN residual and its trailing padding mask,
        # then final_norm, then the model's own trailing mask -- all in the
        # same single site. `need_x=False`: nothing reads the residual again.
        _, out = self._ln_site(
            x, pending, self.final_norm, x.dtype,
            mask_delta=False, mask_x=mask_active,
            mask_out=mask_active, need_x=False,
            valid_flat=valid_flat, invalid_mask=invalid_mask,
            use_fused_ln=use_fused_ln,
        )
        return out

def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass(frozen=True)
class StreamingShardPlan:
    reference_batch_shard: int
    reference_query_tile: int
    optimized_batch_shard: int
    optimized_capacity_batch_shard: int
    target_bytes: int
    estimated_model_bytes: int


def _estimated_streaming_model_bytes(
    config: TransformerConfig, element_bytes: int
) -> int:
    """Estimate parameters for the simultaneously resident model pair."""
    d_model = config.d_model
    per_layer_elements = (
        4 * d_model * d_model
        + 2 * d_model * config.ffn_dim
        + 9 * d_model
        + config.ffn_dim
    )
    one_model_elements = config.num_layers * per_layer_elements + 2 * d_model
    return 2 * one_model_elements * element_bytes


def _query_tile_candidate(value: int, seq_len: int) -> int:
    value = max(1, min(value, seq_len))
    return 1 << (value.bit_length() - 1)


def _streaming_shard_plan(
    config: TransformerConfig, dtype: torch.dtype, device: torch.device
) -> StreamingShardPlan:
    """Choose independent reference/optimized tiles from shape and capacity."""
    total_bytes = _installed_cuda_memory_bytes(device)
    if total_bytes is None:
        raise RuntimeError("capacity streaming requires installed CUDA memory information")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    target_bytes = int(total_bytes * _STREAMING_MEMORY_BUDGET_FRAC)
    model_bytes = _estimated_streaming_model_bytes(config, element_bytes)
    activation_budget = target_bytes - model_bytes
    per_batch_resident = element_bytes * config.seq_len * (
        12 * config.d_model + 4 * config.ffn_dim
    )
    minimum_query_workspace = int(
        2
        * config.num_heads
        * config.seq_len
        * element_bytes
        * _STREAMING_REFERENCE_WORKSPACE_RESERVE
    )
    minimum_streaming_bytes = (
        model_bytes + per_batch_resident + minimum_query_workspace
    )
    if minimum_streaming_bytes > target_bytes:
        raise RuntimeError(
            "capacity streaming cannot fit its minimum reference shard under "
            f"the {_STREAMING_MEMORY_BUDGET_FRAC:.0%} VRAM ceiling: "
            f"required={minimum_streaming_bytes}, target={target_bytes}"
        )
    optimized_capacity_batch_shard = max(
        1, min(config.batch_size, activation_budget // per_batch_resident)
    )
    saturation_batch_shard = max(
        1,
        (config.seq_len + _STREAMING_OPTIMIZED_TOKEN_TARGET - 1)
        // config.seq_len,
    )
    optimized_batch_shard = min(
        optimized_capacity_batch_shard, saturation_batch_shard
    )

    # The reference can have a fixed score workspace and a dynamic probability
    # tensor live together, hence the factor of two. Maximizing batch*query rows
    # is a general proxy for GEMM occupancy and reduces reference tile launches.
    best_reference = (0, 0, 1, 1)
    max_reference_batch = min(config.batch_size, optimized_batch_shard)
    for reference_batch_shard in range(1, max_reference_batch + 1):
        resident_bytes = reference_batch_shard * per_batch_resident
        available_query_bytes = activation_budget - resident_bytes
        bytes_per_query = int(
            2
            * reference_batch_shard
            * config.num_heads
            * config.seq_len
            * element_bytes
            * _STREAMING_REFERENCE_WORKSPACE_RESERVE
        )
        max_queries = max(
            1,
            min(config.seq_len, available_query_bytes // max(1, bytes_per_query)),
        )
        query_tile = _query_tile_candidate(max_queries, config.seq_len)
        candidate = (
            reference_batch_shard * query_tile,
            query_tile,
            reference_batch_shard,
            query_tile,
        )
        if candidate > best_reference:
            best_reference = candidate

    return StreamingShardPlan(
        reference_batch_shard=int(best_reference[2]),
        reference_query_tile=int(best_reference[3]),
        optimized_batch_shard=int(optimized_batch_shard),
        optimized_capacity_batch_shard=int(optimized_capacity_batch_shard),
        target_bytes=target_bytes,
        estimated_model_bytes=model_bytes,
    )


def _generate_streaming_shard(
    config: TransformerConfig,
    batch_size: int,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shard_config = TransformerConfig(
        batch_size=batch_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        causal=config.causal,
    )
    return generate_random_case(
        shard_config, torch.device("cpu"), dtype, seed, padding_ratio, input_scale
    )


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def run_streaming_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
    reference_batch_shard: int,
    optimized_batch_shard: int,
) -> bool:
    """Compare reference subshards with independently sized optimized shards."""
    print("\n=== Accuracy check (STREAMED) ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")
    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            trial_passed = True
            trial_max_abs = 0.0
            trial_max_rel = 0.0
            trial_failed = 0
            trial_elements = 0
            for start in range(0, config.batch_size, optimized_batch_shard):
                shard_size = min(optimized_batch_shard, config.batch_size - start)
                x_cpu, mask_cpu = _generate_streaming_shard(
                    config,
                    shard_size,
                    dtype,
                    seed + trial * config.batch_size + start,
                    padding_ratio,
                    input_scale,
                )
                x = x_cpu.to(device)
                mask = mask_cpu.to(device)
                candidate = optimized(x, mask).cpu()
                del x, mask

                for reference_start in range(0, shard_size, reference_batch_shard):
                    reference_end = min(
                        shard_size, reference_start + reference_batch_shard
                    )
                    reference_x = x_cpu[reference_start:reference_end].to(device)
                    reference_mask = mask_cpu[reference_start:reference_end].to(device)
                    reference = baseline(reference_x, reference_mask).cpu()
                    result = compare_outputs(
                        reference,
                        candidate[reference_start:reference_end],
                        rtol=rtol,
                        atol=atol,
                    )
                    del reference_x, reference_mask, reference

                    trial_passed &= result.passed
                    trial_max_abs = max(trial_max_abs, result.max_abs_error)
                    trial_max_rel = max(trial_max_rel, result.max_relative_error)
                    trial_failed += result.failed_elements
                    trial_elements += result.total_elements
                del x_cpu, mask_cpu, candidate

            all_passed &= trial_passed
            global_max_abs = max(global_max_abs, trial_max_abs)
            global_max_rel = max(global_max_rel, trial_max_rel)
            total_failed += trial_failed
            total_elements += trial_elements
            print(
                f"trial {trial + 1:02d}/{trials}: "
                f"{'PASS' if trial_passed else 'FAIL'} | "
                f"max_abs={trial_max_abs:.6g} | max_rel={trial_max_rel:.6g} | "
                f"failed={trial_failed}/{trial_elements}"
            )

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def benchmark_streaming_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
    reference_batch_shard: int,
    optimized_batch_shard: int,
) -> None:
    """Time full host-to-device/device-to-host streamed passes."""
    print("\n=== Performance benchmark (STREAMED) ===")
    print("timing includes shard transfers; fixed inputs remain resident on CPU")
    print(f"streaming protocol: warmup={warmup}, repeats={repeats}, rounds={rounds}")
    cpu_shards = []
    for start in range(0, config.batch_size, optimized_batch_shard):
        shard_size = min(optimized_batch_shard, config.batch_size - start)
        cpu_shards.append(
            _generate_streaming_shard(
                config,
                shard_size,
                dtype,
                seed + 100000 + start,
                padding_ratio,
                input_scale,
            )
        )

    def run_pass(model: nn.Module, execution_batch_shard: int) -> float:
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            for x_cpu, mask_cpu in cpu_shards:
                for start in range(0, x_cpu.shape[0], execution_batch_shard):
                    end = min(x_cpu.shape[0], start + execution_batch_shard)
                    x = x_cpu[start:end].to(device)
                    mask = mask_cpu[start:end].to(device)
                    output_cpu = model(x, mask).cpu()
                    del x, mask, output_cpu
        torch.cuda.synchronize(device)
        return (time.perf_counter_ns() - started) / 1e6

    for _ in range(warmup):
        run_pass(baseline, reference_batch_shard)
        run_pass(optimized, optimized_batch_shard)

    baseline_samples = []
    optimized_samples = []
    for round_index in range(rounds):
        for _ in range(repeats):
            if round_index % 2 == 0:
                baseline_samples.append(run_pass(baseline, reference_batch_shard))
                optimized_samples.append(run_pass(optimized, optimized_batch_shard))
            else:
                optimized_samples.append(run_pass(optimized, optimized_batch_shard))
                baseline_samples.append(run_pass(baseline, reference_batch_shard))

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens = config.batch_size * config.seq_len
    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={tokens * 1000.0 / baseline_result.median_ms:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={tokens * 1000.0 / optimized_result.median_ms:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--chunk-baseline-ffn",
        action="store_true",
        help=(
            "use ChunkedBaselineTransformer instead of BaselineTransformer: "
            "identical computation, FFN done in row-chunks to bound peak "
            "memory for very large ffn_dim configs. Off by default; does "
            "not affect BaselineTransformer or any existing measurement."
        ),
    )
    parser.add_argument(
        "--baseline-ffn-chunk-size",
        type=int,
        default=4096,
        help="row-chunk size for --chunk-baseline-ffn (rows = batch*seq_len)",
    )
    parser.add_argument(
        "--capacity-streaming",
        action="store_true",
        help=(
            "force the installed-VRAM-derived out-of-core protocol; normally "
            "selected automatically when the static dense peak exceeds capacity"
        ),
    )
    parser.add_argument("--streaming-accuracy-trials", type=int, default=1)
    parser.add_argument("--streaming-warmup", type=int, default=0)
    parser.add_argument("--streaming-repeats", type=int, default=1)
    parser.add_argument("--streaming-rounds", type=int, default=1)
    parser.add_argument(
        "--streaming-accuracy-only",
        action="store_true",
        help="validate a streamed long case without repeating its eager reference for timing",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if args.streaming_accuracy_trials <= 0:
        raise ValueError("streaming_accuracy_trials must be positive")
    if args.streaming_warmup < 0:
        raise ValueError("streaming_warmup must be non-negative")
    if args.streaming_repeats <= 0 or args.streaming_rounds <= 0:
        raise ValueError("streaming repeats and rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    capacity_streaming = args.capacity_streaming or _capacity_streaming_required(
        config, dtype, device
    )
    if capacity_streaming and device.type != "cuda":
        raise ValueError("capacity streaming currently requires a CUDA device")
    if capacity_streaming and (args.compile_baseline or args.compile_user):
        raise ValueError("compiled whole-model bars are not defined for streamed execution")

    streaming_plan = None
    if capacity_streaming:
        streaming_plan = _streaming_shard_plan(config, dtype, device)

    if capacity_streaming:
        baseline = QueryTiledBaselineTransformer(
            config,
            query_tile_size=streaming_plan.reference_query_tile,
            ffn_chunk_size=args.baseline_ffn_chunk_size,
        )
    elif args.chunk_baseline_ffn:
        baseline = ChunkedBaselineTransformer(
            config, ffn_chunk_size=args.baseline_ffn_chunk_size
        )
    else:
        baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    if capacity_streaming:
        element_bytes = torch.empty((), dtype=dtype).element_size()
        linear_bytes, score_bytes, estimated_bytes = _capacity_terms_bytes(
            config, element_bytes
        )
        total_bytes = _installed_cuda_memory_bytes(device)
        print(
            "execution=STREAMED | "
            f"reference_batch_shard={streaming_plan.reference_batch_shard} | "
            f"reference_query_tile={streaming_plan.reference_query_tile} | "
            f"optimized_batch_shard={streaming_plan.optimized_batch_shard} | "
            "optimized_capacity_batch_shard="
            f"{streaming_plan.optimized_capacity_batch_shard} | "
            f"target_vram_fraction={_STREAMING_MEMORY_BUDGET_FRAC:.2f} | "
            f"target_vram_bytes={streaming_plan.target_bytes} | "
            f"estimated_model_bytes={streaming_plan.estimated_model_bytes} | "
            f"linear_bytes={linear_bytes} | score_bytes={score_bytes} | "
            f"estimated_dense_bytes={estimated_bytes} | "
            f"gpu_total_memory_bytes={total_bytes}"
        )

    if capacity_streaming:
        accuracy_passed = run_streaming_accuracy_tests(
            baseline=baseline,
            optimized=optimized,
            config=config,
            device=device,
            dtype=dtype,
            trials=args.streaming_accuracy_trials,
            seed=args.seed,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
            rtol=args.rtol,
            atol=args.atol,
            reference_batch_shard=streaming_plan.reference_batch_shard,
            optimized_batch_shard=streaming_plan.optimized_batch_shard,
        )
    else:
        accuracy_passed = run_accuracy_tests(
            baseline=baseline,
            optimized=optimized,
            config=config,
            device=device,
            dtype=dtype,
            trials=args.accuracy_trials,
            seed=args.seed,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
            rtol=args.rtol,
            atol=args.atol,
        )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    if capacity_streaming and args.streaming_accuracy_only:
        print("\nStreaming performance benchmark skipped by explicit accuracy-only request.")
        return 0 if accuracy_passed else 2

    if capacity_streaming:
        benchmark_streaming_models(
            baseline=baseline,
            optimized=optimized,
            config=config,
            device=device,
            dtype=dtype,
            seed=args.seed,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
            warmup=args.streaming_warmup,
            repeats=args.streaming_repeats,
            rounds=args.streaming_rounds,
            reference_batch_shard=streaming_plan.reference_batch_shard,
            optimized_batch_shard=streaming_plan.optimized_batch_shard,
        )
    else:
        benchmark_models(
            baseline=baseline,
            optimized=optimized,
            config=config,
            device=device,
            dtype=dtype,
            seed=args.seed,
            padding_ratio=args.padding_ratio,
            input_scale=args.input_scale,
            warmup=args.warmup,
            repeats=args.repeats,
            rounds=args.benchmark_rounds,
        )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
