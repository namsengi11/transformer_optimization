#!/usr/bin/env python3
"""
Fused streaming FFN (a "flash-FFN") for the heavy configs.

FFN(x) = gelu(x @ W1^T + b1) @ W2^T + b2, with x[M,D], W1[F,D], W2[D,F].

The standard two-GEMM form materializes the hidden H[M,F] in HBM. For the graded
`14_extreme` row (F = 100000, M = 32768) that intermediate is ~6.5 GB fp16 — it
OOMs the baseline itself and, even when it fits, its write+read dominates the
bandwidth. This kernel STREAMS the hidden dimension F in blocks and accumulates
the output directly, so H[M,F] is never materialized: memory is O(M*D), not
O(M*F), and the hidden-tensor HBM traffic is removed.

  out_tile[BM,BN] = sum over F-blocks of  gelu(x_tile @ W1_blk^T + b1_blk) @ W2_blk^T
  (the inner x@W1^T contraction over D is itself tiled so nothing exceeds SRAM)

fp16 compute, fp32 accumulate — matches the autocast path used elsewhere. Correct
against a reference FFN; targets `14_extreme` (OOM fix) and `06_batch10000`
(bandwidth). Validate end-to-end on the 16 GB Blackwell grading GPU.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_none(x):  # exact erf GELU in fp32, matching F.gelu(approximate="none")
    return x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))


def _configs():
    cfgs = []
    for bm in (64, 128):
        for bn in (64, 128):
            for bk in (64, 128):
                cfgs.append(triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=4, num_stages=2))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "D", "F"])
@triton.jit
def _fused_ffn_kernel(
    X, W1, B1, W2, B2, Out,
    M, D, F,
    sxm, sxd, sw1f, sw1d, sw2d, sw2f, som, son,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # output (D) columns
    offs_d = tl.arange(0, BLOCK_D)                        # input-dim tiles

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for f0 in range(0, F, BLOCK_K):
        offs_f = f0 + tl.arange(0, BLOCK_K)
        # H_blk[BM,BK] = gelu( x @ W1[f_blk]^T + b1 ) — contract over D, tiled
        h = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            dd = d0 + offs_d
            x = tl.load(X + offs_m[:, None] * sxm + dd[None, :] * sxd,
                        mask=(offs_m[:, None] < M) & (dd[None, :] < D), other=0.0).to(tl.float16)
            w1 = tl.load(W1 + offs_f[:, None] * sw1f + dd[None, :] * sw1d,
                         mask=(offs_f[:, None] < F) & (dd[None, :] < D), other=0.0).to(tl.float16)
            h += tl.dot(x, tl.trans(w1))                 # [BM,BD]x[BD,BK]
        h += tl.load(B1 + offs_f, mask=offs_f < F, other=0.0)[None, :]
        h = _gelu_none(h).to(tl.float16)                 # [BM,BK]
        # acc += H_blk @ W2[n_tile, f_blk]^T   (contract over this F-block)
        w2 = tl.load(W2 + offs_n[:, None] * sw2d + offs_f[None, :] * sw2f,
                     mask=(offs_n[:, None] < D) & (offs_f[None, :] < F), other=0.0).to(tl.float16)
        acc += tl.dot(h, tl.trans(w2))                   # [BM,BK]x[BK,BN]
    acc += tl.load(B2 + offs_n, mask=offs_n < D, other=0.0)[None, :]
    tl.store(Out + offs_m[:, None] * som + offs_n[None, :] * son, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < D))


def fused_ffn(x, W1, b1, W2, b2):
    """x[M,D], W1[F,D], b1[F], W2[D,F], b2[D] -> out[M,D]. Hidden never materialized."""
    M, D = x.shape
    F = W1.shape[0]
    out = torch.empty((M, D), device=x.device, dtype=torch.float32)
    BLOCK_D = 64 if D % 64 == 0 else (32 if D % 32 == 0 else 16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(D, meta["BLOCK_N"]))
    _fused_ffn_kernel[grid](
        x, W1, b1, W2, b2, out, M, D, F,
        x.stride(0), x.stride(1), W1.stride(0), W1.stride(1),
        W2.stride(0), W2.stride(1), out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    dev = torch.device("cuda")
    torch.manual_seed(0)

    def ref_ffn(x, W1, b1, W2, b2):
        return F.gelu(x @ W1.t() + b1, approximate="none") @ W2.t() + b2

    def peak_mb(fn):
        torch.cuda.reset_peak_memory_stats()
        fn(); torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2

    # tractable shapes (fit 8 GB, so a reference exists); the real target is
    # M=32768,D=1024,F=100000 which only fits BECAUSE the hidden isn't materialized.
    for name, M, D, Fdim in [("small", 4096, 256, 1024),
                             ("d1024", 4096, 1024, 2048),
                             ("wideF", 8192, 512, 8192)]:
        x = torch.randn(M, D, device=dev)
        W1 = (torch.randn(Fdim, D, device=dev) * (D ** -0.5))
        b1 = torch.randn(Fdim, device=dev) * 0.1
        W2 = (torch.randn(D, Fdim, device=dev) * (Fdim ** -0.5))
        b2 = torch.randn(D, device=dev) * 0.1
        with torch.inference_mode():
            r = ref_ffn(x, W1, b1, W2, b2)
            o = fused_ffn(x, W1, b1, W2, b2)
        ae = (o - r).abs(); rel = ae / r.abs().clamp_min(1e-12)
        ok = bool(((ae <= 2e-3) | (rel <= 2e-2)).all())
        hidden_mb = M * Fdim * 2 / 1024**2
        print(f"[{name:6s} M={M} D={D} F={Fdim}] {'PASS' if ok else 'FAIL'} "
              f"abs={ae.max():.2e} | hidden-if-materialized={hidden_mb:.0f}MB "
              f"| fused-peak={peak_mb(lambda: fused_ffn(x, W1, b1, W2, b2)):.0f}MB "
              f"ref-peak={peak_mb(lambda: ref_ffn(x, W1, b1, W2, b2)):.0f}MB")
