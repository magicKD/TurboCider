#!/usr/bin/env python3
"""Probe the reference tiled MPS Sol-Attention kernel at LTX shapes.

This imports the local reference package without executing its ComfyUI plugin
entry point.  It is a feasibility probe only; the production runtime remains
native Objective-C/Metal.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
import types

import torch


def load_reference(reference: Path, prefill_header: Path):
    package_name = "ltx_sol_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(reference)]
    sys.modules[package_name] = package
    module = __import__(
        f"{package_name}._metal_tiled_fwd",
        fromlist=["_routing_debug_mps", "sol_attn_tiled_mps"],
    )
    header = prefill_header.read_text(encoding="utf-8")
    header = header.replace("#pragma once", "")
    header = header.replace('#include <c10/metal/common.h>', "")
    # torch.mps.compile_shader injects its own c10 Metal common/utils source.
    # The downloaded main-branch header only additionally expects the newer
    # IF_CONSTEXPR compatibility macro, which is absent in PyTorch 2.12.
    compatibility = r"""
#ifndef IF_CONSTEXPR
#define IF_CONSTEXPR
#endif
namespace c10 { namespace metal {
template <typename T>
inline T ceil_div(T a, T b) { return (a + b - 1) / b; }
}}
"""
    include = '#include <ATen/native/mps/kernels/PrefillAttention.h>'
    if include not in module._SOURCE:
        raise RuntimeError("reference Metal source no longer has the expected include")
    module._SOURCE = module._SOURCE.replace(include, compatibility + header)
    return module.sol_attn_tiled_mps, module._routing_debug_mps


def synchronize() -> None:
    torch.mps.synchronize()


def timed(call, iterations: int) -> tuple[torch.Tensor, list[float]]:
    values = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = call()
        synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    assert output is not None
    return output, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prefill-header", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4004)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--tau", type=float, action="append",
        help="routing threshold; may be repeated (default: 1.0)",
    )
    parser.add_argument("--query-block", type=int, choices=(32, 64), default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("this PyTorch build has no torch.mps.compile_shader")

    sol, routing_debug = load_reference(args.reference, args.prefill_header)
    generator = torch.Generator(device="cpu").manual_seed(42)
    shape = (1, args.rows, args.heads, args.head_dim)
    # RMS-normalized Q/K and an order-one V approximate the self-attention
    # core inputs more closely than unbounded random tensors.
    q = torch.randn(shape, generator=generator, dtype=torch.float32)
    k = torch.randn(shape, generator=generator, dtype=torch.float32)
    v = torch.randn(shape, generator=generator, dtype=torch.float32)
    q = torch.nn.functional.normalize(q, dim=-1) * math.sqrt(args.head_dim)
    k = torch.nn.functional.normalize(k, dim=-1) * math.sqrt(args.head_dim)
    q = q.to(device="mps", dtype=torch.bfloat16)
    k = k.to(device="mps", dtype=torch.bfloat16)
    v = v.to(device="mps", dtype=torch.bfloat16)
    scale = args.head_dim**-0.5

    dense_call = lambda: torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2)
    taus = args.tau or [1.0]
    for _ in range(args.warmup):
        dense_call()
        for tau in taus:
            sol(
                q, k, v, scale=scale, tau=tau,
                query_block_size=args.query_block,
            )
        synchronize()
    dense, dense_ms = timed(dense_call, args.iterations)
    dense_sorted = sorted(dense_ms)
    middle = len(dense_sorted) // 2
    dense_f32 = dense.float()
    for tau in taus:
        sol_call = lambda tau=tau: sol(
            q, k, v, scale=scale, tau=tau,
            query_block_size=args.query_block,
        )
        sparse, sparse_ms = timed(sol_call, args.iterations)
        routes, _, _, _ = routing_debug(q, k, scale=scale, tau=tau)
        synchronize()

        sparse_f32 = sparse.float()
        difference = sparse_f32 - dense_f32
        rel_l2 = float(difference.norm() / dense_f32.norm())
        cosine = float(torch.nn.functional.cosine_similarity(
            dense_f32.flatten(), sparse_f32.flatten(), dim=0
        ))
        sparse_sorted = sorted(sparse_ms)
        exact_fraction = float(routes.float().mean())
        print(
            f"shape={shape} tau={tau:g} query_block={args.query_block} "
            f"exact_fraction={exact_fraction:.6f} "
            f"dense_p50_ms={dense_sorted[middle]:.3f} "
            f"sol_p50_ms={sparse_sorted[middle]:.3f} "
            f"speedup={dense_sorted[middle] / sparse_sorted[middle]:.3f}x "
            f"rel_l2={rel_l2:.9g} cosine={cosine:.9g}"
        )


if __name__ == "__main__":
    main()
