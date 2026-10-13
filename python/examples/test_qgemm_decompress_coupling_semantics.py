# Copyright (c) the triton-riscv authors. Licensed under the Apache License 2.0.
"""Randomized semantics preservation for --qgemm-decompress-coupling.

Feature: qgemm-decompress-coupling, Property 4
Validates: Requirements 9.3, 12.7

For each reference format, several random seeds are drawn with the input
generators from benchmark/kernels. Both decompress modes must match the
PyTorch reference, and the two modes must match each other on the final
output (intermediates are internal to the kernel).
"""

import sys
from pathlib import Path

import pytest
import torch

_KERNELS = Path(__file__).resolve().parents[3] / "benchmark" / "kernels"
if str(_KERNELS) not in sys.path:
    sys.path.insert(0, str(_KERNELS))

import q80_q80_strict as q80  # noqa: E402
import q40_q80_canonical as q40  # noqa: E402
import q4k_q8k_canonical as q4k  # noqa: E402
import iq2xxs_q8k_canonical as iq2  # noqa: E402

MODES = ["pre", "on-the-fly"]
SEEDS = [1, 2, 3, 4]
ATOL = 1e-2
RTOL = 1e-2


def _q80(M, N, K, T):
    KB = K // q80.QK8_0
    a_q = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w_q = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    a_d = torch.randn(M, KB, dtype=torch.float16)
    w_d = torch.randn(N, KB, dtype=torch.float16)
    ref = q80.q80_q80_reference(a_q, a_d, w_q, w_d, M, N, K)

    def run(mode):
        out = torch.empty((M, N), dtype=torch.float32)
        q80.q80_q80_strict[(T, 1, 1)](a_q, a_d, w_q, w_d, out, M=M, N=N, K=K,
                                      T=T, qgemm_opt_enable=1,
                                      decompress=mode)
        return out

    return ref, run


def _q40(M, N, K, T):
    a_q, a_d, w_q, w_d = q40._make_inputs(M, N, K)
    ref, _ = q40.q40_q80_reference(a_q, a_d, w_q, w_d, M, N, K)

    def run(mode):
        out = torch.empty((M, N), dtype=torch.float32)
        q40.q40_q80_canonical[(T, 1, 1)](a_q, a_d, w_q, w_d, out, M=M, N=N,
                                         K=K, T=T, qgemm_opt_enable=1,
                                         decompress=mode)
        return out

    return ref, run


def _q4k(M, N, K, T):
    a_q, a_d, w_q, w_d, w_dmin, scales, _, _ = q4k._make_inputs(M, N, K)
    ref, *_ = q4k.q4k_q8k_reference(a_q, a_d, w_q, w_d, w_dmin, scales, M, N,
                                    K)

    def run(mode):
        out = torch.empty((M, N), dtype=torch.float32)
        q4k.q4k_q8k_canonical[(T, 1, 1)](a_q, a_d, w_q, w_d, w_dmin, scales,
                                         out, M=M, N=N, K=K, T=T,
                                         qgemm_opt_enable=1, decompress=mode)
        return out

    return ref, run


def _iq2(M, N, K, T):
    a_q, a_d, qs, w_d = iq2._make_inputs(M, N, K)
    ref, _ = iq2.iq2xxs_q8k_reference(a_q, a_d, qs, w_d, M, N, K)

    def run(mode):
        out = torch.empty((M, N), dtype=torch.float32)
        iq2.iq2xxs_q8k_canonical[(T, 1, 1)](a_q, a_d, qs, w_d,
                                            iq2.IQ2XXS_GRID_BYTES,
                                            iq2.KSIGNS_IQ2XS,
                                            iq2.KMASK_IQ2XS, out, M=M, N=N,
                                            K=K, T=T, qgemm_opt_enable=1,
                                            decompress=mode)
        return out

    return ref, run


FORMATS = {
    "q80_q80": (_q80, (64, 96, 128, 3)),
    "q40_q80": (_q40, (64, 96, 128, 3)),
    "q4k_q8k": (_q4k, (64, 96, 512, 3)),
    "iq2xxs_q8k": (_iq2, (64, 96, 512, 3)),
}


def _close(got, ref, what):
    ref = ref.to(torch.float32)
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
        f"{what}: max_abs_diff={(got - ref).abs().max().item():.6g}")


# Feature: qgemm-decompress-coupling, Property 4
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_semantics_preserved(device, fmt, seed):
    make, shape = FORMATS[fmt]
    torch.manual_seed(seed)
    ref, run = make(*shape)
    outs = {mode: run(mode) for mode in MODES}
    for mode, got in outs.items():
        _close(got, ref, f"{fmt} seed={seed} {mode} vs reference")
    _close(outs["pre"], outs["on-the-fly"],
           f"{fmt} seed={seed} pre vs on-the-fly")
