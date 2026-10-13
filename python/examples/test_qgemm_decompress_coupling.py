# Copyright (c) the triton-riscv authors. Licensed under the Apache License 2.0.
"""End-to-end tests for the `decompress` compile option and the
--qgemm-decompress-coupling pass.

The reference kernels and their PyTorch references live in benchmark/kernels/
and are imported from there rather than duplicated. Their run helpers hardcode
the launch kwargs, so the jit kernels are launched directly here with
qgemm_opt_enable=1 and decompress=<mode>.
"""

import sys
from pathlib import Path

import pytest
import torch
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.compiler import (CPUBackend,
                                                    _build_vectorir_passes)

_KERNELS = Path(__file__).resolve().parents[3] / "benchmark" / "kernels"
if str(_KERNELS) not in sys.path:
    sys.path.insert(0, str(_KERNELS))

import q80_q80_strict as q80  # noqa: E402
import q40_q80_canonical as q40  # noqa: E402
import q4k_q8k_canonical as q4k  # noqa: E402
import iq2xxs_q8k_canonical as iq2  # noqa: E402

MODES = ["pre", "on-the-fly"]

# Per-format tolerance: every reference script compares with check_float's
# defaults (atol=1e-2, rtol=1e-2).
ATOL = 1e-2
RTOL = 1e-2

# --- parse_options resolution (Req 11.3 - 11.6) ---


@pytest.fixture(scope="module")
def backend():
    return CPUBackend(GPUTarget("cpu", "x86_64", 0))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"qgemm_opt_enable": 1}, (1, "pre")),
        ({"qgemm_mr": 8}, (1, "pre")),
        ({"decompress": "on-the-fly"}, (1, "on-the-fly")),
        ({"decompress": "pre"}, (1, "pre")),
        ({}, (0, None)),
        ({"decompress": None}, (0, None)),
        ({"qgemm_opt_enable": 0, "decompress": "on-the-fly"}, (0, None)),
    ],
)
def test_decompress_resolution(backend, raw, expected):
    opts = backend.parse_options(raw)
    assert (opts.qgemm_opt_enable, opts.decompress) == expected


@pytest.mark.parametrize("bad", ["fused", "PRE", "", 1, True])
def test_bad_decompress_raises(backend, bad):
    with pytest.raises(ValueError) as exc:
        backend.parse_options({"decompress": bad})
    msg = str(exc.value)
    assert "decompress" in msg
    assert "on-the-fly" in msg and "pre" in msg


def test_bad_decompress_discarded_when_disabled(backend):
    opts = backend.parse_options({"qgemm_opt_enable": 0, "decompress": "bad"})
    assert opts.decompress is None


@pytest.mark.parametrize("mode", MODES)
def test_pass_position(backend, mode):
    vec = list(_build_vectorir_passes(backend.parse_options(
        {"decompress": mode})))
    entry = f"--qgemm-decompress-coupling=decompress={mode}"
    assert vec.count(entry) == 1
    i = vec.index(entry)
    assert vec[i - 1] == "--qgemm-recognize"
    first_tiling = next(j for j, p in enumerate(vec)
                        if p.startswith("--qgemm-tiling"))
    assert i < first_tiling


def test_pass_absent_when_disabled(backend):
    vec = _build_vectorir_passes(backend.parse_options({}))
    assert not any(p.startswith("--qgemm-decompress-coupling") for p in vec)


# --- End-to-end numerics, 4 formats x 2 modes (Req 12.7) ---


def _check(got, ref):
    ref = ref.to(torch.float32)
    assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={(got - ref).abs().max().item():.6g}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(32, 32, 32, 1), (64, 96, 128, 3)])
def test_q80_q80(device, mode, M, N, K, T):
    torch.manual_seed(0)
    KB = K // q80.QK8_0
    a_q = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w_q = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    a_d = torch.randn(M, KB, dtype=torch.float16)
    w_d = torch.randn(N, KB, dtype=torch.float16)
    ref = q80.q80_q80_reference(a_q, a_d, w_q, w_d, M, N, K)
    got = torch.empty((M, N), dtype=torch.float32)
    q80.q80_q80_strict[(T, 1, 1)](a_q, a_d, w_q, w_d, got, M=M, N=N, K=K,
                                  T=T, qgemm_opt_enable=1, decompress=mode)
    _check(got, ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(32, 32, 32, 1), (64, 96, 128, 3)])
def test_q40_q80(device, mode, M, N, K, T):
    torch.manual_seed(0)
    a_q, a_d, w_q, w_d = q40._make_inputs(M, N, K)
    ref, _ = q40.q40_q80_reference(a_q, a_d, w_q, w_d, M, N, K)
    got = torch.empty((M, N), dtype=torch.float32)
    q40.q40_q80_canonical[(T, 1, 1)](a_q, a_d, w_q, w_d, got, M=M, N=N, K=K,
                                     T=T, qgemm_opt_enable=1,
                                     decompress=mode)
    _check(got, ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(32, 32, 256, 1), (64, 96, 512, 3)])
def test_q4k_q8k(device, mode, M, N, K, T):
    torch.manual_seed(0)
    a_q, a_d, w_q, w_d, w_dmin, scales, _, _ = q4k._make_inputs(M, N, K)
    ref, *_ = q4k.q4k_q8k_reference(a_q, a_d, w_q, w_d, w_dmin, scales, M, N,
                                    K)
    got = torch.empty((M, N), dtype=torch.float32)
    q4k.q4k_q8k_canonical[(T, 1, 1)](a_q, a_d, w_q, w_d, w_dmin, scales, got,
                                     M=M, N=N, K=K, T=T, qgemm_opt_enable=1,
                                     decompress=mode)
    _check(got, ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(32, 32, 256, 1), (64, 96, 512, 3)])
def test_iq2xxs_q8k(device, mode, M, N, K, T):
    torch.manual_seed(0)
    a_q, a_d, qs, w_d = iq2._make_inputs(M, N, K)
    ref, _ = iq2.iq2xxs_q8k_reference(a_q, a_d, qs, w_d, M, N, K)
    got = torch.empty((M, N), dtype=torch.float32)
    iq2.iq2xxs_q8k_canonical[(T, 1, 1)](a_q, a_d, qs, w_d,
                                        iq2.IQ2XXS_GRID_BYTES,
                                        iq2.KSIGNS_IQ2XS, iq2.KMASK_IQ2XS,
                                        got, M=M, N=N, K=K, T=T,
                                        qgemm_opt_enable=1, decompress=mode)
    _check(got, ref)
