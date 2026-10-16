# Copyright (c) the triton-riscv authors. Licensed under the Apache License 2.0.
"""Tests for the `no_alias` compile option.

`no_alias=True` is the user's guarantee that no two pointer arguments of the
kernel overlap. The backend then runs --mark-noalias-args first in the
ttsharedir stage, the arguments carry `llvm.noalias`, and the coupling pass's
load-copy forwarding may drop each operand's private copy. The option is
independent of qgemm_opt_enable.

The kernels and their PyTorch references are shared with
test_qgemm_decompress_coupling.py.
"""

import pytest
import torch
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.compiler import (CPUBackend,
                                                    _build_ttsharedir_passes)

from test_qgemm_decompress_coupling import MODES, _check, iq2, q4k, q40, q80

# --- Option resolution and pass placement (Req 15) ---


@pytest.fixture(scope="module")
def backend():
    return CPUBackend(GPUTarget("cpu", "x86_64", 0))


def test_default_false(backend):
    assert backend.parse_options({}).no_alias is False


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_non_bool_raises(backend, bad):
    with pytest.raises(ValueError, match="no_alias"):
        backend.parse_options({"no_alias": bad})


def test_no_alias_alone_keeps_qgemm_off(backend):
    opts = backend.parse_options({"no_alias": True})
    assert opts.no_alias is True
    assert opts.qgemm_opt_enable == 0


def test_mark_pass_first_when_true(backend, monkeypatch):
    monkeypatch.delenv("TRITON_SHARED_SANITIZER_TYPE", raising=False)
    passes = _build_ttsharedir_passes(backend.parse_options(
        {"no_alias": True, "qgemm_opt_enable": 1}))
    assert passes[0] == "--mark-noalias-args"
    assert passes.count("--mark-noalias-args") == 1


def test_mark_pass_absent_when_false(backend, monkeypatch):
    monkeypatch.delenv("TRITON_SHARED_SANITIZER_TYPE", raising=False)
    passes = _build_ttsharedir_passes(backend.parse_options(
        {"no_alias": False, "qgemm_opt_enable": 1}))
    assert "--mark-noalias-args" not in passes


# --- End-to-end numerics with no_alias=True, 4 formats x 2 modes ---
# Tolerances: every format's reference script uses check_float's defaults,
# which _check applies (atol=1e-2, rtol=1e-2).


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(64, 96, 128, 3)])
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
                                  T=T, qgemm_opt_enable=1, decompress=mode,
                                  no_alias=True)
    _check(got, ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(64, 96, 128, 3)])
def test_q40_q80(device, mode, M, N, K, T):
    torch.manual_seed(0)
    a_q, a_d, w_q, w_d = q40._make_inputs(M, N, K)
    ref, _ = q40.q40_q80_reference(a_q, a_d, w_q, w_d, M, N, K)
    got = torch.empty((M, N), dtype=torch.float32)
    q40.q40_q80_canonical[(T, 1, 1)](a_q, a_d, w_q, w_d, got, M=M, N=N, K=K,
                                     T=T, qgemm_opt_enable=1,
                                     decompress=mode, no_alias=True)
    _check(got, ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(64, 96, 512, 3)])
def test_q4k_q8k(device, mode, M, N, K, T):
    torch.manual_seed(0)
    a_q, a_d, w_q, w_d, w_dmin, scales, _, _ = q4k._make_inputs(M, N, K)
    ref, *_ = q4k.q4k_q8k_reference(a_q, a_d, w_q, w_d, w_dmin, scales, M, N,
                                    K)
    got = torch.empty((M, N), dtype=torch.float32)
    q4k.q4k_q8k_canonical[(T, 1, 1)](a_q, a_d, w_q, w_d, w_dmin, scales, got,
                                     M=M, N=N, K=K, T=T, qgemm_opt_enable=1,
                                     decompress=mode, no_alias=True)
    _check(got, ref)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("M, N, K, T", [(64, 96, 512, 3)])
def test_iq2xxs_q8k(device, mode, M, N, K, T):
    torch.manual_seed(0)
    a_q, a_d, qs, w_d = iq2._make_inputs(M, N, K)
    ref, _ = iq2.iq2xxs_q8k_reference(a_q, a_d, qs, w_d, M, N, K)
    got = torch.empty((M, N), dtype=torch.float32)
    iq2.iq2xxs_q8k_canonical[(T, 1, 1)](a_q, a_d, qs, w_d,
                                        iq2.IQ2XXS_GRID_BYTES,
                                        iq2.KSIGNS_IQ2XS, iq2.KMASK_IQ2XS,
                                        got, M=M, N=N, K=K, T=T,
                                        qgemm_opt_enable=1, decompress=mode,
                                        no_alias=True)
    _check(got, ref)
