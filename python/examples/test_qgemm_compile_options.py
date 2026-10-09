# Copyright (c) the triton-riscv authors. Licensed under the Apache License 2.0.
"""Example tests for QGEMM compile option resolution in CPUBackend."""

import pytest

from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.compiler import CPUBackend, CPUOptions

QGEMM_FIELDS = ("qgemm_opt_enable", "qgemm_mr", "qgemm_nr")


@pytest.fixture(scope="module")
def backend():
    return CPUBackend(GPUTarget("cpu", "x86_64", 0))


def _qgemm(opts):
    return tuple(getattr(opts, k) for k in QGEMM_FIELDS)


def test_cpuoptions_defaults_are_none():
    # Direct construction performs no derivation (Req 1.1, 2.1).
    assert _qgemm(CPUOptions()) == (None, None, None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({}, (0, None, None)),
        ({"qgemm_mr": 8}, (1, 8, 4)),
        ({"qgemm_mr": None}, (0, None, None)),
        ({"qgemm_opt_enable": 0, "qgemm_mr": 8}, (0, None, None)),
        ({"qgemm_opt_enable": True}, (1, 4, 4)),
        ({"qgemm_opt_enable": False, "qgemm_nr": 2}, (0, None, None)),
    ],
)
def test_derivation(backend, raw, expected):
    resolved = _qgemm(backend.parse_options(raw))
    assert resolved == expected
    assert all(type(v) is int for v in resolved if v is not None)


BAD_TUNING_VALUES = [0, -1, "4", 2.0, True]


@pytest.mark.parametrize("key", ["qgemm_mr", "qgemm_nr"])
@pytest.mark.parametrize("bad", BAD_TUNING_VALUES)
def test_bad_tuning_value_raises(backend, key, bad):
    # Passing the bad value alone derives enable=1, which triggers validation.
    with pytest.raises(ValueError, match=key):
        backend.parse_options({key: bad})


@pytest.mark.parametrize("key", ["qgemm_mr", "qgemm_nr"])
@pytest.mark.parametrize("bad", BAD_TUNING_VALUES)
def test_bad_tuning_value_discarded_when_disabled(backend, key, bad):
    # Req 2.9: with enable 0, tuning values are discarded unvalidated.
    opts = backend.parse_options({"qgemm_opt_enable": 0, key: bad})
    assert _qgemm(opts) == (0, None, None)


def test_bad_enable_raises(backend):
    with pytest.raises(ValueError, match="qgemm_opt_enable"):
        backend.parse_options({"qgemm_opt_enable": 2})


@pytest.mark.parametrize(
    "a, b",
    [
        ({"qgemm_mr": 8}, {"qgemm_opt_enable": 1, "qgemm_mr": 8, "qgemm_nr": 4}),
        ({}, {"qgemm_opt_enable": 0}),
        ({"qgemm_opt_enable": 0, "qgemm_mr": 8}, {}),
        ({"qgemm_opt_enable": True}, {"qgemm_opt_enable": 1}),
    ],
)
def test_hash_equalities(backend, a, b):
    assert backend.parse_options(a).hash() == backend.parse_options(b).hash()


def test_hash_differs_on_resolved_values(backend):
    assert (backend.parse_options({"qgemm_mr": 8}).hash() !=
            backend.parse_options({"qgemm_mr": 4}).hash())


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"qgemm_mr": 8},
        {"qgemm_opt_enable": 0, "qgemm_mr": 8},
        {"qgemm_opt_enable": True, "qgemm_nr": 16},
    ],
)
def test_parse_options_fixed_point(backend, raw):
    first = backend.parse_options(raw)
    assert backend.parse_options(first.__dict__) == first


# --- Pass list construction (Req 4.1, 4.2, 4.3, 4.5, 5.5) ---

from triton.backends.triton_shared.compiler import (  # noqa: E402
    _build_ttsharedir_passes, _build_vectorir_passes)

QGEMM_PASS_PREFIXES = ("--decompose-high-level-ops",
                       "--qgemm-simplify-dim-transforms", "--qgemm-recognize",
                       "--qgemm-tiling")


def _qgemm_entries(passes):
    return [p for p in passes if p.startswith(QGEMM_PASS_PREFIXES)]


def _all_passes(opts):
    return list(_build_ttsharedir_passes(opts)) + list(
        _build_vectorir_passes(opts))


def test_passes_disabled(backend):
    opts = backend.parse_options({"qgemm_opt_enable": 0, "qgemm_mr": 8})
    assert _qgemm_entries(_all_passes(opts)) == []


@pytest.mark.parametrize("mr, nr", [(4, 4), (8, 2)])
def test_passes_enabled(backend, mr, nr):
    opts = backend.parse_options({"qgemm_mr": mr, "qgemm_nr": nr})
    assert "--decompose-high-level-ops" in _build_ttsharedir_passes(opts)
    vec = list(_build_vectorir_passes(opts))
    assert "--qgemm-simplify-dim-transforms" in vec
    assert "--qgemm-recognize" in vec
    tiling = [p for p in vec if p.startswith("--qgemm-tiling")]
    assert tiling == ["--qgemm-tiling=tile-m=32 tile-n=32",
                      f"--qgemm-tiling=tile-m={mr} tile-n={nr}"]


def test_env_var_ignored(backend, monkeypatch):
    monkeypatch.setenv("TRITON_ENABLE_QGEMM_OPT", "1")
    assert _qgemm_entries(_all_passes(backend.parse_options({}))) == []


# --- Property-based tests ---

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from triton.backends.triton_shared.compiler import (  # noqa: E402
    _resolve_qgemm_options)

_ABSENT = object()
_ENABLE_VALUES = st.sampled_from([_ABSENT, None, 0, 1, True, False])
_TUNING_VALUES = st.one_of(st.sampled_from([_ABSENT, None]),
                           st.integers(min_value=1, max_value=1 << 16))


def _build(enable, mr, nr):
    raw = {}
    for k, v in zip(QGEMM_FIELDS, (enable, mr, nr)):
        if v is not _ABSENT:
            raw[k] = v
    return raw


def _reference(raw):
    tuning = ("qgemm_mr", "qgemm_nr")
    given_enable = raw.get("qgemm_opt_enable")
    if given_enable is not None:
        enable = int(given_enable)
    else:
        enable = int(any(raw.get(k) is not None for k in tuning))
    if enable == 0:
        return (0, None, None)
    return (1, ) + tuple(4 if raw.get(k) is None else raw[k] for k in tuning)


# Feature: qgemm-compile-options, Property 1: Resolution matches the reference model
# Validates: Requirements 1.2, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 2.9
@settings(max_examples=200)
@given(enable=_ENABLE_VALUES, mr=_TUNING_VALUES, nr=_TUNING_VALUES,
       extra_none=st.sampled_from(QGEMM_FIELDS))
def test_property_resolution_matches_reference(backend, enable, mr, nr,
                                               extra_none):
    raw = _build(enable, mr, nr)
    resolved = _resolve_qgemm_options(raw)
    triple = tuple(resolved[k] for k in QGEMM_FIELDS)
    assert triple == _reference(raw)
    assert all(type(v) is int for v in triple if v is not None)
    if triple[0] == 0:
        assert triple[1:] == (None, None)

    # Inserting a None-valued key equals omitting it.
    if extra_none not in raw:
        with_none = dict(raw, **{extra_none: None})
        assert _resolve_qgemm_options(with_none) == resolved

    # Same result through parse_options (Req 1.2).
    assert _qgemm(backend.parse_options(raw)) == triple


# Feature: qgemm-compile-options, Property 2: parse_options is a fixed point
# Validates: Requirements 2.7, 5.4
@settings(max_examples=200)
@given(enable=_ENABLE_VALUES, mr=_TUNING_VALUES, nr=_TUNING_VALUES)
def test_property_parse_options_fixed_point(backend, enable, mr, nr):
    first = backend.parse_options(_build(enable, mr, nr))
    second = backend.parse_options(first.__dict__)
    assert second == first
    assert second.hash() == first.hash()


_SMALL_TUNING_VALUES = st.sampled_from([_ABSENT, None, 4, 8])


# Feature: qgemm-compile-options, Property 3: Hash equality iff resolved values are equal
# Validates: Requirements 1.3, 2.9, 3.1, 5.2
@settings(max_examples=200)
@given(ea=_ENABLE_VALUES, ma=_SMALL_TUNING_VALUES, na=_SMALL_TUNING_VALUES,
       eb=_ENABLE_VALUES, mb=st.one_of(_SMALL_TUNING_VALUES, _TUNING_VALUES),
       nb=st.one_of(_SMALL_TUNING_VALUES, _TUNING_VALUES))
def test_property_hash_iff_resolved_equal(backend, ea, ma, na, eb, mb, nb):
    a = backend.parse_options(_build(ea, ma, na))
    b = backend.parse_options(_build(eb, mb, nb))
    assert (a.hash() == b.hash()) == (_qgemm(a) == _qgemm(b))
    if _qgemm(a)[0] == 0 and _qgemm(b)[0] == 0:
        assert a.hash() == b.hash()


_INVALID_TUNING_VALUES = st.one_of(
    st.integers(max_value=0),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.booleans(),
)


# Feature: qgemm-compile-options, Property 4: Invalid tuning values are rejected only when enabled
# Validates: Requirements 1.4, 2.9, 5.3
@settings(max_examples=200)
@given(key=st.sampled_from(["qgemm_mr", "qgemm_nr"]),
       bad=_INVALID_TUNING_VALUES)
def test_property_invalid_tuning_rejected_only_when_enabled(backend, key, bad):
    with pytest.raises(ValueError, match=key):
        backend.parse_options({key: bad})
    with pytest.raises(ValueError, match=key):
        backend.parse_options({"qgemm_opt_enable": 1, key: bad})
    opts = backend.parse_options({"qgemm_opt_enable": 0, key: bad})
    assert _qgemm(opts) == (0, None, None)


# Feature: qgemm-compile-options, Property 5: Pass list reflects options
# Validates: Requirements 1.5, 4.1, 4.2, 4.3, 5.5
@settings(max_examples=200)
@given(enable=_ENABLE_VALUES, mr=_TUNING_VALUES, nr=_TUNING_VALUES)
def test_property_pass_list_reflects_options(backend, enable, mr, nr):
    opts = backend.parse_options(_build(enable, mr, nr))
    if opts.qgemm_opt_enable == 0:
        assert _qgemm_entries(_all_passes(opts)) == []
        return
    assert "--decompose-high-level-ops" in _build_ttsharedir_passes(opts)
    vec = list(_build_vectorir_passes(opts))
    assert "--qgemm-simplify-dim-transforms" in vec
    assert "--qgemm-recognize" in vec
    tiling = [p for p in vec if p.startswith("--qgemm-tiling")]
    assert tiling == [
        "--qgemm-tiling=tile-m=32 tile-n=32",
        f"--qgemm-tiling=tile-m={opts.qgemm_mr} tile-n={opts.qgemm_nr}",
    ]


# --- Integration: Q8_0 x Q8_0 canonical-form kernel (Req 5.6) ---

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

QK8_0 = 32
TL_QK8_0 = tl.constexpr(QK8_0)


@triton.jit
def _q80_q80_kernel(a_q_ptr, a_d_ptr, w_q_ptr, w_d_ptr, c_ptr,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    T: tl.constexpr):
    QK: tl.constexpr = TL_QK8_0
    KB: tl.constexpr = K // QK
    n_per_pid: tl.constexpr = N // T
    n_start = tl.program_id(0) * n_per_pid

    a_q_bp = tl.make_block_ptr(a_q_ptr, (M, KB, QK), (K, QK, 1), (0, 0, 0),
                               (M, KB, QK), (2, 1, 0))
    a_d_bp = tl.make_block_ptr(a_d_ptr, (M, KB), (KB, 1), (0, 0), (M, KB),
                               (1, 0))
    w_q_bp = tl.make_block_ptr(w_q_ptr, (KB, QK, N), (QK * N, N, 1),
                               (0, 0, n_start), (KB, QK, n_per_pid),
                               (2, 1, 0))
    w_d_bp = tl.make_block_ptr(w_d_ptr, (KB, N), (N, 1), (0, n_start),
                               (KB, n_per_pid), (1, 0))
    c_bp = tl.make_block_ptr(c_ptr, (M, N), (N, 1), (0, n_start),
                             (M, n_per_pid), (1, 0))

    a_q = tl.broadcast_to(tl.expand_dims(tl.load(a_q_bp), axis=1),
                          (M, n_per_pid, KB, QK))
    w_q = tl.broadcast_to(tl.expand_dims(tl.load(w_q_bp), axis=0),
                          (M, KB, QK, n_per_pid))
    w_q = tl.trans(w_q, 0, 3, 1, 2)
    a_d = tl.broadcast_to(tl.expand_dims(tl.load(a_d_bp), axis=1),
                          (M, n_per_pid, KB))
    w_d = tl.broadcast_to(tl.expand_dims(tl.load(w_d_bp), axis=0),
                          (M, KB, n_per_pid))
    w_d = tl.trans(w_d, 0, 2, 1)

    block_sum = a_q.to(tl.int16) * w_q.to(tl.int16)
    block_sum = tl.sum(block_sum.to(tl.int32), axis=3)
    d = a_d.to(tl.float32) * w_d.to(tl.float32)
    c = tl.sum(block_sum.to(tl.float32) * d, axis=2)
    tl.store(c_bp, c)


def _q80_q80_reference(a_q, a_d, w_q, w_d):
    # Dequantize both operands, then GEMM.
    a = a_q.float() * a_d.float().repeat_interleave(QK8_0, dim=1)
    w = w_q.float() * w_d.float().repeat_interleave(QK8_0, dim=0)
    return a @ w


@pytest.mark.parametrize("qgemm_opt_enable", [1, 0])
def test_q80_q80_end_to_end(device, qgemm_opt_enable):
    # Shape known to pass the QGEMM path (test_q80_q80_qgemm_tiling.py).
    M, N, K, T = 64, 32, 64, 1
    torch.manual_seed(0)
    a_q = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w_q = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    a_d = torch.randn(M, K // QK8_0, dtype=torch.float16)
    w_d = torch.randn(K // QK8_0, N, dtype=torch.float16)

    out = torch.empty((M, N), dtype=torch.float32)
    _q80_q80_kernel[(T, 1, 1)](a_q, a_d, w_q, w_d, out, M=M, N=N, K=K, T=T,
                               qgemm_opt_enable=qgemm_opt_enable)
    ref = _q80_q80_reference(a_q, a_d, w_q, w_d)
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), (
        f"max_abs_diff={(out - ref).abs().max().item():.6g}")


# --- Integration: distinct tile options yield distinct variants (Req 5.7) ---


def test_q80_q80_tile_variants(device):
    M, N, K, T = 64, 32, 64, 1
    torch.manual_seed(0)
    a_q = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w_q = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    a_d = torch.randn(M, K // QK8_0, dtype=torch.float16)
    w_d = torch.randn(K // QK8_0, N, dtype=torch.float16)
    ref = _q80_q80_reference(a_q, a_d, w_q, w_d)

    tiles = [(4, 4), (8, 4)]
    for mr, nr in tiles:
        out = torch.empty((M, N), dtype=torch.float32)
        _q80_q80_kernel[(T, 1, 1)](a_q, a_d, w_q, w_d, out, M=M, N=N, K=K,
                                   T=T, qgemm_mr=mr, qgemm_nr=nr)
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), (
            f"mr={mr} nr={nr} "
            f"max_abs_diff={(out - ref).abs().max().item():.6g}")

    # Collect compiled variants across all device caches; other tests in this
    # module may have added variants, so select by resolved tile options.
    variants = {}
    for kernel_cache, *_ in _q80_q80_kernel.device_caches.values():
        for compiled in kernel_cache.values():
            md = compiled.metadata
            key = (md.qgemm_opt_enable, md.qgemm_mr, md.qgemm_nr)
            if key in {(1, mr, nr) for mr, nr in tiles}:
                variants[key] = compiled
    assert set(variants) == {(1, mr, nr) for mr, nr in tiles}
    a, b = variants.values()
    assert a.hash != b.hash
    opt_hashes = {
        CPUOptions(**{k: getattr(c.metadata, k)
                      for k in CPUOptions.__dataclass_fields__
                      if hasattr(c.metadata, k)}).hash()
        for c in (a, b)
    }
    assert len(opt_hashes) == 2
