"""Resolution, validation, and propagation of exact target VLEN."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from triton.backends.triton_shared import compiler as backend


def resolve(monkeypatch, *, option=None, env=None, bounds=(256, 65536), host=0):
    monkeypatch.delenv("TRITON_RISCV_VLEN", raising=False)
    if env is not None:
        monkeypatch.setenv("TRITON_RISCV_VLEN", env)
    bounds_query = Mock(return_value=bounds)
    host_query = Mock(return_value=host)
    monkeypatch.setattr(backend, "_riscv_vlen_bounds", bounds_query)
    monkeypatch.setattr(backend, "_read_host_vlen", host_query)
    args = dict(target_triple="riscv64-unknown-linux-gnu",
                target_cpu="spacemit-x60", target_features="+v",
                riscv_vlen=option)
    backend._resolve_vlen(args)
    return args["riscv_vlen"], host_query


@pytest.mark.parametrize("option,env,bounds,host,expected,reads", [
    (256, "512", (256, 65536), 1024, 256, 0),
    (None, "512", (256, 65536), 1024, 512, 0),
    (None, None, (512, 512), 256, 512, 0),
    (None, None, (256, 65536), 512, 512, 1),
    (None, None, (256, 65536), 0, 0, 1),
    (0, "512", (256, 65536), 512, 0, 0),
])
def test_resolution_order(monkeypatch, option, env, bounds, host, expected, reads):
    bits, host_query = resolve(monkeypatch, option=option, env=env,
                               bounds=bounds, host=host)
    assert bits == expected
    assert host_query.call_count == reads


@pytest.mark.parametrize("value", [-1, 64, 384, True, 256.5, "bad"])
def test_invalid_vlen(monkeypatch, value):
    with pytest.raises(ValueError, match="zero or a power of two"):
        resolve(monkeypatch, option=value)


@pytest.mark.parametrize("option,env,bounds,host", [
    (128, None, (256, 65536), 0),
    (512, None, (256, 256), 0),
    (None, "512", (256, 256), 0),
    (None, None, (256, 65536), 128),
])
def test_target_conflict(monkeypatch, option, env, bounds, host):
    with pytest.raises(ValueError, match="conflicts with target VLEN range"):
        resolve(monkeypatch, option=option, env=env, bounds=bounds, host=host)


def test_propagation_and_cache(monkeypatch):
    options = backend.CPUOptions(target_triple="riscv64-unknown-linux-gnu",
                                 target_cpu="spacemit-x60", riscv_vlen=256)
    monkeypatch.setenv("TRITON_RISCV_VLEN", "512")
    assert "vlen=256" in backend._set_llvm_target_args(options)[0]
    flags = backend._llvm_target_args(options)
    assert "-riscv-v-vector-bits-min=256" in flags
    assert "-riscv-v-vector-bits-max=256" in flags
    assert options.hash() != replace(options, riscv_vlen=512).hash()


def test_llvm_bounds_query():
    assert backend._riscv_vlen_bounds(
        "riscv64-unknown-linux-gnu", "spacemit-x60", "+v") == (256, 65536)


def test_host_reader():
    # The C++ reader returns zero when the plugin was built without RVV.
    bits = backend._read_host_vlen()
    assert bits == 0 or (bits >= 128 and bits & (bits - 1) == 0)


def test_x86_does_not_probe_host(monkeypatch):
    monkeypatch.delenv("TRITON_RISCV_VLEN", raising=False)
    host = Mock(side_effect=AssertionError("unexpected host probe"))
    monkeypatch.setattr(backend, "_read_host_vlen", host)
    args = dict(target_triple="x86_64-unknown-linux-gnu", riscv_vlen=None)
    backend._resolve_vlen(args)
    assert args["riscv_vlen"] == 0
    args["riscv_vlen"] = 256
    with pytest.raises(ValueError, match="only supported for RISC-V"):
        backend._resolve_vlen(args)


def test_nonzero_vlen_requires_vector_target(monkeypatch):
    with pytest.raises(ValueError, match="nonzero riscv_vlen requires a target with RVV support"):
        resolve(monkeypatch, option=256, bounds=(0, 65536))
