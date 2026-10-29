import pytest
import torch
import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver
import triton.backends.triton_shared.benchmark as bench
from triton.backends.triton_shared.benchmark import do_bench_program, _clone_slots


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def _grid_kernel(out, X: tl.constexpr, scale=1):
    x = tl.program_id(0)
    y = tl.program_id(1)
    z = tl.program_id(2)
    index = (x * 3 + y) * 2 + z
    tl.store(out + index, tl.load(out + index) + scale + index)


@pytest.mark.parametrize("scale", [1, 2.5])
def test_bench_program_traverses_grid_and_rotates_cloned_slots(monkeypatch, scale):
    out = torch.zeros(12, dtype=torch.float32)
    captured = []
    clone_slots = bench._clone_slots

    def capture_slots(*args, **kwargs):
        result = clone_slots(*args, **kwargs)
        captured.extend(result)
        return result

    monkeypatch.setattr(bench, "_clone_slots", capture_slots)
    result = do_bench_program(
        _grid_kernel, (out, scale), constexprs={"X": 2},
        grid=lambda values: (values["X"], 3, 2),
        pool_bytes=out.untyped_storage().nbytes() * 3, warmup_rounds=1,
        repeat_rounds=3, return_mode="all",
    )
    assert len(result) == 3
    assert all(value >= 0 for value in result)
    assert len(captured) == 3
    for slot_args, _ in captured:
        torch.testing.assert_close(slot_args[0], 4 * (torch.arange(12, dtype=torch.float32) + scale))
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_bench_program_clones_aliasing_views_into_rotating_slots():
    base = torch.arange(24, dtype=torch.int32)
    left = base[2:14:2]
    right = base[6:18:2]
    slots = _clone_slots([left, right], pool_bytes=base.untyped_storage().nbytes() * 3)
    assert len(slots) == 3
    for (slot_args, _) in slots:
        a, b = slot_args
        assert a.untyped_storage()._cdata == b.untyped_storage()._cdata
        assert a.storage_offset() == left.storage_offset()
        assert b.storage_offset() == right.storage_offset()
        a[2] = -1
        assert b[0].item() == -1
    assert base[6].item() == 6


