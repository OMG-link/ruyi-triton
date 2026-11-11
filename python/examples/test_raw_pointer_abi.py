"""Check the CPU kernel ABI independently of the generated launcher."""

import ctypes
import re
import subprocess
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.triton_shared import compiler, driver


@pytest.mark.parametrize("addressing", ["pointer", "byte"])
def test_pointer_table_reload_after_mutation(tmp_path, addressing):
    # Both forms used to be mistaken for immutable descriptor fields. They
    # address ordinary mutable user data under the raw-pointer ABI.
    element_type, offset = (
        ("ptr", 1) if addressing == "pointer"
        else ("i8", ctypes.sizeof(ctypes.c_void_p))
    )
    llir = f"""
define i1 @pointer_table_changed(ptr %table, ptr %replacement, ptr %mutate) {{
  %slot = getelementptr {element_type}, ptr %table, i64 {offset}
  %before = load ptr, ptr %slot
  call void %mutate(ptr %table, ptr %replacement)
  %after = load ptr, ptr %slot
  %changed = icmp ne ptr %before, %after
  ret i1 %changed
}}
"""
    backend = compiler.CPUBackend(triton.runtime.driver.active.get_current_target())
    options = backend.parse_options({})
    optimized = compiler._optimize_llir(llir, options)
    obj = tmp_path / "pointer_table.o"
    obj.write_bytes(compiler._llir_to_bin(optimized, {}, options))
    library = tmp_path / "pointer_table.so"
    subprocess.check_call(["g++", "-shared", "-fPIC", str(obj), "-o", str(library)])
    loaded = ctypes.CDLL(str(library))

    pointer_table_type = ctypes.POINTER(ctypes.c_void_p)
    callback_type = ctypes.CFUNCTYPE(None, pointer_table_type, ctypes.c_void_p)

    @callback_type
    def mutate(table, replacement):
        table[1] = replacement

    function = loaded.pointer_table_changed
    function.argtypes = [pointer_table_type, ctypes.c_void_p, callback_type]
    function.restype = ctypes.c_bool
    before, after = ctypes.c_int(1), ctypes.c_int(2)
    table = (ctypes.c_void_p * 2)(None, ctypes.addressof(before))
    assert function(table, ctypes.addressof(after), mutate)
    assert table[1] == ctypes.addressof(after)


@triton.jit
def _store_num_programs(destination):
    singleton = tl.arange(0, 1)
    tl.store(destination + singleton, tl.num_programs(0))
    tl.store(destination + singleton + 1, tl.num_programs(1))
    tl.store(destination + singleton + 2, tl.num_programs(2))


def test_num_programs_with_singleton_stores():
    storage = torch.full((7,), -1, dtype=torch.int32)
    _store_num_programs[(2, 3, 4)](storage[2:])
    expected = torch.tensor([-1, -1, 2, 3, 4, -1, -1], dtype=torch.int32)
    torch.testing.assert_close(storage, expected)


@triton.jit
def _store_with_loop_carried_pointer(destination):
    for index in range(8):
        tl.store(destination, index.to(tl.float32))
        destination += 1


def test_scalar_stores_with_loop_carried_pointer():
    storage = torch.full((12,), -1.0)
    _store_with_loop_carried_pointer[(1,)](storage[2:])
    expected = storage.new_full(storage.shape, -1)
    expected[2:10] = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(storage, expected)


@triton.jit(do_not_specialize=["n", "stride"])
def _raw_pointer_kernel(source, destination, n, stride, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + indices * stride, indices < n, other=0)
    tl.store(destination + indices * stride, values + 7, indices < n)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("stride", [3, -2])
def test_raw_pointer_entry_with_offset_strided_views(tmp_path, alias, stride):
    n, block = 19, 16
    offset = 5 if stride > 0 else 45
    source_storage = torch.arange(80, dtype=torch.int32)
    destination_storage = source_storage if alias else torch.full_like(source_storage, -1)
    source = source_storage[offset:]
    destination = destination_storage[offset:]
    original = destination_storage.clone()
    expected = original.clone()
    indices = offset + torch.arange(n) * stride
    expected[indices] = source_storage[indices] + 7
    grid = (triton.cdiv(n, block),)

    compiled = _raw_pointer_kernel[grid](source, destination, n, stride, BLOCK=block)
    torch.testing.assert_close(destination_storage, expected)
    signature = re.search(
        rf"define[^\n]*@{re.escape(compiled.metadata.name)}\(([^\n]*)\)",
        compiled.asm["llir"],
    )
    assert signature is not None
    parameters = signature.group(1).split(",")
    assert len(parameters) == 10  # Two pointers, two scalars, six grid coordinates.
    assert all(parameter.strip().startswith("ptr ") for parameter in parameters[:2])
    assert all(parameter.strip().startswith("i32 ") for parameter in parameters[2:])

    # Call the object through a handwritten C ABI: no descriptor or rank argument.
    obj = tmp_path / "kernel.o"
    library = tmp_path / "kernel.so"
    obj.write_bytes(compiled.asm["obj"])
    runtime = tmp_path / "runtime.cpp"
    runtime.write_text('#include "ExecutionEngine/CRunnerUtils.cpp"\n')
    include = Path(driver.__file__).resolve().parent / "include"
    subprocess.check_call(["g++", "-shared", "-fPIC", str(obj), str(runtime),
                           f"-I{include}", "-o", str(library)])
    loaded = ctypes.CDLL(str(library))
    kernel = getattr(loaded, compiled.metadata.name)
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int32] * 8
    kernel.restype = None
    destination_storage.copy_(original)
    for program in range(grid[0]):
        kernel(source.data_ptr(), destination.data_ptr(), n, stride,
               grid[0], 1, 1, program, 0, 0)
    torch.testing.assert_close(destination_storage, expected)


@triton.jit
def _raw_pointer_atomic_add(pointer, previous, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    old = tl.atomic_add(pointer + offsets * 2, 1)
    tl.store(previous + offsets, old)


def test_raw_pointer_atomic_add_with_offset_view():
    storage = torch.arange(40, dtype=torch.int32)
    pointer = storage[3:]
    previous = torch.empty(16, dtype=torch.int32)
    expected = storage.clone()
    expected[3:35:2] += 1
    _raw_pointer_atomic_add[(1,)](pointer, previous, BLOCK=16)
    torch.testing.assert_close(previous, torch.arange(3, 35, 2, dtype=torch.int32))
    torch.testing.assert_close(storage, expected)


@triton.jit(do_not_specialize=["row_stride"])
def _raw_pointer_gather(source, indices, destination, row_stride,
                        MASKED: tl.constexpr):
    row_ids = tl.arange(0, 4)
    rows = tl.load(indices + row_ids)
    columns = tl.arange(0, 4)
    offsets = rows[:, None] * row_stride + columns[None, :] * 2
    if MASKED:
        mask = (rows[:, None] < 4) & (columns[None, :] < 3)
        values = tl.load(source + offsets, mask=mask, other=-17)
    else:
        values = tl.load(source + offsets)
    tl.store(destination + row_ids[:, None] * 4 + columns[None, :], values)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_raw_pointer_gather_with_negative_stride_and_mask(masked, alias):
    source = torch.arange(64, dtype=torch.int32)
    # The invalid row would address memory before the allocation if accessed.
    rows = torch.tensor([0, 3, 1, 100 if masked else 4], dtype=torch.int32)
    output = source if alias else torch.full((24,), -99, dtype=torch.int32)
    expected = output.clone()
    for row, index in enumerate(rows.tolist()):
        for column in range(4):
            expected[3 + row * 4 + column] = (
                -17 if masked and (index >= 4 or column >= 3)
                else source[40 - index * 8 + column * 2]
            )
    _raw_pointer_gather[(1,)](source[40:], rows, output[3:], -8, MASKED=masked)
    torch.testing.assert_close(output, expected)
