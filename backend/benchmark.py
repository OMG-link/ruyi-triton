"""Native CPU benchmark helpers for Triton Shared kernels."""

import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from triton.backends.triton_shared import driver
from triton.runtime.cache import get_cache_manager
from triton.testing import _summarize_statistics


def _cpp_type(ty):
    return driver._ty_to_cpp(ty)


def _build_benchmark(compiled, signature, constants):
    runtime = [(i, ty) for i, ty in signature.items()
               if i not in constants and ty != "constexpr"]
    fields = []
    call_args = []
    slot_types = []
    for i, ty in runtime:
        if ty.startswith("*"):
            fields.append(f"  void *arg{i};")
            call_args.append(f"slots[s].arg{i}")
            slot_types.append((i, "pointer", ty))
        else:
            fields.append(f"  {_cpp_type(ty)} arg{i};")
            call_args.append(f"slots[s].arg{i}")
            slot_types.append((i, "scalar", ty))
    kernel_name = compiled.metadata.name
    param_decls = []
    for i, ty in runtime:
        param_decls.append(_cpp_type(ty))
    params = ", ".join(param_decls + ["int", "int", "int", "int", "int", "int"])
    call_args = ", ".join(call_args + ["gx", "gy", "gz", "x", "y", "z"])
    src = f'''#include <chrono>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"
struct Slot {{
{os.linesep.join(fields)}
}};
extern "C" void {kernel_name}({params});
extern "C" void benchmark(Slot *slots, int count, int gx, int gy, int gz,
                           int warmups, int repeats, double *out) {{
  for (int r = 0; r < warmups; ++r)
    for (int s = 0; s < count; ++s)
      for (int x = 0; x < gx; ++x)
        for (int y = 0; y < gy; ++y)
          for (int z = 0; z < gz; ++z) {kernel_name}({call_args});
  for (int r = 0; r < repeats; ++r) {{
    auto begin = std::chrono::steady_clock::now();
    for (int s = 0; s < count; ++s)
      for (int x = 0; x < gx; ++x)
        for (int y = 0; y < gy; ++y)
          for (int z = 0; z < gz; ++z) {kernel_name}({call_args});
    auto end = std::chrono::steady_clock::now();
    out[r] = std::chrono::duration<double, std::milli>(end - begin).count() / count;
  }}
}}
'''
    identity = hashlib.sha256(src.encode() + compiled.asm["obj"]).hexdigest()
    name = f"triton_shared_bench_{identity[:16]}"
    cache = get_cache_manager(identity)
    so_path = cache.get_file(name + ".so")
    if so_path is None:
        # In installed builds, locate the backend include directory through driver.py.
        include = Path(driver.__file__).resolve().parent / "include"
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / "bench.cpp"
            obj = Path(tmp) / "kernel.o"
            so = Path(tmp) / "bench.so"
            cpp.write_text(src)
            obj.write_bytes(compiled.asm["obj"])
            cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(cpp), str(obj), f"-I{include}", "-o", str(so)]
            if os.name != "nt":
                cmd.insert(1, "-Wl,-Bsymbolic")
            driver._append_vector_math_link_args(cmd)
            subprocess.check_call(cmd)
            so_path = cache.put(so.read_bytes(), name + ".so", binary=True)
    lib = ctypes.CDLL(so_path)
    struct_fields = [(f"arg{i}", ctypes.c_void_p if kind == "pointer" else {
        "i1": ctypes.c_int32, "i8": ctypes.c_int8, "i16": ctypes.c_int16,
        "i32": ctypes.c_int32, "i64": ctypes.c_int64, "u1": ctypes.c_uint32,
        "u8": ctypes.c_uint8, "u16": ctypes.c_uint16, "u32": ctypes.c_uint32,
        "u64": ctypes.c_uint64, "fp32": ctypes.c_float, "f32": ctypes.c_float,
        "fp64": ctypes.c_double,
    }[ty]) for i, kind, ty in slot_types]
    class Slot(ctypes.Structure):
        _fields_ = struct_fields
    fn = lib.benchmark
    fn.argtypes = [ctypes.POINTER(Slot), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    fn.restype = None
    return Slot, fn, slot_types


def _clone_slots(args, pool_bytes):
    storages = {}
    for value in args:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages.setdefault(storage._cdata, (storage, storage.nbytes()))
    slot_bytes = sum(size for _, size in storages.values())
    count = max(1, math.ceil(pool_bytes / slot_bytes)) if slot_bytes else 1
    slots = []
    for _ in range(count):
        copies = {}
        owners = []
        for key, (storage, nbytes) in storages.items():
            source = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (nbytes,), (1,))
            # Keep the sample pointer's low alignment bits for specialization guards.
            sample_ptr = storage.data_ptr()
            base = torch.empty(nbytes + 64, dtype=torch.uint8)
            offset = (sample_ptr - base.data_ptr()) % 64
            target = base[offset:offset + nbytes]
            target.copy_(source)
            copies[key] = (target, storage)
            owners.append(base)
        slot_args = []
        for value in args:
            if isinstance(value, torch.Tensor):
                target, storage = copies[value.untyped_storage()._cdata]
                element_offset = (target.data_ptr() - target.untyped_storage().data_ptr()) // value.element_size()
                view = torch.empty(0, dtype=value.dtype).set_(target.untyped_storage(),
                    value.storage_offset() + element_offset, tuple(value.shape), tuple(value.stride()))
                slot_args.append(view)
            else:
                slot_args.append(value)
        slots.append((slot_args, owners))
    return slots


def do_bench_program(kernel, args, *, constexprs=None, grid, pool_bytes=64 << 20,
                     warmup_rounds=1, repeat_rounds=10, return_mode="median", quantiles=None):
    """Benchmark a complete CPU kernel grid without Python launch timing.

    Example::

        from triton.backends.triton_shared.benchmark import do_bench_program
        ms = do_bench_program(kernel, (input, output), constexprs={"N": n}, grid=(4,))

    ``args`` supplies positional runtime arguments in kernel order, skipping
    names present in ``constexprs``; omitted arguments use their Python defaults.
    The grid is traversed serially in x, y, z order, without OpenMP. Each native
    round visits every slot once and reports milliseconds per full-grid launch,
    averaged over slots. Slots preserve tensor storage aliases, views, and pointer
    alignment, and kernel state carries between rounds.
    """
    constexprs = {} if constexprs is None else dict(constexprs)
    bound = kernel.signature.bind_partial()
    runtime_names = [name for name in kernel.arg_names if name not in constexprs]
    if len(args) > len(runtime_names):
        raise TypeError(f"expected at most {len(runtime_names)} positional runtime arguments, got {len(args)}")
    bound.arguments.update(zip(runtime_names, args))
    bound.arguments.update(constexprs)
    bound.apply_defaults()
    compile_args = [bound.arguments[name] for name in kernel.arg_names]
    compiled = kernel.warmup(*compile_args, grid=grid)
    def arg_index(key):
        if isinstance(key, str):
            return kernel.arg_names.index(key)
        if isinstance(key, tuple):
            return key[0]
        return key
    signature = {arg_index(k): v for k, v in compiled.src.signature.items()}
    constants = {arg_index(k): v for k, v in compiled.src.constants.items()}
    runtime_indices = [i for i in range(len(compile_args))
                       if i not in constants and signature.get(i) != "constexpr"]
    runtime_args = [compile_args[i] for i in runtime_indices]
    slots = _clone_slots(runtime_args, pool_bytes)
    gdims = tuple(grid(bound.arguments)) if callable(grid) else tuple(grid)
    if not 1 <= len(gdims) <= 3:
        raise ValueError("grid must have between one and three dimensions")
    gdims = gdims + (1,) * (3 - len(gdims))
    Slot, native, slot_types = _build_benchmark(compiled, signature, constants)
    native_slots = (Slot * len(slots))()
    keepalive = []
    for n, (slot_args, owners) in enumerate(slots):
        keepalive.extend(owners)
        for i, kind, ty in slot_types:
            value = slot_args[runtime_indices.index(i)]
            if kind == "pointer":
                setattr(native_slots[n], f"arg{i}", value.data_ptr())
            else:
                setattr(native_slots[n], f"arg{i}", value)
    output = (ctypes.c_double * repeat_rounds)()
    native(native_slots, len(slots), *gdims, warmup_rounds, repeat_rounds, output)
    return _summarize_statistics(list(output), quantiles, return_mode)
