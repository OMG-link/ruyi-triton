"""Index-offset conversion and compatibility with the CPU pipeline."""

import ctypes
import subprocess

import pytest
from triton.backends.triton_shared import compiler


def native_options():
    """The host target as the driver resolves it, so the object links."""
    args = {}
    compiler._resolve_target(args)
    return compiler.CPUOptions(**args)


def compile_ir(ttir, target):
    shared = compiler._ttir_to_ttsharedir(ttir)
    return compiler._optimize_llir(compiler._ttsharedir_to_llir(shared, target), target)


def test_index_shape_ops_native(tmp_path):
    triple = compiler._host_triple()
    if not triple:
        pytest.skip("native target unavailable")
    target = native_options()
    ttir = """module {
      tt.func public @shape_ops(%base: !tt.ptr<f32>, %out: !tt.ptr<f32>, %offset: index) {
        %splat = tt.splat %offset : index -> tensor<2xindex>
        %expanded = tt.expand_dims %splat {axis = 1 : i32} : tensor<2xindex> -> tensor<2x1xindex>
        %broadcast = tt.broadcast %expanded : tensor<2x1xindex> -> tensor<2x2xindex>
        %bases = tt.splat %base : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
        %pointers = tt.addptr %bases, %broadcast : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xindex>
        %values = tt.load %pointers : tensor<2x2x!tt.ptr<f32>>
        %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
        %indices = tt.reshape %range : tensor<4xi32> -> tensor<2x2xi32>
        %outputs = tt.splat %out : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
        %destinations = tt.addptr %outputs, %indices : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %destinations, %values : tensor<2x2x!tt.ptr<f32>>
        tt.return
      }
    }
    """
    llir = compile_ir(ttir, target)
    obj = tmp_path / "shape.o"
    obj.write_bytes(compiler._llir_to_bin(llir, {}, target))
    library = tmp_path / "shape.so"
    subprocess.check_call(["g++", "-shared", str(obj), "-o", str(library)])
    loaded = ctypes.CDLL(str(library))
    function = loaded.shape_ops
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t] + [ctypes.c_int32] * 6
    function.restype = None
    source = (ctypes.c_float * 4)(0, 1, 2, 3)
    destination = (ctypes.c_float * 4)()
    for offset in [-1, 1]:
        function(ctypes.byref(source, 2 * ctypes.sizeof(ctypes.c_float)),
                 destination, offset, *([0] * 6))
        assert list(destination) == [2 + offset] * 4
