import hashlib
import tempfile
import sysconfig

import os, subprocess, platform
import importlib.util
import sys

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")

    return sanitizer_type


def _sanitizer_available(sanitizer_type):
    if "LD_PRELOAD" not in os.environ:
        return False
    if f"libclang_rt.{sanitizer_type}.so" not in os.environ["LD_PRELOAD"]:
        return False

    return True


def _sanitizer_flags(sanitizer_type):
    if sanitizer_type == "asan":
        return ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
    if sanitizer_type == "tsan":
        return ["-g", "-fsanitize=thread"]
    return []


# -------------------- CPU runtime -------------------------
# The worker pool that executes launch grids lives in its own shared library
# (backend/runtime/triton_cpu_rt.*).  It is built on first use on the machine
# that runs the kernels and cached like any other Triton artifact.  Every
# kernel launcher links against the same library, so a process has a single
# pool no matter how many kernels it launches.

_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
_RUNTIME_FILES = ("triton_cpu_rt.h", "triton_cpu_rt.cpp")
_runtime_libraries = {}


def _runtime_source_bytes():
    return b"".join((_RUNTIME_DIR / name).read_bytes() for name in _RUNTIME_FILES)


def _cxx_for(sanitizer_type):
    return _get_llvm_bin_path("clang++") if sanitizer_type != "" else "g++"


def _get_runtime_library(sanitizer_type):
    """Returns (path, key) of the CPU runtime library, building it if needed.

    The file name embeds the key and is also used as soname: the dynamic loader
    identifies libraries by soname, so two different builds must never share
    one, otherwise the first one loaded would silently serve both.
    """
    cached = _runtime_libraries.get(sanitizer_type)
    if cached is not None and os.path.exists(cached[0]):
        return cached

    cxx = _cxx_for(sanitizer_type)
    flags = [
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-pthread",
        *_sanitizer_flags(sanitizer_type),
    ]
    key = hashlib.sha256(
        _runtime_source_bytes() + " ".join([cxx, *flags]).encode("utf-8")
    ).hexdigest()
    name = f"libtriton_cpu_rt.{key[:16]}.so"
    cache = get_cache_manager(key)
    path = cache.get_file(name)
    if path is None:
        if platform.system() == "Darwin":
            soname = f"-Wl,-install_name,@rpath/{name}"
        else:
            soname = f"-Wl,-soname,{name}"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, name)
            subprocess.check_call(
                [
                    cxx,
                    *flags,
                    str(_RUNTIME_DIR / "triton_cpu_rt.cpp"),
                    f"-I{_RUNTIME_DIR}",
                    soname,
                    "-o",
                    out_path,
                    # dlopen of libspine_tcm (TCM of the SpacemiT K3).
                    *(["-ldl"] if platform.system() == "Linux" else []),
                ]
            )
            path = cache.put(Path(out_path).read_bytes(), name, binary=True)

    _runtime_libraries[sanitizer_type] = (path, key)
    return path, key


# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        # Proper support for bfloat16 and float16 is not yet handled.
        # https://github.com/microsoft/triton-shared/issues/348
        # "fp16": "TODO",
        # "bf16": "TODO",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _extracted_type(ty):
    if ty[0] == "*":
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


# Replaced by the threads per program instance of the kernel when the launcher
# is specialized for it (see compile_module).
_CPU_THREADS_PLACEHOLDER = "TRITON_CPU_THREADS_PLACEHOLDER"


def _generate_launcher(constants, signature, kernel_name):
    arg_decls = "".join(
        f", {_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items()
    )
    args_format = "".join(
        [_format_of(_extracted_type(ty)) for ty in signature.values()]
    )
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    kernel_arg_decls = ", ".join(
        _ty_to_cpp(ty) if ty[0] != "*" else "int64_t, void*"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_arg_decls += ", " if kernel_arg_decls else ""

    # Arguments forwarded to the kernel; constexprs are folded into the kernel
    # at compile time and are not passed.
    kernel_args = [(i, ty) for i, ty in signature.items() if ty != "constexpr"]

    args_fields = "".join(f"  {_ty_to_cpp(ty)} arg{i};\n" for i, ty in kernel_args)
    args_init = "".join(f"  kargs.arg{i} = arg{i};\n" for i, ty in kernel_args)
    # Pointer type (=Memref) becomes int64_t + MemRef struct.  Use some random
    # type "char" here.
    memref_decls = "".join(
        f"  StridedMemRefType<char, 0> ptr_arg{i} = "
        f"{{static_cast<char *>(a->arg{i}), static_cast<char *>(a->arg{i}), 0}};\n"
        for i, ty in kernel_args
        if ty[0] == "*"
    )
    kernel_parameters = "".join(
        f"a->arg{i}, " if ty[0] != "*" else f"0, &ptr_arg{i}, "
        for i, ty in kernel_args
    )

    launch_args_str = "".join(
        f", ptr_info{i}.dev_ptr" if ty[0] == "*" else f", _arg{i}"
        for i, ty in signature.items()
    )
    get_ptr_checks_str = "; ".join(
        (
            f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); "
            f"if (!ptr_info{i}.valid) return NULL;"
            if ty[0] == "*"
            else ""
        )
        for i, ty in signature.items()
    )

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"
#include "triton_cpu_rt.h"

#include <cstdlib>

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

// ---------------------------------------------------------------------------
// Program instance level parallelism
//
// The launch grid is executed by the CPU runtime (triton_cpu_rt): tcrt_launch()
// runs every program instance exactly once on the runtime's worker pool and
// returns when the whole grid is done.  The launcher only packs the kernel
// arguments and provides the per instance entry point below.
// ---------------------------------------------------------------------------

struct _KernelArgs {{
  int gridX, gridY, gridZ;
{args_fields}}};

static void _instance(void *raw, int x, int y, int z) {{
  const _KernelArgs *a = static_cast<const _KernelArgs *>(raw);
{memref_decls}  {kernel_name}({kernel_parameters}
                a->gridX, a->gridY, a->gridZ, x, y, z);
}}

static int _launch(int gridX, int gridY, int gridZ{arg_decls}) {{
  _KernelArgs kargs;
  kargs.gridX = gridX;
  kargs.gridY = gridY;
  kargs.gridZ = gridZ;
{args_init}
  tcrt_launch_desc desc;
  desc.abi_version = TCRT_ABI_VERSION;
  desc.grid[0] = gridX;
  desc.grid[1] = gridY;
  desc.grid[2] = gridZ;
  // Threads per program instance (metadata["cpu_threads"], see compiler.py).
  desc.warps = {_CPU_THREADS_PLACEHOLDER};
  desc.fn = _instance;
  desc.args = &kargs;
  // CSV timeline of the launch, see docs/05-Program-Instance-Parallelism.md.
  desc.trace_path = getenv("TRITON_CPU_TRACE_FILE");
  return tcrt_launch(&desc);
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas,
  //      &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {get_ptr_checks_str};
  const int _rc = _launch(gridX, gridY, gridZ{launch_args_str});
  if (_rc == TCRT_ERROR_NO_TCM) {{
    PyErr_SetString(PyExc_RuntimeError,
                    "cpu.alloc buffers could not be put in the TCM "
                    "(TRITON_CPU_TCM=require; see the triton-cpu-rt warnings)");
    return NULL;
  }}
  if (_rc != 0) {{
    PyErr_SetString(PyExc_RuntimeError,
                    "triton-cpu-rt rejected the launch (runtime ABI mismatch?)");
    return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, "include")
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib = "{name}{major}{minor}.lib".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    else:
        py_include_dir = os.path.join(
            sys.base_prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        py_lib_dir = os.path.join(sys.base_prefix, "lib")
        py_lib = "{name}{major}.{minor}".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile one kernel every time.
        # The cu_function parameter actually contains our kernel obj.
        # See CPUUtils.load_binary method.
        kernel_obj = cu_function
        # See pack_metadata in compiler.py.
        kernel_name = kernel_metadata[6]
        cpu_threads = int(kernel_metadata[7]) if len(kernel_metadata) > 7 else 1
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)
        src = src.replace(_CPU_THREADS_PLACEHOLDER, str(max(cpu_threads, 1)))

        sanitizer_type = _get_sanitizer_type()
        if platform.system() == "Windows":
            if sanitizer_type != "":
                raise Exception(
                    "Sanitizers are not supported on Windows with triton-shared."
                )
            # No shared runtime library on Windows: the runtime is compiled
            # into every launcher (one pool per kernel signature).
            runtime_key = hashlib.sha256(_runtime_source_bytes()).hexdigest()
        else:
            runtime_path, runtime_key = _get_runtime_library(sanitizer_type)

        key = hashlib.sha256(
            src.encode("utf-8") + kernel_obj + runtime_key.encode("utf-8")
        ).hexdigest()
        cache = get_cache_manager(key)
        name = "__triton_shared_ref_cpu_kernel_launcher"

        if platform.system() == "Windows":
            filename = f"{name}.pyd"
        else:
            filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                if platform.system() == "Windows":
                    obj_path = os.path.join(tmpdir, "kernel.obj")
                    launcher_src_path = os.path.join(tmpdir, "main.cxx")
                    so_path = os.path.join(tmpdir, "kernel.pyd")
                    Path(obj_path).write_bytes(kernel_obj)
                    Path(launcher_src_path).write_text(src)
                    # Compile it together.
                    subprocess.check_call(
                        [
                            "cl",
                            "/LD",
                            "/std:c++17",
                            launcher_src_path,
                            str(_RUNTIME_DIR / "triton_cpu_rt.cpp"),
                            obj_path,
                            f"-I{py_include_dir}",
                            f"-I{include_dir}",
                            f"-I{_RUNTIME_DIR}",
                            "/link",
                            f"/LIBPATH:{py_lib_dir}",
                            "/link",
                            f"{py_lib}",
                            f"/OUT:{so_path}",
                        ]
                    )
                else:
                    obj_path = os.path.join(tmpdir, "kernel.o")
                    launcher_src_path = os.path.join(tmpdir, "main.cxx")
                    so_path = os.path.join(tmpdir, "kernel.so")
                    Path(obj_path).write_bytes(kernel_obj)
                    Path(launcher_src_path).write_text(src)

                    if sanitizer_type != "" and not _sanitizer_available(
                        sanitizer_type
                    ):
                        raise Exception(
                            'Use LD_PRELOAD="path/to/libclang_rt.'
                            + sanitizer_type
                            + '.so" TRITON_SHARED_SANITIZER_TYPE='
                            + sanitizer_type
                            + " python ..."
                        )

                    # Compile it together.  The launcher links against the
                    # shared CPU runtime; the rpath lets the dynamic loader find
                    # it in the Triton cache.
                    subprocess_args = [
                        _cxx_for(sanitizer_type),
                        "-std=c++17",
                        launcher_src_path,
                        obj_path,
                        f"-I{py_include_dir}",
                        f"-I{include_dir}",
                        f"-I{_RUNTIME_DIR}",
                        f"-L{py_lib_dir}",
                        "-shared",
                        f"-l{py_lib}",
                        "-fPIC",
                        runtime_path,
                        f"-Wl,-rpath,{os.path.dirname(runtime_path)}",
                        *_sanitizer_flags(sanitizer_type),
                        "-o",
                        so_path,
                    ]
                    subprocess.check_call(subprocess_args)

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(
            gridX,
            gridY,
            gridZ,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            *args,
        )

    return launch


class CPULauncher(object):

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            return src.fn.arg_names.index(i) if isinstance(i, str) else i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(launcher_src, kernel_placeholder_name)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # obj of the kernel so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
            None,  # module
            kernel_obj,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )


class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "obj"

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
