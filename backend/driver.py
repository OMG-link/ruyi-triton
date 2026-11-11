import hashlib
import tempfile

import os
import subprocess
import platform
import importlib.util
import sys
import triton
from triton._C.libtriton import native_specialize_impl

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from .paths import _get_llvm_bin_path
from . import board, cpu_runtime, spm


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



def _append_vector_math_link_args(subprocess_args):
    # LLVM's LIBMVEC vector-library mapping emits glibc vector ABI symbols such
    # as _ZGVdN8v_expf. Link them explicitly into the generated launcher.
    if platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"}:
        subprocess_args.append("-lmvec")
    return subprocess_args



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



_WARPS_PLACEHOLDER = "__TRITON_CPAR_WARPS__"


def _generate_launcher(constants, signature, kernel_name, spm_size=0, spm_align=1):
    """The C++ source of a kernel's Python launcher.

    The launch grid is executed by the CPU runtime (cpu_runtime.py):
    tcrt_launch() runs every program instance exactly once on the runtime's
    worker pool and returns when the whole grid is done. The launcher packs
    the kernel arguments and provides the per instance entry point. A kernel
    that uses tl.dev.alloc(placement="spm") takes its slot's scratchpad
    region as trailing argument. _WARPS_PLACEHOLDER is replaced at launch by
    the threads per program instance (metadata cpu_threads).
    """
    arg_decls = "".join(f", {_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join(
        [_format_of(_extracted_type(ty)) for ty in signature.values()]
    )
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )
    runtime_args = [(i, ty) for i, ty in signature.items() if ty != "constexpr"]

    kernel_arg_decls = "".join(f"{_ty_to_cpp(ty)}, " for _, ty in runtime_args)
    spm_param = ", void *" if spm_size else ""
    spm_arg = ", spm" if spm_size else ""
    args_fields = "".join(f"  {_ty_to_cpp(ty)} arg{i};\n" for i, ty in runtime_args)
    args_init = "".join(
        (f"  kargs.arg{i} = arg{i};\n" if ty[0] == "*"
         else f"  kargs.arg{i} = static_cast<{_ty_to_cpp(ty)}>(arg{i});\n")
        for i, ty in runtime_args)
    kernel_parameters = "".join(f"a->arg{i}, " for i, _ in runtime_args)

    launch_args_str = ", ".join(
        f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"
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
#include <stdlib.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"
#include "triton_cpu_rt.h"

extern "C" {{
  // Each pointer argument uses the raw data-pointer ABI.
  void {kernel_name}({kernel_arg_decls}int, int, int, int, int, int{spm_param});
}}

struct _KernelArgs {{
  int gridX, gridY, gridZ;
{args_fields}}};

static void _instance(void *raw, void *spm, int x, int y, int z) {{
  (void)spm;
  const _KernelArgs *a = static_cast<const _KernelArgs *>(raw);
  {kernel_name}({kernel_parameters}a->gridX, a->gridY, a->gridZ, x, y, z{spm_arg});
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
  desc.warps = {_WARPS_PLACEHOLDER};
  desc.spm_bytes = {spm_size};
  desc.spm_align = {spm_align};
  desc.fn = _instance;
  desc.args = &kargs;
  // CSV timeline of the launch, see docs/09-Program-Instance-Parallelism.md.
  desc.trace_path = getenv("TRITON_CPAR_TRACE_FILE");
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
  static PyObject *data_ptr_name = NULL;
  if (!data_ptr_name)
    data_ptr_name = PyUnicode_InternFromString("data_ptr");
  if (data_ptr_name) {{
    // Call the method through CPython's vectorcall path. This avoids both the
    // temporary bound-method object and the empty argument tuple per pointer.
    PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_name);
    if (!ret) {{
      PyErr_Clear();
    }} else {{
      if (!PyLong_Check(ret)) {{
        PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
        Py_DECREF(ret);
        ptr_info.valid = false;
        return ptr_info;
      }}
      ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
      Py_DECREF(ret);
      return ptr_info;
    }}
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {" ".join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
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
  int rc = _launch(gridX, gridY, gridZ, {launch_args_str});
  if (rc == TCRT_ERROR_SPM) {{
    // driver._raise_spm_error builds the report and raises SpmOutOfMemoryError.
    PyObject *mod = PyImport_ImportModule("triton.backends.triton_shared.driver");
    if (mod) {{
      PyObject *ret = PyObject_CallMethod(mod, "_raise_spm_error", "snn",
                                          "{kernel_name}", (Py_ssize_t){spm_size},
                                          (Py_ssize_t){spm_align});
      Py_XDECREF(ret);
      Py_DECREF(mod);
    }}
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError, "SPM unavailable");
    return NULL;
  }}
  if (rc != TCRT_OK) {{
    PyErr_Format(PyExc_RuntimeError, "tcrt_launch failed with %d", rc);
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




def _raise_spm_error(kernel_name, bytes_, align):
    """Called by a launcher whose launch the runtime refused for lack of SPM."""
    spm.raise_launch_error(cpu_runtime.spm_status(), kernel_name, bytes_, align,
                           cpu_runtime.active().config)


def _launcher_cache_key(src, kernel_obj, runtime_path):
    cache_input = (
        src.encode("utf-8")
        + kernel_obj
        + _LAUNCHER_CACHE_VERSION
        + os.path.basename(runtime_path).encode("utf-8")
    )
    return hashlib.sha256(cache_input).hexdigest()


_LAUNCHER_CACHE_VERSION = b"triton_shared_launcher_v4"


def compile_module(launcher_src, kernel_placeholder_name):
    if platform.system() != "Linux":
        raise RuntimeError("the triton-riscv CPU runtime requires Linux")
    py_version = sys.version_info
    py_include_dir = os.path.join(
        sys.base_prefix,
        "include",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    )
    py_lib_dir = os.path.join(sys.base_prefix, "lib")
    py_lib = "{name}{major}.{minor}".format(
        name="python", major=py_version.major, minor=py_version.minor
    )
    include_dir = str(cpu_runtime.INCLUDE_DIR)
    runtime_dir = str(cpu_runtime.RUNTIME_DIR)
    # Loading an extension module from disk costs O(100 us) even when its
    # compiled shared object is already in Triton's cache. Keep the imported
    # module and bound launch function alive for repeated kernel invocations.
    loaded_launchers = {}
    loaded_kernels = {}

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
        cpu_threads = kernel_metadata[7]
        config = board.CONFIGS[kernel_metadata[8]]
        sanitizer_type = _get_sanitizer_type()
        runtime_path = cpu_runtime.configure(config, sanitizer_type)
        kernel_key = (kernel_name, kernel_obj, sanitizer_type, cpu_threads)
        launch_fn = loaded_kernels.get(kernel_key)
        if launch_fn is not None:
            return launch_fn(
                gridX,
                gridY,
                gridZ,
                kernel_metadata,
                launch_metadata,
                launch_enter_hook,
                launch_exit_hook,
                *args,
            )
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)
        src = src.replace(_WARPS_PLACEHOLDER, str(max(int(cpu_threads), 1)))

        key = _launcher_cache_key(src, kernel_obj, runtime_path)
        cache = get_cache_manager(key)
        name = f"__triton_shared_ref_cpu_kernel_launcher_{key[:16]}"
        src = src.replace("__triton_shared_ref_cpu_kernel_launcher", name)

        loaded = loaded_launchers.get(key)
        if loaded is not None:
            return loaded[1](
                gridX,
                gridY,
                gridZ,
                kernel_metadata,
                launch_metadata,
                launch_enter_hook,
                launch_exit_hook,
                *args,
            )

        filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                obj_path = os.path.join(tmpdir, "kernel.o")
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                so_path = os.path.join(tmpdir, "kernel.so")
                Path(obj_path).write_bytes(kernel_obj)
                Path(launcher_src_path).write_text(src)
                # The runtime library is linked by path and found again by its
                # unique soname; the process already loaded it (cpu_runtime).
                runtime_link_args = [
                    runtime_path,
                    f"-Wl,-rpath,{os.path.dirname(runtime_path)}",
                ]

                # Compile it together.
                if sanitizer_type != "":
                    if not _sanitizer_available(sanitizer_type):
                        raise Exception(
                            'Use LD_PRELOAD="path/to/libclang_rt.'
                            + sanitizer_type
                            + '.so" TRITON_SHARED_SANITIZER_TYPE='
                            + sanitizer_type
                            + " python ..."
                        )
                    subprocess_args = [
                        _get_llvm_bin_path("clang++"),
                        "-std=c++17",
                        launcher_src_path,
                        obj_path,
                        f"-I{py_include_dir}",
                        f"-I{include_dir}",
                        f"-I{runtime_dir}",
                        f"-L{py_lib_dir}",
                        f"-Wl,-rpath,{py_lib_dir}",
                        "-shared",
                        f"-l{py_lib}",
                        "-fPIC",
                        "-o",
                        so_path,
                        "-Wl,-Bsymbolic",
                    ]
                    if sanitizer_type == "asan":
                        subprocess_args.extend(
                            ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
                        )
                    elif sanitizer_type == "tsan":
                        subprocess_args.extend(["-g", "-fsanitize=thread"])
                else:
                    subprocess_args = [
                        "g++",
                        "-std=c++17",
                        launcher_src_path,
                        obj_path,
                        f"-I{py_include_dir}",
                        f"-I{include_dir}",
                        f"-I{runtime_dir}",
                        "-shared",
                        "-fPIC",
                        "-o",
                        so_path,
                        "-Wl,-Bsymbolic",
                    ]
                subprocess_args += runtime_link_args
                subprocess_args = _append_vector_math_link_args(subprocess_args)
                subprocess.check_call(subprocess_args)

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        launch_fn = mod.launch
        loaded_launchers[key] = (mod, launch_fn)
        loaded_kernels[kernel_key] = launch_fn
        return launch_fn(
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


_compiled_launcher_cache = {}
_launcher_signature_cache = {}


class CPULauncher(object):
    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            return src.fn.arg_names.index(i) if isinstance(i, str) else i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        # A kernel's scratchpad requirement (compiler.py) is part of its
        # launcher, so it is part of the key.
        spm_size = int(getattr(metadata, "spm_size", 0))
        spm_align = int(getattr(metadata, "spm_align", 1))
        launcher_key = (
            tuple(sorted(constants.items())),
            tuple(signature.items()),
            spm_size,
            spm_align,
            _get_sanitizer_type(),
        )
        self.launch = _launcher_signature_cache.get(launcher_key)
        if self.launch is not None:
            return
        launcher_src = _generate_launcher(
            constants, signature, kernel_placeholder_name, spm_size, spm_align)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = _compiled_launcher_cache.get(launcher_src)
        if self.launch is None:
            self.launch = compile_module(launcher_src, kernel_placeholder_name)
            _compiled_launcher_cache[launcher_src] = self.launch
        _launcher_signature_cache[launcher_key] = self.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class PreparedCPUKernel:
    """A fixed-specialization CPU launch path with runtime pointer arguments."""

    def __init__(self, jit_kernel, grid, *sample_args, **compile_kwargs):
        # Separate kernel parameters from backend/compiler options before
        # binding the function signature. JITFunction.warmup still receives
        # both sets and remains the authority for specialization and codegen.
        parameter_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key in jit_kernel.arg_names
        }
        bound = jit_kernel.signature.bind(*sample_args, **parameter_kwargs)
        bound.apply_defaults()
        argument_template = [bound.arguments[name] for name in jit_kernel.arg_names]

        compiled = jit_kernel.warmup(
            *sample_args,
            grid=grid,
            **compile_kwargs,
        )
        # Accessing .run initializes the CPU launcher and module lifecycle once.
        launcher = compiled.run

        if callable(grid):
            grid = grid(bound.arguments)
        grid = tuple(grid)
        if not 1 <= len(grid) <= 3:
            raise ValueError("CPU launch grid must have between one and three axes")
        self.grid = grid + (1,) * (3 - len(grid))
        self.compiled = compiled
        self.launcher = launcher
        self.jit_kernel = jit_kernel
        self.original_grid = grid
        self.compile_kwargs = compile_kwargs.copy()
        self.backend_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key not in jit_kernel.arg_names
        }
        self.argument_template = argument_template
        constexprs = set(jit_kernel.constexprs)
        self.runtime_indices = [
            index for index in range(len(argument_template)) if index not in constexprs
        ]
        # Use Triton's generated native binder as the source of truth for
        # dtype, scalar-value and alignment specialization. Keeping only the
        # resulting tuples avoids retaining the sample tensors.
        device = triton.runtime.driver.active.get_current_device()
        _, _, _, backend, _ = jit_kernel.device_caches[device]
        self.runtime_specializations = []
        for index in self.runtime_indices:
            param = jit_kernel.params[index]
            specialize = not param.do_not_specialize
            if param.annotation_type == "u1" or param.annotation_type[:2] in {
                "fp",
                "bf",
            }:
                specialize = False
            self.runtime_specializations.append(
                (
                    backend,
                    param.is_const,
                    specialize,
                    not param.do_not_specialize_on_alignment,
                    native_specialize_impl(
                        backend,
                        argument_template[index],
                        param.is_const,
                        specialize,
                        not param.do_not_specialize_on_alignment,
                    ),
                )
            )
        # Runtime tensors can be very large. They are only examples used for
        # specialization and must not be retained by a long-lived runner.
        for index in self.runtime_indices:
            self.argument_template[index] = None

    def _call_kwargs(self, args):
        kwargs = dict(zip(self.jit_kernel.arg_names, args))
        kwargs.update(self.backend_kwargs)
        return kwargs

    def _standard_launch(self, args):
        return self.jit_kernel.run(
            grid=self.original_grid,
            warmup=False,
            **self._call_kwargs(args),
        )

    def __call__(self, *runtime_args):
        if len(runtime_args) != len(self.runtime_indices):
            raise TypeError(
                f"prepared CPU kernel expects {len(self.runtime_indices)} runtime "
                f"arguments, got {len(runtime_args)}"
            )
        args = self.argument_template.copy()
        for index, value in zip(self.runtime_indices, runtime_args):
            args[index] = value
        from triton import knobs

        if (
            knobs.runtime.launch_enter_hook.calls
            or knobs.runtime.launch_exit_hook.calls
            or self.jit_kernel.pre_run_hooks
        ):
            return self._standard_launch(args)

        for value, guard in zip(runtime_args, self.runtime_specializations):
            backend, is_const, specialize, align, expected = guard
            if (
                native_specialize_impl(backend, value, is_const, specialize, align)
                != expected
            ):
                return self._standard_launch(args)

        not_present = object()
        for (name, _), (
            expected,
            globals_dict,
        ) in self.jit_kernel.used_global_vals.items():
            if globals_dict.get(name, not_present) != expected:
                return self._standard_launch(args)
        compiled = self.compiled
        self.launcher(
            self.grid[0],
            self.grid[1],
            self.grid[2],
            None,
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *args,
        )


def prepare_cpu_kernel(jit_kernel, grid, *sample_args, **compile_kwargs):
    """Compile one specialization and return its low-overhead CPU runner.

    The returned callable accepts only non-constexpr kernel arguments, in the
    original signature order. It intentionally omits launch hooks and dynamic
    grids; callers needing those features should use the standard Triton API.
    """

    return PreparedCPUKernel(jit_kernel, grid, *sample_args, **compile_kwargs)


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
            kernel_obj,  # non-null module/lifecycle sentinel
            kernel_obj,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )

    @staticmethod
    def unload_module(module):
        # The object code is owned by CompiledKernel and the imported launcher
        # module is retained by compile_module's cache, so there is no native
        # CPU module handle to release here.
        return None


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
