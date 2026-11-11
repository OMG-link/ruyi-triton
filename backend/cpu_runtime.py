"""Build, load and configure the CPU runtime (backend/runtime/triton_cpu_rt.*).

The runtime is a shared library built on first use on the machine that runs
the kernels, for one board configuration's scratchpad driver, and cached like
any other Triton artifact. It is loaded once per process with RTLD_GLOBAL and
every kernel launcher links against it, so a process has a single worker
pool. The first launch configures it (configure()); see board.py for the
configurations.
"""

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from triton.runtime.cache import get_cache_manager

from . import board
from .paths import _get_llvm_bin_path

BACKEND_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BACKEND_DIR / "runtime"
INCLUDE_DIR = BACKEND_DIR / "include"
_RUNTIME_FILES = (RUNTIME_DIR / "triton_cpu_rt.h", RUNTIME_DIR / "triton_cpu_rt.cpp")

# TCRT_* of triton_cpu_rt.h.
ABI_VERSION = 2
OK, ERROR_INVALID, ERROR_RECONFIGURE, ERROR_SPM = 0, -1, -2, -3


class _Config(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_int), ("ncpus", ctypes.c_int),
                ("cpus", ctypes.POINTER(ctypes.c_int)), ("ndomains", ctypes.c_int),
                ("domain_sizes", ctypes.POINTER(ctypes.c_int)),
                ("enter_ai_cores", ctypes.c_int), ("spm_bytes_per_cpu", ctypes.c_int64),
                ("spm_first_cpu", ctypes.c_int)]


class SpmStatus(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("busy_block", ctypes.c_int),
                ("sys_errno", ctypes.c_int), ("bytes", ctypes.c_int64),
                ("slot_bytes", ctypes.c_int64), ("slot_cpu", ctypes.c_int)]


def _sanitizer_flags(sanitizer_type):
    if sanitizer_type == "asan":
        return ["-g", "-fsanitize=address"]
    if sanitizer_type == "tsan":
        return ["-g", "-fsanitize=thread"]
    return []


def library_path(config, sanitizer_type="") -> str:
    """Path of the runtime library for `config`, building it if needed.

    The file name embeds the build key and is also the soname: the dynamic
    loader identifies libraries by soname, so two different builds must never
    share one.
    """
    driver = config.spm_driver
    cxx = _get_llvm_bin_path("clang++") if sanitizer_type else "g++"
    flags = ["-O2", "-std=c++17", "-fPIC", "-shared", "-pthread",
             f"-I{INCLUDE_DIR}", f"-I{RUNTIME_DIR}",
             f'-DTCRT_SPM_DRIVER_H="spm/{driver.header}"',
             *_sanitizer_flags(sanitizer_type)]
    sources = b"".join(p.read_bytes() for p in _RUNTIME_FILES)
    sources += b"".join(p.read_bytes() for p in sorted((INCLUDE_DIR / "spm").glob("*.h")))
    key = hashlib.sha256(sources + " ".join([cxx, *flags, *driver.link_args]).encode()).hexdigest()
    name = f"libtriton_cpu_rt.{key[:16]}.so"
    cache = get_cache_manager(key)
    path = cache.get_file(name)
    if path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, name)
            subprocess.check_call([cxx, *flags, str(RUNTIME_DIR / "triton_cpu_rt.cpp"),
                                   f"-Wl,-soname,{name}", *driver.link_args, "-o", out])
            path = cache.put(Path(out).read_bytes(), name, binary=True)
    return path


_lock = threading.Lock()
_lib = None
_lib_path = None
_active = None  # the ResolvedConfig the runtime was configured with


def _load(path):
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    lib.tcrt_configure.argtypes = [ctypes.POINTER(_Config)]
    lib.tcrt_spm_status.argtypes = [ctypes.POINTER(SpmStatus)]
    lib.tcrt_num_slots.argtypes = [ctypes.c_int]
    lib.tcrt_slot_cpus.argtypes = [ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    return lib


def configure(config, sanitizer_type=""):
    """Loads the runtime and configures it with `config`, once per process.

    Returns the runtime library path, which launchers link against. A kernel
    whose configuration differs from the process's raises RuntimeError.
    """
    global _lib, _lib_path, _active
    with _lock:
        if _active is not None:
            if _active.config != config:
                raise RuntimeError(
                    f"this process runs kernels with the board configuration "
                    f"'{_active.config.name}'; a kernel compiled for "
                    f"'{config.name}' cannot run in it")
            return _lib_path
        resolved = board.resolve(config)
        path = library_path(config, sanitizer_type)
        lib = _load(path)
        cpus = [c for d in resolved.domains for c in d]
        cpu_array = (ctypes.c_int * len(cpus))(*cpus)
        size_array = (ctypes.c_int * len(resolved.domains))(*map(len, resolved.domains))
        desc = _Config(ABI_VERSION, len(cpus), cpu_array, len(resolved.domains), size_array,
                       int(resolved.enter_ai_cores), config.spm_bytes_per_cpu,
                       config.spm_first_cpu)
        rc = lib.tcrt_configure(ctypes.byref(desc))
        if rc != OK:
            raise RuntimeError(f"tcrt_configure failed with {rc} for {resolved}")
        if lib.tcrt_num_workers() == 0:
            raise RuntimeError(f"the CPU runtime could not start any worker on "
                               f"{resolved.domains}")
        _lib, _lib_path, _active = lib, path, resolved
        return path


def active():
    """The ResolvedConfig of the process, or None before the first launch."""
    return _active


def spm_status() -> SpmStatus:
    status = SpmStatus()
    _lib.tcrt_spm_status(ctypes.byref(status))
    return status


def num_slots(warps) -> int:
    return _lib.tcrt_num_slots(warps)


def slot_cpus(warps, slot):
    cpus = (ctypes.c_int * 64)()
    n = _lib.tcrt_slot_cpus(warps, slot, cpus, 64)
    return list(cpus[:n]) if n >= 0 else None


def num_workers() -> int:
    return _lib.tcrt_num_workers()
