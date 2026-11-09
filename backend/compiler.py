from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import platform
import re
import shutil
import subprocess
import functools
import triton
from pathlib import Path


def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _get_buddy_opt_path() -> str:
    """Path to buddy-opt (from buddy-mlir build)."""
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path == "":
        raise Exception("BUDDY_MLIR_BINARY_DIR is not set.")
    return os.path.join(path, "buddy-opt")


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    for f in files:
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")

    return sanitizer_type


def _is_qgemm_opt_enabled():
    """Check whether QGEMM optimization passes are enabled.

    Reads the TRITON_ENABLE_QGEMM_OPT environment variable.
    This switch controls all QGEMM-related optimization passes
    (currently: decompose-high-level-ops; more may be added in the future).

    Development-phase default: ENABLED (True when env var is unset).
    For production release, change the default to False.
    """
    val = os.getenv("TRITON_ENABLE_QGEMM_OPT")
    if val is None:
        # Development-phase default: enabled.
        # TODO: Change to False for production release.
        return True
    return val.lower() in ("1", "true")


# ----------------------- Threads per program instance -----------------------
# `num_warps` is the number of threads of a program instance:
#   * num_warps > 0: used as is.
#   * num_warps = 0: inferred by _infer_num_warps().
# The resolved value is stored in metadata["num_warps"].  Loops marked with
# `triton.language.extra.cpu.parallel` are distributed over that many threads
# (buddy-mlir --cpar-distribute-loops / --cpar-outline-to-runtime).  The
# runtime reserves cores for the threads only if the kernel actually contains
# a parallel region (metadata["cpu_threads"], see pack_metadata).

# Symbol of the fork/join entry point of the CPU runtime (backend/runtime).
_CPAR_RUNTIME_SYMBOL = "__cpar_parallel"


def _l2_domain_size(cpu: int, available) -> int:
    """Number of available CPUs sharing the L2 cache (or cluster) of `cpu`."""
    base = f"/sys/devices/system/cpu/cpu{cpu}"

    def parse(text):
        cpus = set()
        for item in text.strip().split(","):
            if not item:
                continue
            first, _, last = item.partition("-")
            cpus.update(range(int(first), int(last or first) + 1))
        return cpus

    try:
        for index in sorted(Path(base, "cache").glob("index*")):
            if (index / "level").read_text().strip() == "2":
                return len(parse((index / "shared_cpu_list").read_text()) & available)
        return len(parse(Path(base, "topology", "cluster_cpus_list").read_text()) & available)
    except (OSError, ValueError):
        return 0


@functools.lru_cache()
def _infer_num_warps() -> int:
    """Placeholder inference of the threads per program instance (num_warps=0).

    Uses the number of CPUs sharing the L2 cache of the first available CPU,
    i.e. one full cluster per program instance (4 on the SpacemiT K1), so
    that the threads of an instance share an L2.  Falls back to 1.

    TODO: replace by an inference based on the kernel (work per instance,
    grid size, memory footprint).
    """
    try:
        available = set(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return 1
    if not available:
        return 1
    return max(1, _l2_domain_size(min(available), available))


def _resolve_num_warps(num_warps) -> int:
    num_warps = int(num_warps or 0)
    if num_warps < 0:
        raise ValueError(f"num_warps must be >= 0, got {num_warps}")
    return num_warps if num_warps > 0 else _infer_num_warps()


def _build_ttsharedir_passes():
    passes = []

    if _get_sanitizer_type() != "":
        print("Building with sanitizer support...")
        passes.append("--add-llvm-debug-info")

    if _is_qgemm_opt_enabled():
        passes.append("--decompose-high-level-ops")

    passes.append("--triton-to-linalg-experimental")

    return passes


def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        _dump_ir_if_needed([src_path])
        triton_shared_opt_path = _get_triton_shared_opt_path()

        subprocess_args = [
            triton_shared_opt_path,
            src_path,
            *_build_ttsharedir_passes(),
            "--mlir-print-debuginfo",
            "-o",
            dst_path,
        ]

        subprocess.check_call(subprocess_args)
        _dump_ir_if_needed([dst_path])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        buddy_opt_path = _get_buddy_opt_path()
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                buddy_opt_path,
                ttshared_path,
                "--convert-linalg-to-affine-loops",
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--matmul-vectorization",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--expand-strided-metadata",
                "--convert-scf-to-cf",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--convert-cf-to-llvm",
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
                "--reconcile-unrealized-casts",
                "--mlir-print-debuginfo",
                "-o",
                llmlir_path,
            ]
        )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        _dump_ir_if_needed([ttshared_path, llmlir_path, llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     llir_path = os.path.join(tmpdir, "ll.ir")
    #     lliropt_path = os.path.join(tmpdir, "llopt.ir")
    #     Path(llir_path).write_text(llir)

    #     opt_path = _get_llvm_bin_path("opt")
    #     subprocess.check_call([opt_path, llir_path, "-O2", "-S", "-o", lliropt_path])

    #     _dump_ir_if_needed([lliropt_path])
    #     return Path(lliropt_path).read_text()
    return llir

def _ttsharedir_to_vectorir(ttsharedir: str, num_threads: int):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        vector_path = os.path.join(tmpdir, "vector.mlir")
        Path(ttshared_path).write_text(ttsharedir)
        buddy_opt_path = _get_buddy_opt_path()

        passes = []
        if _is_qgemm_opt_enabled():
            passes.append("--qgemm-simplify-dim-transforms")
            passes.append("--qgemm-recognize")
            passes.append("--qgemm-tiling=tile-m=32 tile-n=32")
            passes.append("--qgemm-tiling=tile-m=4 tile-n=4")
        passes.extend([
            # Note: eliminate-empty-tensors fails when there are multiple func.return ops
            # in a single kernel which are the results of early returns.
            # See python/examples/test_early_return.py for examples.
            # We disable this pass for now since performance on CPU isn't the main
            # focus at the moment.
            # "--eliminate-empty-tensors",
            # Loops marked by cpu.parallel are distributed over num_threads
            # threads.  This has to happen on tensors: the distributed loop is
            # an scf.forall, which bufferization treats as a parallel region,
            # so every thread gets private buffers for what it writes.
            f"--cpar-distribute-loops=num-threads={num_threads}",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            # The foralls become cpar.parallel regions (outlined in
            # _vectorir_to_llir).
            "--cpar-forall-to-parallel",
            "--lower-linalg-to-vir",
            "--lower-vir-to-vector=vector-width=4",
            "--convert-linalg-to-affine-loops",
        ])

        subprocess.check_call(
            [
                buddy_opt_path,
                ttshared_path,
                *passes,
                "--mlir-print-debuginfo",
                "-o",
                vector_path,
            ]
        )
        _dump_ir_if_needed([vector_path])
        return Path(vector_path).read_text()

def _vectorir_to_llir(vectorir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_path = os.path.join(tmpdir, "vector.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(vector_path).write_text(vectorir)
        buddy_opt_path = _get_buddy_opt_path()
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                buddy_opt_path,
                vector_path,
                # "--convert-linalg-to-affine-loops",
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                # "--empty-tensor-to-alloc-tensor",
                # "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                # "--matmul-vectorization",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--expand-strided-metadata",
                "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                "--convert-vector-to-scf",
                "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                "--convert-ub-to-llvm",
                "--convert-scf-to-cf",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--convert-cf-to-llvm",
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
                "--reconcile-unrealized-casts",
                # Outline cpar.parallel regions into functions called through
                # the runtime's __cpar_parallel (needs pure LLVM dialect).
                "--cpar-outline-to-runtime",
                "--mlir-print-debuginfo",
                "-o",
                llmlir_path,
            ]
        )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        _dump_ir_if_needed([llmlir_path, llir_path])
        return Path(llir_path).read_text()


def _record_cpu_threads(llir: str, metadata):
    """Threads the runtime reserves per program instance.

    num_warps if the kernel forks parallel regions, 1 otherwise: a kernel
    without parallel region would leave the other cores of its slot idle.
    """
    forks = re.search(rf"\bcall void @{_CPAR_RUNTIME_SYMBOL}\(", llir) is not None
    metadata["cpu_threads"] = metadata["num_warps"] if forks else 1
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)

        sanitizer_type = _get_sanitizer_type()

        if sanitizer_type != "":
            # using a sanitizer
            # invoke pass to append sanitizer attributes
            instrumented_src_path = os.path.join(tmpdir, "kernel-instrumented.ll")

            opt_path = _get_llvm_bin_path("opt")
            top_level_triton_path = os.path.dirname(triton.__file__)
            sanitizer_attributes_pass_path = str(
                next(
                    Path(top_level_triton_path).rglob("libSanitizerAttributes.so"), None
                )
            )

            if not sanitizer_attributes_pass_path:
                raise Exception("libSanitizerAttributes.so does not exist.")

            subprocess.check_call(
                [
                    opt_path,
                    "-load-pass-plugin",
                    sanitizer_attributes_pass_path,
                    "-passes=sanitizer-attributes",
                    f"-sanitizer-type={sanitizer_type}",
                    "-S",
                    src_path,
                    "-o",
                    instrumented_src_path,
                ]
            )

            # compile to object file
            clang_path = _get_llvm_bin_path("clang++")

            subprocess_args = [clang_path, "-c", instrumented_src_path, "-o", dst_path]

            if sanitizer_type == "asan":
                subprocess_args.extend(
                    ["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"]
                )
            elif sanitizer_type == "tsan":
                subprocess_args.extend(["-g", "-fsanitize=thread"])

            subprocess.check_call(subprocess_args)
        else:
            llc_path = _get_llvm_bin_path("llc")
            llc_args = [
                llc_path,
                src_path,
                "-filetype=obj",
                "-relocation-model=pic",
                "-o",
                dst_path,
            ]
            # On RISC-V Linux, the system ABI is lp64d (hardware double-precision float).
            # Without explicitly enabling +f,+d, llc defaults to soft-float, causing
            # "can't link soft-float modules with double-float modules" linker errors.
            if platform.machine() == "riscv64":
                llc_args.extend(["-mattr=+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb"])
            subprocess.check_call(llc_args)

        return Path(dst_path).read_bytes()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = "obj"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        )
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # The launch function in driver.py uses the name (index 6) and the
        # threads per program instance (index 7).  The other entries are kept
        # for consistency with the other backends.
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
            getattr(metadata, "cpu_threads", 1),
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def _num_threads(metadata):
        # Resolved once per compilation; also covers compilations that start
        # from an IR file and skip make_ttir.
        metadata["num_warps"] = _resolve_num_warps(metadata.get("num_warps", 0))
        return metadata["num_warps"]

    def add_stages(self, stages, options, language):
        def ttir(src, metadata):
            self._num_threads(metadata)
            return self.make_ttir(src, metadata, options)

        stages["ttir"] = ttir
        stages["ttsharedir"] = lambda src, metadata: _ttir_to_ttsharedir(src)
        stages["ttsharedir.opt"] = lambda src, metadata: _optimize_ttsharedir(src)
        stages["vectorir"] = lambda src, metadata: _ttsharedir_to_vectorir(
            src, self._num_threads(metadata)
        )
        stages["llir"] = lambda src, metadata: _record_cpu_threads(
            _vectorir_to_llir(src), metadata
        )
        stages["llir.opt"] = lambda src, metadata: _optimize_llir(src)
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
