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

from .paths import _get_buddy_opt_path, _get_llvm_bin_path, _get_triton_shared_opt_path
from .riscv import DEFAULT_LLC_FEATURES


def _get_buddy_translate_path() -> str:
    """Path to buddy-translate (from buddy-mlir build)."""
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path == "":
        raise Exception("BUDDY_MLIR_BINARY_DIR is not set.")
    return os.path.join(path, "buddy-translate")


def _get_buddy_llc_path() -> str:
    """Path to buddy-llc (from buddy-mlir build)."""
    path = os.getenv("BUDDY_MLIR_BINARY_DIR", "")
    if path == "":
        raise Exception("BUDDY_MLIR_BINARY_DIR is not set.")
    return os.path.join(path, "buddy-llc")


def _use_ime_pipeline() -> bool:
    """Return True when the IME (RISC-V matrix-extension) lowering path should be used.

    Set the environment variable TRITON_RISCV_USE_IME=1 to activate.  The IME
    pipeline lowers linalg.matmul to ime.vfmadot (fp16) / ime.vmadot (int)
    instructions via buddy-mlir and cross-compiles to a RISC-V ELF object.
    """
    return os.getenv("TRITON_RISCV_USE_IME", "") == "1"


def _get_openmp_num_threads() -> int:
    value = os.getenv("TRITON_RISCV_OPENMP_THREADS", "0")
    if value == "":
        return 0
    try:
        threads = int(value)
    except ValueError as exc:
        raise ValueError(
            "TRITON_RISCV_OPENMP_THREADS must be an integer thread count"
        ) from exc
    if threads < 0:
        raise ValueError("TRITON_RISCV_OPENMP_THREADS must be non-negative")
    return threads


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
            "--triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu",
            "--mlir-print-debuginfo",
            "-o",
            dst_path,
        ]

        if _get_sanitizer_type() != "":
            print("Building with sanitizer support...")

            # has to run before the other passes as operates on the tt dialect
            subprocess_args.insert(2, "--add-llvm-debug-info")

        subprocess.check_call(subprocess_args)
        _dump_ir_if_needed([dst_path])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str, options=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        ime_pre_llvm_path = os.path.join(tmpdir, "ime-pre-llvm.mlir")
        atomic_cas_path = os.path.join(tmpdir, "ttshared-atomic-cas.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        buddy_opt_path = _get_buddy_opt_path()

        if _use_ime_pipeline():
            # ---------------------------------------------------------------
            # IME path: Triton-Shared MLIR → buddy-mlir IME dialect → LLVM IR
            # Lowers linalg.matmul (f16) to ime.vfmadot via buddy-mlir passes,
            # then cross-compiles to a RISC-V ELF object with +xsmtvdot (K1 IME).
            # ---------------------------------------------------------------
            ime_lowering_passes = [
                *_tile_to_vector_args(options, ime=True),
                # Bufferize tensor ops before IME lowering
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                # One-shot bufferization creates heap-backed temporary
                # memrefs. Insert and lower their deallocations while the
                # memref-level ownership information is still available.
                "--buffer-deallocation-pipeline",
                # Lower linalg.matmul (memref) → ime.vfmadot / ime.vmadot
                "--lower-linalg-to-ime",
                # Lower IME dialect → vector / loop ops
                "--lower-ime=target="
                + ("a100" if _ime_extension(options) == "xsmtvdotii" else "generic"),
                # Lower any remaining linalg / affine / SCF ops
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--expand-strided-metadata",
                "--convert-scf-to-cf",
            ]
            llvm_lowering_passes = [
                # Lower to LLVM dialect
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-ub-to-llvm",
                "--convert-index-to-llvm",
                "--convert-func-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                # Lowering memrefs creates more affine.apply ops; run again.
                "--lower-affine",
                "--convert-arith-to-llvm",
                "--reconcile-unrealized-casts",
            ]

            buddy_input_path = ttshared_path
            buddy_passes = [*ime_lowering_passes, *llvm_lowering_passes]
            if "__triton_shared_atomic_cas_" in ttsharedir:
                # LLVM cmpxchg is not bufferizable, so lower helper calls only
                # after Buddy has completed its tensor-to-memref work.
                subprocess.check_call(
                    [
                        buddy_opt_path,
                        ttshared_path,
                        *_set_llvm_target_args(options),
                        *_remarks_args(),
                        *ime_lowering_passes,
                        "--mlir-print-debuginfo",
                        "-o",
                        ime_pre_llvm_path,
                    ]
                )
                subprocess.check_call(
                    [
                        _get_triton_shared_opt_path(),
                        ime_pre_llvm_path,
                        "--lower-atomic-cas-to-llvm",
                        "--canonicalize",
                        "--mlir-print-debuginfo",
                        "-o",
                        atomic_cas_path,
                    ]
                )
                buddy_input_path = atomic_cas_path
                buddy_passes = llvm_lowering_passes

            subprocess.check_call(
                [
                    buddy_opt_path,
                    buddy_input_path,
                    *_set_llvm_target_args(options),
                    *_remarks_args(),
                    *buddy_passes,
                    "--mlir-print-debuginfo",
                    "-o",
                    llmlir_path,
                ]
            )
            # LLVM-MLIR → LLVM-IR via buddy-translate (handles buddyext dialect)
            buddy_translate_path = _get_buddy_translate_path()
            subprocess.check_call(
                [
                    buddy_translate_path,
                    "--buddy-to-llvmir",
                    llmlir_path,
                    "-o",
                    llir_path,
                ]
            )
        else:
            # ---------------------------------------------------------------
            # Standard path: buddy-mlir vectorisation → host LLVM IR
            # ---------------------------------------------------------------
            subprocess.check_call(
                [
                    buddy_opt_path,
                    ttshared_path,
                    *_set_llvm_target_args(options),
                    *_remarks_args(),
                    # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                    # in a single kernel which are the results of early returns.
                    # See python/examples/test_early_return.py for examples.
                    # We disable this pass for now since performance on CPU isn't the main
                    # focus at the moment.
                    # "--eliminate-empty-tensors",
                    # Fuse tensor-level elementwise chains before bufferization
                    # turns every intermediate and scalar broadcast into a
                    # separate temporary buffer.
                    "--linalg-fuse-elementwise-ops",
                    *_tile_to_vector_args(options),
                    "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                    "--buffer-deallocation-pipeline",
                    "--eliminate-memref-copy",
                    # Triton programs commonly materialize small, statically
                    # sized tiles for loads and scalar broadcasts.  Paying for
                    # several heap allocations on every grid invocation
                    # dominates elementwise CPU kernels, so keep bounded tile
                    # temporaries in the launcher thread's stack frame.
                    "--promote-buffers-to-stack=max-alloc-size-in-bytes=65536",
                    "--matmul-vectorization",
                    "--convert-linalg-to-affine-loops",
                    "--lower-affine",
                    "--convert-linalg-to-loops",
                    "--expand-strided-metadata",
                    "--convert-scf-to-cf",
                    "--convert-arith-to-llvm",
                    "--convert-math-to-llvm",
                    "--convert-complex-to-llvm",
                    "--convert-vector-to-llvm",
                    "--convert-ub-to-llvm",
                    "--convert-index-to-llvm",
                    "--memref-expand",
                    "--finalize-memref-to-llvm",
                    "--convert-cf-to-llvm",
                    "--convert-func-to-llvm",
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


_FADD_INSTRUCTION = re.compile(
    r"^(?P<prefix>\s*%[-\w.$\"]+\s*=\s*fadd)[ \t]+"
    r"(?P<flags>(?:(?:fast|reassoc|nnan|ninf|nsz|arcp|contract|afn)[ \t]+)*)"
    r"(?P<operands>\S.*)$",
    re.MULTILINE,
)


def _enable_fadd_reassociation(llir: str) -> str:
    """Add reassoc only to complete fadd instructions that do not have it."""

    def add_flag(match):
        flags = match.group("flags").split()
        if "fast" in flags or "reassoc" in flags:
            return match.group(0)
        flags.insert(0, "reassoc")
        return f"{match.group('prefix')} {' '.join(flags)} {match.group('operands')}"

    return _FADD_INSTRUCTION.sub(add_flag, llir)


_HOST_TRIPLES = {
    "x86_64": "x86_64-unknown-linux-gnu",
    "AMD64": "x86_64-unknown-linux-gnu",
    "riscv64": "riscv64-unknown-linux-gnu",
}


@functools.lru_cache()
def _get_host_cpu() -> str:
    """LLVM's name for the host CPU, e.g. spacemit-x60 on the SpacemiT K1."""
    output = subprocess.check_output(
        [_get_llvm_bin_path("llc"), "--version"], text=True
    )
    match = re.search(r"Host CPU:\s*(\S+)", output)
    return match.group(1) if match else "generic"


def _host_triple():
    return _HOST_TRIPLES.get(platform.machine())


def _arch_of(triple: str) -> str:
    return triple.split("-", 1)[0]


@functools.lru_cache()
def _riscv_vlen_bounds(triple, cpu, features):
    """Query LLVM-derived bounds before applying an explicit exact VLEN."""
    output = subprocess.run(
        [_get_buddy_opt_path(),
         f"--set-llvm-target=triple={triple} chip={cpu} features={features}",
         "--test-target-info", "-o", os.devnull],
        input="module {}\n", capture_output=True, text=True, check=True,
    ).stdout
    facts = dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)
    return int(facts["min-vlen"]), int(facts["max-vector-bits"])


def _read_host_vlen():
    from triton._C.libtriton import triton_shared
    return triton_shared.read_host_vlen()


def _validate_vlen(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("riscv_vlen must be zero or a power of two >= 128")
    try:
        bits = int(value)
    except ValueError as exc:
        raise ValueError("riscv_vlen must be zero or a power of two >= 128") from exc
    if bits != 0 and (bits < 128 or bits & (bits - 1)):
        raise ValueError("riscv_vlen must be zero or a power of two >= 128")
    return bits


def _resolve_vlen(args):
    explicit = args.get("riscv_vlen")
    if explicit is None:
        value = os.getenv("TRITON_RISCV_VLEN", "").strip()
        explicit = value if value else None
    if not args["target_triple"].startswith("riscv"):
        if explicit is not None and _validate_vlen(explicit):
            raise ValueError("riscv_vlen is only supported for RISC-V targets")
        args["riscv_vlen"] = 0
        return
    lower, upper = _riscv_vlen_bounds(
        args["target_triple"], args["target_cpu"], args["target_features"])
    if explicit is not None:
        bits = _validate_vlen(explicit)
    elif lower != 0 and lower == upper:
        bits = lower
    else:
        bits = _validate_vlen(_read_host_vlen())
    if bits and lower == 0:
        raise ValueError("nonzero riscv_vlen requires a target with RVV support")
    if bits and not lower <= bits <= upper:
        raise ValueError(
            f"riscv_vlen={bits} conflicts with target VLEN range [{lower}, {upper}]")
    args["riscv_vlen"] = bits


def _resolve_target(args: dict) -> None:
    """Resolve the target and exact VLEN into `args`.

    This is the only place the compilation target is decided; every tool
    (buddy-opt, opt, llc) receives exactly this target. Each field comes from
    the caller's option, then its environment variable, then the host:

      triple    TRITON_RISCV_TARGET_TRIPLE   the host triple
      cpu       TRITON_RISCV_MCPU            the host CPU LLVM detects, when
                                             the target is the host's arch
      features  TRITON_RISCV_LLC_FEATURES    the RISC-V Linux baseline for
                                             RISC-V, none for x86
      vlen      TRITON_RISCV_VLEN            equal ISA bounds, then host VLEN,
                                             then zero (unknown)

    Compiling for an arch other than the host's is the only cross
    compilation; nothing else changes how the target is chosen. On a host
    LLVM has no triple for and with no triple given, the target stays unset.
    """
    triple = (
        args.get("target_triple")
        or os.getenv("TRITON_RISCV_TARGET_TRIPLE")
        or _host_triple()
    )
    if not triple:
        args["riscv_vlen"] = 0
        return
    args["target_triple"] = triple
    host = _host_triple()
    native = host is not None and _arch_of(host) == _arch_of(triple)
    if not args.get("target_cpu"):
        args["target_cpu"] = os.getenv("TRITON_RISCV_MCPU") or (
            _get_host_cpu() if native else "generic"
        )
    if args.get("target_features") is None:
        default = DEFAULT_LLC_FEATURES if triple.startswith("riscv") else ""
        args["target_features"] = os.getenv("TRITON_RISCV_LLC_FEATURES", default)
    _resolve_vlen(args)


def _target(options):
    """(triple, cpu, features) of the compilation target, or None if unset."""
    triple = getattr(options, "target_triple", None)
    if not triple:
        return None
    return (
        triple,
        getattr(options, "target_cpu", None) or "generic",
        getattr(options, "target_features", None) or "",
    )


def _is_riscv(options) -> bool:
    target = _target(options)
    return target is not None and target[0].startswith("riscv")


def _llvm_target_args(options) -> list:
    """-mtriple/-mcpu/-mattr for opt and llc, plus the RISC-V VLEN bounds."""
    target = _target(options)
    if target is None:
        return []
    triple, cpu, features = target
    args = [f"-mtriple={triple}"]
    if cpu != "generic":
        args.append(f"-mcpu={cpu}")
    if features:
        args.append(f"-mattr={features}")
    if triple.startswith("riscv"):
        args += _riscv_vector_bits_args(options)
    return args


@functools.lru_cache()
def _target_has_extension(triple: str, cpu: str, features: str, ext: str) -> bool:
    """Whether buddy's TargetInfo, the facts the passes read, lists `ext`."""
    output = subprocess.run(
        [
            _get_buddy_opt_path(),
            f"--set-llvm-target=triple={triple} chip={cpu} features={features}",
            "--test-target-info",
            "-o",
            os.devnull,
        ],
        input="module {}\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    facts = " ".join(
        line for line in output.splitlines() if line.startswith(("isa:", "features:"))
    )
    return re.search(rf"(^|[_,: ]){re.escape(ext)}(\d|,|$|_| )", facts) is not None


def _ime_extension(options):
    """The SpacemiT dot extension of the target: "xsmtvdotii", "xsmtvdot" or None.

    XSMTVDotII (the A100 cores of the K3) is a superset of the dot forms the
    pipeline uses from XSMTVDot (the K1); the tile shapes follow from the
    extension and the exact VLEN, which the buddy passes read themselves.
    """
    target = _target(options)
    if target is None or not target[0].startswith("riscv"):
        return None
    for ext in ("xsmtvdotii", "xsmtvdot"):
        if _target_has_extension(*target, ext):
            return ext
    return None


def _check_ime_target(options) -> None:
    """The IME pipeline lowers matmuls to the SpacemiT dot extensions."""
    if _ime_extension(options) is None:
        raise ValueError(
            "TRITON_RISCV_USE_IME=1 needs a RISC-V target with XSMTVDot or "
            f"XSMTVDotII, but the target is {_target(options)}. Run on an IME "
            "core (a SpacemiT K1, or an A100 core of the K3), or set "
            "TRITON_RISCV_MCPU=spacemit-x60 / spacemit-a100."
        )


def _set_llvm_target_args(options) -> list:
    """buddy-opt arguments that attach the target as #llvm.target."""
    target = _target(options)
    if target is None:
        return []
    triple, cpu, features = target
    return [f"--set-llvm-target=triple={triple} chip={cpu} features={features} "
            f"vlen={options.riscv_vlen or 0}"]


def _tile_to_vector_args(options, ime=False) -> list:
    """Keep register-sized tensor tiles in vector registers.

    `--convert-tile-tensor-to-vector` turns tile tensors into vectors of the
    same shape, sized from the #llvm.target attribute, so it runs whenever a
    target is attached (RISC-V and x86). `--flatten-vector-tiles` then maps
    the n-D vectors and their transfers onto 1-D registers. The IME pipeline
    also converts matmul(A, transpose(B)) of the target's IME tile shapes
    into vector.contract, including those whose K is below the tile's when
    their operands are loads a masked full-K tile can be read from;
    `--promote-ime-tile-k` widens those to the tile's K, and
    `--lower-vector-contract-to-ime-tile` turns all of them into vmadot or
    vfwmadot tile dots that its `--lower-ime` lowers. `--lower-cpu-sched-barrier` lowers `cpu.sched_barrier` to inline asm that
    holds its values in registers.
    TRITON_RISCV_REMARKS=<regex> prints the optimization remarks whose
    category matches, e.g. `tile-to-vector`; remarks are silent otherwise.
    """
    if _target(options) is None:
        return []
    if ime:
        return [
            "--convert-tile-tensor-to-vector=allow-ime-tile-contracts=true",
            "--canonicalize",
            "--cse",
            "--promote-ime-tile-k",
            "--canonicalize",
            "--lower-vector-contract-to-ime-tile",
            "--flatten-vector-tiles",
            "--lower-cpu-sched-barrier",
            "--canonicalize",
            "--cse",
        ]
    return [
        "--convert-tile-tensor-to-vector",
        "--canonicalize",
        "--cse",
        "--flatten-vector-tiles",
        "--lower-cpu-sched-barrier",
        "--canonicalize",
        "--cse",
    ]


def _remarks_args() -> list:
    regex = os.getenv("TRITON_RISCV_REMARKS", "")
    return [f"--remarks-filter={regex}"] if regex else []


def _riscv_vector_bits_args(options=None) -> list:
    """Use only the VLEN resolved into the compilation options."""
    vlen = getattr(options, "riscv_vlen", 0)
    if vlen:
        return [f"-riscv-v-vector-bits-min={vlen}",
                f"-riscv-v-vector-bits-max={vlen}"]
    target = _target(options)
    named = target is not None and target[1] != "generic"
    return [] if named else ["-riscv-v-vector-bits-min=128"]


def _optimize_llir(llir: str, options=None):
    # llc's -O3 controls code-generation optimizations but does not run the
    # target-aware LLVM middle-end pipeline. In particular, scalar linalg
    # loops emitted for elementwise Triton tiles remain scalar unless opt sees
    # the target's vector width. Run the O3 pipeline for the target the
    # options describe, native or not.
    if _target(options) is None:
        return llir

    if getattr(options, "allow_fp_reassoc", False):
        # Per-kernel opt-in for reduction-heavy code whose numerical contract
        # permits reassociation. LLVM otherwise preserves scalar accumulation
        # order and cannot vectorize these loops.
        llir = _enable_fadd_reassociation(llir)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel-opt.ll")
        Path(src_path).write_text(llir)
        command = [_get_llvm_bin_path("opt"), "-O3", *_llvm_target_args(options)]
        if _target(options)[0].startswith("x86_64"):
            # glibc's x86 vector math ABI; the launcher links libmvec.
            command.append("-vector-library=LIBMVEC")
        command.extend(["-S", src_path, "-o", dst_path])
        subprocess.check_call(command)
        _dump_ir_if_needed([dst_path])
        return Path(dst_path).read_text()


def _ttsharedir_to_vectorir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        vector_path = os.path.join(tmpdir, "vector.mlir")
        Path(ttshared_path).write_text(ttsharedir)
        buddy_opt_path = _get_buddy_opt_path()
        subprocess.check_call(
            [
                buddy_opt_path,
                ttshared_path,
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--buffer-deallocation-pipeline",
                "--lower-linalg-to-vir",
                "--lower-vir-to-vector=vector-width=16",
                "--cse",
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
        transformed_vector_path = os.path.join(tmpdir, "vector-transformed.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(vector_path).write_text(vectorir)
        transform_passes = []
        if "__triton_shared_atomic_cas_" in vectorir:
            transform_passes.append("--lower-atomic-cas-to-llvm")
        transform_passes.append("--expand-float8-conversions")
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess.check_call(
            [
                triton_shared_opt_path,
                vector_path,
                *transform_passes,
                "--canonicalize",
                "--mlir-print-debuginfo",
                "-o",
                transformed_vector_path,
            ]
        )
        _dump_ir_if_needed([transformed_vector_path])
        vector_path = transformed_vector_path
        buddy_opt_path = _get_buddy_opt_path()
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                buddy_opt_path,
                vector_path,
                "--convert-linalg-to-affine-loops",
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                # "--empty-tensor-to-alloc-tensor",
                # "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                # "--matmul-vectorization",
                "--expand-strided-metadata",
                "--lower-affine",
                "--convert-math-to-llvm",
                "--convert-math-to-libm",
                "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                "--convert-vector-to-scf",
                "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
                "--convert-ub-to-llvm",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
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
        _dump_ir_if_needed([llmlir_path, llir_path])
        return Path(llir_path).read_text()


_LLVM_SYMBOL = r'(?:"([^"]+)"|([A-Za-z$._][A-Za-z0-9$._-]*))'
_LLVM_HELPER_SYMBOLS = {"dealloc_helper"}


def _find_kernel_name(llir: str) -> str:
    """Return the single externally callable function defined in LLVM IR.

    Buffer deallocation may add helper definitions such as ``dealloc_helper``
    to the module.  Helpers are called by the kernel, whereas the kernel is the
    sole definition that is not called from within the module.
    """
    definitions = {
        quoted or unquoted
        for quoted, unquoted in re.findall(
            rf"^define\b[^@\n]*@{_LLVM_SYMBOL}\s*\(", llir, re.MULTILINE
        )
    }
    callees = {
        quoted or unquoted
        for quoted, unquoted in re.findall(
            rf"\b(?:call|invoke)\b[^@\n]*@{_LLVM_SYMBOL}\s*\(", llir
        )
    }
    # LLVM O3 can prove a generated deallocation helper unnecessary and erase
    # all calls without deleting the externally visible helper definition.
    # Such runtime helpers are never Triton kernel entry points.
    candidates = sorted(definitions - callees - _LLVM_HELPER_SYMBOLS)
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one externally callable kernel definition, "
            f"found {candidates}; all definitions: {sorted(definitions)}"
        )
    return candidates[0]


def _llir_to_bin(llir: str, metadata, options=None):
    metadata["name"] = _find_kernel_name(llir)
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
                    f"-passes=sanitizer-attributes<{sanitizer_type}>",
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
            # buddy-llc and llc share one LLVM; buddy-llc also registers
            # buddy's dialect-independent codegen extensions the IME path
            # relies on.
            llc_path = (
                _get_buddy_llc_path()
                if _use_ime_pipeline()
                else _get_llvm_bin_path("llc")
            )
            subprocess.check_call(
                [
                    llc_path,
                    src_path,
                    "-filetype=obj",
                    "-O3",
                    "-relocation-model=pic",
                    *_llvm_target_args(options),
                    "-o",
                    dst_path,
                ]
            )

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
    # The RISC-V backend supports fp8_e4m3fn storage/conversion through
    # Triton's fp8e4nv IR type. Keep the list explicit so unsupported fp8
    # variants still fail at frontend type legalization.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv",)
    allow_fp8e4nv: bool = True
    max_num_imprecise_acc_default: int = 0
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    target_triple: str = None
    target_features: str = None
    target_cpu: str = None
    riscv_vlen: int = None
    openmp_num_threads: int = 0
    allow_fp_reassoc: bool = False

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
        args.setdefault("openmp_num_threads", _get_openmp_num_threads())
        _resolve_target(args)
        options = CPUOptions(**args)
        if _use_ime_pipeline():
            _check_ime_target(options)
        return options

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
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
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttir")
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _ttir_to_ttsharedir(src)
        stages["ttsharedir.opt"] = lambda src, metadata: _optimize_ttsharedir(src)
        stages["llir"] = lambda src, metadata: _ttsharedir_to_llir(src, options)
        stages["llir.opt"] = lambda src, metadata: _optimize_llir(src, options)
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
