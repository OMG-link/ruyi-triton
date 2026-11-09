"""Scratchpad memory (SPM) drivers for ``tl.dev.alloc(placement="spm")``.

The compiler needs nothing from the hardware: a kernel that uses ``spm`` gets
one hidden trailing argument, the base of a region of ``spm_size`` bytes, and
records ``spm_size`` and ``spm_align`` in its metadata. A driver provides that
region when the kernel is launched, from the scratchpad local to the core the
launch runs on, and reports every reason it cannot as an
``SpmOutOfMemoryError``.

Each driver has a C header under ``include/spm/`` that the launcher includes,
and the link flags it needs. The launcher is built for the board it runs on.

Run ``python -m triton.backends.triton_shared.spm_tool status`` to see who
holds the scratchpad, and ``... release-stale`` to clear records left by
processes that exited without releasing them (K3 only).
"""

import ctypes
import functools
import os
import sys
import warnings

KiB = 1 << 10

# Return codes of triton_riscv_spm_acquire, include/spm/spm_common.h.
_OK, _NO_DRIVER, _CPU, _CAPACITY, _ALIGN, _OCCUPIED, _DEVICE = range(7)


class SpmOutOfMemoryError(RuntimeError):
    """A launch could not get the scratchpad region its kernel needs."""


class SpmDriver:
    """How one board provides the scratchpad region of a launch."""

    name = "none"
    board = None                 # Device Tree `compatible` prefix
    header = "spm_none.h"
    link_args = ()
    budget_per_program = 0       # bytes, for the compile-time warning
    cpus = ""                    # cores that have a scratchpad

    @classmethod
    def detect(cls):
        return False

    def describe_holders(self):
        """Who holds the scratchpad, for error reports. Empty if unknown."""
        return ""


class NoSpmDriver(SpmDriver):
    pass


class SpacemitK1TcmDriver(SpmDriver):
    name = "spacemit-k1-tcm"
    board = "spacemit,k1"
    header = "spm_spacemit_k1.h"
    budget_per_program = 512 * KiB
    cpus = "0-3"

    @classmethod
    def detect(cls):
        return _compatible().startswith(cls.board)


class SpacemitK3TcmDriver(SpmDriver):
    name = "spacemit-k3-tcm"
    board = "spacemit,k3"
    header = "spm_spacemit_k3.h"
    link_args = ("-lspine_tcm",)
    budget_per_program = 768 * KiB
    cpus = "8-15"

    @classmethod
    def detect(cls):
        return _compatible().startswith(cls.board)

    def describe_holders(self):
        try:
            blocks = k3_blocks()
        except OSError as e:
            return f"TCM blocks: cannot query spine_tcm ({e})."
        return _format_k3_blocks(blocks)


_DRIVERS = (SpacemitK1TcmDriver, SpacemitK3TcmDriver)

# LLVM processor names that identify a board, for the compile-time warning.
# LLVM names both K1 core types spacemit-x60.
_TARGET_BOARDS = {"spacemit-x60": SpacemitK1TcmDriver, "spacemit-a100": SpacemitK3TcmDriver}


@functools.lru_cache()
def _compatible():
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            return f.read().split(b"\0")[0].decode()
    except OSError:
        return ""


@functools.lru_cache()
def runtime_driver():
    """The driver of the board this process runs on."""
    for cls in _DRIVERS:
        if cls.detect():
            return cls()
    return NoSpmDriver()


def driver_for_target(target_cpu):
    """The driver a compile target implies, or None. Only used for warnings."""
    board = os.getenv("TRITON_RISCV_BOARD")
    if board:
        for cls in _DRIVERS:
            if cls.name == board or (cls.board and board.startswith(cls.board)):
                return cls()
        return None
    cls = _TARGET_BOARDS.get(target_cpu)
    return cls() if cls else None


def warn_if_unlikely_to_run(kernel_name, spm_size, target_cpu):
    """Compile-time warning for a kernel whose scratchpad will not fit."""
    if not spm_size:
        return
    driver = driver_for_target(target_cpu)
    if driver is None:
        warnings.warn(
            f"kernel '{kernel_name}' uses placement=\"spm\", which no SPM driver "
            f"provides on target '{target_cpu}'; it will fail at launch",
            stacklevel=2)
    elif spm_size > driver.budget_per_program:
        warnings.warn(
            f"kernel '{kernel_name}' needs {_kib(spm_size)} of SPM, more than the "
            f"{_kib(driver.budget_per_program)} available per program instance on "
            f"{driver.name}; it will fail at launch",
            stacklevel=2)


def _kib(n):
    return f"{n} bytes" if n < KiB else f"{n / KiB:g} KiB"


def raise_acquire_error(code, kernel_name, bytes_, align, cpu, domain, busy_block, sys_errno):
    """Called by the launcher when triton_riscv_spm_acquire fails."""
    driver = runtime_driver()
    head = f"kernel '{kernel_name}' needs {_kib(bytes_)} of SPM"
    if code == _NO_DRIVER:
        reason = f"{head}, but this board has no SPM driver ({_compatible() or sys.platform})."
    elif code == _CPU:
        reason = (f"{head}, but CPU {cpu} has no scratchpad on {driver.name} "
                  f"(scratchpad cores: {driver.cpus}).")
    elif code == _CAPACITY:
        reason = (f"{head}, but a program instance on {driver.name} can use at most "
                  f"{_kib(driver.budget_per_program)}.")
    elif code == _ALIGN:
        reason = f"{head} aligned to {align} bytes, more than {driver.name} can align."
    elif code == _OCCUPIED:
        where = f"CPU {cpu}, domain {domain}"
        if busy_block >= 0:
            reason = f"{head} on {driver.name} ({where}), but block {busy_block} is occupied."
        else:
            reason = (f"{head} on {driver.name} ({where}), but the TCM is mapped by "
                      f"another process.")
    elif code == _DEVICE:
        reason = (f"{head}, but the {driver.name} scratchpad cannot be opened or mapped: "
                  f"{os.strerror(sys_errno) if sys_errno else 'unavailable'}.")
    else:
        reason = f"{head}, but the driver returned an unknown error {code}."
    holders = driver.describe_holders() if code in (_OCCUPIED, _DEVICE) else ""
    raise SpmOutOfMemoryError(reason + ("\n\n" + holders if holders else ""))


# --------------------------------------------------------------------------
# K3: spine_tcm block records.
# --------------------------------------------------------------------------

class _SpineBlockInfo(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("va", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("phys_addr", ctypes.c_uint64), ("cpu_affinity_mask", ctypes.c_uint64),
                ("owner_tid", ctypes.c_int), ("is_acquired", ctypes.c_int)]


@functools.lru_cache()
def _spine():
    lib = ctypes.CDLL("libspine_tcm.so")
    lib.spine_tcm_runtime_mem_info.argtypes = [ctypes.c_int, ctypes.POINTER(_SpineBlockInfo)]
    lib.spine_tcm_runtime_mem_force_release.argtypes = [ctypes.c_int]
    lib.spine_tcm_runtime_version.restype = ctypes.c_char_p
    return lib


def _thread_process(tid):
    """(pid, name) of a live thread, or None when it no longer exists."""
    try:
        with open(f"/proc/{tid}/status") as f:
            fields = dict(line.split(":", 1) for line in f if ":" in line)
        return int(fields["Tgid"]), fields["Name"].strip()
    except (OSError, KeyError, ValueError):
        return None


def k3_blocks():
    """The spine_tcm record of each K3 TCM block."""
    lib = _spine()
    blocks = []
    for i in range(8):
        info = _SpineBlockInfo()
        if lib.spine_tcm_runtime_mem_info(i, ctypes.byref(info)) != 0:
            raise OSError(f"spine_tcm_runtime_mem_info({i}) failed")
        mask = info.cpu_affinity_mask
        cpus = ",".join(str(c) for c in range(64) if mask >> c & 1)
        owner = _thread_process(info.owner_tid) if info.is_acquired else None
        blocks.append(dict(id=i, cpus=cpus, acquired=bool(info.is_acquired),
                           tid=info.owner_tid, owner=owner))
    return blocks


def _format_k3_blocks(blocks):
    version = _spine().spine_tcm_runtime_version().decode()
    lines = [f"TCM blocks (spine_tcm {version}):",
             "  block  cpus   state     owner tid  owner alive"]
    stale, live = [], []
    for b in blocks:
        if not b["acquired"]:
            lines.append(f"  {b['id']:<5}  {b['cpus']:<5}  free")
            continue
        if b["owner"] is None:
            stale.append(b)
            lines.append(f"  {b['id']:<5}  {b['cpus']:<5}  acquired  {b['tid']:<9}  no   <- stale")
        else:
            live.append(b)
            pid, name = b["owner"]
            lines.append(f"  {b['id']:<5}  {b['cpus']:<5}  acquired  {b['tid']:<9}  yes  "
                         f"({name}, pid {pid})")
    if stale:
        ids = ", ".join(str(b["id"]) for b in stale)
        lines += ["", f"Block(s) {ids} are held by threads that no longer exist. The records "
                  "are stale, left by processes that exited without releasing them, and are "
                  "safe to clear:",
                  f"    {sys.executable} -m triton.backends.triton_shared.spm_tool release-stale",
                  "This releases only blocks whose owner thread no longer exists."]
    if live:
        ids = ", ".join(str(b["id"]) for b in live)
        lines += ["", f"Block(s) {ids} are held by live threads. Do not clear them; wait for "
                  "those processes to finish, or run on a core whose pair of blocks is free."]
        free = [d for d in range(4) if not any(b["acquired"] for b in blocks[2 * d:2 * d + 2])]
        if free:
            lines.append("Free pairs: " + ", ".join(f"CPU {8 + 2 * d},{9 + 2 * d}" for d in free) + ".")
    return "\n".join(lines)


def release_stale_k3():
    """Release the K3 blocks whose owner thread no longer exists."""
    lib = _spine()
    released = []
    for b in k3_blocks():
        if b["acquired"] and b["owner"] is None:
            if lib.spine_tcm_runtime_mem_force_release(b["id"]) == 0:
                released.append(b["id"])
    return released


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="python -m triton.backends.triton_shared.spm_tool",
                                     description="Inspect the board's scratchpad memory.")
    parser.add_argument("command", choices=["status", "release-stale"])
    args = parser.parse_args(argv)
    driver = runtime_driver()
    if isinstance(driver, SpacemitK3TcmDriver):
        if args.command == "release-stale":
            released = release_stale_k3()
            print("released blocks: " + (", ".join(map(str, released)) or "none"))
        print(driver.describe_holders())
    elif isinstance(driver, SpacemitK1TcmDriver):
        print("spacemit-k1-tcm: the kernel driver allows one mapping at a time and frees it "
              "when its process exits; it does not report the owner, and there are no "
              "records to clear.")
    else:
        print(f"no SPM driver for this board ({_compatible() or sys.platform}).")
    return 0
