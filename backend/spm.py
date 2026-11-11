"""Scratchpad memory (SPM) drivers for ``tl.dev.alloc(placement="spm")``.

The compiler needs nothing from the hardware: a kernel that uses ``spm`` gets
one hidden trailing argument, the base of a region of ``spm_size`` bytes, and
records ``spm_size`` and ``spm_align`` in its metadata.

The CPU runtime acquires the board's whole scratchpad once, when it is
configured, and holds it until the process exits. Every slot (the CPUs that
run one program instance) gets the share of its CPUs (board.py), and every
program instance gets the share of its slot. Every reason a launch cannot get
its region is reported as an ``SpmOutOfMemoryError``.

Each driver has a C header under ``include/spm/`` the runtime is built with,
and the link flags it needs. The runtime is built for the board it runs on.

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

# SPM_* codes of include/spm/spm_common.h.
_OK, _NO_DRIVER, _CPU, _CAPACITY, _ALIGN, _OCCUPIED, _DEVICE = range(7)


class SpmOutOfMemoryError(RuntimeError):
    """A launch could not get the scratchpad region its kernel needs."""


class SpmDriver:
    """How one board provides its scratchpad to the runtime."""

    name = "none"
    header = "spm_none.h"
    link_args = ()

    def describe_holders(self):
        """Who holds the scratchpad, for error reports. Empty if unknown."""
        return ""


class NoSpmDriver(SpmDriver):
    pass


class SpacemitK1TcmDriver(SpmDriver):
    name = "spacemit-k1-tcm"
    header = "spm_spacemit_k1.h"


class SpacemitK3TcmDriver(SpmDriver):
    name = "spacemit-k3-tcm"
    header = "spm_spacemit_k3.h"
    link_args = ("-lspine_tcm",)

    def describe_holders(self):
        try:
            blocks = k3_blocks()
        except OSError as e:
            return f"TCM blocks: cannot query spine_tcm ({e})."
        return _format_k3_blocks(blocks)


@functools.lru_cache()
def _compatible():
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            return f.read().split(b"\0")[0].decode()
    except OSError:
        return ""


def warn_if_unlikely_to_run(kernel_name, spm_size, config, warps):
    """Compile-time warning for a kernel whose scratchpad will not fit."""
    if not spm_size:
        return
    if not config.spm_bytes_per_cpu:
        warnings.warn(
            f"kernel '{kernel_name}' uses placement=\"spm\", which the board "
            f"configuration '{config.name}' does not provide; it will fail at launch",
            stacklevel=2)
        return
    ncpus = sum(len(d) for d in config.domains or ())
    budget = config.spm_bytes_per_cpu * max(1, min(warps, ncpus or warps))
    if spm_size > budget:
        warnings.warn(
            f"kernel '{kernel_name}' needs {_kib(spm_size)} of SPM, more than the "
            f"{_kib(budget)} a program instance of {warps} thread(s) gets on "
            f"'{config.name}'; it will fail at launch",
            stacklevel=2)


def _kib(n):
    return f"{n} bytes" if n < KiB else f"{n / KiB:g} KiB"


def raise_launch_error(status, kernel_name, bytes_, align, config):
    """Raises the SpmOutOfMemoryError of a launch the runtime refused.

    `status` is the runtime's tcrt_spm_status_t of the failure.
    """
    driver = config.spm_driver()
    code = status.code
    head = f"kernel '{kernel_name}' needs {_kib(bytes_)} of SPM"
    if code == _NO_DRIVER:
        reason = (f"{head}, but the board configuration '{config.name}' has no "
                  f"scratchpad ({_compatible() or sys.platform}).")
    elif code == _CPU:
        reason = (f"{head}, but the program instance on CPU {status.slot_cpu} runs on "
                  f"CPUs without a scratchpad share on '{config.name}'.")
    elif code == _CAPACITY:
        reason = (f"{head}, but a program instance on CPU {status.slot_cpu} gets only "
                  f"{_kib(status.slot_bytes)} on '{config.name}' (its CPUs' shares).")
    elif code == _ALIGN:
        reason = f"{head} aligned to {align} bytes, more than {driver.name} can align."
    elif code == _OCCUPIED:
        if status.busy_block >= 0:
            reason = (f"{head}, but {driver.name} block {status.busy_block} is held by "
                      f"another process.")
        else:
            reason = f"{head}, but the TCM is mapped by another process."
    elif code == _DEVICE:
        reason = (f"{head}, but the {driver.name} scratchpad cannot be opened or mapped: "
                  f"{os.strerror(status.sys_errno) if status.sys_errno else 'unavailable'}.")
    else:
        reason = f"{head}, but the runtime returned an unknown error {code}."
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
                  "those processes to finish."]
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
    board = _compatible()
    if board.startswith("spacemit,k3"):
        driver = SpacemitK3TcmDriver()
        if args.command == "release-stale":
            released = release_stale_k3()
            print("released blocks: " + (", ".join(map(str, released)) or "none"))
        print(driver.describe_holders())
    elif board.startswith("spacemit,k1"):
        print("spacemit-k1-tcm: the kernel driver allows one mapping at a time and frees it "
              "when its process exits; it does not report the owner, and there are no "
              "records to clear.")
    else:
        print(f"no SPM driver for this board ({_compatible() or sys.platform}).")
    return 0
