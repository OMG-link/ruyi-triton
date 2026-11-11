"""Board configurations: which CPUs run kernels, and how.

A configuration names the compute CPUs a process runs kernels on, grouped
into domains (CPUs sharing an L2 cache), how threads reach those CPUs, and
the scratchpad (SPM) each CPU owns. The CPU runtime
(backend/runtime/triton_cpu_rt.cpp) is generic; everything it knows about the
board comes from here, through tcrt_configure().

A kernel's configuration is fixed when it is compiled, from its compile
target (config_for_target), and recorded in its metadata. A process runs
every kernel with one configuration: the first launch configures the
runtime, and a later kernel compiled for another configuration fails to
launch.

TRITON_CPAR_CPUS replaces the domains of the configuration, e.g.
``TRITON_CPAR_CPUS=8`` (one CPU, a serial baseline) or ``0-3;4-7`` (on the
K3, the X100 cores instead of the A100 cores). Domains are separated by
``;``, CPUs inside one by ``,`` and ranges by ``-``. A CPU keeps its
scratchpad share; CPUs that have none get none.
"""

import dataclasses
import os
from typing import Optional, Tuple

from . import spm

KiB = 1 << 10

Domains = Tuple[Tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class BoardConfig:
    name: str
    # Compute CPUs in order, partitioned into domains. None: the CPUs of the
    # process, grouped by the L2 caches sysfs reports.
    domains: Optional[Domains]
    # The CPUs are reachable only through /proc/set_ai_thread (K3 A100).
    enter_ai_cores: bool = False
    # Scratchpad driver and the share of it that belongs to each CPU, the
    # share of `spm_first_cpu` first.
    spm_driver: type = spm.NoSpmDriver
    spm_bytes_per_cpu: int = 0
    spm_first_cpu: int = 0
    # SpacemiT IME generation of the cores, informational.
    ime: Optional[str] = None


def _cpus(first, last):
    return tuple(range(first, last + 1))


K1_X60 = BoardConfig("k1-x60", (_cpus(0, 3), _cpus(4, 7)))
K1_A60 = BoardConfig("k1-a60", (_cpus(0, 3),), spm_driver=spm.SpacemitK1TcmDriver,
                     spm_bytes_per_cpu=128 * KiB, spm_first_cpu=0, ime="ime1")
K3_X100 = BoardConfig("k3-x100", (_cpus(0, 3), _cpus(4, 7)), ime="ime1")
K3_A100 = BoardConfig("k3-a100", (_cpus(8, 11), _cpus(12, 15)), enter_ai_cores=True,
                      spm_driver=spm.SpacemitK3TcmDriver, spm_bytes_per_cpu=384 * KiB,
                      spm_first_cpu=8, ime="ime2")
GENERIC = BoardConfig("generic", None)

CONFIGS = {c.name: c for c in (K1_X60, K1_A60, K3_X100, K3_A100, GENERIC)}

# LLVM processor names of the compile target. LLVM has one name for both K1
# core types; K1 kernels run on the A60 cores.
_TARGET_CONFIGS = {
    "spacemit-x60": K1_A60,
    "spacemit-x100": K3_X100,
    "spacemit-a100": K3_A100,
}


def config_for_target(target_cpu) -> BoardConfig:
    """The configuration kernels compiled for `target_cpu` run with."""
    board = os.getenv("TRITON_RISCV_BOARD")
    if board:
        if board not in CONFIGS:
            raise ValueError(f"TRITON_RISCV_BOARD={board!r} is not one of "
                             f"{', '.join(sorted(CONFIGS))}")
        return CONFIGS[board]
    return _TARGET_CONFIGS.get(target_cpu or "", GENERIC)


# --------------------------------------------------------------------------
# Resolution of the domains a process uses.
# --------------------------------------------------------------------------


def parse_cpu_list(text) -> Tuple[int, ...]:
    """Parses a Linux CPU list such as "0-3,8,10-11"."""
    cpus = set()
    for item in text.strip().split(","):
        item = item.strip()
        if not item:
            continue
        first, _, last = item.partition("-")
        first, last = int(first), int(last or first)
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU range {item!r}")
        cpus.update(range(first, last + 1))
    return tuple(sorted(cpus))


def parse_domains(text) -> Domains:
    """Parses TRITON_CPAR_CPUS: CPU lists separated by ';'."""
    domains = tuple(d for d in (parse_cpu_list(part) for part in text.split(";")) if d)
    seen = [c for d in domains for c in d]
    if not domains or len(seen) != len(set(seen)):
        raise ValueError(f"TRITON_CPAR_CPUS={text!r}: expected disjoint, non-empty "
                         "CPU lists separated by ';', e.g. '0-3;4-7'")
    return domains


def _l2_domain(cpu):
    base = f"/sys/devices/system/cpu/cpu{cpu}"
    try:
        for index in sorted(os.listdir(f"{base}/cache")):
            path = f"{base}/cache/{index}"
            with open(f"{path}/level") as f:
                if f.read().strip() != "2":
                    continue
            with open(f"{path}/shared_cpu_list") as f:
                return parse_cpu_list(f.read())
    except OSError:
        pass
    try:
        with open(f"{base}/topology/cluster_cpus_list") as f:
            return parse_cpu_list(f.read())
    except OSError:
        return (cpu,)


def detect_domains() -> Domains:
    """The CPUs of the process grouped by shared L2 cache."""
    cpus = sorted(os.sched_getaffinity(0))
    domains, assigned = [], set()
    for cpu in cpus:
        if cpu in assigned:
            continue
        domain = tuple(c for c in _l2_domain(cpu) if c in cpus and c not in assigned)
        domain = domain or (cpu,)
        assigned.update(domain)
        domains.append(domain)
    return tuple(domains)


@dataclasses.dataclass(frozen=True)
class ResolvedConfig:
    """What the runtime is configured with (tcrt_config)."""
    config: BoardConfig
    domains: Domains
    enter_ai_cores: bool


def resolve(config: BoardConfig) -> ResolvedConfig:
    """Applies TRITON_CPAR_CPUS and the detection of a generic configuration."""
    override = os.getenv("TRITON_CPAR_CPUS", "").strip()
    if override:
        domains = parse_domains(override)
    elif config.domains is not None:
        domains = config.domains
    else:
        domains = detect_domains()
    # Threads reach the AI cores only through /proc/set_ai_thread, and once
    # moved there they cannot reach other cores.
    own = {c for d in config.domains or () for c in d}
    enter = config.enter_ai_cores and all(c in own for d in domains for c in d)
    return ResolvedConfig(config, domains, enter)


def cluster_size(config_name) -> int:
    """CPUs of the first domain of a configuration, for num_warps=0."""
    return len(resolve(CONFIGS[config_name]).domains[0])
