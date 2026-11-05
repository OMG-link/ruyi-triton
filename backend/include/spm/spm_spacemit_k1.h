// Scratchpad driver for the SpacemiT K1: its 512 KiB TCM, shared by the A60
// cores 0-3.
//
// Each launch maps the whole TCM when it starts and unmaps it when it ends,
// so the TCM is held only while a kernel runs. The kernel driver allows one
// mapping at a time and frees it when the process exits, so a mapping that
// fails with EINVAL at this fixed, valid size means another process holds the
// TCM. An X60 core (4-7) has no TCM, and touching the mapping there is a
// SIGSEGV, so such a launch is refused before any program instance runs.
#ifndef TRITON_RISCV_SPM_DRIVER_H
#define TRITON_RISCV_SPM_DRIVER_H

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "spm_affinity.h"
#include "spm_common.h"

#define SPM_K1_BYTES (512u << 10)
#define SPM_K1_FIRST_CPU 0
#define SPM_K1_NCPUS 4

static int triton_riscv_spm_acquire(size_t bytes, size_t align,
                                    spm_lease_t *lease) {
  lease->base = NULL;
  lease->cpu = sched_getcpu();
  lease->domain = -1;
  lease->blocks[0] = lease->blocks[1] = -1;
  lease->busy_block = -1;
  lease->sys_errno = 0;
  lease->pinned = 0;
  if (lease->cpu < SPM_K1_FIRST_CPU ||
      lease->cpu >= SPM_K1_FIRST_CPU + SPM_K1_NCPUS)
    return SPM_ERR_CPU;
  lease->domain = 0;
  if (bytes > SPM_K1_BYTES)
    return SPM_ERR_CAPACITY;
  if (align > (size_t)sysconf(_SC_PAGESIZE))
    return SPM_ERR_ALIGN;
  int fd = open("/dev/tcm", O_RDWR | O_SYNC);
  if (fd < 0) {
    lease->sys_errno = errno;
    return SPM_ERR_DEVICE;
  }
  void *m = mmap(NULL, SPM_K1_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  int map_errno = errno;
  close(fd);
  if (m == MAP_FAILED) {
    lease->sys_errno = map_errno;
    return map_errno == EINVAL ? SPM_ERR_OCCUPIED : SPM_ERR_DEVICE;
  }
  spm_restrict_to(lease, SPM_K1_FIRST_CPU, SPM_K1_NCPUS);
  lease->base = m;
  return SPM_OK;
}

static void triton_riscv_spm_release(spm_lease_t *lease) {
  spm_restore_affinity(lease);
  if (lease->base)
    munmap(lease->base, SPM_K1_BYTES);
  lease->base = NULL;
}

#endif
