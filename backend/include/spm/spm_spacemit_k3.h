// Scratchpad driver for the SpacemiT K3: its 3 MiB TCM, in eight 384 KiB
// blocks. Blocks 2d and 2d+1 are local to the A100 cores 8+2d and 9+2d;
// reaching another pair's blocks is correct but slower than DDR.
//
// Ownership goes through spine_tcm (libspine_tcm.so), the vendor library that
// records which thread holds which block. A launch takes the blocks it needs
// from its core's pair with spine_tcm_mem_try_wait(id, 0), so it never waits
// for and never preempts another user; a block already held fails the launch.
// The kernel driver itself does not refuse a second mapping, so these records
// are the only exclusion.
//
// Access goes through one mapping of the whole TCM, made once per process: a
// pair's two blocks are adjacent there, in order, which spine_tcm's own
// per-block mappings do not promise.
#ifndef TRITON_RISCV_SPM_DRIVER_H
#define TRITON_RISCV_SPM_DRIVER_H

#define SPINE_TCM_DIRECT_LINK
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <spine_tcm.h>
#include <sys/mman.h>
#include <unistd.h>

#include "spm_affinity.h"
#include "spm_common.h"

#define SPM_K3_BLOCK (384u << 10)
#define SPM_K3_BLOCKS 8
#define SPM_K3_FIRST_CPU 8
#define SPM_K3_DOMAIN_CPUS 2

static void *spm_k3_map = NULL;
static int spm_k3_map_errno = 0;
static pthread_once_t spm_k3_once = PTHREAD_ONCE_INIT;

static void spm_k3_map_once(void) {
  int fd = open("/dev/tcm", O_RDWR | O_SYNC);
  if (fd < 0) {
    spm_k3_map_errno = errno;
    return;
  }
  void *m = mmap(NULL, (size_t)SPM_K3_BLOCK * SPM_K3_BLOCKS,
                 PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  spm_k3_map_errno = m == MAP_FAILED ? errno : 0;
  close(fd);
  if (m != MAP_FAILED)
    spm_k3_map = m;
}

static int triton_riscv_spm_acquire(size_t bytes, size_t align,
                                    spm_lease_t *lease) {
  lease->base = NULL;
  lease->cpu = sched_getcpu();
  lease->domain = -1;
  lease->blocks[0] = lease->blocks[1] = -1;
  lease->busy_block = -1;
  lease->sys_errno = 0;
  lease->pinned = 0;
  int index = lease->cpu - SPM_K3_FIRST_CPU;
  if (index < 0 || index >= SPM_K3_BLOCKS)
    return SPM_ERR_CPU;
  int d = index / SPM_K3_DOMAIN_CPUS;
  lease->domain = d;
  if (bytes > 2 * (size_t)SPM_K3_BLOCK)
    return SPM_ERR_CAPACITY;
  if (align > (size_t)sysconf(_SC_PAGESIZE))
    return SPM_ERR_ALIGN;
  if (!spine_tcm_is_available()) {
    lease->sys_errno = ENODEV;
    return SPM_ERR_DEVICE;
  }
  pthread_once(&spm_k3_once, spm_k3_map_once);
  if (!spm_k3_map) {
    lease->sys_errno = spm_k3_map_errno;
    return SPM_ERR_DEVICE;
  }
  int nblocks = bytes > SPM_K3_BLOCK ? 2 : 1;
  for (int i = 0; i < nblocks; ++i) {
    int id = 2 * d + i;
    if (!spine_tcm_mem_try_wait(id, 0)) {
      for (int j = 0; j < i; ++j)
        spine_tcm_mem_release(lease->blocks[j]);
      lease->blocks[0] = lease->blocks[1] = -1;
      lease->busy_block = id;
      return SPM_ERR_OCCUPIED;
    }
    lease->blocks[i] = id;
  }
  spm_restrict_to(lease, SPM_K3_FIRST_CPU + SPM_K3_DOMAIN_CPUS * d,
                  SPM_K3_DOMAIN_CPUS);
  lease->base = (char *)spm_k3_map + (size_t)2 * d * SPM_K3_BLOCK;
  return SPM_OK;
}

static void triton_riscv_spm_release(spm_lease_t *lease) {
  spm_restore_affinity(lease);
  for (int i = 0; i < 2; ++i)
    if (lease->blocks[i] >= 0) {
      spine_tcm_mem_release(lease->blocks[i]);
      lease->blocks[i] = -1;
    }
  lease->base = NULL;
}

#endif
