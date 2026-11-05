// Scratchpad driver for boards without one: every acquire fails.
#ifndef TRITON_RISCV_SPM_DRIVER_H
#define TRITON_RISCV_SPM_DRIVER_H

#include "spm_common.h"

static int triton_riscv_spm_acquire(size_t bytes, size_t align,
                                    spm_lease_t *lease) {
  (void)bytes;
  (void)align;
  lease->base = NULL;
  lease->cpu = -1;
  lease->domain = -1;
  lease->blocks[0] = lease->blocks[1] = -1;
  lease->busy_block = -1;
  lease->sys_errno = 0;
  lease->pinned = 0;
  return SPM_ERR_NO_DRIVER;
}

static void triton_riscv_spm_release(spm_lease_t *lease) { (void)lease; }

#endif
