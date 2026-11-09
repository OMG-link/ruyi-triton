// Common interface of the scratchpad (SPM) drivers that a CPU kernel launcher
// includes. A launch of a kernel that uses `tl.dev.alloc(placement="spm")`
// calls triton_riscv_spm_acquire before any program instance runs, passes
// lease.base as the kernel's trailing argument, and calls
// triton_riscv_spm_release on every exit path.
//
// Exactly one driver header defines the two functions for a launcher:
// spm_none.h, spm_spacemit_k1.h or spm_spacemit_k3.h, chosen by
// triton/backends/triton_shared/spm.py.
#ifndef TRITON_RISCV_SPM_COMMON_H
#define TRITON_RISCV_SPM_COMMON_H

#include <stddef.h>
#include <stdint.h>

// Why an acquire failed. spm.py turns each into an SpmOutOfMemoryError.
enum {
  SPM_OK = 0,
  SPM_ERR_NO_DRIVER = 1, // no scratchpad on this board
  SPM_ERR_CPU = 2,       // the current core has no scratchpad
  SPM_ERR_CAPACITY = 3,  // more bytes than one program instance may use
  SPM_ERR_ALIGN = 4,     // an alignment the region cannot provide
  SPM_ERR_OCCUPIED = 5,  // another process holds the region
  SPM_ERR_DEVICE = 6,    // the device cannot be opened or mapped
};

typedef struct {
  void *base;              // the region passed to the kernel
  int cpu;                 // the core the launch started on
  int domain;              // the sharing domain of that core
  int blocks[2];           // driver blocks held, -1 when unused
  int busy_block;          // on SPM_ERR_OCCUPIED, the block found held, or -1
  int sys_errno;           // on SPM_ERR_DEVICE, the failing call's errno
  int pinned;              // whether `saved_mask` must be restored
  uint64_t saved_mask[16]; // the thread's affinity before the launch
} spm_lease_t;

#endif
