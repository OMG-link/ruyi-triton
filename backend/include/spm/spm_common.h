//===- spm_common.h - Scratchpad driver interface -------------*- C -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Common interface of the scratchpad (SPM) drivers. The CPU runtime
// (backend/runtime/triton_cpu_rt.cpp) is built with exactly one driver header
// (spm_none.h, spm_spacemit_k1.h or spm_spacemit_k3.h, chosen by
// backend/board.py), which defines
//
//   static int spm_hold(spm_hold_t *hold);    // SPM_OK or an SPM_ERR_* code
//   static void spm_release(spm_hold_t *hold);
//
// spm_hold acquires the whole scratchpad of the board for the process and
// maps it; spm_release gives it back. The runtime calls both on the same
// thread, once each: when it is configured and when the process exits.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_RISCV_SPM_COMMON_H
#define TRITON_RISCV_SPM_COMMON_H

#include <stddef.h>
#include <stdint.h>

// Why the scratchpad cannot serve a launch. spm.py turns each into an
// SpmOutOfMemoryError.
enum {
  SPM_OK = 0,
  SPM_ERR_NO_DRIVER = 1, // no scratchpad on this board
  SPM_ERR_CPU = 2,       // the slot's CPUs have no scratchpad
  SPM_ERR_CAPACITY = 3,  // more bytes than one program instance's share
  SPM_ERR_ALIGN = 4,     // an alignment the region cannot provide
  SPM_ERR_OCCUPIED = 5,  // another process holds the scratchpad
  SPM_ERR_DEVICE = 6,    // the device cannot be opened or mapped
};

typedef struct {
  void *base;      // the whole scratchpad, mapped
  int64_t bytes;   // its size
  int busy_block;  // on SPM_ERR_OCCUPIED, a block found held, or -1
  int sys_errno;   // on SPM_ERR_DEVICE, the failing call's errno
  int blocks_held; // driver specific
} spm_hold_t;

#endif
