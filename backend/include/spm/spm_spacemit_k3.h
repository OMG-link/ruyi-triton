//===- spm_spacemit_k3.h - Scratchpad driver of the SpacemiT K3 *- C -*-===//
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
// Scratchpad driver for the SpacemiT K3: its 3 MiB TCM, in eight 384 KiB
// blocks. Block i is local to the A100 core 8+i; reaching another pair's
// blocks is correct but slower than DDR.
//
// Ownership goes through spine_tcm (libspine_tcm.so), the vendor library that
// records which thread holds which block. The process takes all eight blocks
// with spine_tcm_mem_try_wait(id, 0), so it never waits for and never
// preempts another user; a block already held fails the acquisition. The
// kernel driver itself does not refuse a second mapping, so these records
// are the only exclusion. Only the acquiring thread may release a block.
//
// Access goes through one mapping of the whole TCM, in block order, which
// spine_tcm's own per-block mappings do not promise.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_RISCV_SPM_DRIVER_H
#define TRITON_RISCV_SPM_DRIVER_H

#define SPINE_TCM_DIRECT_LINK
#include <errno.h>
#include <fcntl.h>
#include <spine_tcm.h>
#include <sys/mman.h>
#include <unistd.h>

#include "spm_common.h"

#define SPM_K3_BLOCK (384u << 10)
#define SPM_K3_BLOCKS 8

static void spm_release(spm_hold_t *hold) {
  for (int i = 0; i < hold->blocks_held; ++i)
    spine_tcm_mem_release(i);
  hold->blocks_held = 0;
  if (hold->base)
    munmap(hold->base, (size_t)hold->bytes);
  hold->base = NULL;
}

static int spm_hold(spm_hold_t *hold) {
  hold->base = NULL;
  hold->bytes = (int64_t)SPM_K3_BLOCK * SPM_K3_BLOCKS;
  hold->busy_block = -1;
  hold->sys_errno = 0;
  hold->blocks_held = 0;
  if (!spine_tcm_is_available()) {
    hold->sys_errno = ENODEV;
    return SPM_ERR_DEVICE;
  }
  for (int id = 0; id < SPM_K3_BLOCKS; ++id) {
    if (!spine_tcm_mem_try_wait(id, 0)) {
      hold->busy_block = id;
      spm_release(hold);
      return SPM_ERR_OCCUPIED;
    }
    hold->blocks_held = id + 1;
  }
  int fd = open("/dev/tcm", O_RDWR | O_SYNC);
  if (fd < 0) {
    hold->sys_errno = errno;
    spm_release(hold);
    return SPM_ERR_DEVICE;
  }
  void *m = mmap(NULL, (size_t)hold->bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                 fd, 0);
  int map_errno = errno;
  close(fd);
  if (m == MAP_FAILED) {
    hold->sys_errno = map_errno;
    spm_release(hold);
    return SPM_ERR_DEVICE;
  }
  hold->base = m;
  return SPM_OK;
}

#endif
