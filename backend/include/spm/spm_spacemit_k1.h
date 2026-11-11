//===- spm_spacemit_k1.h - Scratchpad driver of the SpacemiT K1 *- C -*-===//
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
// Scratchpad driver for the SpacemiT K1: its 512 KiB TCM, shared by the A60
// cores 0-3.
//
// The kernel driver allows one mapping at a time and frees it when the
// process exits, so a mapping that fails with EINVAL at this fixed, valid
// size means another process holds the TCM. An X60 core (4-7) has no TCM,
// and touching the mapping there is a SIGSEGV; the board configuration only
// gives SPM to slots of A60 cores.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_RISCV_SPM_DRIVER_H
#define TRITON_RISCV_SPM_DRIVER_H

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "spm_common.h"

#define SPM_K1_BYTES (512u << 10)

static int spm_hold(spm_hold_t *hold) {
  hold->base = NULL;
  hold->bytes = SPM_K1_BYTES;
  hold->busy_block = -1;
  hold->sys_errno = 0;
  hold->blocks_held = 0;
  int fd = open("/dev/tcm", O_RDWR | O_SYNC);
  if (fd < 0) {
    hold->sys_errno = errno;
    return SPM_ERR_DEVICE;
  }
  void *m = mmap(NULL, SPM_K1_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  int map_errno = errno;
  close(fd);
  if (m == MAP_FAILED) {
    hold->sys_errno = map_errno;
    return map_errno == EINVAL ? SPM_ERR_OCCUPIED : SPM_ERR_DEVICE;
  }
  hold->base = m;
  return SPM_OK;
}

static void spm_release(spm_hold_t *hold) {
  if (hold->base)
    munmap(hold->base, SPM_K1_BYTES);
  hold->base = NULL;
}

#endif
