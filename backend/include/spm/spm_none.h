//===- spm_none.h - Scratchpad driver for boards without one ----*- C -*-===//
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
// Every acquisition fails.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_RISCV_SPM_DRIVER_H
#define TRITON_RISCV_SPM_DRIVER_H

#include "spm_common.h"

static int spm_hold(spm_hold_t *hold) {
  hold->base = NULL;
  hold->bytes = 0;
  hold->busy_block = -1;
  hold->sys_errno = 0;
  hold->blocks_held = 0;
  return SPM_ERR_NO_DRIVER;
}

static void spm_release(spm_hold_t *hold) { (void)hold; }

#endif
