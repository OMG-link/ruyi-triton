// Keeps the launching thread on the cores of its scratchpad domain for the
// length of a launch, and restores its affinity afterwards. A program that
// moves to another domain mid-launch stays correct, but each scratchpad
// access becomes several times slower than DDR; on a K1 a move to an X60
// core would crash. TRITON_RISCV_SPM_PIN=0 turns the restriction off.
#ifndef TRITON_RISCV_SPM_AFFINITY_H
#define TRITON_RISCV_SPM_AFFINITY_H

#include <sched.h>
#include <stdlib.h>
#include <string.h>

#include "spm_common.h"

static void spm_restrict_to(spm_lease_t *lease, int first_cpu, int ncpus) {
  const char *pin = getenv("TRITON_RISCV_SPM_PIN");
  if (pin && strcmp(pin, "0") == 0)
    return;
  cpu_set_t saved, domain;
  if (sched_getaffinity(0, sizeof saved, &saved) != 0)
    return;
  CPU_ZERO(&domain);
  for (int c = first_cpu; c < first_cpu + ncpus; ++c)
    CPU_SET(c, &domain);
  if (sched_setaffinity(0, sizeof domain, &domain) != 0)
    return;
  memcpy(lease->saved_mask, &saved,
         sizeof saved < sizeof lease->saved_mask ? sizeof saved
                                                 : sizeof lease->saved_mask);
  lease->pinned = 1;
}

static void spm_restore_affinity(spm_lease_t *lease) {
  if (!lease->pinned)
    return;
  cpu_set_t saved;
  CPU_ZERO(&saved);
  memcpy(&saved, lease->saved_mask,
         sizeof saved < sizeof lease->saved_mask ? sizeof saved
                                                 : sizeof lease->saved_mask);
  sched_setaffinity(0, sizeof saved, &saved);
  lease->pinned = 0;
}

#endif
