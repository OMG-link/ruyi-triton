#include <pybind11/pybind11.h>

#if defined(__riscv) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace py = pybind11;

/// VLEN of the CPU this process runs on, in bits, or 0 when it has no vector
/// unit. The plugin itself is built for the baseline ISA, so the read enables
/// V for this one instruction, and only after the kernel reports V.
static unsigned readHostVLEN() {
#if defined(__riscv) && defined(__linux__)
  constexpr unsigned long kHwcapV = 1UL << ('V' - 'A');
  if (!(getauxval(AT_HWCAP) & kHwcapV))
    return 0;
  unsigned long bytes;
  asm volatile(".option push\n\t"
               ".option arch, +v\n\t"
               "csrr %0, vlenb\n\t"
               ".option pop"
               : "=r"(bytes));
  return bytes * 8;
#else
  return 0;
#endif
}

void init_triton_triton_shared(py::module &&m) {
  m.def("read_host_vlen", &readHostVLEN);
}
