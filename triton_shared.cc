#include <pybind11/pybind11.h>

namespace py = pybind11;

static unsigned readHostVLEN() {
#if defined(__riscv) && defined(__riscv_vector)
  unsigned long bytes;
  asm volatile("csrr %0, vlenb" : "=r"(bytes));
  return bytes * 8;
#else
  return 0;
#endif
}

void init_triton_triton_shared(py::module &&m) {
  m.def("read_host_vlen", &readHostVLEN);
}
