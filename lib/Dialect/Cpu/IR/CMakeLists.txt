add_triton_library(CpuIR
  CpuDialect.cpp
  DEPENDS CpuIRIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces MLIRInferTypeOpInterface
)
