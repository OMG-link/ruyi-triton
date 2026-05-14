add_triton_library(DecomposeHighLevelOps
  DecomposeHighLevelOpsPass.cpp

  DEPENDS
  DecomposeHighLevelOpsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
