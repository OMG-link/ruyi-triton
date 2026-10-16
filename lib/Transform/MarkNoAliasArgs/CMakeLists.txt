add_triton_library(MarkNoAliasArgs
  MarkNoAliasArgsPass.cpp

  DEPENDS
  MarkNoAliasArgsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFunctionInterfaces
)
