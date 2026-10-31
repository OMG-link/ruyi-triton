#ifndef TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES
#define TRITON_TO_UNSTRUCTURED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToUnstructured : Pass<"triton-to-unstructured", "mlir::ModuleOp"> {
  let summary = "Transforms tt.addptr ops into offset accumulation ops";
  let constructor = "triton::createTritonToUnstructuredPass()";
}

#endif
