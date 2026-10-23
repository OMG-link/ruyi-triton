#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// The sanitizer type is a pass parameter, `sanitizer-attributes<asan>` or
// `sanitizer-attributes<tsan>`: LLVM 24's opt parses its command line before
// loading plugins, so a plugin-registered cl::opt is no longer reachable.

namespace {

struct SanitizerAttributes : OptionalPassInfoMixin<SanitizerAttributes> {
  explicit SanitizerAttributes(StringRef type) : SanitizerType(type) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    // add an attribute to the function depending on the type of sanitizer
    if (SanitizerType == "asan") {
      F.addFnAttr(Attribute::SanitizeAddress);
    } else if (SanitizerType == "tsan") {
      F.addFnAttr(Attribute::SanitizeThread);
    }

    // this pass modifies all function attributes
    return PreservedAnalyses::none();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }

private:
  std::string SanitizerType;
};
} // namespace

llvm::PassPluginLibraryInfo getSanitizerAttributesPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SanitizerAttributes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  for (StringRef type : {"asan", "tsan"}) {
                    if (Name == ("sanitizer-attributes<" + type + ">").str()) {
                      FPM.addPass(SanitizerAttributes(type));
                      return true;
                    }
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSanitizerAttributesPluginInfo();
}
