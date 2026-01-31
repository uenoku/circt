class MockturtlePassBase<string name, string op> : Pass<name, op> {
  let dependentDialects = ["::circt::comb::CombDialect", 
                           "::circt::hw::HWDialect",
                           "::circt::synth::SynthDialect"];
}

def MockturtleRefactor : MockturtlePassBase<"synth-mockturtle-refactor", "hw::HWModuleOp"> {
  let summary = "Mockturtle refactoring transformations";
  let description = [{
    This pass applies refactoring transformations to the Synth dialect
    operations using the Mockturtle library.
  }];
}

def MockturtleEmap : MockturtlePassBase<"synth-mockturtle-emap", "mlir::ModuleOp"> {
  let summary = "Mockturtle extended technology mapping";
  let description = [{
    This pass applies technology mapping to the Synth dialect operations
    using the Mockturtle library's emap algorithm.
  }];
  let options = [
    Option<"genlibPath", "genlib-path", "std::string", "",
           "Path to the Genlib file describing the target library">
  ];
}

def MockturtleFunctionalReduction :
  MockturtlePassBase<"synth-mockturtle-functional-reduction", "hw::HWModuleOp"> {
  let summary = "Mockturtle function reduction transformations";
}

def MockturtleMIGAlgebraicRewriteDepth :
  MockturtlePassBase<"synth-mockturtle-mig-algebraic-rewrite-depth", "hw::HWModuleOp"> {
  let summary = "Mockturtle MIG algorithm transformations";
}

def MockturtleSOPBalancing :
  MockturtlePassBase<"synth-mockturtle-sop-balancing", "hw::HWModuleOp"> {
  let summary = "Mockturtle SOP balancing transformations";
}
