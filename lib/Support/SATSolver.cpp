#include "circt/Support/SatSolver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/llvm-config.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if LLVM_WITH_Z3
#include <z3++.h>
#else
#endif

namespace circt {

namespace {

#if LLVM_WITH_Z3
class Z3SATSolver : public IncrementalSATSolver {
public:
  Z3SATSolver() : context(), solver(z3::tactic(context, "sat").mk_solver()) {}

  void add(int lit) override {
    if (lit == 0) {
      addClauseInternal(clauseBuf);
      ++numClauses;
      clauseBuf.clear();
      return;
    }

    const int absLit = std::abs(lit);
    if (absLit > maxVariable)
      reserveVars(absLit);

    clauseBuf.push_back(lit);
  }

  void assume(int lit) override {
    if (lit == 0)
      return;
    assumptions.push_back(lit);
  }

  Result solve(int64_t confLimit) override {
    uint64_t rlimit = confLimit;
    solver.set("rlimit", static_cast<unsigned>(rlimit));
    auto result = solver.check(toExprs(assumptions));
    assumptions.clear();
    switch (result) {
    case z3::sat:
      return kSAT;
    case z3::unsat:
      return kUNSAT;
    default:
      return kUNKNOWN;
    }
  }

  int val(int v) const override {
    if (v <= 0 || v > maxVariable)
      return 0;

    auto assignment = solver.get_model().eval(variables[v - 1], true);
    if (assignment.is_true())
      return v;
    if (assignment.is_false())
      return -v;
    return 0;
  }

  void reserveVars(int maxVar) override {
    if (maxVar <= maxVariable)
      return;
    while (static_cast<int>(variables.size()) < maxVar)
      newVariable();
    maxVariable = maxVar;
  }

private:
  int newVariable() {
    int varIndex = static_cast<int>(variables.size()) + 1;
    variables.push_back(
        context.bool_const(("v" + std::to_string(varIndex)).c_str()));
    return varIndex;
  }

  z3::expr literalToExpr(int lit) {
    int absLit = std::abs(lit);
    // Ensure variable exists for this literal.
    reserveVars(absLit);
    z3::expr variable = variables[absLit - 1];
    return lit > 0 ? variable : !variable;
  }

  z3::expr_vector toExprs(llvm::ArrayRef<int> lits) {
    z3::expr_vector exprs(context);
    for (int lit : lits) {
      assert(lit != 0 && "Literals must be non-zero");
      exprs.push_back(literalToExpr(lit));
    }
    return exprs;
  }

  void addClauseInternal(llvm::ArrayRef<int> lits) {
    if (lits.empty()) {
      solver.add(context.bool_val(false));
      return;
    }

    z3::expr_vector exprs(context);
    for (int lit : lits) {
      if (lit == 0)
        continue;
      exprs.push_back(literalToExpr(lit));
    }

    if (exprs.empty()) {
      solver.add(context.bool_val(false));
      return;
    }

    if (exprs.size() == 1) {
      solver.add(exprs[0]);
      return;
    }

    solver.add(z3::mk_or(exprs));
  }

  z3::context context;
  z3::solver solver;
  llvm::SmallVector<z3::expr> variables;
  llvm::SmallVector<int> assumptions, clauseBuf;
  uint64_t numClauses = 0;
  int maxVariable = 0;
};
#endif // LLVM_WITH_Z3

} // namespace

std::unique_ptr<IncrementalSATSolver> createZ3SATSolver() {
#if LLVM_WITH_Z3
  return std::make_unique<Z3SATSolver>();
#else
  return {};
#endif
}

} // namespace circt
