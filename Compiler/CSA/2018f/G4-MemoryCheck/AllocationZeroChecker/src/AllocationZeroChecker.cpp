
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/AST/ExprCXX.h"
#include <iostream>

using namespace clang;
using namespace ento;

namespace {
class AllocationZeroChecker : public Checker<check::PostStmt<CXXNewExpr>> {
  mutable std::unique_ptr<BuiltinBug> BT;
  void reportBug(const char *Msg, ProgramStateRef StateZero,
                 CheckerContext &C) const;

public:
  void checkPostStmt(const CXXNewExpr *NewExpr, CheckerContext &C) const;
};
} // end anonymous namespace

void AllocationZeroChecker::reportBug(const char *Msg, ProgramStateRef StateZero,
                               CheckerContext &C) const {
  if (ExplodedNode *N = C.generateErrorNode(StateZero)) {
    if (!BT)
      BT.reset(new BuiltinBug(this, "Size of allocated memory is zero"));

    auto R = llvm::make_unique<BugReport>(*BT, Msg, N);
    bugreporter::trackNullOrUndefValue(N, bugreporter::GetDenomExpr(N), *R);
    C.emitReport(std::move(R));
  }
}

void AllocationZeroChecker::checkPostStmt(const CXXNewExpr *NewExpr,
                                  CheckerContext &C) const {
  if (!NewExpr->isArray())//not allocate an array
    return;

  const Expr *ArraySize = NewExpr->getArraySize();
  SVal Denom = C.getState()->getSVal(ArraySize, C.getLocationContext());
  Optional<DefinedSVal> size = Denom.getAs<DefinedSVal>();
  if (!size)
    return;

  ConstraintManager &CM = C.getConstraintManager();
  ProgramStateRef stateNotZero, stateZero;
  std::tie(stateNotZero, stateZero) = CM.assumeDual(C.getState(), *size);

  if (!stateNotZero) {
    reportBug("Size of allocated memory is zero", stateZero, C);
    return;
  }

  bool TaintedD = C.getState()->isTainted(*size);
  if ((stateNotZero && stateZero && TaintedD)) {
    reportBug("Allocate a tainted value of memory, possibly zero", stateZero, C);
    return;
  }

  // If we get here, then the denom should not be zero. We abandon the implicit
  // zero denom case for now.
  C.addTransition(stateNotZero);
}

void ento::registerAllocationZeroChecker(CheckerManager &mgr) {
  mgr.registerChecker<AllocationZeroChecker>();
}
