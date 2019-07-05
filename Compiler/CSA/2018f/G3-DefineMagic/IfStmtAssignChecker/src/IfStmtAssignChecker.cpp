// This defines IfStmtAssignChecker, checks whether there is assignment in if-conditions.
// GNU C Coding Standards
// Section 5.3
// see https://www.gnu.org/prep/standards/html_node/Syntactic-Conventions.html for detail

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace ento;

namespace {
    class IfStmtAssignChecker
     : public Checker<check::BranchCondition> {
     mutable std::unique_ptr<BuiltinBug> BT;

    public:
        void checkBranchCondition(const Stmt *CS, CheckerContext &C) const;
    private:
        bool checkCondContainsAssign(const Stmt *stmt, ASTContext &ASTC) const;
    };
}

void IfStmtAssignChecker::checkBranchCondition(const Stmt *CS, 
                                        CheckerContext &C) const {
    ASTContext &ASTC = C.getASTContext();      
    const Stmt* S = CS;
    do {
        const auto& parents = ASTC.getParents(*S);
        if (parents.empty())
            return;
        S = parents[0].get<Stmt>();
        if (dyn_cast<IfStmt>(S))
            break;
    } while (S);
    bool flag = checkCondContainsAssign(CS, ASTC);
    if (!flag) return;
    if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
        if (!BT)
            BT.reset(
                new BuiltinBug(this, "assign in if statement", 
                "Try to avoid assignments inside if-conditions "
                "(assignments inside while-conditions are ok)")
            );
            auto R = llvm::make_unique<BugReport>(*BT, BT->getDescription(), N);
            R->addRange(CS->getSourceRange());
            C.emitReport(std::move(R));
    }
}

bool IfStmtAssignChecker::checkCondContainsAssign(const Stmt *stmt, ASTContext &ASTC) const {
    if (const Expr *E = dyn_cast<Expr>(stmt)) {
        stmt = E->IgnoreParenCasts()->IgnoreConversionOperator();
    }
    if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(stmt)) {
        if (BO->getOpcode() == BO_Assign) {
            return true;
        }
    }
    for (Stmt::const_child_iterator
        i = stmt->child_begin(), e = stmt->child_end(); i != e; ++i) {
        const Stmt *child = *i;
        if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(child)) {
            if (BO->getOpcode() == BO_Assign) {
                return true;
            }
        }
        else {
            // OpaqueValueExpr holds another expression, therefore we extract wrapped expression
            // otherwise recursion will stop
            if (const OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(child)) {
                child = OVE->getSourceExpr();
            }
            bool flag = checkCondContainsAssign(child, ASTC);
            if (flag) {
                return true;
            }
        }
    }
    return false;
}

void ento::registerIfStmtAssignChecker(CheckerManager &mgr) {
  mgr.registerChecker<IfStmtAssignChecker>();
}