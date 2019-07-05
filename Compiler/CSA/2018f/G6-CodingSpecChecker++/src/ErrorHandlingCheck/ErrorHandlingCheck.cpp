////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file ErrorHandlingCheck.cpp
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang, <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#include "ErrorHandlingCheck.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMap.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/TaintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include <fstream>
#include <string>
#include <unordered_set>


using namespace clang;
using namespace clang::ento;

REGISTER_TRAIT_WITH_PROGRAMSTATE(ErrorState, bool);

extern std::string ErrorFunctionListFilePath;

std::unordered_set<std::string> ErrorHandlingChecker::fName;

void ErrorHandlingChecker::StaticInit(const std::string &ErrorFunctionListFilePath) {
  std::ifstream funcName(ErrorFunctionListFilePath);
  if (!funcName) {
    llvm::errs() << "[ErrorHandlingCheck]: funcName file not found\n";
  } else {
    std::string line;
    while (getline(funcName, line)) {
      fName.insert(line);
    }
  }
}

bool IsInIfStmt(const Stmt *S, CheckerContext &Ctx) {
	ParentMap &PM = Ctx.getLocationContext()->getParentMap();
	const Stmt *P = PM.getParent(S);
	const Stmt *C = S;

	while (P) {
    if (P->getStmtClass() == Stmt::IfStmtClass) {
      return C == cast<IfStmt>(P)->getCond();
    }
		C = P;
		P = PM.getParent(C);
	}
	return false;
}

bool IsInDeclStmt(const Stmt *S, CheckerContext &Ctx) {
	ParentMap &PM = Ctx.getLocationContext()->getParentMap();
	const Stmt *P = PM.getParent(S);
	const Stmt *C = S;

	while (P) {
    if (P->getStmtClass() == Stmt::DeclStmtClass) {
      return 1;
    }
		C = P;
		P = PM.getParent(C);
	}
	return false;
}

void ErrorHandlingChecker::checkPostCall(const CallEvent &Call, CheckerContext &C) const {
  static int z = 1;
  static SourceLocation line;
  ProgramStateRef State = C.getState();
  BugReporter &BR = C.getBugReporter();
  DiagnosticsEngine &DE = BR.getDiagnostic();
  const Expr* CEX = Call.getOriginExpr();
  SourceLocation location = CEX->getExprLoc();
  if(z)
    z = 0;
  else if(location==line)
    return;
  
  if(IsInDeclStmt(CEX,C)||IsInIfStmt(CEX,C)||CEX->getStmtClass() == Stmt::DeclStmtClass)
    return;

  if(Call.getResultType()->isVoidType()==0
  &&Call.getCalleeIdentifier()->getName()!="printf")
  {
    line = location;
    const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                                 "The return value of the function is not checked");
          auto DB = DE.Report(location, ID);
    return;
  }

  if (const IdentifierInfo *cc = Call.getCalleeIdentifier())
    {
      StringRef SR = cc->getName();
      if (fName.find(SR.data()) == fName.end()) {
        if (State->get<ErrorState>()) {
          const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                                 "The return value of the function is not checked");
          auto DB = DE.Report(location, ID);
        }
        return;
      }
      if (State->get<ErrorState>()) {
        const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                               "The return value of the function is not checked");
        // auto DB = DE.Report(location, ID);
        auto DB = DE.Report(location.getLocWithOffset(-10), ID);
      }


      SymbolRef VarDesc = Call.getReturnValue().getAsSymbol();

      State = State->set<ErrorState>(true);
    }
	if (State != C.getState()) {
		C.addTransition(State);
	}
  
}

// check whether the function called in IFCondition
void ErrorHandlingChecker::checkBranchCondition(const Stmt *Condition, CheckerContext &C) const {
  if (!IsInIfStmt(Condition, C))
    return;

	ProgramStateRef State = C.getState();
  SVal X = C.getSVal(Condition);
  SymbolRef VarRef = X.getAsSymbol();
  if (State->get<ErrorState>()) {
    State = State->set<ErrorState>(false);
  }
	if (State != C.getState()) {
		C.addTransition(State);
	}
}
