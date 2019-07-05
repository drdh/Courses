////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file InitInNeedCheck.h
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang, <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

namespace clang {
namespace ento {

class InitInNeedChecker : public Checker<
	check::Location, check::Bind, eval::Call, 
	check::BranchCondition, check::ASTDecl<VarDecl>, 
	check::BeginFunction, check::EndFunction,
	check::EndAnalysis> {

public:
	InitInNeedChecker();
	void checkBeginFunction(CheckerContext &Ctx) const;
	void checkLocation(
		SVal Loc, bool IsLoad, const Stmt *S, CheckerContext &Ctx) const;
	void checkBind(SVal Loc, SVal Val, const Stmt *S, CheckerContext &Ctx) const;
	bool evalCall(const CallExpr *CE, CheckerContext &Ctx) const;
	void checkBranchCondition(const Stmt *Condition, CheckerContext &Ctx) const;
	void checkASTDecl(const VarDecl *D, AnalysisManager &Mgr, BugReporter &BR) const;
	void checkEndFunction(const ReturnStmt *R, CheckerContext &Ctx) const;
	void checkEndAnalysis(ExplodedGraph &G, BugReporter &BR, ExprEngine &Eng) const;

private:
	void reportReduntantWrite(ExplodedNode *N, CheckerContext &Ctx) const;

private:
	std::unique_ptr<BugType> RedundantWriteBugType;
	mutable std::set<const VarDecl*> GlobalVarDecl;
};

} // end namespace ento
} // end namespace clang