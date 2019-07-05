////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file ModuleCheck.h
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

// ///
// /// @brief
// ///
// class ModuleCheckASTConsumer : public clang::ASTConsumer {
// public:
//   void HandleTranslationUnit(clang::ASTContext &Context) override;
// };

namespace clang
{
namespace ento
{

class ModuleChecker : public Checker<
						  check::BeginFunction,
						  check::PreStmt<CallExpr>,
						  check::PostStmt<CallExpr>, check::BranchCondition>
{

  public:
	ModuleChecker();
	void checkBeginFunction(CheckerContext &Ctx) const;
	void checkPostStmt(const CallExpr *DS, CheckerContext &Ctx) const;
	void checkPreStmt(const CallExpr *CE, CheckerContext &Ctx) const;
	void checkBranchCondition(const Stmt *Condition, CheckerContext &Ctx) const;

  private:
	mutable bool NeedCheckParam;
	void reportReadUncheckedBug(ExplodedNode *ErrNode, CheckerContext &Ctx) const;
	void reportPassUncheckedParamBug(ExplodedNode *ErrNode, CheckerContext &Ctx) const;

	void TryToRemoveTaintDueToCheck(const Stmt *Condition, CheckerContext &Ctx) const;
	void TryToReportReduntantParamCheck(const Stmt *Condition, CheckerContext &Ctx) const;

	std::unique_ptr<BugType> ReadUncheckedBugType;
	std::unique_ptr<BugType> PassUncheckedParamBugType;
};

} // end namespace ento
} // end namespace clang
