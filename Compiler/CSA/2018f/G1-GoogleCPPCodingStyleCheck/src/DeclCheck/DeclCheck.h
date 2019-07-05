////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
///\file ClassCheck.h
///\brief The tool that check the style of class definition.
///
///\version 0.1.0
///\author Zihan Wang<a href = "linkURL">wzh99@mail.ustc.edu.cn</a>
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <regex>
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

namespace clang {
namespace ento {
class DeclChecker : public Checker<
	check::ASTDecl<VarDecl>, check::ASTDecl<FunctionDecl>
	> {
  public:
//    DeclChecker();
    void checkASTDecl(const VarDecl *D, AnalysisManager &Mgr, BugReporter &BR) const;
    void checkASTDecl(const FunctionDecl *D, AnalysisManager &Mgr, BugReporter &BR) const;

  private:
    void referencecheck(clang::ParmVarDecl *param, clang::DiagnosticsEngine &DE)const;
    void defaultcheck(clang::ParmVarDecl *param, clang::DiagnosticsEngine &DE)const;
    void intcheck(const VarDecl *decl, clang::DiagnosticsEngine &DE)const;
    void autocheck(const VarDecl *decl, clang::DiagnosticsEngine &DE)const;
    void inlinestmtcheck(const Stmt *stmt, clang::DiagnosticsEngine &DE)const;
    int getline(const clang::SourceLocation SL, const clang::SourceManager &SM)const;
};
}
}
