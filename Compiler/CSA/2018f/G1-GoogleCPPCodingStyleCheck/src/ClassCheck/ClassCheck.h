////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
///\file ClassCheck.h
///\brief The tool that check the style of class definition.
///
///\version 0.1.0
///\author ypliu<a href = "linkURL">ypliu88@mail.ustc.edu.cn</a>
///\author Citrine(srGui)<a href="mailto:agnesgsr@mail.ustc.edu.cn"> GsrMailBox </a>
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <unordered_set>
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
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

namespace clang{
namespace ento{

///It require at least two checker, why?
class ClassChecker: public Checker<
            check::ASTDecl<CXXRecordDecl>,
            check::PreCall,
            check::EndAnalysis
             >{

public:
    //ClassCheck();


public:
    void checkASTDecl(const CXXRecordDecl*, AnalysisManager &, BugReporter &)const;
    void checkPreCall(const CallEvent &call, CheckerContext &C)const;
    void checkEndAnalysis(ExplodedGraph &, BugReporter &, ExprEngine &)const;
private:
    ///\note customize function should be named with format different to callback function
    void cxxConstructorDeclCheck(const CXXConstructorDecl*, AnalysisManager &, BugReporter &)const;
    void cxxDestructorDeclCheck(const CXXDestructorDecl*,AnalysisManager &, BugReporter &) const;
    void cxxMethodDeclCheck(const CXXMethodDecl *, AnalysisManager &, BugReporter &)const;
    void baseClassCheck(const CXXRecordDecl *, const CXXBaseSpecifier *, AnalysisManager &, BugReporter &) const;
    void friendCheck(const FriendDecl *, AnalysisManager &, BugReporter &) const;
    void fieldDeclCheck(const FieldDecl*, AnalysisManager&, BugReporter&) const;

    void workInCtorCheck(const CallEvent &call, CheckerContext &C)const;
    void implicitConversionsCheck_Ctor(const CXXConstructorDecl*, AnalysisManager &, BugReporter &)const;
    void implicitConversionsCheck_Operator(const CXXMethodDecl*, AnalysisManager&,BugReporter&)const;
    ///FIXME: I am not sure what the type of the first parameter should is.
    void copyableAndMovableTypeCheck(const CXXConstructorDecl *, AnalysisManager &, BugReporter &)const;
    void structsVsClassCheck(const CXXRecordDecl*, AnalysisManager &, BugReporter &) const;
    void inheritanceCheck(const CXXRecordDecl *, const CXXBaseSpecifier*, AnalysisManager &, BugReporter &)const;
    void virtualspecifierCheck(const CXXMethodDecl *, AnalysisManager &, BugReporter &)const;
    void operatorOverloadingCheck(const CXXMethodDecl*, AnalysisManager &, BugReporter &)const;
    void accessControl(const FieldDecl*, AnalysisManager &, BugReporter &)const;
    void declarationOrder(const CXXRecordDecl *, AnalysisManager &, BugReporter &)const;
    ///
private:

};
}
}




