////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
///\file StmtCheck.h
///\brief The tool that check stmt
///
///\version 0.1.0
///\author yixinyu<a href = "linkURL">yixinyu@mail.ustc.edu.cn</a>
////////////////////////////////////////////////////////////////////////////////

#ifndef __STMT_CHECK_H__
#define __STMT_CHECK_H__

#include <string>
#include <vector>
#include <regex>
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

class StmtCheckASTConsumer : public clang::ASTConsumer {

public:
    StmtCheckASTConsumer(std::vector<std::string> &sl);
    void HandleTranslationUnit(clang::ASTContext &Context) override;

private:
    clang::DiagnosticsEngine *DE;
    unsigned int WARNING_EXCEPTION_ID;
    unsigned int WARNING_RTTI_ID;
    unsigned int WARNING_C_STYLE_CAST_ID;
    unsigned int WARNING_OVLPOSTINC_ID;
    unsigned int WARNING_OVLPOSTDEC_ID;
    unsigned int WARNING_POSTINC_ID;
    unsigned int WARNING_POSTDEC_ID;
    unsigned int WARNING_LAMBDA_CAPTURES_ID;
    unsigned int WARNING_LAMBDA_RESULT_ID;
    unsigned int WARNING_LAMBDA_BODY_ID;
    unsigned int WARNING_ALLOCA_ID;
    unsigned int WARNING_NULL2INT_ID;
    unsigned int WARNING_NULL2FLOAT_ID;
    unsigned int WARNING_FLOATZERO2INT_ID;
    unsigned int WARNING_INTZERO2FLOAT_ID;
    unsigned int WARNING_INTZERO2POINTER_ID;
    unsigned int WARNING_CHARZERO2INT_ID;
    unsigned int WARNING_INTZERO2CHAR_ID;
    unsigned int WARNING_CHARZERO2FLOAT_ID;
    unsigned int WARNING_FLOATZERO2CHAR_ID;
    unsigned int WARNING_NULL2CHAR_ID;
    void setDiagnosticsEngineAndID(const clang::ASTContext &ctx);

    const clang::SourceManager *SM;
    int getline(const clang::SourceLocation SL, const clang::SourceManager &SM) const;
    std::vector<std::string> SourceList;
    bool isFileInSource(std::string &file);
    std::string getFullPathFromLocation(const clang::SourceLocation &location, const clang::SourceManager &sm);

    bool inVarDecl;
    void DFSTraverseDecls(llvm::iterator_range<clang::DeclContext::decl_iterator> decls);
    void DFSTraverseTree(clang::Stmt *&stmt, bool flag);

    bool isCharType(std::string type);
    clang::Stmt *removeParenRecursively(clang::Stmt *&stmt);
    void checkAlloca(clang::Stmt *&stmt);
    void checkException(clang::Stmt *&stmt);
    void checkRTTI(clang::Stmt *&stmt);
    void checkCStyleCast(clang::Stmt *&stmt);
    void checkPostIncDec(clang::Stmt *&stmt, bool flag);
    void checkOverloadedPostIncDec(clang::Stmt *&stmt);
    void checkLambda(clang::Stmt *&stmt);
    void checkZero(clang::Stmt *&stmt);


};

#endif //__STMT_CHECK_H__
