////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file NamingCheck.h
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "clang/Frontend/FrontendActions.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
using namespace clang::ast_matchers;
using namespace clang;
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <algorithm>
#include <fstream>
#include <experimental/filesystem>
using namespace std;

#include "NameChecker.h"
#include "NamingCheckCallback.h"

/*
 * Only consume AST and hold two static members.
 */
class NamingCheckASTConsumer : public clang::ASTConsumer {
public:
    NamingCheckASTConsumer(std::shared_ptr<clang::Preprocessor> prep) {
        preprocessor = prep;
    }

    void HandleTranslationUnit(clang::ASTContext &Context) override;

    static void StaticInit(const string &dict_dir, const std::string &naming_prefix) {
        name_checker = std::make_shared<NameChecker>(dict_dir, naming_prefix);
        callback = std::make_shared<NamingCheckCallback>(name_checker);
    }

    static void StaticFinalization() {
    }
private:
    static std::shared_ptr<NameChecker> name_checker;
    static std::shared_ptr<NamingCheckCallback> callback;
    std::shared_ptr<clang::Preprocessor> preprocessor;
};
