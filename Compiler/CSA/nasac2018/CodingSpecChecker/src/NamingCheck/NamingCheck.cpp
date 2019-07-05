////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file NamingCheck.cpp
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#include "NamingCheck.h"
#include "NamingCheckCallback.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"

std::shared_ptr<NameChecker> NamingCheckASTConsumer::name_checker;
std::shared_ptr<NamingCheckCallback> NamingCheckASTConsumer::callback;

void NamingCheckASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    // Check macro names
    for (auto &macro: preprocessor->macros()) {
        auto info = macro.first;
        std::string name = info->getName().str();
        auto macro_info = preprocessor->getMacroInfo(info);
        if (macro_info == nullptr)
            continue;
        if (macro_info->isBuiltinMacro())
            continue;
        if (preprocessor->getSourceManager().isInSystemHeader(macro_info->getDefinitionLoc()))
            continue;
        if (name_checker->is_valid_name(name))
            continue;
        const unsigned ID = preprocessor->getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Warning,
                        "`%0` is neither an English word nor a common abbreviation");
        auto DB = preprocessor->getDiagnostics().Report(macro_info->getDefinitionLoc(), ID);
        DB.AddString(name);
    }

    // Create a match finder that matches all declarations and apply name checker when match
    MatchFinder m;
    m.addMatcher(varDecl().bind("varDecl"), callback.get());
    m.addMatcher(functionDecl().bind("functionDecl"), callback.get());
    m.addMatcher(typedefDecl().bind("typedefDecl"), callback.get());
    m.addMatcher(enumDecl().bind("enumDecl"), callback.get());
    m.addMatcher(enumConstantDecl().bind("enumConstantDecl"), callback.get());

    m.matchAST(Context);
}
