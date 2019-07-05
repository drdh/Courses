#pragma once

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "NameChecker.h"

/*
 * Simply do type conversion and pass check to checker. Because the checker might work on preprocess.
 */
class NamingCheckCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    NamingCheckCallback(std::shared_ptr<NameChecker> checker)
    : name_checker(checker) {
    }

    virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) final {
        // Deliver checks
        if (const auto decl = Result.Nodes.getNodeAs<clang::FunctionDecl>("functionDecl")) {
            name_checker->check(decl);
        } else if (const auto decl = Result.Nodes.getNodeAs<clang::VarDecl>("varDecl")) {
            name_checker->check(decl);
        } else if (const auto decl = Result.Nodes.getNodeAs<clang::TypedefDecl>("typedefDecl")) {
            name_checker->check(decl);
        } else if (const auto decl = Result.Nodes.getNodeAs<clang::EnumDecl>("enumDecl")) {
            name_checker->check(decl);
        } else if (const auto decl = Result.Nodes.getNodeAs<clang::EnumConstantDecl>("enumConstantDecl")) {
            name_checker->check(decl);
        }
    }
private:
    std::shared_ptr<NameChecker> name_checker;
};