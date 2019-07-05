////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file FullCommentCheck.cpp
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#include "FullCommentCheck.h"
#include <regex>
using namespace std;

using namespace clang;

// TODO: a temporary solution
typedef unsigned fuid_t;
static std::set<fuid_t> scanned_headers;

void FullCommentCheckASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    std::set<fuid_t> headers_this_run;

    for (auto decl: Context.getTranslationUnitDecl()->decls()) {
        auto func_decl = dyn_cast<FunctionDecl>(decl);
        if (func_decl == nullptr)
            continue;
        
        auto &sm = Context.getSourceManager();
        std::string fname = sm.getFilename(decl->getLocation()).str();
        if (!fname.empty()) {  // If this is not a file in memory, check if it has appeared
            auto uid = sm.getFileEntryForID(sm.getFileID(decl->getLocation()))->getUID();
            if (scanned_headers.find(uid) != scanned_headers.end())
                continue;
            headers_this_run.insert(uid);
        }

        // Check decls that are
        // * not from system header
        if (sm.isInSystemHeader(decl->getBeginLoc()))
            continue;
        // * is externally visible
        if (!func_decl->isExternallyVisible())
            continue;
        check(func_decl);
    }
    scanned_headers.merge(headers_this_run);
}

void FullCommentCheckASTConsumer::check(clang::FunctionDecl *decl) {
    const auto &ctx = decl->getASTContext();
    auto &DE = ctx.getDiagnostics();

    // TODO: merge these
    check_mix_style(*decl, ctx, DE);
    check_style_inconsistency(*decl, ctx, DE);
    check_useless_format(*decl, ctx, DE);
    check_need_comment(*decl, ctx, DE);
    check_pointer_param(*decl, ctx, DE);
    check_pointer_return(*decl, ctx, DE);
    check_redundant_function_name(*decl, ctx, DE);
    check_slash_star_misuse(*decl, ctx, DE);
}

bool FullCommentCheckASTConsumer::check_slash_star_misuse(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
    if (raw_comment == nullptr)
        return false;
    std::string raw_text = raw_comment->getRawText(ctx.getSourceManager()).str();
    std::regex style("\\*/\\n");
    if (std::regex_search(raw_text, style)) {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "misuse /* comment style in function comment, consider using //"
        );
        DE.Report(decl.getBeginLoc(), ID);
        return true;
    }
    return false;
}

bool FullCommentCheckASTConsumer::check_mix_style(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
    if (raw_comment == nullptr)
        return false;
    std::string raw_text = raw_comment->getRawText(ctx.getSourceManager()).str();
    std::regex style1("//");
    std::regex style2("/\\*");
    if (std::regex_search(raw_text, style1) && std::regex_search(raw_text, style2)) {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "mix comment style in function annotation"
        );
        DE.Report(decl.getBeginLoc(), ID);
        return true;
    }
    return false;
}

#include <regex>

bool FullCommentCheckASTConsumer::check_redundant_function_name(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
    if (raw_comment == nullptr)
        return false;
    std::regex reg(":\\s?" + decl.getNameAsString());
    if (std::regex_search(raw_comment->getRawText(ctx.getSourceManager()).str(), reg)) {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "redundant function name in comment"
        );
        DE.Report(decl.getBeginLoc(), ID);
        return true;
    }
    return false;
}

bool FullCommentCheckASTConsumer::check_need_comment(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    if (ctx.getRawCommentForDeclNoCache(&decl) != nullptr)
        return false;
    if (decl.isExternallyVisible() && (decl.getNameAsString() != "main")) {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Remark,
            "might need annotation"
        );
        DE.Report(decl.getBeginLoc(), ID);
        return true;
    }
    return false;
}

bool FullCommentCheckASTConsumer::check_pointer_return(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    if (!decl.getReturnType()->isPointerType())
        return false;
    const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
    // Check if the comment describes the returned value
    if (raw_comment != nullptr && regex_search(raw_comment->getRawText(ctx.getSourceManager()).str(), regex("(返回|return|Return|RETURN)")))
        return false;
    const unsigned int ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Remark,
        "might need documentation about return value"
    );
    DE.Report(decl.getBeginLoc(), ID);
    return true;
}

bool FullCommentCheckASTConsumer::check_pointer_param(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    bool ret = false;
    for (auto &param: decl.parameters()) {
        // Only check non-function, non-array, non-const pointer params
        if (!(param->getType()->isPointerType() && !param->getType()->isArrayType() && !param->getType()->isFunctionPointerType() && !param->getType()->getPointeeType().isConstQualified())) {
            continue;
        } else {
            const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
            if (raw_comment != nullptr && regex_search(raw_comment->getRawText(ctx.getSourceManager()).str(), regex(param->getNameAsString())))
                continue;
        }
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Remark,
            "might need comment about pointer `%0`"
        );
        auto DB = DE.Report(param->getLocation(), ID);
        DB.AddString(param->getNameAsString());
        ret = true;
    }
    return ret;
}

bool FullCommentCheckASTConsumer::check_useless_format(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    if (ctx.getCommentForDecl(&decl, nullptr) == nullptr)
        return false;
    regex reg("(：|:)(\\s*\\n|$)");
    const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
    if (raw_comment == nullptr || !regex_search(raw_comment->getRawText(ctx.getSourceManager()).str(), reg))
        return false;

    const unsigned int ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "function header comment contains only format but no content"
    );
    DE.Report(decl.getNameInfo().getBeginLoc(), ID);

    return true;
}

bool FullCommentCheckASTConsumer::check_style_inconsistency(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE) {
    // If both exist, don't report any more
    if (slash_star_comment_exist && double_slash_comment_exist)
        return false;

    const auto *raw_comment = ctx.getRawCommentForDeclNoCache(&decl);
    if (raw_comment == nullptr)
        return false;
    if (raw_comment->getRawText(ctx.getSourceManager()).str()[1] == '*') {
        slash_star_comment_exist = true;
    } else {
        double_slash_comment_exist = true;
    }

    if (!(slash_star_comment_exist && double_slash_comment_exist))  // Consistent
        return false;

    const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Remark,
            "function header comment differ from previous ones"
        );
    DE.Report(decl.getNameInfo().getBeginLoc(), ID);
    return true;
}
