////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file FullCommentCheck.h
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang, <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
using namespace clang;

#include <set>

/*
 * 函数头注释检查
 *
 * 比赛要求：
 *     函数命名无法表达的信息需要加函数头注释辅助说明。
 *     函数头注释放在函数声明或定义上方。
 *     注释风格应统一。
 *     按需写函数头注释。
 *     不写空有格式的函数头。
 *     函数头注释内容可选，不限于：
 *         功能说明
 *         返回值
 *         性能约束
 *         用法
 *         内存约定
 *         算法实现
 *         可重入性
 *     模块对外头文件中的函数接口声明，其函数头注释，应当将重要、有用的信息表达清楚。
 * 
 * 函数命名无法表达的信息加头注释辅助说明：
 *     如果函数有指针类型的形参，则要求对指针有所说明。我们在注释内容中直接查找指针变量的名称是否存在。
 * 
 * 函数头注释放在声明或定义的上方：
 *     已经由 clang 满足。
 * 
 * 注释风格检查：
 *     只在首次发现注释风格不统一时报出错误。
 *     我们假定每个头注释自身是格式统一的。
 * 
 * 判断函数头是否空有格式：
 *     查找形如“冒号+空白或字符串结束”是否存在。
 * 
 * 按需写头注释：
 *     如果函数体的定义超过一定长度（默认为 15），则要求提供函数头注释。
 * 
 * 模块对外头文件中的函数接口声明：
 *     预留接口，尚未实现
 */
class FullCommentCheckASTConsumer : public clang::ASTConsumer {
public:
    void HandleTranslationUnit(clang::ASTContext &Context) override;
private:
    bool slash_star_comment_exist = false, double_slash_comment_exist = false;

    void check(clang::FunctionDecl *decl);
    bool check_pointer_return(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_pointer_param(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_useless_format(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_style_inconsistency(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_need_comment(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_redundant_function_name(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_mix_style(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
    bool check_slash_star_misuse(const FunctionDecl &decl, const ASTContext &ctx, DiagnosticsEngine &DE);
};