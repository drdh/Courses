#include "StmtCheck.h"

using namespace std;
using namespace clang;

#define MAX_LINENUM_IN_LAMBDA_BODY 5

StmtCheckASTConsumer::StmtCheckASTConsumer(std::vector<std::string> &sl){
    SourceList = sl;
    DE = nullptr;
}

void StmtCheckASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    DFSTraverseDecls(Context.getTranslationUnitDecl()->decls());
}

void StmtCheckASTConsumer::DFSTraverseDecls(llvm::iterator_range<clang::DeclContext::decl_iterator> decls) {
    for (auto decl: decls) {

        const auto &ctx = decl->getASTContext();
        setDiagnosticsEngineAndID(ctx);

        SM = &ctx.getSourceManager();

        string path = getFullPathFromLocation(
            decl->getLocation(),
            *SM
        );
        if(!isFileInSource(path)) continue;

        auto func_decl = dyn_cast<FunctionDecl>(decl);
        if (func_decl != nullptr){
            auto stmts = decl->getBody();
            // stmts->dump();
            if(stmts == nullptr) continue;
            for(auto child: stmts->children()){
                DFSTraverseTree(child, false);
            }
        }

        auto method_decl = dyn_cast<CXXMethodDecl>(decl);
        if (method_decl != nullptr){
            auto stmts = method_decl->getBody();
            if(stmts == nullptr) continue;
            for(auto child: stmts->children()){
                DFSTraverseTree(child, false);
            }
        }

        auto record_decl = dyn_cast<CXXRecordDecl>(decl);
        if (record_decl != nullptr){
            for(auto method: record_decl->methods()){
                auto stmts = method->getBody();
                if(stmts == nullptr) continue;
                for(auto child: stmts->children()){
                    DFSTraverseTree(child, false);
                }
            }
        }

        auto namespace_decl = dyn_cast<NamespaceDecl>(decl);
        if (namespace_decl != nullptr){
            DFSTraverseDecls(namespace_decl->decls());
//            if(!namespace_decl->isStdNamespace()){      // exclude std namespace
//                if(namespace_decl->getNameAsString() != "<other_exclude_name_space>"){
//                    DFSTraverseDecls(namespace_decl->decls());
//                }
//            }
        }
    }
}

bool StmtCheckASTConsumer::isFileInSource(std::string &file){
    auto source_begin = SourceList.begin();
    auto source_end = SourceList.end();
    while(source_begin != source_end){
        if(source_begin->compare(file)==0)
            return true;
        source_begin++;
    }
    return false;
}

std::string StmtCheckASTConsumer::getFullPathFromLocation(const clang::SourceLocation &location, const clang::SourceManager &sm){
    auto loc_str = location.printToString(sm);
    int cut_index = loc_str.find(':');
    string cut_loc = loc_str.substr(0,cut_index);
    return std::move(cut_loc);
}

/// Get line number using pos information.
int StmtCheckASTConsumer::getline(const clang::SourceLocation SL, const clang::SourceManager &SM) const
{
    auto loc = SL.printToString(SM);
    /// The location message is something like "^/PATH/name.cpp:line:pos$".
    /// Get ":line:pos$".
    std::regex pattern(":[0-9]+:[0-9]+$");
    std::smatch result;
    std::regex_search(loc, result, pattern);
    int line, pos, i = 1;
    /// Get "line"
    sscanf(result[0].str().c_str(), ":%d:%d", &line, &pos);
    return line;
}

void StmtCheckASTConsumer::setDiagnosticsEngineAndID(const clang::ASTContext &ctx){
    DE = &ctx.getDiagnostics();

    WARNING_EXCEPTION_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using unrecommended C++ exception"
    );

    WARNING_RTTI_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using unrecommended C++ RTTI"
    );

    WARNING_C_STYLE_CAST_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using unrecommended C-style cast"
    );

    WARNING_OVLPOSTINC_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Prefix ++ is recommended when the operand is an object"
    );

    WARNING_OVLPOSTDEC_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Prefix -- is recommended when the operand is an object"
    );

    WARNING_POSTINC_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Prefix ++ is recommended when not using the expression's value"
    );

    WARNING_POSTDEC_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Prefix -- is recommended when not using the expression's value"
    );

    WARNING_LAMBDA_CAPTURES_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Explicit parameters are recommended"
    );

    WARNING_LAMBDA_RESULT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Explicit result type is recommended if it is readable"
    );

    WARNING_LAMBDA_BODY_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Lambda body is too large, better to assign it to an object or use functions instead"
    );

    WARNING_ALLOCA_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using unrecommended function alloca()"
    );

    WARNING_NULL2INT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using NULL as integer. 0 is recommended"
    );

    WARNING_NULL2FLOAT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using NULL as float. 0.0 is recommended"
    );

    WARNING_FLOATZERO2INT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using 0.0 as integer. 0 is recommended"
    );

    WARNING_INTZERO2FLOAT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using 0 as float. 0.0 is recommended"
    );

    WARNING_CHARZERO2INT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using '\\0' as int. 0 is recommended"
    );

    WARNING_INTZERO2CHAR_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using 0 as char. '\\0' is recommended"
    );

    WARNING_FLOATZERO2CHAR_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using 0.0 as char. '\\0' is recommended"
    );

    WARNING_CHARZERO2FLOAT_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using '\\0' as float. 0.0 is recommended"
    );

    WARNING_INTZERO2POINTER_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using 0 as pointer. nullptr is recommended"
    );

    WARNING_NULL2CHAR_ID = DE->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Using NULL as char. '\\0' is recommended"
    );
}

void StmtCheckASTConsumer::checkException(clang::Stmt *&stmt){
    // throw
    auto throwexprstmt = dyn_cast<CXXThrowExpr>(stmt);
    if(throwexprstmt != nullptr){
        DiagnosticBuilder builder = DE->Report(throwexprstmt->getBeginLoc(), WARNING_EXCEPTION_ID);
        builder.setForceEmit();
    }

    // try
    auto trystmt = dyn_cast<CXXTryStmt>(stmt);
    if(trystmt != nullptr){
        DiagnosticBuilder builder = DE->Report(trystmt->getBeginLoc(),WARNING_EXCEPTION_ID);
        builder.setForceEmit();
    }

    // catch
    auto catchstmt = dyn_cast<CXXCatchStmt>(stmt);
    if(catchstmt != nullptr){
        DiagnosticBuilder builder = DE->Report(catchstmt->getBeginLoc(),WARNING_EXCEPTION_ID);
        builder.setForceEmit();
    }
}

void StmtCheckASTConsumer::checkRTTI(clang::Stmt *&stmt){
    // dynamic_cast
    auto dyncaststmt = dyn_cast<CXXDynamicCastExpr>(stmt);
    if(dyncaststmt != nullptr){
        DiagnosticBuilder builder = DE->Report(dyncaststmt->getBeginLoc(), WARNING_RTTI_ID);
        builder.setForceEmit();
    }

    // typeid
    auto typeidstmt = dyn_cast<CXXTypeidExpr>(stmt);
    if(typeidstmt != nullptr){
        DiagnosticBuilder builder = DE->Report(typeidstmt->getBeginLoc(),WARNING_RTTI_ID);
        builder.setForceEmit();
    }
}

void StmtCheckASTConsumer::checkCStyleCast(clang::Stmt *&stmt){
    // C-style cast
    auto cstylecaststmt = dyn_cast<CStyleCastExpr>(stmt);
    if(cstylecaststmt != nullptr){
        DiagnosticBuilder builder = DE->Report(cstylecaststmt->getBeginLoc(),WARNING_C_STYLE_CAST_ID);
        builder.setForceEmit();
    }
}

void StmtCheckASTConsumer::checkOverloadedPostIncDec(clang::Stmt *&stmt){
    // overloaded ++ | --
    auto ovlunopstmt = dyn_cast<CXXOperatorCallExpr>(stmt);
    if(ovlunopstmt != nullptr){
        OverloadedOperatorKind op = ovlunopstmt->getOperator();
        if(op == OO_PlusPlus){
            if(! (ovlunopstmt->getOperatorLoc() < ovlunopstmt->getEndLoc())){   // postfix ++
                DiagnosticBuilder builder = DE->Report(ovlunopstmt->getBeginLoc(),WARNING_OVLPOSTINC_ID);
                builder.setForceEmit();
            }
        }
        else if(op == OO_MinusMinus){
            if(! (ovlunopstmt->getOperatorLoc() < ovlunopstmt->getEndLoc())){   // postfix --
                unsigned int ID = DE->getCustomDiagID(
                    clang::DiagnosticsEngine::Warning,
                    "Prefix -- is recommended when the operand is an object"
                );
                DiagnosticBuilder builder = DE->Report(ovlunopstmt->getBeginLoc(),WARNING_OVLPOSTDEC_ID);
                builder.setForceEmit();
            }
        }
    }
}

void StmtCheckASTConsumer::checkPostIncDec(clang::Stmt *&stmt, bool flag){
    // ++ | --
    auto unaryopstmt = dyn_cast<UnaryOperator>(stmt);
    if(unaryopstmt != nullptr){
        if(unaryopstmt->isPostfix()){
            if(unaryopstmt->isIncrementOp()){   // a ++
                if(flag == false){
                    DiagnosticBuilder builder = DE->Report(unaryopstmt->getBeginLoc(),WARNING_POSTINC_ID);
                    builder.setForceEmit();
                }
            }
            else {  // a --
                if(flag == false){
                    DiagnosticBuilder builder = DE->Report(unaryopstmt->getBeginLoc(),WARNING_POSTDEC_ID);
                    builder.setForceEmit();
                }
            }
        }
    }
}

void StmtCheckASTConsumer::checkLambda(clang::Stmt *&stmt){
    // LambdaExpr
    auto lambdastmt = dyn_cast<LambdaExpr>(stmt);
    if(lambdastmt != nullptr){
        // check capture list
        if(lambdastmt->implicit_capture_begin() < lambdastmt->implicit_capture_end()){
            DiagnosticBuilder builder = DE->Report(lambdastmt->getBeginLoc(),WARNING_LAMBDA_CAPTURES_ID);
            builder.setForceEmit();
        }

//        // show warning where implicit captures occur (in function body)
//        for(auto ic: lambdastmt->implicit_captures()){
//            unsigned int ID = DE->getCustomDiagID(
//                clang::DiagnosticsEngine::Note,
//                "Implicit parameter"
//            );
//            DiagnosticBuilder builder = DE->Report(ic.getLocation(),ID);
//            builder.setForceEmit();
//        }

        // check result type
        if(!lambdastmt->hasExplicitResultType()){
            DiagnosticBuilder builder = DE->Report(lambdastmt->getBeginLoc(),WARNING_LAMBDA_RESULT_ID);
            builder.setForceEmit();
        }

        // check line number
        // emit warning when not wrapped in vardecl, or wrapped in a callexpr inside vardecl
        CompoundStmt *body = lambdastmt->getBody();
        int linenum = getline(body->getEndLoc(), *SM) - getline(body->getBeginLoc(), *SM) - 1;
        if(!inVarDecl && linenum > MAX_LINENUM_IN_LAMBDA_BODY){
            DiagnosticBuilder builder = DE->Report(body->getBeginLoc(),WARNING_LAMBDA_BODY_ID);
            builder.setForceEmit();
        }
    }
}

void StmtCheckASTConsumer::checkAlloca(clang::Stmt *&stmt){
    auto funcstmt = dyn_cast<CallExpr>(stmt);
    if(funcstmt != nullptr){
        FunctionDecl *callee = funcstmt->getDirectCallee();
        if(callee != NULL){
            // function name is alloca
            string funcname = callee->getNameAsString();
            if(funcname == "__builtin_alloca" || funcname == "alloca" || funcname == "_alloca"){
                DiagnosticBuilder builder = DE->Report(funcstmt->getBeginLoc(),WARNING_ALLOCA_ID);
                builder.setForceEmit();
            }
        }
    }

}

clang::Stmt *StmtCheckASTConsumer::removeParenRecursively(clang::Stmt *&stmt){
    for(auto child: stmt->children()){
        auto childstmt = dyn_cast<ParenExpr>(child);
        if(childstmt == nullptr) return child;
        else return removeParenRecursively(child);
    }
}

bool StmtCheckASTConsumer::isCharType(std::string type){
    return (type == "char") || (type == "unsigned char");
}

void StmtCheckASTConsumer::checkZero(clang::Stmt *&stmt){
    auto caststmt = dyn_cast<ImplicitCastExpr>(stmt);
    if(caststmt != nullptr){
        string targetType = caststmt->getType().getAsString();
        auto childstmt = removeParenRecursively(stmt);
        unsigned int id = 0;
        switch(caststmt->getCastKind()){

            case CK_IntegralCast:           // NULL to int
                if(dyn_cast<GNUNullExpr>(childstmt) != nullptr) {
                    if(isCharType(targetType)) id = WARNING_NULL2CHAR_ID;
                    else id = WARNING_NULL2INT_ID;
                }
                if(
                    dyn_cast<CharacterLiteral>(childstmt) != nullptr &&
                    dyn_cast<CharacterLiteral>(childstmt)->getValue() == 0
                )
                    id = WARNING_CHARZERO2INT_ID;
                if(
                    dyn_cast<IntegerLiteral>(childstmt) != nullptr &&
                    dyn_cast<IntegerLiteral>(childstmt)->getValue().isNullValue() &&
                    isCharType(targetType)
                )
                    id = WARNING_INTZERO2CHAR_ID;

                break;

            case CK_NullToPointer:
                if(
                    dyn_cast<IntegerLiteral>(childstmt) != nullptr &&
                    dyn_cast<IntegerLiteral>(childstmt)->getValue().isNullValue()
                )
                    id = WARNING_INTZERO2POINTER_ID;
                break;

            case CK_FloatingToIntegral:
                if(
                    dyn_cast<FloatingLiteral>(childstmt) != nullptr &&
                    dyn_cast<FloatingLiteral>(childstmt)->getValue().isZero()
                ){
                    if(isCharType(targetType)) id = WARNING_FLOATZERO2CHAR_ID;
                    else id = WARNING_FLOATZERO2INT_ID;;
                }

                break;

            case CK_IntegralToFloating:
                if(dyn_cast<GNUNullExpr>(childstmt) != nullptr)
                    id = WARNING_NULL2FLOAT_ID;
                if(
                    dyn_cast<IntegerLiteral>(childstmt) != nullptr &&
                    dyn_cast<IntegerLiteral>(childstmt)->getValue().isNullValue()
                )
                    id = WARNING_INTZERO2FLOAT_ID;
                if(
                    dyn_cast<CharacterLiteral>(childstmt) != nullptr &&
                    dyn_cast<CharacterLiteral>(childstmt)->getValue() == 0
                )
                    id = WARNING_CHARZERO2FLOAT_ID;
                break;
        }

        if(id != 0){
            DiagnosticBuilder builder = DE->Report(childstmt->getBeginLoc(), id);
            builder.setForceEmit();
        }
    }
}

// flag: does parent use the value generated by children
void StmtCheckASTConsumer::DFSTraverseTree(clang::Stmt *&stmt, bool flag) {
    if(stmt == nullptr) return;

    checkAlloca(stmt);
    checkException(stmt);
    checkRTTI(stmt);
    checkCStyleCast(stmt);
    checkPostIncDec(stmt, flag);
    checkOverloadedPostIncDec(stmt);
    checkLambda(stmt);
    checkZero(stmt);

    if(dyn_cast<DeclStmt>(stmt) != nullptr) inVarDecl = true;
    if(dyn_cast<CallExpr>(stmt) != nullptr) inVarDecl = false;

    // DFS traverse AST
    for(auto child: stmt->children()){
        DFSTraverseTree(
            child,
            // most cases that children's value is useless
            dyn_cast<ForStmt>(stmt) == nullptr &&
            dyn_cast<CompoundStmt>(stmt) == nullptr &&
            dyn_cast<IfStmt>(stmt) == nullptr &&
            dyn_cast<WhileStmt>(stmt) == nullptr &&
            dyn_cast<DefaultStmt>(stmt) == nullptr &&
            dyn_cast<LabelStmt>(stmt) == nullptr &&
            dyn_cast<CaseStmt>(stmt) == nullptr &&
            dyn_cast<DoStmt>(stmt) == nullptr
        );
    }

    if(dyn_cast<DeclStmt>(stmt) != nullptr) inVarDecl = false;
}
