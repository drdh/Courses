#include "DeclCheck.h"

//#define __MYDEBUG__
#ifdef __MYDEBUG__
#define MYDEBUG(format, ...) printf(format, ##__VA_ARGS__)
#else
#define MYDEBUG(format, ...)
#endif

using namespace clang;
using namespace clang::ento;

///\brief Check for function declaration.
/// Check for:
///     1 any default arguments
///     2 any overloaded functions
///     3 unconstant reference arguments.
///     4 inline functions with more than 10 lines or contain loop or switch
void DeclChecker::checkASTDecl(
    const FunctionDecl *decl, AnalysisManager &Mgr, BugReporter &BR) const
{
    const auto &ctx = decl->getASTContext();
    auto &DE = ctx.getDiagnostics();
    MYDEBUG("CHECK FUNCTION: %s\n", decl->getNameAsString().c_str());

    

    /// Check for inline functions.
    if (decl->isInlineSpecified())
    {
        auto &sm = decl->getASTContext().getSourceManager();
        int startline = getline(decl->getLocStart(), sm);
        int endline = getline(decl->getLocEnd(), sm);
        if (endline - startline > 10)
        {
            const unsigned int ID = DE.getCustomDiagID(
                clang::DiagnosticsEngine::Warning,
                "Too long inline function, use 'inline' keyword for functions within 10 lines");
            DE.Report(decl->getBeginLoc(), ID);
        }
        /// Search AST for loop or switch.
        inlinestmtcheck(decl->getBody(), DE);
    }

    /// Check if it is overloaded.
    auto DC = decl->getDeclContext();
    auto result = const_cast<DeclContext *>(DC)->noload_lookup(decl->getDeclName());
    int i, overloadedflag = 0;
    for (i = 0; i < result.size(); i++)
    {
        MYDEBUG("%d", i);
        /// If a same-name function exists, might be an overloaded function.
        MYDEBUG(":%s\n", result[i]->getDeclKindName());
        if (result[i]->getKind() == clang::Decl::Kind::Function)
        {
            overloadedflag++;
        }
    }
    MYDEBUG("\n");

    /// overloadedflag is the number of same-name functions.
    if (overloadedflag > 1)
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Remark,
            "Using overloaded function, consider naming with parameter information");
        DE.Report(decl->getBeginLoc(), ID);
    }

    for (auto param : decl->parameters())
    {
        /// Check reference arguments.
        referencecheck(param, DE);
        /// Check if there is any default arguments.
        defaultcheck(param, DE);
    }
    MYDEBUG("FUNCTION CHECK END\n\n");
}

/// Get line number using pos information.
int DeclChecker::getline(const clang::SourceLocation SL, const clang::SourceManager &SM) const
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
    MYDEBUG("LINE:%d\n", line);
    return line;
}

///\brief Inline statements check. This check must do deep into AST to check ALL stmt nodes.
/// There are 20+ classes in "Stmt.h" for statements, we only go deep into nodes which may have
/// children.
void DeclChecker::inlinestmtcheck(const Stmt *stmt, clang::DiagnosticsEngine &DE) const
{
    if (stmt == nullptr)
        return;

    if (dyn_cast<SwitchStmt>(stmt))
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Using 'switch' statement in inline function");
        DE.Report(stmt->getBeginLoc(), ID);
    }

    if (dyn_cast<ForStmt>(stmt) || dyn_cast<WhileStmt>(stmt) || dyn_cast<DoStmt>(stmt))
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Using loop statement in inline function");
        DE.Report(stmt->getBeginLoc(), ID);
    }

    /// Stmt type for "{ ... }"
    if (dyn_cast<CompoundStmt>(stmt))
    {
        for (auto st : dyn_cast<CompoundStmt>(stmt)->body())
            inlinestmtcheck(st, DE);
    }

    /// LabelStmt - Represents a label, which has a substatement.  For example:
    ///    foo: return;
    if (auto st = dyn_cast<LabelStmt>(stmt))
        inlinestmtcheck(st->getSubStmt(), DE);

    /// Represents an attribute applied to a statement. For example:
    ///   [[omp::for(...)]] for (...) { ... }
    if (auto st = dyn_cast<AttributedStmt>(stmt))
        inlinestmtcheck(st->getSubStmt(), DE);

    /// IfStmt - This represents an if/then/else.
    if (auto st = dyn_cast<IfStmt>(stmt))
    {
        inlinestmtcheck(st->getThen(), DE);
        inlinestmtcheck(st->getElse(), DE);
    }

    /// Finally statement.
    if (auto st = dyn_cast<SEHFinallyStmt>(stmt))
        inlinestmtcheck(st->getBlock(), DE);

    ///Try statement. It has two children - TRY and HANDLER
    if (auto st = dyn_cast<SEHTryStmt>(stmt))
    {
        inlinestmtcheck(st->getHandler(), DE);
        inlinestmtcheck(st->getTryBlock(), DE);
    }
}

/// Reference arguments check.
void DeclChecker::referencecheck(
    clang::ParmVarDecl *param, clang::DiagnosticsEngine &DE) const
{
    MYDEBUG("%s:\n", param->getNameAsString().c_str());
    MYDEBUG("Is param reference?%d Constant?%d\n",
            param->getType()->isLValueReferenceType(),
            param->getType().isConstQualified() ||
                param->getType().getNonReferenceType().isConstQualified());
    if (param->getType()->isLValueReferenceType() &&
        !param->getType().getNonReferenceType().isConstQualified())
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Not-constant reference argument, consider labeling 'const' or using pointer instead");
        DE.Report(param->getBeginLoc(), ID);
    }
}

/// Default arguments check.
void DeclChecker::defaultcheck(
    clang::ParmVarDecl *param, clang::DiagnosticsEngine &DE) const
{
    /// If a argument has an initializer
    if (param->getInit())
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Using unrecommended default arguments, consider overloading the function instead");
        DE.Report(param->getBeginLoc(), ID);
    }
}

///\brief Check for variable declaration.
/// Check for:
///     1 use unsigned int
///     2 use 'auto' type globally
///     3 use 'auto' for init list
void DeclChecker::checkASTDecl(
    const VarDecl *decl, AnalysisManager &Mgr, BugReporter &BR) const
{
    MYDEBUG("CHECK VAR: %s\n", decl->getNameAsString().c_str());
    const auto &ctx = decl->getASTContext();
    auto &DE = ctx.getDiagnostics();
    intcheck(decl, DE);
    autocheck(decl, DE);
    MYDEBUG("VAR CHECK END\n\n");
}

/// Check for "unsigned int"
void DeclChecker::intcheck(
    const VarDecl *decl, clang::DiagnosticsEngine &DE) const
{
    auto type = decl->getType();
    if (type->isIntegerType() && type->isUnsignedIntegerType() &&
        !type->isEnumeralType())
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Using unsigned interger, may causes unexpected result");
        DE.Report(decl->getBeginLoc(), ID);
    }
}

/// Check for "auto"
void DeclChecker::autocheck(
    const VarDecl *decl, clang::DiagnosticsEngine &DE) const
{
    MYDEBUG("%s VARDECL:%d,", decl->getNameAsString().c_str(),
            decl->getType()->getTypeClass());
    /// Check if declare an "auto" type, not-local variable.
    if (
        decl->getType()->getTypeClass() == clang::Type::TypeClass::Auto &&
        !decl->isLocalVarDecl())
    {
        const unsigned int ID = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Using 'auto' globally, consider specifying the type");
        DE.Report(decl->getBeginLoc(), ID);
    }
    ///\brief Check if use initialization list for "auto" type.
    /// Current solution is check AST if the initialization nodes
    /// are "ExprWithCleanups"--"CXXStdInitializerListExpr".
    /// This solution is based on tests since the initialization node type
    /// is NOT "InitListExpr" when the declaration type is "auto".
    if (decl->getType()->getTypeClass() == clang::Type::TypeClass::Auto &&
        decl->hasInit())
    {
        if (auto temp = dyn_cast<ExprWithCleanups>(decl->getInit()))
        {
            if (dyn_cast<CXXStdInitializerListExpr>(temp->getSubExpr()))
            {
                MYDEBUG("is init list\n");
                const unsigned int ID = DE.getCustomDiagID(
                    clang::DiagnosticsEngine::Warning,
                    "Using listinit to initialize 'auto' type, consider specifying the type");
                DE.Report(decl->getBeginLoc(), ID);
            }
        }
    }
}
