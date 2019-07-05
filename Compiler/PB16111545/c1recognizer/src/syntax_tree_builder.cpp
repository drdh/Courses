
#include "syntax_tree_builder.h"
#include <memory>

using namespace c1_recognizer;
using namespace c1_recognizer::syntax_tree;

syntax_tree_builder::syntax_tree_builder(error_reporter &_err) : err(_err) {}

antlrcpp::Any syntax_tree_builder::visitCompilationUnit(C1Parser::CompilationUnitContext *ctx)
{
    //std::cout<<"visitCompilationUnit"<<std::endl;
    //compilationUnit: (decl | funcdef)+ EOF ;
    auto result = new assembly; // Root node of an ordinary syntax tree.
    result->line = ctx->getStart()->getLine();
    result->pos = ctx->getStart()->getCharPositionInLine(); //设置行、列

    //遍历所有子节点
    for(auto &child : ctx->children){
        //decl
        if (antlrcpp::is<C1Parser::DeclContext*>(child)){
            auto Decl_list = visit(child).as<std::vector<var_def_stmt_syntax*>>();
            for (auto &decl : Decl_list)
                result->global_defs.push_back(static_cast<ptr<var_def_stmt_syntax>>(decl));
        }
        //funcdef
        else if (antlrcpp::is<C1Parser::FuncdefContext*>(child)){
            //result->global_defs.push_back(visit(child).as<func_def_syntax *>());
            //auto temp = static_cast<C1Parser::FuncdefContext*>(child);
            auto func = visit(child).as<func_def_syntax*>();
            result->global_defs.push_back(static_cast<ptr<func_def_syntax>>(func));
        }
    }
    return result;
}

antlrcpp::Any syntax_tree_builder::visitDecl(C1Parser::DeclContext *ctx)
{
    //std::cout<<"visitDecl"<<std::endl;
    //decl:  constdecl | vardecl;
    //constdecl
    if (auto constdecl = ctx->constdecl())
        return visit(constdecl);
    //vardecl
    else 
        return visit(ctx->vardecl());
}

antlrcpp::Any syntax_tree_builder::visitConstdecl(C1Parser::ConstdeclContext *ctx)
{
    //std::cout<<"visitConstdecl"<<std::endl;
    //constdecl:  Const (Int | Float) constdef (Comma constdef)* SemiColon;
    std::vector<var_def_stmt_syntax*>result;
    for(auto &child:ctx->constdef()){
        auto temp=visit(child).as<var_def_stmt_syntax *>();
        temp->is_constant=true;
        if(ctx->Int())
            temp->is_int=true;
        else
            temp->is_int=false;
        result.push_back(temp);
    }
    return result;
}

antlrcpp::Any syntax_tree_builder::visitConstdef(C1Parser::ConstdefContext *ctx)
{
    //std::cout<<"visitConstdef"<<std::endl;
    //constdef: 
    // Identifier Assign exp
    //  | Identifier LeftBracket  exp? RightBracket Assign LeftBrace exp ( Comma exp)* RightBrace;
    auto result = new var_def_stmt_syntax;
    auto id = ctx->Identifier();
    result->pos = id->getSymbol()->getCharPositionInLine();
    result->line = id->getSymbol()->getLine();
    result->name = id->getSymbol()->getText();
    result->is_constant = true;

    //array[]
    if (ctx->LeftBracket()){
        auto exps = ctx->exp();
        int nnum = ctx->Comma().size() + 1;//Comma个数
        //[2]显式表明了长度
        if (exps.size() == nnum+1){
            //std::cout<<"[explictlt]"<<std::endl;

            result->array_length.reset(visit(exps[0]).as<expr_syntax*>());
            /*
            for(auto &init:exps){
                auto temp=visit(init).as<expr_syntax*>();
                result->initializers.push_back(static_cast<ptr<expr_syntax>>(temp));
            }
            */
            
            //size_t i = 1;
            for (size_t i=1; i < exps.size(); i++){
                auto expression = visit(exps[i]).as<expr_syntax*>();
                result->initializers.push_back(static_cast<ptr<expr_syntax>>(expression));
            }
            
        }
        //[]没有表明长度
        else{
            //std::cout<<"[implicitly]"<<std::endl;
            for(auto &init : exps){
                auto temp = visit(init).as<expr_syntax*>();
                result->initializers.push_back(static_cast<ptr<expr_syntax>>(temp));
            }
            auto length = new literal_syntax;
            length->is_int=true;
            length->intConst = nnum;
            length->line = ctx->RightBracket()->getSymbol()->getLine();
            length->pos = ctx->RightBracket()->getSymbol()->getCharPositionInLine();
            result->array_length.reset(static_cast<expr_syntax*>(length));
        }
    }
    //无[]
    else{
        result->array_length.reset();//nullptr
        auto expression = visit(ctx->exp(0)).as<expr_syntax*>();
        result->initializers.push_back(static_cast<ptr<expr_syntax>>(expression));
    }
    return result;
}

antlrcpp::Any syntax_tree_builder::visitVardecl(C1Parser::VardeclContext *ctx)
{
    //std::cout<<"visitVardecl"<<std::endl;
    //vardecl: (Int | Float ) vardef (Comma vardef)* SemiColon;
    std::vector<var_def_stmt_syntax *> result;
    for(auto &child:ctx->vardef()){
        //std::cout<<"visitVardecl-before"<<std::endl;
        auto temp=visit(child).as<var_def_stmt_syntax *>();
        //std::cout<<"visitVardecl-after"<<std::endl;
        temp->is_constant=false;
        if(ctx->Int())
            temp->is_int=true;
        else
            temp->is_int=false;
        result.push_back(temp);
    }
    return result;
}

antlrcpp::Any syntax_tree_builder::visitVardef(C1Parser::VardefContext *ctx)
{
    //std::cout<<"visitVardef"<<std::endl;
    //vardef: 
    //Identifier 
    //| Identifier LeftBracket exp RightBracket 
    //| Identifier Assign exp 
    //| Identifier LeftBracket  exp? RightBracket Assign LeftBrace exp ( Comma exp)* RightBrace;
    auto result = new var_def_stmt_syntax;
    result->is_constant = false;
    auto id = ctx->Identifier();
    result->name = id->getSymbol()->getText();
    result->pos = id->getSymbol()->getCharPositionInLine();
    result->line = id->getSymbol()->getLine();

    result->array_length.reset();
    //non-array variable declaration
    if (!ctx->LeftBracket()){
        if (ctx->Assign()){//id=4
            auto expression = visit(ctx->exp(0)).as<expr_syntax*>();
            result->initializers.push_back(static_cast<ptr<expr_syntax>>(expression));
        }
    }
    else{//带有[]
        if (ctx->Assign()){//[] =
            auto exps = ctx->exp();
            int nnum = ctx->Comma().size() + 1;
            //explicitly declare the array length
            if (exps.size() == nnum+1){//id[2]={1,2}
                result->array_length.reset(visit(exps[0]).as<expr_syntax*>());
                /*
                for(auto &e:exps){
                    auto temp=visit(e).as<expr_syntax*>();
                    result->initializers.push_back(static_cast<ptr<expr_syntax>>(temp));
                }
                */
                
                for (size_t i=1; i < exps.size(); i++){
                    auto expression = visit(exps[i]).as<expr_syntax*>();
                    result->initializers.push_back(static_cast<ptr<expr_syntax>>(expression));
                }
                
            }
            //not explicitly declare the array length
            else{//id[]={1,2}
                for(auto init : exps){
                    auto expression = visit(init).as<expr_syntax*>();
                    result->initializers.push_back(static_cast<ptr<expr_syntax>>(expression));
                }
                auto length = new literal_syntax;
                length->line = ctx->RightBracket()->getSymbol()->getLine();
                length->pos = ctx->RightBracket()->getSymbol()->getCharPositionInLine(); 
                length->is_int=true;
                length->intConst = nnum;
                result->array_length.reset(static_cast<expr_syntax*>(length));
            }
        }//id[2];
        else{
            auto expression = ctx->exp(0);
            result->array_length.reset(visit(expression).as<expr_syntax*>());
        }
    }
    return result;
}

antlrcpp::Any syntax_tree_builder::visitFuncdef(C1Parser::FuncdefContext *ctx)
{
    //std::cout<<"visitFuncdef"<<std::endl;
    //funcdef: Void Identifier LeftParen RightParen block;
    auto result = new func_def_syntax;
    //auto block = ctx->block();
    result->line = ctx->getStart()->getLine();
    result->pos = ctx->getStart()->getCharPositionInLine();
    result->name = ctx->Identifier()->getSymbol()->getText();
    result->body.reset(visit(ctx->block()).as<block_syntax*>());
    return result;
}

antlrcpp::Any syntax_tree_builder::visitBlock(C1Parser::BlockContext *ctx)
{
    //std::cout<<"visitBlock"<<std::endl;
    //block: LeftBrace (decl | stmt)* RightBrace;
    auto result = new block_syntax;
    result->line = ctx->getStart()->getLine();
    result->pos = ctx->getStart()->getCharPositionInLine();
    for (auto &child : ctx->children){
        if (antlrcpp::is<C1Parser::DeclContext*>(child)){//decl
            auto list = visit(child).as<std::vector<var_def_stmt_syntax *> >();
            for (auto &decl : list){
                //auto stmt = static_cast<stmt_syntax*>(decl);
                result->body.push_back(static_cast<ptr<stmt_syntax>>(decl));
            }
        }
        else if (antlrcpp::is<C1Parser::StmtContext*>(child)){//stmt
            //auto temp = dynamic_cast<C1Parser::StmtContext*>(child);
            auto stmt = visit(child).as<stmt_syntax*>();
            result->body.push_back(static_cast<ptr<stmt_syntax>>(stmt));
        }
    }
    return result;
}

antlrcpp::Any syntax_tree_builder::visitStmt(C1Parser::StmtContext *ctx)
{
    //std::cout<<"visitStmt"<<std::endl;
    //stmt: 
    //lval Assign exp SemiColon
    //| Identifier LeftParen RightParen SemiColon
    //| block
    //| If LeftParen cond RightParen stmt (Else stmt)? 
    //| While LeftParen cond RightParen stmt
    //| SemiColon;
    if (ctx->Assign()){//a=1+2;
        auto result = new assign_stmt_syntax;
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        result->target.reset(visit(ctx->lval()).as<lval_syntax*>());
        //std::cout<<"visitStmt-before-1"<<std::endl;
        result->value.reset(visit(ctx->exp()).as<expr_syntax*>());
        //std::cout<<"visitStmt-after-1"<<std::endl;
        return static_cast<stmt_syntax*>(result); 
    }
    if (ctx->Identifier()){//func();
        auto result = new func_call_stmt_syntax;
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        result->name = ctx->Identifier()->getSymbol()->getText();
        return static_cast<stmt_syntax*>(result);
    }
    if (auto block = ctx->block()){//{}
        auto result = visit(block).as<block_syntax*>();
        return static_cast<stmt_syntax*>(result);
    }
    if (ctx->If()){//if(..)..
        auto result = new if_stmt_syntax;
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        result->pred.reset(visit(ctx->cond()).as<cond_syntax*>());
        auto stmt = ctx->stmt();
        result->then_body.reset(visit(stmt[0]).as<stmt_syntax*>());
        result->else_body.reset();
        if (stmt.size() == 2)
            result->else_body.reset(visit(stmt[1]).as<stmt_syntax*>());
        return static_cast<stmt_syntax*>(result);
    }
    if (ctx->While()){//while
        auto result = new while_stmt_syntax;
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        result->pred.reset(visit(ctx->cond()).as<cond_syntax*>());
        result->body.reset(visit(ctx->stmt(0)).as<stmt_syntax*>());
        return static_cast<stmt_syntax*>(result);
    }
    else{//;
        auto result = new empty_stmt_syntax;
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        return static_cast<stmt_syntax*>(result);
    }
}

antlrcpp::Any syntax_tree_builder::visitLval(C1Parser::LvalContext *ctx)
{
    //std::cout<<"visitLval"<<std::endl;
    //lval: 
    //Identifier
    //| Identifier LeftBracket exp RightBracket;
    auto result = new lval_syntax;
    auto id = ctx->Identifier();
    result->line = id->getSymbol()->getLine();
    result->pos = id->getSymbol()->getCharPositionInLine();
    result->name = id->getSymbol()->getText();
    result->array_index.reset();
    if (auto expression = ctx->exp())
        result->array_index.reset(visit(expression).as<expr_syntax *>());
    return result;
}

antlrcpp::Any syntax_tree_builder::visitCond(C1Parser::CondContext *ctx)
{
    //std::cout<<"visitCond"<<std::endl;
    //cond: exp ( Equal | NonEqual | Less | Greater | LessEqual | GreaterEqual ) exp;
    auto result = new cond_syntax;
    //auto exps = ctx->exp();
    result->line = ctx->getStart()->getLine();
    result->pos = ctx->getStart()->getCharPositionInLine();
    result->lhs.reset(visit(ctx->exp(0)).as<expr_syntax*>());
    result->rhs.reset(visit(ctx->exp(1)).as<expr_syntax*>());

    if (ctx->Equal())
        result->op = relop::equal;
    if (ctx->NonEqual())
        result->op = relop::non_equal;
    if (ctx->Less())
        result->op = relop::less;
    if (ctx->Greater())
        result->op = relop::greater;
    if (ctx->LessEqual())
        result->op = relop::less_equal;
    if (ctx->GreaterEqual())
        result->op = relop::greater_equal;
    return result;
}

// Returns antlrcpp::Any, which is constructable from any type.
// However, you should be sure you use the same type for packing and depacking the `Any` object.
// Or a std::bad_cast exception will rise.
// This function always returns an `Any` object containing a `expr_syntax *`.
antlrcpp::Any syntax_tree_builder::visitExp(C1Parser::ExpContext *ctx)
{
    //std::cout<<"visitExp"<<std::endl;
    //exp:
    //(Plus | Minus) exp
    //| exp (Multiply | Divide | Modulo) exp
    //| exp (Plus | Minus) exp
    //| LeftParen exp RightParen
    //| number
    //| lval;
    // Get all sub-contexts of type `exp`.
    auto expressions = ctx->exp();
    // Two sub-expressions presented: this indicates it's a expression of binary operator, aka `binop`.
    if (expressions.size() == 2)
    {
        auto result = new binop_expr_syntax;
        // Set line and pos.
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        // visit(some context) is equivalent to calling corresponding visit method; dispatching is done automatically
        // by ANTLR4 runtime. For this case, it's equivalent to visitExp(expressions[0]).
        // Use reset to set a new pointer to a std::shared_ptr object. DO NOT use assignment; it won't work.
        // Use `.as<Type>()' to get value from antlrcpp::Any object; notice that this Type must match the type used in
        // constructing the Any object, which is constructed from (usually pointer to some derived class of
        // syntax_node, in this case) returning value of the visit call.
        //std::cout<<"visitExp-before-1"<<std::endl;
        result->lhs.reset(visit(expressions[0]).as<expr_syntax *>());
        //std::cout<<"visitExp-after-1"<<std::endl;
        // Check if each token exists.
        // Returnd value of the calling will be nullptr (aka NULL in C) if it isn't there; otherwise non-null pointer.
        if (ctx->Plus())
            result->op = binop::plus;
        if (ctx->Minus())
            result->op = binop::minus;
        if (ctx->Multiply())
            result->op = binop::multiply;
        if (ctx->Divide())
            result->op = binop::divide;
        if (ctx->Modulo())
            result->op = binop::modulo;
        //std::cout<<"visitExp-before-2"<<std::endl;
        result->rhs.reset(visit(expressions[1]).as<expr_syntax *>());
        //std::cout<<"visitExp-after-2"<<std::endl;
        return static_cast<expr_syntax *>(result);
    }
    // Otherwise, if `+` or `-` presented, it'll be a `unaryop_expr_syntax`.
    if (ctx->Plus() || ctx->Minus())
    {
        auto result = new unaryop_expr_syntax;
        result->line = ctx->getStart()->getLine();
        result->pos = ctx->getStart()->getCharPositionInLine();
        if (ctx->Plus())
            result->op = unaryop::plus;
        if (ctx->Minus())
            result->op = unaryop::minus;
        //std::cout<<"visitExp-before-3"<<std::endl;
        result->rhs.reset(visit(expressions[0]).as<expr_syntax *>());
        //std::cout<<"visitExp-after-3"<<std::endl;
        return static_cast<expr_syntax *>(result);
    }
    // In the case that `(` exists as a child, this is an expression like `'(' expressions[0] ')'`.
    if (ctx->LeftParen()){
        //std::cout<<"visitExp-before-4"<<std::endl;
        return visit(expressions[0]); // Any already holds expr_syntax* here, no need for dispatch and re-patch with casting.
    }
       
    // If `number` exists as a child, we can say it's a literal integer expression.
    if (auto number = ctx->number()){
        //std::cout<<"visitExp-before-5"<<std::endl;
        return visit(number);
    }
    if (auto lval = ctx->lval())
    {
        //std::cout<<"visitExp-before-6"<<std::endl;
        auto result = visit(lval).as<lval_syntax*>();
        return static_cast<expr_syntax *>(result);
    }
}

antlrcpp::Any syntax_tree_builder::visitNumber(C1Parser::NumberContext *ctx)
{
    //std::cout<<"visitNumber"<<std::endl;
    //number: 
    //FloatConst
    //| IntConst;
    auto result = new literal_syntax;
    if (auto intConst = ctx->IntConst())
    {
        result->is_int = true;
        result->line = intConst->getSymbol()->getLine();
        result->pos = intConst->getSymbol()->getCharPositionInLine();
        auto text = intConst->getSymbol()->getText();
        if (text[0] == '0' &&  (text[1] == 'x' || text[1] == 'X'))                // Hexadecimal
            result->intConst = std::stoi(text, nullptr, 16); // std::stoi will eat '0x'
        /* you need to add other situations here */
        else                                               // Decimal
            result->intConst = std::stoi(text, nullptr, 10);
        return static_cast<expr_syntax *>(result);
    }
    // else FloatConst
    else
    {   
        auto floatConst=ctx->FloatConst();
        result->is_int=false;
        result->line=floatConst->getSymbol()->getLine();
        result->pos=floatConst->getSymbol()->getCharPositionInLine();
        auto text=floatConst->getSymbol()->getText();
        result->floatConst=std::stod(text,nullptr);
        return static_cast<expr_syntax *>(result);
    }
}

ptr<syntax_tree_node> syntax_tree_builder::operator()(antlr4::tree::ParseTree *ctx)
{
    auto result = visit(ctx);
    if (result.is<syntax_tree_node *>())
        return ptr<syntax_tree_node>(result.as<syntax_tree_node *>());
    if (result.is<assembly *>())
        return ptr<syntax_tree_node>(result.as<assembly *>());
    if (result.is<global_def_syntax *>())
        return ptr<syntax_tree_node>(result.as<global_def_syntax *>());
    if (result.is<func_def_syntax *>())
        return ptr<syntax_tree_node>(result.as<func_def_syntax *>());
    if (result.is<cond_syntax *>())
        return ptr<syntax_tree_node>(result.as<cond_syntax *>());
    if (result.is<expr_syntax *>())
        return ptr<syntax_tree_node>(result.as<expr_syntax *>());
    if (result.is<binop_expr_syntax *>())
        return ptr<syntax_tree_node>(result.as<binop_expr_syntax *>());
    if (result.is<unaryop_expr_syntax *>())
        return ptr<syntax_tree_node>(result.as<unaryop_expr_syntax *>());
    if (result.is<lval_syntax *>())
        return ptr<syntax_tree_node>(result.as<lval_syntax *>());
    if (result.is<literal_syntax *>())
        return ptr<syntax_tree_node>(result.as<literal_syntax *>());
    if (result.is<stmt_syntax *>())
        return ptr<syntax_tree_node>(result.as<stmt_syntax *>());
    if (result.is<var_def_stmt_syntax *>())
        return ptr<syntax_tree_node>(result.as<var_def_stmt_syntax *>());
    if (result.is<assign_stmt_syntax *>())
        return ptr<syntax_tree_node>(result.as<assign_stmt_syntax *>());
    if (result.is<func_call_stmt_syntax *>())
        return ptr<syntax_tree_node>(result.as<func_call_stmt_syntax *>());
    if (result.is<block_syntax *>())
        return ptr<syntax_tree_node>(result.as<block_syntax *>());
    if (result.is<if_stmt_syntax *>())
        return ptr<syntax_tree_node>(result.as<if_stmt_syntax *>());
    if (result.is<while_stmt_syntax *>())
        return ptr<syntax_tree_node>(result.as<while_stmt_syntax *>());
    return nullptr;
}
