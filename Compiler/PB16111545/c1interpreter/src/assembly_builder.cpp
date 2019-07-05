
#include "assembly_builder.h"

#include <vector>

using namespace llvm;
using namespace c1_recognizer::syntax_tree;

void assembly_builder::visit(assembly &node)
{
    in_global=true;
    bb_count=0;
    error_flag=false;
    value_result = ConstantInt::get(Type::getInt32Ty(context), 0);
    int_const_result=0;
    float_const_result=0;
    constexpr_expected=false;
    lval_as_rval=true;
    for(auto &def:node.global_defs){
        def->accept(*this);
    }
}

void assembly_builder::visit(func_def_syntax &node)
{
    in_global=false;
    //检查是否重复定义
    auto name=node.name;
    if(functions.count(name)){
        err.error(node.line,node.pos,"Multiple Declaration: "+name);
        error_flag=true;
        return;
    }
    //下面代表无重复定义错误
    current_function = Function::Create(FunctionType::get(Type::getVoidTy(context), {}, false),
                                    GlobalValue::LinkageTypes::ExternalLinkage,
                                    name, module.get());
    functions[name]=current_function;

    //builder.SetInsertPoint(BasicBlock::Create(context,"entry_BB"+std::to_string(bb_count++),current_function));
    builder.SetInsertPoint(BasicBlock::Create(context,"entry",current_function));
    bb_count++;
    node.body->accept(*this);
    builder.CreateRetVoid();

    in_global=true;
}

void assembly_builder::visit(cond_syntax &node)
{   
    constexpr_expected=false;
    node.lhs->accept(*this);
    auto lhs=value_result;
    auto lhs_is_int=is_result_int;
    node.rhs->accept(*this);
    auto rhs=value_result;
    auto rhs_is_int=is_result_int;

    if(lhs_is_int && rhs_is_int){//全部为int
        if(node.op == relop::equal)
            value_result = builder.CreateICmpEQ(lhs, rhs);
        else if(node.op == relop::non_equal)
            value_result = builder.CreateICmpNE(lhs, rhs);
        else if(node.op == relop::less)
            value_result = builder.CreateICmpSLT(lhs, rhs);
        else if(node.op == relop::less_equal)
            value_result = builder.CreateICmpSLE(lhs, rhs);
        else if(node.op == relop::greater)
            value_result = builder.CreateICmpSGT(lhs, rhs);
        else if(node.op == relop::greater_equal)
            value_result = builder.CreateICmpSGE(lhs, rhs);
        }
    else{//至少一个为float，则需要全部转化成float
        if(lhs_is_int){
            lhs=builder.CreateSIToFP(lhs,Type::getDoubleTy(context));
        }
        if(rhs_is_int){
            rhs=builder.CreateSIToFP(rhs,Type::getDoubleTy(context));
        }
        if(node.op == relop::equal){
            err.error(node.line,node.pos,"Cannot '==' on float");
            error_flag=true;
            return;
            //value_result=builder.CreateFCmpOEQ(lhs,rhs);
        }
        else if(node.op == relop::non_equal){
            err.error(node.line,node.pos,"Cannot '!=' on float");
            error_flag=true;
            return;
            //value_result=builder.CreateFCmpONE(lhs,rhs);
        }
        else if(node.op == relop::less)
            value_result = builder.CreateFCmpOLT(lhs, rhs);
        else if(node.op == relop::less_equal)
            value_result = builder.CreateFCmpOLE(lhs, rhs);
        else if(node.op == relop::greater)
            value_result = builder.CreateFCmpOGT(lhs, rhs);
        else if(node.op == relop::greater_equal)
            value_result = builder.CreateFCmpOGE(lhs, rhs);
    }
}

void assembly_builder::visit(binop_expr_syntax &node)
{
    if(constexpr_expected){//const
        //std::cout<<"binop constexpr_expected"<<std::endl;
        int lhs_int,rhs_int;
        double lhs_float,rhs_float;
        bool is_int_left,is_int_right;

        node.lhs->accept(*this);
        is_int_left=is_result_int;
        if(is_int_left){
            lhs_int=int_const_result;
            lhs_float=int_const_result;
        }
        else{
            lhs_float=float_const_result;
            lhs_int=float_const_result;
        }
        node.rhs->accept(*this);
        is_int_right=is_result_int;
        if(is_int_right){
            rhs_int=int_const_result;
            rhs_float=int_const_result;
        }
        else{
            rhs_float=float_const_result;
            rhs_int=float_const_result;
        }
        is_result_int=is_int_left && is_int_right;

        if(is_result_int){//即为两个int相加
            if(node.op == binop::plus)
                int_const_result = lhs_int + rhs_int;
            else if(node.op == binop::minus)
                int_const_result = lhs_int - rhs_int;
            else if(node.op == binop::multiply)
                int_const_result = lhs_int * rhs_int;
            else if(node.op == binop::divide)
                int_const_result = lhs_int / rhs_int;
            else if(node.op == binop::modulo)
                int_const_result = lhs_int % rhs_int;
        }
        else{//至少有一个为fp:float point
            if(node.op == binop::plus)
                float_const_result = lhs_float + rhs_float;
            else if(node.op == binop::minus)
                float_const_result = lhs_float - rhs_float;
            else if(node.op == binop::multiply)
                float_const_result = lhs_float * rhs_float;
            else if(node.op == binop::divide)
                float_const_result = lhs_float / rhs_float;
            else if(node.op == binop::modulo){
                err.error(node.line,node.pos,"Cannot Modulo on Float");
                error_flag=true;
            }
        }        
    }
    else{//not const
        //std::cout<<"binop non constexpr_expected"<<std::endl;
        node.lhs->accept(*this);
        auto lhs=value_result;
        bool is_int_left=is_result_int;
        node.rhs->accept(*this);
        auto rhs=value_result;
        bool is_int_right=is_result_int;
        is_result_int=is_int_left && is_int_right;
        
        if(is_result_int){//两边都是int
            //std::cout<<"binop:non const:int"<<std::endl;
            /*
            if(node.op == binop::plus)
                value_result = builder.CreateNSWAdd(lhs, rhs);
            else if(node.op == binop::minus)
                value_result = builder.CreateNSWSub(lhs, rhs);
            else if(node.op == binop::multiply)
                value_result = builder.CreateNSWMul(lhs, rhs);
            else if(node.op == binop::divide)
                value_result = builder.CreateSDiv(lhs, rhs);
            else if(node.op == binop::modulo)
                value_result = builder.CreateSRem(lhs, rhs);
            */
            if(node.op == binop::plus)
                value_result = builder.CreateAdd(lhs, rhs);
            else if(node.op == binop::minus)
                value_result = builder.CreateSub(lhs, rhs);
            else if(node.op == binop::multiply)
                value_result = builder.CreateMul(lhs, rhs);
            else if(node.op == binop::divide)
                value_result = builder.CreateSDiv(lhs, rhs);
            else if(node.op == binop::modulo)
                value_result = builder.CreateSRem(lhs, rhs);
        }
        else{//至少有一个为fp
            //std::cout<<"binop:non const:fp"<<std::endl;
            if(is_int_left){//先转化
                lhs=builder.CreateSIToFP(lhs,Type::getDoubleTy(context));
            }
            if(is_int_right){
                rhs=builder.CreateSIToFP(rhs,Type::getDoubleTy(context));
            }

            if(node.op == binop::plus)
                value_result = builder.CreateFAdd(lhs, rhs);
            else if(node.op == binop::minus)
                value_result = builder.CreateFSub(lhs, rhs);
            else if(node.op == binop::multiply)
                value_result = builder.CreateFMul(lhs, rhs);
            else if(node.op == binop::divide)
                value_result = builder.CreateFDiv(lhs, rhs);
            else if(node.op == binop::modulo){
                err.error(node.line,node.pos,"Cannot '%' on float");
                error_flag=true;
                return;
                //value_result = builder.CreateFRem(lhs, rhs);
            }
        }
        
    }
}

void assembly_builder::visit(unaryop_expr_syntax &node)
{
    if(constexpr_expected){
        node.rhs->accept(*this);
        if(is_result_int){
            if(node.op==unaryop::minus)
                int_const_result=-int_const_result;
        }
        else{
            if(node.op==unaryop::minus)
                float_const_result=-float_const_result;
        }
    }
    else{
        node.rhs->accept(*this);
        if(is_result_int){
            if(node.op==unaryop::minus)
                value_result=builder.CreateNeg(value_result);
        }
        else{
            if(node.op==unaryop::minus)
                value_result=builder.CreateFNeg(value_result);
        }
    }
}

void assembly_builder::visit(lval_syntax &node)
{
    //变量地址在 LLVM 数据结构中的表示、是否为常量、是否为数组、是否为整型
    auto variable_tuple=lookup_variable(node.name);
    auto lval=std::get<0>(variable_tuple);
    bool is_const=std::get<1>(variable_tuple);
    bool is_array=std::get<2>(variable_tuple);
    bool is_int=std::get<3>(variable_tuple);
    is_result_int=is_int;

    //没有声明
    if(!lval){
        err.error(node.line,node.pos,"Undeclared Variable: "+node.name);
        error_flag=true;
        return;
    }

    //不可为const
    
    if(constexpr_expected && !lval_as_rval){
        err.error(node.line,node.pos,"Constexper cannot be Lval");
        error_flag=true;
        return;
    }
    

    if(!is_array){//不是数组
        if(node.array_index){//类型不符
            err.error(node.line,node.pos,"Type mismatch, expect a non-array");
            error_flag=true;
            return;
        }
        if(lval_as_rval){//作为右值
            //value_result=builder.CreateLoad(lval);//取出其值,存入
            if(is_int){
                value_result=builder.CreateLoad(Type::getInt32Ty(context),lval);
            }
            else{
                value_result=builder.CreateLoad(Type::getDoubleTy(context),lval);
            }
            //is_result_int=is_int;
        }
        else{//作为左值
            if(is_const){//const 不能为左值
                err.error(node.line,node.pos,"Const cannot be Lval");
                error_flag=true;
                return;
            }
            value_result=lval;//即取出其地址,存入
        }
    }
    else{//是数组
        if(!node.array_index){//类型不符
            err.error(node.line,node.pos,"Type mismatch, expect an array");
            error_flag=true;
            return;
        }
        std::vector<Value*>index;
        auto temp_lval_as_rval=lval_as_rval;
        auto temp_is_result_int=is_result_int;
        lval_as_rval=true;
        node.array_index->accept(*this);
        //索引不是int
        if(!is_result_int){
            err.error(node.line,node.pos,"Index cannot be float");
            error_flag=true;
            return;
        }
        lval_as_rval=temp_lval_as_rval;
        is_result_int=temp_is_result_int;
        Value* int_index=value_result;
        index.push_back(ConstantInt::get(Type::getInt32Ty(context),0));
        index.push_back(int_index);
        auto element=builder.CreateGEP(lval,index);//即getelementptr
        if(lval_as_rval){//作为右值
            //std::cout<<"lval:array::as_rval"<<std::endl;
            //value_result=builder.CreateLoad(element);
            if(is_int){
                value_result=builder.CreateLoad(Type::getInt32Ty(context),element);
            }
            else{
                value_result=builder.CreateLoad(Type::getDoubleTy(context),element);
            }
            //std::cout<<"lval:array::as_rval-after"<<std::endl;
        }
        else{//作为左值
            if(is_const){//左值不能为const
                err.error(node.line,node.pos,"Const cannot be Lval");
                error_flag=true;
                return;
            }
            //std::cout<<"lval:array"<<std::endl;
            value_result=element;
        }
    }

}

void assembly_builder::visit(literal_syntax &node)
{
    is_result_int=node.is_int;
    if(constexpr_expected){//为const
        if(is_result_int){
            int_const_result=node.intConst;
        }
        else{
            float_const_result=node.floatConst;
        }
    }
    else{//非const
        if(is_result_int){
            value_result=ConstantInt::get(Type::getInt32Ty(context), node.intConst);
        }
        else{
            value_result=ConstantFP::get(Type::getDoubleTy(context), node.floatConst);
        }
    }
}

void assembly_builder::visit(var_def_stmt_syntax &node)
{
    auto var_name=node.name;
    bool is_int=node.is_int;
    bool is_const=node.is_constant;
    bool is_array=false;
    if(node.array_length!=nullptr)
        is_array=true;
    
    Value *var_ptr;
    //int array_length=0;
    if(in_global){//global
        constexpr_expected=true;//需要对index求值
        if(!is_array){//不是array
            if(is_int){//int
                ConstantInt *init_value;
                if(node.initializers.empty()){//没有显式初值
                    init_value=ConstantInt::get(Type::getInt32Ty(context), 0);
                }
                else{//有显式初始值
                    node.initializers[0]->accept(*this);
                    
                    if(!is_result_int){//应该是int，却用float赋值
                        /*err.error(node.line,node.pos,"Initialize "+var_name+" with a float");
                        error_flag=true;
                        return;
                        */
                       int_const_result=float_const_result;
                    }
                    
                    init_value=ConstantInt::get(Type::getInt32Ty(context),int_const_result);
                }
                var_ptr=new GlobalVariable(*module,Type::getInt32Ty(context),is_const,
                                            GlobalValue::ExternalLinkage,init_value,"");
            }
            else{//float
                Constant *init_value;
                if(node.initializers.empty()){//没有显式初值
                    init_value=ConstantFP::get(Type::getDoubleTy(context), 0);
                }
                else{
                    node.initializers[0]->accept(*this);
                    
                    if(is_result_int){//应该是float，却用int赋值
                        /*err.error(node.line,node.pos,"Initialize "+var_name+" with an int");
                        error_flag=true;
                        return;
                        */
                       float_const_result=int_const_result;
                    }
                    
                    init_value=ConstantFP::get(Type::getDoubleTy(context),float_const_result);
                }
                var_ptr=new GlobalVariable(*module,Type::getDoubleTy(context),is_const,
                                            GlobalValue::ExternalLinkage,init_value,"");
            }
        }
        else{//是array
            lval_as_rval=true;
            node.array_length->accept(*this);
            if(!is_result_int){//index不为int
                err.error(node.line,node.pos,"Float cannot be index of "+var_name);
                error_flag=true;
                return;
            }
            int array_length=int_const_result;
            int init_length=node.initializers.size();
            if(array_length<=0){//数组大小必须大于0
                err.error(node.line,node.pos,"Index of array must greater than 0 in "+var_name);
                error_flag=true;
                return;
            }
            if(array_length<init_length){//初始值比数组大小大
                err.error(node.line,node.pos,"Too much initial value of "+var_name);
                error_flag=true;
                return;
            }
            std::vector<Constant *>init_array;
            //constexpr_expected=true;
            if(is_int){//数组为int
                for(int i=0;i<array_length;i++){
                    if(i<init_length){//有初始值的情况下
                        node.initializers[i]->accept(*this);
                        if(!is_result_int){//初始值不为int
                        /*    err.error(node.line,node.pos,"Cannot initialize int array with float in "+var_name);
                            error_flag=true;
                            return;
                        */
                            int_const_result=float_const_result;
                        }
                        init_array.push_back(ConstantInt::get(Type::getInt32Ty(context),int_const_result));
                    }
                    else{//超过初始值
                        init_array.push_back(ConstantInt::get(Type::getInt32Ty(context),0));
                    }
                }
                var_ptr=new GlobalVariable(*module.get(),ArrayType::get(Type::getInt32Ty(context),array_length),is_const,
                                            GlobalValue::ExternalLinkage,ConstantArray::get(ArrayType::get(Type::getInt32Ty(context),array_length),init_array),"");
            }
            else{//数组为float
                for(int i=0;i<array_length;i++){
                    if(i<init_length){//有初始值的情况下
                        node.initializers[i]->accept(*this);
                        if(is_result_int){//初始值不为float
                        /*    err.error(node.line,node.pos,"Cannot initialize float array with int in "+var_name);
                            error_flag=true;
                            return;
                        */
                            float_const_result=int_const_result;
                        }
                        init_array.push_back(ConstantFP::get(Type::getDoubleTy(context),float_const_result));
                    }
                    else{//超过初始值
                        init_array.push_back(ConstantFP::get(Type::getDoubleTy(context),0));
                    }
                }
                var_ptr=new GlobalVariable(*module.get(),ArrayType::get(Type::getDoubleTy(context),array_length),is_const,
                                            GlobalValue::ExternalLinkage,ConstantArray::get(ArrayType::get(Type::getInt32Ty(context),array_length),init_array),"");
            }
        }
        constexpr_expected=false;//note
    }
    else{//local
        if(!is_array){//非数组情况
            if(is_int){//int
                var_ptr=builder.CreateAlloca(Type::getInt32Ty(context),nullptr,"");
                constexpr_expected=false;
                if(!node.initializers.empty()){
                    node.initializers[0]->accept(*this);
                    if(!is_result_int){//初始值不为int
                        /*    err.error(node.line,node.pos,"Cannot initialize int with float in "+var_name);
                            error_flag=true;
                            return;
                        */
                       value_result=builder.CreateFPToSI(value_result,Type::getInt32Ty(context));

                    }
                    
                    builder.CreateStore(value_result,var_ptr);
                    
                }
                else{//默认初值0
                    builder.CreateStore(ConstantInt::get(Type::getInt32Ty(context),0),var_ptr);
                }
            }
            else{//float
                var_ptr=builder.CreateAlloca(Type::getDoubleTy(context),nullptr,"");
                constexpr_expected=false;
                if(!node.initializers.empty()){
                    node.initializers[0]->accept(*this);
                    if(is_result_int){//初始值不为float
                        /*    err.error(node.line,node.pos,"Cannot initialize float array with int in "+var_name);
                            error_flag=true;
                            return;
                        */
                       value_result=builder.CreateSIToFP(value_result,Type::getDoubleTy(context));
                    }
                    
                    builder.CreateStore(value_result,var_ptr);
                }
                else{//默认初值为0
                    builder.CreateStore(ConstantFP::get(Type::getDoubleTy(context),0),var_ptr);
                }
            }
        }
        else{//数组
            //std::cout<<"var_def::local::array"<<std::endl;
            constexpr_expected=true;//求数组大小
            node.array_length->accept(*this);
            constexpr_expected=false;//note
            if(!is_result_int){//index不为int
                err.error(node.line,node.pos,"Float cannot be index of "+var_name);
                error_flag=true;
                return;
            }
            int array_length=int_const_result;
            int init_length=node.initializers.size();
            if(array_length<=0){//数组大小必须大于0
                err.error(node.line,node.pos,"Index of array must greater than 0 in "+var_name);
                error_flag=true;
                return;
            }
            if(array_length<init_length){//初始值比数组大小大
                err.error(node.line,node.pos,"Too much initial value of "+var_name);
                error_flag=true;
                return;
            }
            constexpr_expected=false;//对于局部变量，无论变量为常量或可变量，其初始化表达式均不要求是常量表达式
            Value *element;
            std::vector<Value*>index;
            index.push_back((Value*)ConstantInt::get(Type::getInt32Ty(context),0));
            if(is_int){//int 
                //std::cout<<"var_def::local::array::int"<<std::endl;
                var_ptr=builder.CreateAlloca(ArrayType::get(Type::getInt32Ty(context),array_length),nullptr,"");
                for(int i=0;i<array_length;i++){
                    index.push_back((Value*)ConstantInt::get(Type::getInt32Ty(context),i));
                    element=builder.CreateGEP(var_ptr,index);
                    //element=builder.CreateConstGEP1_32(var_ptr,i);
                    index.pop_back();
                    if(i<init_length){
                        node.initializers[i]->accept(*this);
                        if(!is_result_int){//初始值不为int
                        /*
                                err.error(node.line,node.pos,"Cannot initialize int array with float in"+var_name);
                                error_flag=true;
                                return;
                        */
                            value_result=builder.CreateFPToSI(value_result,Type::getInt32Ty(context));        
                        }
                        //std::cout<<"var_def::local::array::int::init: "<<i<<std::endl;
                        //Value *init=value_result;
                        //builder.CreateStore(init,element);
                        builder.CreateStore(value_result,element);
                    }
                    else{
                        Value *init=ConstantInt::get(Type::getInt32Ty(context),0);
                        builder.CreateStore(init,element);
                    }
                }
            }
            else{//float
                var_ptr=builder.CreateAlloca(ArrayType::get(Type::getDoubleTy(context),array_length),nullptr,"");
                for(int i=0;i<array_length;i++){
                    index.push_back((Value*)ConstantInt::get(Type::getInt32Ty(context),i));
                    element=builder.CreateGEP(var_ptr,index);
                    //element=builder.CreateConstGEP1_32(var_ptr,i);
                    index.pop_back();
                    if(i<init_length){
                        node.initializers[i]->accept(*this);
                        if(is_result_int){//初始值不为float
                        /*
                                err.error(node.line,node.pos,"Cannot initialize int array with float in"+var_name);
                                error_flag=true;
                                return;
                        */
                            value_result=builder.CreateSIToFP(value_result,Type::getDoubleTy(context));              
                        }
                        Value *init=value_result;
                        builder.CreateStore(init,element);
                    }
                    else{
                        Value *init=ConstantInt::get(Type::getDoubleTy(context),0);
                        builder.CreateStore(init,element);
                    }
                }
            }
        }
    }
    if(!declare_variable(var_name,var_ptr,is_const,is_array,is_int))
    {
        err.error(node.line, node.pos, "variable: " + node.name + " has already been declared.");
        error_flag = true;
        return;
    }
}

void assembly_builder::visit(assign_stmt_syntax &node)
{
    //变量地址在 LLVM 数据结构中的表示、是否为常量、是否为数组、是否为整型
    auto variable_tuple=lookup_variable(node.target->name);
    auto lval=std::get<0>(variable_tuple);
    bool is_const=std::get<1>(variable_tuple);
    bool is_array=std::get<2>(variable_tuple);
    bool is_int=std::get<3>(variable_tuple);

    //左边不存在
    if(!lval){
        err.error(node.line,node.pos,"Undeclared Variable: "+node.target->name);
        error_flag=true;
        return;
    }
    //左边不可为const
    if(is_const){
        err.error(node.line,node.pos,"Assign to a const "+node.target->name);
        error_flag=true;
        return;
    }
    //以下判断类型，关于数组是否符合声明
    if((!is_array && node.target->array_index)||(is_array && !node.target->array_index)){
        err.error(node.line,node.pos,"Type dismatch: "+node.target->name);
        error_flag=true;
        return;
    }
    //std::cout<<"assign::"<<std::endl;
    lval_as_rval=false;
    node.target->accept(*this);
    auto target=value_result;
    
    lval_as_rval=true;
    node.value->accept(*this);
    auto value=value_result;
    
    //target与value类型不同时，需要进行类型转换
    if(is_int != is_result_int){
        if(is_int){//target为int
            auto cast_value=builder.CreateFPToSI(value,Type::getInt32Ty(context));
            builder.CreateStore(cast_value,target);
        }
        else{//target为float
            auto cast_value=builder.CreateSIToFP(value,Type::getDoubleTy(context));
            builder.CreateStore(cast_value,target);
        }
    }
    else{
        builder.CreateStore(value,target);
    }
}

void assembly_builder::visit(func_call_stmt_syntax &node)
{
    auto name=node.name;
    //先检查是否声明了这个函数
    if(!functions.count(name)){
        err.error(node.line,node.pos,"Function "+name+" not found");
        error_flag=true;
        return;
    }
    builder.CreateCall(functions[name],{});
}

void assembly_builder::visit(block_syntax &node)
{
    enter_scope();
    for(auto &body:node.body){
        body->accept(*this);
    }
    exit_scope();
}

void assembly_builder::visit(if_stmt_syntax &node)
{
    /*
    auto cond_block=BasicBlock::Create(context,"cond_BB",current_function);
    auto then_block=BasicBlock::Create(context,"then_BB",current_function);
    auto else_block=BasicBlock::Create(context,"else_BB",current_function);
    auto next_block=BasicBlock::Create(context,"next_BB",current_function);
    */
    
    if(node.else_body){
        auto cond_block=BasicBlock::Create(context,"IfB",current_function);
        bb_count++;
        auto then_block=BasicBlock::Create(context,"ThenB",current_function);
        bb_count++;
        auto else_block=BasicBlock::Create(context,"ElseB",current_function);
        bb_count++;
        auto next_block=BasicBlock::Create(context,"AfterIf",current_function);
        bb_count++;

        builder.CreateBr(cond_block);
        //cond 部分
        builder.SetInsertPoint(cond_block);
        node.pred->accept(*this);
        builder.CreateCondBr(value_result,then_block,else_block);
        //then 部分
        builder.SetInsertPoint(then_block);
        node.then_body->accept(*this);
        builder.CreateBr(next_block);//直接跳转
        //else 部分
        builder.SetInsertPoint(else_block);
        node.else_body->accept(*this);
        builder.CreateBr(next_block);//直接跳转
        //next 部分
        builder.SetInsertPoint(next_block);
    }
    else{
        auto cond_block=BasicBlock::Create(context,"IfB",current_function);
        bb_count++;
        auto then_block=BasicBlock::Create(context,"ThenB",current_function);
        bb_count++;
        auto next_block=BasicBlock::Create(context,"AfterIf",current_function);
        bb_count++;

        builder.CreateBr(cond_block);
        //cond 部分
        builder.SetInsertPoint(cond_block);
        node.pred->accept(*this);
        
        builder.CreateCondBr(value_result,then_block,next_block);
        //then 部分
        builder.SetInsertPoint(then_block);
        node.then_body->accept(*this);
        builder.CreateBr(next_block);//直接跳转
        //next 部分
        builder.SetInsertPoint(next_block);
    }
}

void assembly_builder::visit(while_stmt_syntax &node)
{
    auto cond_block=BasicBlock::Create(context,"while_cond_BB"+std::to_string(bb_count++),current_function);
    auto true_block=BasicBlock::Create(context,"while_true_BB"+std::to_string(bb_count++),current_function);
    auto next_block=BasicBlock::Create(context,"whilw_next_BB"+std::to_string(bb_count++),current_function);

    builder.CreateBr(cond_block);
    //while_cond
    builder.SetInsertPoint(cond_block);
    node.pred->accept(*this);
    builder.CreateCondBr(value_result,true_block,next_block);//设置跳转
    //while_true
    builder.SetInsertPoint(true_block);
    node.body->accept(*this);
    builder.CreateBr(cond_block);//直接跳转到语句判断部分
    //while_next
    builder.SetInsertPoint(next_block);
}

void assembly_builder::visit(empty_stmt_syntax &node)
{
}
