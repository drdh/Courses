#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>

#include <memory>

using namespace llvm;

LLVMContext context;

//由于中间需要用到数字
Value *Num(int n){
    return ConstantInt::get(Type::getInt32Ty(context), n);
}

int main()
{
    
    IRBuilder<> builder(context);
    auto module = new Module("fib_gen", context);

    //fib 函数
    auto fib_func=Function::Create(FunctionType::get(Type::getInt32Ty(context),
                                    Type::getInt32Ty(context),false),
                                    GlobalValue::LinkageTypes::ExternalLinkage,
                                    "fib",module);
    auto fib_bb0=BasicBlock::Create(context,"",fib_func);
    //%0 作为参数
    auto fib_i0=fib_func->arg_begin();

    builder.SetInsertPoint(fib_bb0);
    //%1为第一个label
    //%2 = alloca i32, align 4;return
    auto fib_i2=builder.CreateAlloca(Type::getInt32Ty(context));
    fib_i2->setAlignment(4);
	
	//判断n==0 ?
    auto fib_i3=builder.CreateICmpEQ(fib_i0, Num(0));
    auto fib_i4 = BasicBlock::Create(context, "", fib_func);
    auto fib_i5 = BasicBlock::Create(context, "", fib_func);
    builder.CreateCondBr(fib_i3, fib_i4, fib_i5);//由n==0 设置跳转

    //label 4
    builder.SetInsertPoint(fib_i4);	
    builder.CreateAlignedStore(Num(0),fib_i2, 4);
    auto fib_i14 = BasicBlock::Create(context, "", fib_func);	
    builder.CreateBr(fib_i14);	//n==0 情况，跳转到返回代码

    //label 5
    builder.SetInsertPoint(fib_i5);
    auto fib_i6=builder.CreateICmpEQ(fib_i0,Num(1)); //判断n==1 ?
    auto fib_i7 = BasicBlock::Create(context, "", fib_func);
    auto fib_i8 = BasicBlock::Create(context, "", fib_func);
    builder.CreateCondBr(fib_i6,fib_i7,fib_i8);	//

    //label 7
    builder.SetInsertPoint(fib_i7);
    builder.CreateAlignedStore(Num(1),fib_i2, 4);
    builder.CreateBr(fib_i14); //n==1 情况，设置返回值为1后跳转到返回代码

    //label 8
    builder.SetInsertPoint(fib_i8);
    auto fib_i9=builder.CreateNSWSub(fib_i0,Num(1));	//n-1
    auto fib_i10=builder.CreateCall(fib_func,{fib_i9});//fib(n-1)
    auto fib_i11=builder.CreateNSWSub(fib_i0,Num(2)); //n-2
    auto fib_i12=builder.CreateCall(fib_func,{fib_i11});//fib(n-2)

    auto fib_i13=builder.CreateNSWAdd(fib_i10,fib_i12); //fib(n-1)+fib(n-2)
    builder.CreateAlignedStore(fib_i13,fib_i2, 4);
    builder.CreateBr(fib_i14);	

    //label 14
    builder.SetInsertPoint(fib_i14);	//设置返回值
    auto fib_i15=builder.CreateAlignedLoad(fib_i2, 4);
    builder.CreateRet(fib_i15);


    //main 函数
    auto main_func = Function::Create(FunctionType::get(Type::getInt32Ty(context), 
                                    std::vector<Type *>(), false),
                                    GlobalValue::LinkageTypes::ExternalLinkage,
                                    "main", module);

    //label 0
    auto main_bb0 = BasicBlock::Create(context, "", main_func);
    builder.SetInsertPoint(main_bb0);
    
    auto main_i1 = builder.CreateAlloca(Type::getInt32Ty(context));//return value 
    main_i1 -> setAlignment(4);
    builder.CreateAlignedStore(Num(0),main_i1, 4);
    auto main_i2 = builder.CreateAlloca(Type::getInt32Ty(context)); //x
    main_i2 -> setAlignment(4);
    builder.CreateAlignedStore(Num(0),main_i2, 4);
    auto main_i3 = builder.CreateAlloca(Type::getInt32Ty(context)); //i
    main_i3 -> setAlignment(4);
    builder.CreateAlignedStore(Num(0),main_i3, 4);
    auto main_i4 = BasicBlock::Create(context, "", main_func);
    builder.CreateBr(main_i4);

    //label 4
    builder.SetInsertPoint(main_i4);
    auto main_i5=builder.CreateAlignedLoad(main_i3, 4);
    auto main_i6=builder.CreateICmpSLT(main_i5,Num(8));//i<8 ?

    auto main_i7 = BasicBlock::Create(context, "", main_func);
    auto main_i15 = BasicBlock::Create(context, "", main_func);
    builder.CreateCondBr(main_i6,main_i7,main_i15);//由i<8? 来设置跳转

    //label 7
    builder.SetInsertPoint(main_i7);	//i<8
    auto main_i8=builder.CreateAlignedLoad(main_i3, 4);
    auto main_i9=builder.CreateCall(fib_func,{main_i8}); //fib(i)
    auto main_i10=builder.CreateAlignedLoad(main_i2, 4);
    auto main_i11=builder.CreateNSWAdd(main_i9,main_i10);//x+=fib(i)
    builder.CreateAlignedStore(main_i11,main_i2, 4);
    auto main_i12 = BasicBlock::Create(context, "", main_func);
    builder.CreateBr(main_i12);

    //label 12
    builder.SetInsertPoint(main_i12);
    auto main_i13=builder.CreateAlignedLoad(main_i3, 4);
    auto main_i14=builder.CreateNSWAdd(main_i13,Num(1));	//i++	
    builder.CreateAlignedStore(main_i14,main_i3, 4);
    builder.CreateBr(main_i4);

    //label 15
    builder.SetInsertPoint(main_i15);//i>=8
    auto main_i16=builder.CreateAlignedLoad(main_i2, 4);//设置返回值
    builder.CreateRet(main_i16);

    module->print(outs(), nullptr);
    delete module;
    return 0;
}
