
#include <iostream>
#include "runtime.h"
#include "runtime/io.h"

#include <llvm/IR/Type.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>

using namespace std;
using namespace llvm;

runtime_info::runtime_info(Module *module)
{
    input_ivar = new GlobalVariable(*module,
                                   Type::getInt32Ty(module->getContext()),
                                   false,
                                   GlobalValue::ExternalLinkage,
                                   ConstantInt::get(Type::getInt32Ty(module->getContext()), 0),
                                   "input_ivar");
    input_fvar = new GlobalVariable(*module,
                                   Type::getDoubleTy(module->getContext()),
                                   false,
                                   GlobalValue::ExternalLinkage,
                                   ConstantFP::get(Type::getDoubleTy(module->getContext()), 0),
                                   "input_fvar");
    output_ivar = new GlobalVariable(*module,
                                    Type::getInt32Ty(module->getContext()),
                                    false,
                                    GlobalValue::ExternalLinkage,
                                    ConstantInt::get(Type::getInt32Ty(module->getContext()), 0),
                                    "output_ivar");
    output_fvar = new GlobalVariable(*module,
                                    Type::getDoubleTy(module->getContext()),
                                    false,
                                    GlobalValue::ExternalLinkage,
                                    ConstantFP::get(Type::getDoubleTy(module->getContext()), 0),
                                    "output_fvar");
    auto inputInt_impl = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()),
                                                         {Type::getInt32PtrTy(module->getContext())},
                                                         false),
                                       GlobalValue::LinkageTypes::ExternalLinkage,
                                       "inputInt_impl",
                                       module);
    auto inputFloat_impl = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()),
                                                         {Type::getDoublePtrTy(module->getContext())},
                                                         false),
                                       GlobalValue::LinkageTypes::ExternalLinkage,
                                       "inputFloat_impl",
                                       module);
    auto outputInt_impl = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()),
                                                          {Type::getInt32PtrTy(module->getContext())},
                                                          false),
                                        GlobalValue::LinkageTypes::ExternalLinkage,
                                        "outputInt_impl",
                                        module);
    auto outputFloat_impl = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()),
                                                          {Type::getDoublePtrTy(module->getContext())},
                                                          false),
                                        GlobalValue::LinkageTypes::ExternalLinkage,
                                        "outputFloat_impl",
                                        module);

    IRBuilder<> builder(module->getContext());

    inputInt_func = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()), {}, false),
                                  GlobalValue::LinkageTypes::ExternalLinkage,
                                  "inputInt",
                                  module);
    builder.SetInsertPoint(BasicBlock::Create(module->getContext(), "entry", inputInt_func));
    builder.CreateCall(inputInt_impl, {input_ivar});
    builder.CreateRetVoid();

    inputFloat_func = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()), {}, false),
                                  GlobalValue::LinkageTypes::ExternalLinkage,
                                  "inputFloat",
                                  module);
    builder.SetInsertPoint(BasicBlock::Create(module->getContext(), "entry", inputFloat_func));
    builder.CreateCall(inputFloat_impl, {input_fvar});
    builder.CreateRetVoid();

    outputInt_func = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()), {}, false),
                                   GlobalValue::LinkageTypes::ExternalLinkage,
                                   "outputInt",
                                   module);
    builder.SetInsertPoint(BasicBlock::Create(module->getContext(), "entry", outputInt_func));
    builder.CreateCall(outputInt_impl, {output_ivar});
    builder.CreateRetVoid();

    outputFloat_func = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()), {}, false),
                                   GlobalValue::LinkageTypes::ExternalLinkage,
                                   "outputFloat",
                                   module);
    builder.SetInsertPoint(BasicBlock::Create(module->getContext(), "entry", outputFloat_func));
    builder.CreateCall(outputFloat_impl, {output_fvar});
    builder.CreateRetVoid();
}

using namespace string_literals;

vector<tuple<string, llvm::GlobalValue *, bool, bool, bool, bool>> runtime_info::get_language_symbols()
{
    return {
        make_tuple("input_ivar"s, input_ivar, false, false, false, true),
        make_tuple("input_fvar"s, input_fvar, false, false, false, false),
        make_tuple("output_ivar"s, output_ivar, false, false, false, true),
        make_tuple("output_fvar"s, output_fvar, false, false, false, false),
        make_tuple("inputInt"s, inputInt_func, true, false, false, true),
        make_tuple("inputFloat"s, inputFloat_func, true, false, false, false),
        make_tuple("outputInt"s, outputInt_func, true, false, false, true),
        make_tuple("outputFloat"s, outputFloat_func, true, false, false, false)};
}

vector<tuple<string, void *>> runtime_info::get_runtime_symbols()
{
    return {
        make_tuple("inputInt_impl"s, (void *)&::inputInt),
        make_tuple("inputFloat_impl"s, (void *)&::inputFloat),
        make_tuple("outputInt_impl"s, (void *)&::outputInt),
        make_tuple("outputFloat_impl"s, (void *)&::outputFloat) };
}
