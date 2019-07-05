
#ifndef _C1_RUNTIME_H_
#define _C1_RUNTIME_H_

#include <vector>
#include <tuple>

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

class runtime_info
{
    llvm::GlobalVariable *input_ivar;
    llvm::GlobalVariable *input_fvar;
    llvm::GlobalVariable *output_ivar;
    llvm::GlobalVariable *output_fvar;
    llvm::Function *inputInt_func;
    llvm::Function *inputFloat_func;
    llvm::Function *outputInt_func;
    llvm::Function *outputFloat_func;

  public:
    runtime_info(llvm::Module *module);

    std::vector<std::tuple<std::string, llvm::GlobalValue *, bool, bool, bool, bool>> get_language_symbols();

    std::vector<std::tuple<std::string, void *>> get_runtime_symbols();
};

#endif
