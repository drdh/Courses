#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <fstream>
using namespace llvm;
using namespace std;

#define DEBUG_TYPE "hello"

STATISTIC(HelloCounter, "Counts number of functions greeted");

struct attribute
{
    bool load;
    Value* val;
    vector<Value*> alias;
    attribute() : load(false), val(NULL)
    {
        alias.clear();
    }
};

namespace {
    // Hello - The first implementation, without getAnalysisUsage.
    struct Hello : public ModulePass {
        static char ID; // Pass identification, replacement for typeid
        Hello() : ModulePass(ID) {}
        bool runOnModule(Module &M) override
        {
            errs() << M << '\n';
            return false;
        }
    };
}

char Hello::ID = 0;
static RegisterPass<Hello> X("hello", "Hello World Pass");

namespace
{
    struct labopt : public ModulePass
    {
        static char ID; // Pass identification, replacement for typeid
        labopt() : ModulePass(ID) {}

        using vmap = map<Value*, Value*>;

        template<class T1, class T2>
        T2 simblock(T1 BB, T2 lookup, T2 &alias)
        {
            //map<Value*, Value*> lookup;
            //map<Value*, Value*> alias;
            for (auto inst = BB->getInstList().begin(); inst != BB->getInstList().end(); inst++)
            {
                int num = inst->getNumOperands();
                for (int i = 0; i < num; i++)
                {
                    if (alias.find(inst->getOperand(i)) != alias.end() && alias[inst->getOperand(i)])
                    {
                        inst->setOperand(i, alias[inst->getOperand(i)]);
                    }
                }
                if (inst->getOpcode() == Instruction::Load)
                {
                    if (lookup.find(inst->getOperand(0)) != lookup.end()) // found
                    {
                        if (lookup[inst->getOperand(0)] != nullptr)
                        {
                            alias.insert(pair<Value*, Value*>(dyn_cast<Value>(inst), lookup[inst->getOperand(0)]));
                        }
                        else
                        {
                            lookup[inst->getOperand(0)] = dyn_cast<Value>(inst);
                        }
                    }
                    else // not found
                    {
                        lookup.insert(pair<Value*, Value*>(inst->getOperand(0), dyn_cast<Value>(inst)));
                    }
                }
                else if (inst->getOpcode() == Instruction::Store)
                {
                    if (lookup.find(inst->getOperand(1)) != lookup.end())
                    {
                        lookup[inst->getOperand(1)] = nullptr;
                    }
                }
                else if (inst->getOpcode() == Instruction::Alloca)
                {
                    if (lookup.find(dyn_cast<Value>(inst)) != lookup.end())
                    {
                        lookup[dyn_cast<Value>(inst)] = nullptr;
                    }
                }
            }
            return lookup;
        }

        template<class T>
        bool erase(T BB)
        {
            bool ans = false;
            for (auto inst = BB->getInstList().begin(); inst != BB->getInstList().end(); inst++)
            {
                auto code = inst->getOpcode();
                if (!inst->isUsedInBasicBlock(inst->getParent()) && !inst->isUsedOutsideOfBlock(inst->getParent()) && code != Instruction::Ret && code != Instruction::Store && code != Instruction::Br && code != Instruction::Call)
                {
                    inst = inst->eraseFromParent();
                    inst--;
                    ans = true;
                }
            }
            return ans;
        }

        template<class T>
        void dofunction(T F)
        {
            //++HelloCounter;
            int is = 0;
            int i;
            int count = 0;
            vector<Value *> table;
            table.clear();
            vmap alias;
            vector<vmap> lookup0;
            vector<vmap> lookup1;
            //vector<vmap> lookup2;
            map<BasicBlock*, int> index;
            vector<bool> check;
            alias.clear();
            lookup0.resize(F->getBasicBlockList().size());
            lookup1.resize(F->getBasicBlockList().size());
            //lookup2.resize(F->getBasicBlockList().size());
            index.clear();
            check.resize(F->getBasicBlockList().size(), true);
            i = 0;
            for (auto BB = F->getBasicBlockList().begin(); BB != F->getBasicBlockList().end(); BB++, i++)
            {
                index.insert(pair<BasicBlock*, int>(dyn_cast<BasicBlock>(BB), i));
            }
            is = F->getBasicBlockList().size();
            while (is > 0)
            {
                count++;
                //errs() << "; " << is << '\n';
                is--;
                i = 0;
                for (auto BB = F->getBasicBlockList().begin(); BB != F->getBasicBlockList().end(); BB++, i++)
                {
                    //if (!check[i]) continue;
                    //is = true;
                    auto num = distance(F->getBasicBlockList().begin(), BB);
                    //lookup0[num] = lookup2[num];
                    lookup1[num] = simblock(BB, lookup0[num], alias);
                    if (erase(BB)) is = F->getBasicBlockList().size();
                }
                i = 0;
                for (auto BB = F->getBasicBlockList().begin(); BB != F->getBasicBlockList().end(); BB++, i++)
                {
                    auto num = distance(F->getBasicBlockList().begin(), BB);
                    lookup0[num].clear();
                    for (auto pred : predecessors(dyn_cast<BasicBlock>(BB)))
                    {
                        int n = index[pred];
                        for (auto iter : lookup1[n])
                        {
                            if (lookup0[i].find(iter.first) == lookup0[i].end())
                            {
                                lookup0[i].insert(iter);
                            }
                            else if (lookup0[i][iter.first] != iter.second)
                            {
                                lookup0[i][iter.first] = nullptr;
                            }
                        }
                    }
                    //check[i] = !(lookup2[i] == lookup0[i]);
                }
                /*
                for (int i = 0; i < lookup2.size(); i++)
                {
                    if (!check[i])
                    {
                        //errs() << ";\t\t\tsame\n";
                        continue;
                    }
                    errs() << "; \t" << i << ":\n";
                    for (auto iter : lookup2[i])
                    {
                        errs() << ";\t\t\t";
                        iter.first->printAsOperand(errs());
                        errs() << " -> ";
                        if (iter.second == nullptr)
                        {
                            errs() << "nullptr";
                        }
                        else
                        {
                            iter.second->printAsOperand(errs());
                        }
                        errs() << '\n';
                        //errs() << " -> " << iter.second << '\n';
                    }
                }
                errs() << '\n';
                */
            }
        }
        bool runOnModule(Module &M) override
        {
            for (auto func = M.getFunctionList().begin(); func != M.getFunctionList().end(); func++)
            {
                if (func->getBasicBlockList().size() > 0)
                {
                    dofunction(func);
                }
            }
            errs() << M << '\n';
            return false;
        }
        // We don't modify the program, so we preserve all analyses.
        void getAnalysisUsage(AnalysisUsage &AU) const override {
            AU.setPreservesAll();
        }
    };
}

char labopt::ID = 0;
static RegisterPass<labopt>
Y("labopt", "Optimize by merging and hoisting load instructions.");