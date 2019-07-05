#include <algorithm>
#include <vector>
#include <set>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace std;
using namespace llvm;

namespace {
    struct Node {
        /// LLVM Value
        Value* value;
        // 多项式的系数poly[i]对应x^i的系数
        vector<int> poly;
        /// Dependencies
        vector<Node*> deps;
        set<Node*> users;
    };

    struct Algebraic : public FunctionPass {
        static char ID;
        Algebraic() : FunctionPass(ID) {}

        bool runOnFunction(Function& BB) override;

    private:
        vector<Node*> Graph;
        Node* findOrCreateInGraph(Value* v);
        void addInstrToGraph(Instruction& I);
        Node* constructGraph(BasicBlock& BB);
    };
}

char Algebraic::ID = 0;
static RegisterPass<Algebraic> X("algebraic", "Algebraic Optimizations");

// Pass 主函数
bool Algebraic::runOnFunction(Function& func)
{
    bool modified = false;
    Node *ans;

    // 对每个函数块建立一个图
    if(func.size()>1)
        return modified;

    // errs() << "Algebraic: before";
    // errs() << func;

    for(auto& BB : func) {
        ans = constructGraph(BB);
        if(ans==nullptr)
            continue;
        if(ans->poly.size()==1) {
            // 如果计算的多项式恒为常数
            // errs() << ans->poly.size() << '\n';
            auto bb_1 = BasicBlock::Create(func.getContext(), "Opt", &func, &BB);
            IRBuilder<> builder(func.getContext());
            builder.SetInsertPoint(bb_1);
            builder.CreateRet(builder.getInt32(ans->poly[0]));
            // errs() << ans->poly[0] << '\n';
        }
    }
    // errs() << "After:";
    // errs() << func;
    // outs() << func;
    /*
    // 删除节点释放内存
    for (auto node : Graph) {
        delete node;
    }
*/
    return modified;
}

Node *Algebraic::constructGraph(BasicBlock& BB)
{
    int leafnum = 0;
    int rootnum = 0;
    int count = 0;
    Node *root;
    /// 
    int debug = 0;

    // 对函数块进行筛选，若存在除法等操作，就跳过不进行优化
    for(auto& I:BB) {
        if(I.getOpcode() != Instruction::Add && I.getOpcode() != Instruction::Sub && I.getOpcode() != Instruction::Xor 
            && I.getOpcode() != Instruction::Mul && I.getOpcode() != Instruction::Ret)
            return nullptr;
    }
    for (auto& I : BB) {
        if (I.getOpcode() == Instruction::Add || I.getOpcode() == Instruction::Sub || I.getOpcode() == Instruction::Xor
            || I.getOpcode() == Instruction::Mul) {
            addInstrToGraph(I);
        }
    }

    for(auto& g : Graph)
    {
        count++; 
        if(g->deps.empty() && dyn_cast<Constant>(g->value)==nullptr) {
            leafnum++;
            if(debug) {
                errs() << *(g->value) << '\n';
                errs() << count << " leaf\n";
                errs() << '\n';
            }
        }
        if(g->users.empty()) {
            rootnum++;
            if(debug) {
                errs() << *(g->value) << '\n';
                errs() << count << " root\n";
                errs() << '\n';
            }
        }   
    }
    if(debug)
        errs() << leafnum <<' ' << rootnum <<' '<<count<<'\n';
    if(leafnum != 1 && rootnum != 1) {
        // 对于单个变量的优化
        return nullptr;
    }
    else {
        errs() << "Pattern recognized !!!\n";
        // 提取多项式的系数
        for(auto& g : Graph) {
            if(g->users.empty())
                root = g;
            if(dyn_cast<ConstantInt>(g->value)!=nullptr)
                // 如果是常数 那么转为int存入
                g->poly.push_back(dyn_cast<ConstantInt>(g->value)->getLimitedValue());
            else {
                if(g->deps.empty()) {
                    // g为叶子节点的情况
                    g->poly.push_back(0);
                    g->poly.push_back(1);
                } 
                 else {
                    // g为根节点
                    // 如果g这个节点表示的是加法
                    if(dyn_cast<Instruction>(g->value)->getOpcode() == Instruction::Add){
                        if(g->deps.size()==1) {
                            // 这个语句的加法项相同
                            auto it1 = g->deps.begin();
                            int sz=(*it1)->poly.size();
                            vector<int> v(sz);
                            // 相同的次数的系数相加
                            for(int i=0;i<sz;i++) 
                                v[i]+=(*it1)->poly[i];
                            for(int i=0;i<sz;i++) 
                                v[i]+=(*it1)->poly[i];
                            // 把最高次系数为0时，直接删去
                            while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                            
                            g->poly=v;
                        }
                        else {
                            // 这个语句的加法项不同
                            auto it1 = g->deps.begin();
                            auto it2 = it1;
                            it2++;
                            int sz=max((*it1)->poly.size(),(*it2)->poly.size());
                            vector<int> v(sz);
                            // 相同的次数的系数相加
                            for(int i=0;i<(*it1)->poly.size();i++) 
                                v[i]+=(*it1)->poly[i];
                            for(int i=0;i<(*it2)->poly.size();i++) 
                                v[i]+=(*it2)->poly[i];
                            while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                            g->poly=v;
                        }
                    } else if(dyn_cast<Instruction>(g->value)->getOpcode() == Instruction::Mul){
                        // 这个语句是乘法的情况
                        if(g->deps.size()==1){
                            auto it1=g->deps.begin();
                            int sz1=(*it1)->poly.size();
                            int sz=sz1+sz1-1;
                            vector<int> v(sz);
                            for(int i=0;i<sz1;i++)
                                for(int j=0;j<sz1;j++)
                                    v[i+j]+=(*it1)->poly[i]*(*it1)->poly[j];
                            while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                            g->poly=v;
                        }else {
                            auto it1=g->deps.begin();
                            auto it2=it1;
                            it2++;
                            int sz1=(*it1)->poly.size();
                            int sz2=(*it2)->poly.size();
                            int sz=sz1+sz2-1;
                            vector<int> v(sz);
                            for(int i=0;i<sz1;i++)
                                for(int j=0;j<sz2;j++)
                                    v[i+j]+=(*it1)->poly[i]*(*it2)->poly[j];
                            while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                            g->poly=v;
                        }
                    } else if(dyn_cast<Instruction>(g->value)->getOpcode() == Instruction::Xor) {
                        if(g->deps.size()==1) {
                            // 两个项相同xor
                            vector<int> v;
                            v.push_back(0);
                            g->poly=v;
                        }
                        else {
                            auto it1=g->deps.begin();
                            auto it2=it1;
                            it2++;
                            int sz1=(*it1)->poly.size();
                            int sz2=(*it2)->poly.size();
                            if(sz1==1 && (*it1)->poly[0]==-1)
                            {
                                vector<int> v(sz2);
                                for(int i=0;i<sz2;i++)
                                    v[i] = -(*it2)->poly[i];
                                v[0]--;
                                while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                                g->poly=v;
                            }
                            if(sz2==1 && (*it2)->poly[0]==-1)
                            {
                                vector<int> v(sz1);
                                for(int i=0;i<sz1;i++)
                                    v[i] = -(*it1)->poly[i];
                                v[0]--;
                                while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                                g->poly=v;
                            }
                        }
                    } else if(dyn_cast<Instruction>(g->value)->getOpcode() == Instruction::Sub) {
                         if(g->deps.size()==1) {
                            // 这个语句的减法项相同
                            vector<int> v;
                            v.push_back(0);
                            g->poly=v;
                        }
                        else {
                            // 这个语句的减法项不同
                            auto it1 = g->deps.begin();
                            auto it2 = it1;
                            it2++;
                            int sz=max((*it1)->poly.size(),(*it2)->poly.size());
                            vector<int> v(sz);
                            // 相同的次数的系数相加
                            for(int i=0;i<(*it1)->poly.size();i++) 
                                v[i]+=(*it1)->poly[i];
                            for(int i=0;i<(*it2)->poly.size();i++) 
                                v[i]-=(*it2)->poly[i];
                            while(v[v.size()-1]==0 && v.size()>1) v.pop_back();
                            g->poly=v;
                        }
                    }
                }
            }
        }
        if(debug) {
            for(auto & g :Graph) {
                errs() << *(g->value) <<'\n';
                for(int i=0;i<g->poly.size();i++)
                    if(g->poly[i] != 0)
                    errs() << i << "::" << g->poly[i] << '\n';
                errs() << '\n';
            }
        }
    }
    return root;
}


void Algebraic::addInstrToGraph(Instruction& I)
{
    Node* lhs = findOrCreateInGraph(I.getOperand(0));
    Node* rhs = findOrCreateInGraph(I.getOperand(1));
    Node* node = findOrCreateInGraph(dyn_cast<Value>(&I));

    /*
    errs()<<*(I.getOperand(0))<<" * "<<*(I.getOperand(1))<<" * ";
    errs()<<I.getOpcode()<<'\n';
    */

    node->deps.push_back(lhs);
    node->deps.push_back(rhs);
    lhs->users.insert(node);
    rhs->users.insert(node);
}

Node* Algebraic::findOrCreateInGraph(Value* v)
{
    for (Node* node : Graph) {
        if (node->value == v)
            return node;
    }

    auto node = new Node;
    node->value = v;
    node->deps.clear();     // 叶子节点的deps为空
    node->users.clear();    // 根节点的users为空
    Graph.push_back(node);
    return node;
}

