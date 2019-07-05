#include <string.h>
#include <iostream>
#include <set>
#include "Lexer.h"
#include "Level.h"
#define N 300
#define IFINITY 9999
#define Visited 1
using namespace std;

void insert_codes_qubit(Instruction a, set<Qubit*>& Q) {
    for (Qubit *p : a.getQubits()) {
        Q.insert(p);
    }
}

bool check_code_overlap_set(Instruction a, set<Qubit*> Q) {  //检查一个ins和一个set是否重合
    for (Qubit *q : a.getQubits()) {
        if (Q.find(q) != Q.end()) return true;
    }
    return false;
}
/**/
int find_next_level(set<Qubit*> qubits_set,
                    int l) {  //找到level l之后和集合qubits_set重合的最近的level
    auto start = qubits_set.begin();
    int min = IFINITY;
    for (; start != qubits_set.end(); start++) {
        auto p = *start;
        for (auto i : p->getInstructions()) {
            if (i->getLine() > l) {
                if(i->getLine() < min)
                    min = i->getLine();
                break;
            }
        }
    }
    return min;
}

int translate(Level tower[]) {
    Instruction codes[N];
    Qubit qubits[N];
    int total_qubits, total_ins;
    Lexer(codes, qubits, total_qubits, total_ins);
    //Level tower[N];
    printf("分层处理后：\n\n");
    int total_level = levelize(tower, codes, qubits);
    for (int i = 1; i < total_level; i++) tower[i].print();
    int i;
    printf("\n");
    for (i = 0; qubits[i].sign; i++) {
        qubits[i].print();
        cout << endl;
        qubits[i].mark = 0;
    }
    /**/
    printf("\n代码变换后：\n\n");
    for (i = 1; i <= total_level; i++) {
        // tower[i].print();
        if (tower[i].is_operation == false) continue;
        auto p = tower[i].head->next;
        while (p != NULL) {  //一次while处理一条指令的合并

            if (p->judgeQubits() == true) {  //如果指令的所有比特都已经处于叠加态，则不需要优化合并
                p = p->next;
                continue;
            }
            // p->print();
            p->setQubits(Visited);  //将p的所有比特置位

            if (p->mark == Visited) {
                continue;
                p = p->next;
            }
            Instruction* ins_unit[10] = {p};  // ins_unit装着所有可以并到一起的ins的指针
            int unit_begin = 1;               // ins_unit数组下一个加入的ins存放的下标
            p->mark = Visited;
            set<Qubit*> qubits_set;
            insert_codes_qubit(*p, qubits_set);
            p = p->next;
            int cur_level = i;
            /*
            if (tower[i].line==30){
                cout<<"***************"<<endl;
                p->printcode();
                printf("\n##########\n");
                for (int u=0;u<9;u++){
                    tower[i+u].print();
                }
                cout<<"***************"<<endl;
            }*/
            while (qubits_set.size() < total_qubits) {
                auto tmp_level =
                    find_next_level(qubits_set, cur_level);  //找到和当前set重合的下一个level
                if (tmp_level == IFINITY) break;
                cur_level = tmp_level;

                auto next_instruction = tower[cur_level].find_overlap_instruction(qubits_set);
                next_instruction->mark = Visited;
                ins_unit[unit_begin] = next_instruction;
                unit_begin++;
                insert_codes_qubit(*next_instruction, qubits_set);
            }
            cur_level = ins_unit[unit_begin - 1]->line;
            unit_begin--;
            // printf("begin: %d, cur\n",unit_begin);
            for (int u = unit_begin - 1; u >= 0; u--) {
                auto ins = ins_unit[u];
                int l = ins->line;
                if (l == cur_level - unit_begin + u) continue;
                tower[l].del(ins);
                /*
            if (tower[i].line==30){
                printf("break\n");
            }                */
                tower[cur_level - unit_begin + u].insert(ins);
            }

            // return 0;
        }
    }
    for (i = 1; i <= total_level; i++) {
        tower[i].print();
    }

    return total_level;
}