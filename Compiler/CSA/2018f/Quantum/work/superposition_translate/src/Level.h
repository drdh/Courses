#pragma once

#include "Instruction.h"
#include <set>

class Level {
   public:
    Instruction level_head;
    Instruction* head = &level_head;
    int line = 0;
    int length;
    bool is_operation = true;
    Instruction* tail = head;

    inline void insert(Instruction* a) {
        tail->next = a;
        a->next = NULL;
        tail = a;
    }

    void print();

    void del(Instruction* a);

    //找到level l中和Q重合的指令
    Instruction* find_overlap_instruction(std::set<Qubit*> Q);
};

int levelize(Level tower[], Instruction codes[], Qubit qubits[]);
