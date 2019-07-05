#include <string.h>
#include <iostream>
#include "Level.h"
#define N 300
using namespace std;

bool overlap(Instruction a, Instruction b) {  //两条指令有重叠返回true，否则false
    if (!a.isOperation() || !b.isOperation()) return true;
    for (Qubit *p : a.getQubits()) {
        for (Qubit *q : b.getQubits())
            if (p == q) return true;
    }
    return false;
}

int levelize(Level tower[], Instruction codes[], Qubit qubits[]) {
    int lastline, i;
    int newline = 1;
    for (i = 2, lastline = 1; codes[lastline].line; i++) {
        int j;
        tower[newline].is_operation = codes[lastline].isOperation();
        for (j = lastline; j < i; j++) {
            if (overlap(codes[i], codes[j])) break;
        }
        if (j < i || codes[i].line == 0) {
            for (j = lastline; j < i; j++) {
                tower[newline].insert(&(codes[j]));
                codes[j].line = newline;
            }
            lastline = i;
            tower[newline].line = newline;
            newline++;
        }
    }
    
    return newline - 1;
}

void Level::print() {
        Instruction* p = head->next;
        cout << line << ": ";
        while (p != NULL) {
            p->printCode();
            cout << ' ';
            p = p->next;
        }
        cout << endl;
    }

void Level::del(Instruction* a) {
    Instruction* p = head;
    while (p->next != NULL) {
        if (p->next == a) break;
        p=p->next;
    }
    if (p->next == NULL){
        return;
    } 
    if (p->next == tail) tail = p;
    p->next = p->next->next;
}

Instruction* Level::find_overlap_instruction(set<Qubit*> Q) {  
    auto p = head->next;
    while (p) {
        for (Qubit *q : p->getQubits()) {
            if (Q.find(q) != Q.end()) return p;
        }
        p = p->next;
    }
    return NULL;
}