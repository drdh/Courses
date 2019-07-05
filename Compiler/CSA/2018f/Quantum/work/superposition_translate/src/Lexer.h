#pragma once

#include "Qubit.h"
#include "Instruction.h"

#define N 300


inline void Lexer(Instruction codes[], Qubit qubits[], int& total_qubit, int& total_instruction) {
    int j = 1/*line number*/;
    // Qubit qubits[N];
    // int findqubit[100] = {};
    int u = 0;
    while (codes[j].input(u, j, qubits));

    total_instruction = j - 1;
    total_qubit = u;
    /*
    for (i = 0; i < u; i++) {
        qubitqueue[i] = qubits[findqubit[i]];
    }
    cout << endl;*/
}