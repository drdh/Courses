#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "Qubit.h"

class Instruction {
private:
    enum InstructionType {
        Operation = 0,
        Measure = 1,
        Decline = 2
    };

    inline static InstructionType judge(std::string &instruction) {
        if (instruction == "qreg" || instruction == "creg")
            return Decline;
        else if (instruction == "measure")
            return Measure;
        else
            return Operation;
    }


    inline static int hashi(Qubit* qubits, std::string &name) {
        int i, sum;
        for (i = 0; qubits[i].sign; i++) {
            if (name == qubits[i].getName()) return i;
        }
        return i;
        /*
        for (i = 0, sum = 0; name[i]; i++) {
            if (name[i] == ' ') continue;
            sum += name[i] - 'A';
        }
        while (qubits[sum].sign) {
            /*cout << "new:" << name << "\norigin:" << qubits[sum].getname()
                << "compare:" << strcpy(qubits[sum].getname(), name) << endl;
            if (strcmp(qubits[sum].getname(), name) == 0) break;
            sum++;
        }
        return sum;*/
    }

    // 从code[pos]开始解析出一个以空格结尾的子串，然后移动pos到下一位置
    // 需要保证pos不超过code的有效范围
    inline std::string getSubstr(size_t &pos) {
        size_t start = pos;
        while(pos < code.size() && !isblank(code[pos]) && code[pos] != ',' && code[pos] != ';')
            pos++;
        size_t end = pos;
        do {
            pos++;
        } while(pos < code.size() && isblank(code[pos]));
        return code.substr(start, end - start);
    }
    
    inline void insertQubit(Qubit* q) {
        qubits.push_back(q);
    }
    
    inline void printCode(std::ostream &os = std::cout) const {
        if (is_void)
            return;
        os << code;
    }

    inline bool isEnd() {
        return code == "end";
    }

    inline bool isEmpty() {
        for(char c : code)
            if(!isblank(c)) return false;
        return true;
    }

public:
    inline Qubit* getQubit(int i) {
         return qubits[i];
    }

    inline const std::vector<Qubit *> &getQubits() {
        return qubits;
    }

    inline bool isOperation() {
        return is_operation;
    }

    inline void print(std::ostream &os = std::cout) const {
        os << line << ": ";
        printCode();
        os << std::endl;
    }

    inline int getLine() {
        return line;
    }

    // u: 当前已使用qubit数
    // line: 行号
    // qubits: qubit列表
    inline bool input(int &u, int &line, Qubit *qubits, std::istream &is = std::cin) {
        if(!(std::getline(is, code)) || isEnd())
            return false;
        if(isEmpty())
            return true;

        this->line = line++;

        size_t pos = 0;
        auto prefix = getSubstr(pos);
        if (judge(prefix) != Operation) {  //判断是否为门操作
            is_operation = false;
            return true;
        }
        
        is_operation = true;

        while (pos < code.size()) {
            auto qub = getSubstr(pos);
            int tmp = hashi(qubits, qub);
            if (qubits[tmp].sign == 0) {
                qubits[tmp].setName(qub);
                qubits[tmp].sign = 1;
                u++;
            }
            qubits[tmp].insert(this);
            insertQubit(&qubits[tmp]);
        }
        return true;
    }

    //判断是否所有比特的mark位都不为1，是则返回true
    inline bool judgeQubits() const {
        for (Qubit *p : qubits) {
            if (p->mark == 0)
                return false;
        }
        return true;
    }

    //将所有qubit的信息置为state
    void setQubits(int state) {
        for (Qubit *p : qubits) {
            p->mark = state;
        }
    }

public:
    int line = 0;
    int mark = 0;  // can do every thing you want, is not used in class
    Instruction* next = NULL;
private:
    bool is_operation = false;
    bool is_void = false;
    std::vector<Qubit*> qubits;
    std::string prefix;
    std::string code;

};
