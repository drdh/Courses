#pragma once
#include <iostream>
#include <vector>
#include <string>

class Instruction;

class Qubit {
public:
    inline const std::string &getName() const {
        return name; 
    }

    inline void setName(const char* newname) {
        name = newname;
    }

    inline void setName(const std::string& newname) {
        name = newname;
    }

    inline const std::vector<Instruction *> &getInstructions() const{
        return instructions;
    }

    inline void insert(Instruction* n) {
        instructions.push_back(n);
    }

    inline bool operator==(const Qubit& a) const {
        return name == a.name;
    }

    inline bool operator<(const Qubit& a) const {
         return name < a.name; 
    }

    void print(std::ostream &os = std::cout) const;

public:
    int sign = 0;
    int mark = 0;  // can do every thing you want, is not used in class

private:
    std::string name;
    std::vector<Instruction *> instructions;    //该比特出现在哪些instruction中的指针
};
