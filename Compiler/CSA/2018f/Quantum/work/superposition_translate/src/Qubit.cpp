#include "Qubit.h"
#include "Instruction.h"

void Qubit::print(std::ostream &os) const {
    os << name << ": ";
    for (auto i : instructions) {
        os << i->getLine() << " ";
    }
    os << std::endl;
}