#include "Call.h"

Call::Call(std::string mangle_name, std::string demangle_name, bool is_caller, clock_t begin_time,
           clock_t end_time, double exec_time) {
    this->mangle_name = mangle_name;
    this->demangle_name = demangle_name;
    this->is_caller = is_caller;
    this->begin_time = begin_time;
    this->end_time = end_time;
    this->exec_time = exec_time;
}