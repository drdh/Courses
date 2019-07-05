#include "Function.h"

//函数类的构造函数
Function::Function() {
    call_num = 0;
    type = DEFAULT_TYPE;
    total_runtime = 0;
    min_runtime = 0;
    max_runtime = 0;
}

void Function::set_name(std::string mangle, std::string demangle) {
    mangle_name = mangle;
    demangle_name = demangle;
}

void Function::inc_callnum() {
    call_num++;
}

void Function::push_entertime(clock_t entertime) {
    entertime_list.push(entertime);
}

clock_t Function::pop_entertime() {
    clock_t entertime;
    entertime = entertime_list.top();
    entertime_list.pop();
    return entertime;
}

void Function::add_newruntime(double runtime) {
    total_runtime += runtime;
    if (min_runtime > runtime)
        min_runtime = runtime;
    else if (min_runtime < 1e-8)
        min_runtime = runtime;
    if (max_runtime < runtime)
        max_runtime = runtime;
    else if (max_runtime < 1e-8)
        max_runtime = runtime;
}

void Function::add_newcallee(Call callee) {
    callee_list.push_back(callee);
}

void Function::add_newcaller(Call caller) {
    caller_list.push_back(caller);
}

void Function::set_type(char func_type) {
    type = func_type;
}

void Function::set_language(char func_language){
    language = func_language;
}

void Function::print() {
    std::cout << "name: " << demangle_name << std::endl;
    std::cout << "call_num: " << call_num << std::endl;
    std::cout << "type: " << (int) type << std::endl;
    std::cout << "total_runtime: " << total_runtime << std::endl;
    std::cout << "min_runtime: " << min_runtime << std::endl;
    std::cout << "max_runtime: " << max_runtime << std::endl;
    std::cout << "language:" << (int) language << std::endl;
    std::cout << "callers: " << std::endl;
    for (auto iter = caller_list.begin(); iter != caller_list.end(); iter++) {
        std::cout << iter->demangle_name << ":\tbegintime:" << iter->begin_time <<
                  "\tendtime:" << iter->end_time << "\texectime:" << iter->exec_time << std::endl;
    }
    std::cout << "callees: " << std::endl;
    for (auto iter = callee_list.begin(); iter != callee_list.end(); iter++) {
        std::cout << iter->demangle_name << ":\tbegintime:" << iter->begin_time <<
                  "\tendtime:" << iter->end_time << "\texectime:" << iter->exec_time << std::endl;
    }
}
