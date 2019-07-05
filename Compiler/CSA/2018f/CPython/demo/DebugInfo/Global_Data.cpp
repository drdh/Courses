#include "Global_Data.h"


//根据函数名获取函数对象
Function &Global_data::get_function(std::string mangle, std::string demangle = "", char language = CPP) {
    if (function_list.count(mangle) == 0 && demangle.size() != 0)
        add_function(mangle, demangle, language);
    return function_list[mangle];
}

//当函数不在函数列表中时,加入函数列表
void Global_data::add_function(std::string mangle, std::string demangle, char language) {
    Function func;
    func.set_name(mangle, demangle);
    func.set_language(language);
    function_list.insert(std::pair<std::string, Function>(mangle, func));
}

void Global_data::enter_function(std::string mangle, std::string demangle, char language) {
    //获取被调函数
    Function &callee = get_function(mangle, demangle,language);
    if (!call_stack.empty())
        last_function = call_stack.back();
    call_stack.push_back(mangle);
    //记录被调函数开始执行时间
    clock_t entertime = clock();
    callee.push_entertime(entertime);
    callee.inc_callnum();
}

void Global_data::exit_function() {
    //将被调函数弹出调用栈
    std::string callee_mangle_name = call_stack.back();
    Function &callee = get_function(callee_mangle_name);
    std::string callee_demangle_name = callee.getdemangle_name();
    last_function = callee_mangle_name;
    call_stack.pop_back();
    //获取主调函数
    std::string caller_mangle_name;
    std::string caller_demangle_name;
    if (!call_stack.empty()) {
        caller_mangle_name = call_stack.back();
        caller_demangle_name = get_function(caller_mangle_name).getdemangle_name();
    } else {
        caller_mangle_name = "%root";
        caller_demangle_name = "root";
    }
    Function &caller = get_function(caller_mangle_name);
    //计算被调函数开始执行时间、结束执行时间、执行时间
    clock_t entertime, exittime = clock();
    entertime = callee.pop_entertime();

    double exectime = double((exittime - entertime)) / CLOCKS_PER_SEC;
    Call callee_edge(callee_mangle_name, callee_demangle_name, false, entertime, exittime, exectime);
    Call caller_edge(caller_mangle_name, caller_demangle_name, true, entertime, exittime, exectime);

    callee.add_newcaller(caller_edge);
    caller.add_newcallee(callee_edge);
    callee.add_newruntime(exectime);
}

Global_data::Global_data() {
    last_function = "%root";
    add_function("%root", "root", ROOT);
    call_stack.push_back("%root");
}