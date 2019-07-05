#ifndef CLASS_CALL
#define CLASS_CALL

#include <iostream>


class Call {
    //全局变量
    //被调用者边
public:


    std::string mangle_name; //mangle函数名用于唯一确定函数
    std::string demangle_name;//demangle函数名用于显示
    bool is_caller;//true为caller,false为callee
    clock_t begin_time; //函数开始时间(用于生成调用路径)
    clock_t end_time; //函数结束时间
    double exec_time; //本次调用持续时间
    Call(std::string mangle_name = "", std::string demangle_name = "", bool is_caller = false,
         clock_t begin_time = 0, clock_t end_time = 0, double exec_time = 0.);
    template<class Archive>
    void serialize(Archive &ar) {
        ar(mangle_name, demangle_name, is_caller, begin_time, end_time, exec_time);
    }
};

#endif