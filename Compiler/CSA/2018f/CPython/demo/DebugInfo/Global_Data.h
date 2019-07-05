#ifndef CLASS_GLOBALDATA
#define CLASS_GLOBALDATA


#include <map>
#include <sstream>
#include <iostream>
#include <fstream>

#include "Function.h"


#define CPP 1
#define PYTHON 2
#define ROOT 3

class Global_data {
protected:
    std::map<std::string, Function> function_list;
    //函数列表std::string last_function; //上一次执行的函数名
    std::vector<std::string> call_stack; //函数调用栈
    std::string last_function;
public:
    template<class Archive>
    void serialize(Archive &ar) {
        ar(function_list, call_stack, last_function);
    }

    Global_data(); //构造函数
    void Save();//序列化后存到共享内存
    Global_data *Load();//从共享内存取出后反序列化
    Function &get_function(std::string mangle, std::string demangle, char language); //通过函数名获取函数对象
    void add_function(std::string mangle, std::string demangle, char language); //向函数列表中添加函数
    void enter_function(std::string mangle, std::string demangle, char language); //进入函数前的统计操作
    void exit_function(); //退出函数前的统计操作
    std::map<std::string, Function> get_function_list() { return function_list; };
};

#endif