#ifndef CLASS_FUNCTION
#define CLASS_FUNCTION


#include <vector>
#include <stack>

#include "Call.h"

#define DEFAULT_TYPE 0
#define CPP 1
#define PYTHON 2

#define PropertyBuilderByName(type, name)\
    public:\
    inline type get##name() {\
        return name;\
    }\


class Function {
protected:
    std::string mangle_name; //mangle函数名用于唯一确定函数
    std::string demangle_name;//demangle函数名用于显示
    int call_num; //函数被调用的次数
    char type; //函数类型
    char language;  //函数在c++侧还是python侧
    double total_runtime; //函数总执行时间
    double min_runtime; //函数单次最小执行时间
    double max_runtime; //函数单次最大执行时间
    std::stack<clock_t> entertime_list; //函数进入时间栈(用于递归)
    std::vector<Call> callee_list; //函数的被调用者序列
    std::vector<Call> caller_list; //函数的调用者序列
public:
    template<class Archive>
    void serialize(Archive &ar) {
        ar(mangle_name, demangle_name, call_num, type, language, total_runtime, min_runtime, max_runtime, entertime_list,
           callee_list, caller_list);
    }

    Function(); //构造函数
PropertyBuilderByName(std::string, mangle_name)//宏替换生成get/set不需要分号
PropertyBuilderByName(std::string, demangle_name)//宏替换生成get/set不需要分号
PropertyBuilderByName(char, language)
PropertyBuilderByName(int, call_num)//宏替换生成get/set不需要分号
PropertyBuilderByName(char, type)//宏替换生成get/set不需要分号
PropertyBuilderByName(double, total_runtime)//宏替换生成get/set不需要分号
PropertyBuilderByName(double, min_runtime)//宏替换生成get/set不需要分号
PropertyBuilderByName(double, max_runtime)//宏替换生成get/set不需要分号
PropertyBuilderByName(std::stack<clock_t>, entertime_list)//宏替换生成get/set不需要分号
PropertyBuilderByName(std::vector<Call>, callee_list)//宏替换生成get/set不需要分号
PropertyBuilderByName(std::vector<Call>, caller_list)//宏替换生成get/set不需要分号


    void set_name(std::string mangle, std::string demangle); //设定函数名
    void inc_callnum(); //增加函数被调次数
    void add_newruntime(double runtime); //增加一次函数的执行时间
    clock_t pop_entertime(); //弹出并返回一个进入时间
    void push_entertime(clock_t entertime); //弹入一个进入时间
    void add_newcallee(Call callee); //增加被调用者
    void add_newcaller(Call caller); //增加调用者
    void set_type(char func_type); //设定函数类型
    void set_language(char language);
    void print(); //打印函数相关信息

};

#endif
