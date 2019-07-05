# 对存储数据结构的修改

代码更新在nm_extrator

1. 加入了序列化和共享内存部分

   > 这里引入了两个库boost.Interprocess用于共享内存管理，cereal用于序列化/反序列化
   >
   > 如果需要在共享内存中直接放置STL容器和自定义类，需要对每一种类型构建新的allocator和shm
   >
   > _type，因此对比较复杂的嵌套结构，工作量是非常大的。
   >
   > 我们这里采用了序列化成字符串再存进共享内存的方式，只需要为string这个STL构建allocator，取出时可以相应地反序列化后写入类。
   >
   > 序列化时，我们在boost.serialize和cereal中选择了后者，因为7前者需要对boost进行预先编译，后者可以直接引入头文件和项目一起编译。

2. 将struct Caller和struct Callee合并成了class Call，并增加了构造函数，仅在实例化时通过新增的is_caller变量区分

   > 考虑到Caller和Callee的成员变量完全相同，为了减少序列化时的工作和减小代码的冗余，进行了这个修改

3. 将内部所有的指针存储修改为值存储，【在副本上操作】的问题通过在操作时采用引用解决

   > 显然，共享内存无法放置指针，原因如下：
   >
   > ```mermaid
   > graph LR
   > A[A process]-->|put|B[shared memory]
   > A[A process]-->|pointer|D[A memory]
   > C[B process]-->|get|B[shared memory]
   > C[B process]-->|not valid!|D[A memory]
   > 
   > 
   > ```
   >
   > 对于进程A，其数据结构中若存储指针，该指针会指向A所属的内存区域，如果这时对指针进行序列化，而进程B访问时，获得该指针指向的对象位于A所属的内存区域而不是共享内存，因此会出现非法访问。
   >
   > P.S.相关资料同时提到，如果一定要使用指针的话，应该使用C++的智能指针代替，考虑到demo应该尽量简洁，我们选择了直接消除指针

4. 将name分割成mangle_name和demangle_name

   > 由于C++的多态特性，函数名并不能唯一指定一个函数，因此我们引进mangle_name用于作为key来唯一确定一个函数对象，demangle_name作为阅读友好的结构用于打印输出。
   >
   > 但是，这里我只是简单地给部分位置增加了两个新属性，可能引入了一些冗余，在后面的优化过程中可能需要修改。

# 遇到的问题以及解决方案

使用nm时，对于定义在namespace中的纯函数，nm将会忽略命名空间和函数参数，因此我们无法获得函数具体的命名空间和其参数。

例如对于下面的代码(extension0/eculid.cpp)：

```c++
#include "eculid.h"

namespace eculid {
    int gcd(int m, int n)
    {
        int i;
        if (m  < n)
            i = m;
        else
            i = n;
        for (; i > 0; i--)
            if (m % i == 0 && n % i == 0)
                break;
        return i;
    }

    Net::Net()
    {
        printf("This is constructor.\n");
    }

    void Net::forward()
    {
        printf("This is Net::forward\n");
    }
}
```

其nm解析的符号表如下：

```
0000000000201028 B __bss_start
                 w __cxa_finalize
0000000000201028 D _edata
0000000000201030 B _end
0000000000000794 T _fini
0000000000000710 T gcd
                 w __gmon_start__
00000000000005c0 T _init
                 w _ITM_deregisterTMCloneTable
                 w _ITM_registerTMCloneTable
                 w _Jv_RegisterClasses
                 U puts
0000000000000778 T eculid::Net::forward()
000000000000075c T eculid::Net::Net()
000000000000075c T eculid::Net::Net()
```

注意到，其中的gcd命名空间和参数类型被完全消去。

原因[链接](https://stackoverflow.com/questions/4186165/best-practise-and-semantics-of-namespace-nested-functions-and-the-use-of-extern)及其具体解释：

因为在eculid.h中命名空间被extern "C" 修饰，其中所有函数被视作C风格，对于顶层的函数，因为C中没有命名空间和多态的概念，所以说所有的命名空间和参数不同的函数会被认为是同一个函数，自然其符号表就不会显示命名空间和参数。

暂定的解决方案如下(Todo)：

1. 对于命名空间的问题，按照上述文中所述，使用extern "C" 修饰的函数，编译时其所在的命名空间会被完全忽略，因此我们可以不写命名空间在全局定义，函数本身使用extern "C" 修饰(已经测试确认可用)
2. 对于参数的问题，我们拟在hook函数中采用模板类可变参数(等待测试)

# 下一步工作

1. 完成以上问题的解决
2. 将代码生成部分和debug存储的数据结构完成连接和相关的测试工作
3. 和python端的测试连接