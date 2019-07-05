# ClangCheckerGuide

> 说明：此处记录对编写clang checker 相关知识的梳理

[TOC]

## 1.定义Checker的流程

1. 声明一个checker 类，需继承`Checker<...>`，定义要实现的回调函数。通常将该类声明于匿名命名空间中以避免名字冲突。

   ```c++
   namespace {
   class MainCallChecker : public Checker < check :: PreCall > {
      mutable std :: unique_ptr < BugType > BT;    
   public :
      void checkPreCall ( const CallEvent & Call , CheckerContext & C ) const ;
   };
   }
   ```

2. 实现回调函数(实例化)

3. 报告错误

4. 注册checker

   ```c++
   void ento::registerMainCallChecker(CheckerManager &Mgr){
       Mgr.registerChecker<MainCallChecker>();
   }
   ```

## 2. bug report 流程

1. 定义BugType

   ```c++
   BugType(CheckName Check, StringRef Name, StringRef cat);
   BugType(const CheckerBase *Checker, StirngRef Name, StringRef cat);
   
   //example:
   std::unique_ptr<BugType> DoubleCloseBugType;
   DoubleCloseBugType.reset(
       new BugType(this, "Double fclose", "Unix Stream API Error"));
   ```

2. Generate the report

   ```c++
   BugReport(BugType &bt, StringRef desc, const ExplodedNode *errornode);
   
   auto R = llvm::make_unique<BugReport>(*DoubleCloseBugType,
                                         "Closing a previously closed file stream", ErrNode);
   R->addRange(Call.getSourceRange());
   R->markInteresting(FileDescSym);
   C.emitReport(std::move(R));
   ```

> 注：貌似BugReporter只能在Checker里面用，ASTConsumer不太行(我没弄明白)

## 3. ASTConsumer

定义方式：

```c++
//example:
class ClassChecker : public clang::ASTConsumer{
public:
    ClassChecker();
    void HandleTranslationUnit(clang::ASTContext &Context)override;//这个接口必须实现
}
```

> 到底是用Checker模板还是ASTConsumer，应该可以根据实际需要自己选择。

## 4. 一些类

1. StringRef

   ```c++
   StringRef(const char *str);
   StringRef(const char *data, size_t length);
   ```

   这个类一般会用在函数参数，直接用字符串当作参数即可。

## Sunday 01:25 30th December, 2018

1. 建议使用clang7版本，因为两者在AST的方法名上都有所不同。（再议）

2. 首先得跑起来对吧：

官方manual里的Getting Started部分的：

```
clang -cc1 -analyze -analyzer-checker=core.DivideZero test.c
```

是跑不起来的（如果有像include<stdio.h>这样的）（test.c自己编写），因为-cc1选项使用的是clang前端（至于有什么区别，主要在于有的命令行参数clang -cc1有，但是clang没有），使用前端时会清空clang所有的默认选项：主要指的是include的路径等。因此简单的方法就是使用-Xclang选项，使得我们可以在clang使用保留了所有默认选项的-cc1才可以使用的参数（ps：-Xclang只对下一个参数有效）

综上，官方manual的这句话应该改为：

```
clang --analyze -Xclang -analyzer-checker=core.DivideZero test.c
```

便可以执行了.

各位可以直接复制粘贴跑一跑。

参考：

> http://clang-developers.42468.n3.nabble.com/clang-cc1-error-file-not-found-td4044576.html
>
> https://www.cnblogs.com/wangyuxia/p/6568917.html



## 在项目中开始编写一个检查子项目

目前修改的`CmakeLists.txt`和`ToolDriver.cpp`已经可以编译运行，可能还需要改进。

1. 命令行选项：

   在`ToolDriver.cpp`中`HelpMessage`字符串最后加上自己的命令行选项编号，比如`类`在规范中为第五章，所以编号为`5`

   ```c++
     "-c <checker-id> is used to do special check. \n"
     "\t 0 is default that do all checking\n"
     "\t 5 for class check\n";
   ```

2. 添加checker进最后生成的可执行文件中：

   如果是`ASTConsumer`:

   ```c++
     if(/*根据CheckType这个opt判断*/){
         Consumers.push_back(std::move(
           std::unique_ptr<ClassChecker>(new ClassChecker)
         ));
     }
   ```

   如果是`Checker`:

   ```c++
     if (/*根据op进行判断*/) {
       AnalysisConsumer->AddCheckerRegistrationFn([] (CheckerRegistry& Registry) {
         Registry.addChecker<ModuleChecker>("nasac.ModuleChecker","No desc");
       });
       AnalyzerOptions->CheckersControlList.push_back({"nasac.ModuleChecker", true});
     }
   ```

   > 为每章的检查建立一个子文件夹，并定义一个类，然后在ToolDriver.cpp中include

3. 关于测试：

   ```bash
   # 在G1-*文件夹下
   mkdir build ; cd build
   cmake ..
   make
   # 可执行文件在src/目录下，即G1-*/build/src
   ```

   在`test`目录下创建相关子目录，比如`ClassCheckTest`，然后在该目录下存放测试用例，建议编写测试脚本(强制？)，比如：

   ```bash
   # test/ClassCheckTest/test,sh
   checker='../../build/src/google-cpp-coding-style-checker'
   ${checker} . -c=5
   ```

## 利用clang-check查看测试文件的AST

使用`clang-check`这个工具可以查看测试文件的`ast`，这样可以知道应该调用哪个函数来对相关节点进行访问。使用方法如下(假设处于测试目录下，比如ClassCheckTest)：

### 1. 创建compilation database

`clang-check`需要compilation database才能工作，这一步由`cmake`完成：

首先要编写CMakeLists.txt文件，大致内容为：

```cmake
cmake_minimum_required(VERSION 3.4.3)

project(ClassCheckTest)

add_executable(hello hello.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) ## 这个命令是必须的 
```

创建build目录，防止对测试目录的污染：

```bash
mkdir build ; cd build
cmake ..
make
```

### 2. 使用clang-check

完成上面步骤后在`build`目录下：

```bash
clang-check ../hello.cpp -ast-dump -ast-dump-filter=Hello > dump.txt
```

随后查看`dump.txt`即可。结果示例：

```
|-CXXRecordDecl 0x5561f0115c48 <col:1, col:7> col:7 implicit referenced class Hello
|-AccessSpecDecl 0x5561f0115ce0 <line:2:1, col:7> col:1 public
|-CXXConstructorDecl 0x5561f0115db0 <line:3:5, col:11> col:5 Hello 'void ()'
|-CXXDestructorDecl 0x5561f0115eb0 <line:4:5, col:12> col:5 ~Hello 'void ()' noexcept-unevaluated 0x5561f0115eb0
|-AccessSpecDecl 0x5561f0115f60 <line:5:1, col:7> col:1 public
|-FieldDecl 0x5561f0115fa0 <line:6:5, col:9> col:9 referenced pub_data 'int'
|-CXXMethodDecl 0x5561f0116078 <line:7:5, col:45> col:9 get_pub_data 'int () const'
```

> 注：不要把这里创建的文件push到仓库中

## 推荐的debug方式

1. 在debug的时候可能需要知道函数调用流程的信息，单步调试显然效率太低，单纯的printf大法后面又得将所有的printf注释掉，这时候可以使用下面这种方式debug:

```c++
//在文件头定义一些宏
#if 1 // debug需要输出的时候为1，不需要的时候为0.也可以用#ifdef DEBUG 这个需要在前面DEBUG宏
#define PRINTF(str) printf("%s\n",str)
#define BEGIN(str) printf("%s begin\n",str)
#define END(str) printf("%s end\n",str)
#else 
#define PRINTF(str)
#define BEGIN(str)
#define END(str)
#endif
//使用
void func(){
    BEGIN("func");
    //some stmts
    PRINTF(str);
    //some stmts
    END("func");
}

//记得在文件末尾使用#undef
#undef PRINTF
#undef BEGIN
#undef END
```

2. 使用可边长参数调试时输出调试信息，正式发布时则不输出，可以这样

   ```
   #ifdef DEBUG
   #define LOG(format, ...)     printf(stdout, format, ##__VA_ARGS__)
   #else
   #define LOG(format, ...)
   #endif
   ```

   在调试环境下，LOG宏是一个变参输出宏，以自定义的格式输出；

   在发布环境下，LOG宏是一个空宏，不做任何事情。

3. 最后还是建议使用gdb，可以在不需要重新编译增加print的情况下多次调试，且有不逊色于IDE的代码界面，单步执行功能，能够极大提高调试效率，请务必尝试（准备工作都帮做好了）XD

## ASTMatcher使用方式

### 1. 包含必要文件

```c++
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatcherFinder.h"
```

### 2. 添加需要抽取的AST节点

```c++
//此句可放于文件前面作全局变量
std::unique_ptr<ClassCheckCallBack> callback(new ClassCheckCallBack);
	MatchFinder m;
	//callback是啥见3
	//cxxRecordDecl即为AST节点，在clang-check中class Hello的节点属性为CXXRecordDecl,
	//其它同理
	//bind的内容与后面callback的getNodes对应
    m.addMatcher(cxxRecordDecl().bind("cxxRecordDecl"),callback.get());
    m.addMatcher(accessSpecDecl().bind("accessSpecDecl"),callback.get());
    m.addMatcher(cxxConstructorDecl().bind("cxxConstructorDecl"),callback.get());
    m.addMatcher(cxxDestructorDecl().bind("cxxDestructorDecl"),callback.get());
    m.addMatcher(fieldDecl().bind("fieldDecl"),callback.get());
    m.addMatcher(cxxMethodDecl().bind("cxxMethonDecl"),callback.get());
    m.matchAST(context);//必须
```

### 3. 定义callback类

需要继承`clang::ast_matchers::MatchFinder::MatchCallback`，然后覆盖`run`方法：

```c++
class ClassCheckCallBack: public clang::ast_matchers::MatchFinder::MatchCallback{
public:
    ClassCheckCallBack(){}

public:
    virtual void run(const clang::ast_matchers::
                MatchFinder::MatchResult &Result)final{
        //函数参数与前面的bind对应
        if(const auto decl = Result.Nodes.getNodeAs<
                    clang::CXXRecordDecl>("cxxRecordDecl")){
            
            COUT(decl->getIdentifier()->getName().data());
            auto specifier = decl->getAccess();
            
        }
        else if(const auto decl = Result.Nodes.getNodeAs<
                    clang::AccessSpecDecl>("accessSpecDecl")){
            
            COUT(decl->getDeclKindName());
        }
        else if(const auto decl = Result.Nodes.getNodeAs<clang::CXXConstructorDecl>("cxxConstructorDecl")){
            COUT(decl->getDeclName().getAsString());
        }
        else if(const auto decl = Result.Nodes.getNodeAs<clang::CXXDestructorDecl>("cxxDestructorDecl")){
            COUT(decl->getDeclName().getAsString());
        }
        else if(const auto decl = Result.Nodes.getNodeAs<clang::FieldDecl>("fieldDecl")){
            COUT(decl->getDeclName().getAsString());
        }
        else if(const auto decl = Result.Nodes.getNodeAs<clang::CXXMethodDecl>("cxxMethodDecl")){
            COUT(decl->getDeclName().getAsString());
        }
        else{
            COUT("Error Occur!");
        }

    }
};

```

More:

> https://llvm.org/devmtg/2013-04/klimek-slides.pdf

