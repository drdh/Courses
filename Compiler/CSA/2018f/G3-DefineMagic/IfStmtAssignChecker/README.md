# if语句条件中赋值行为检查
## 功能说明
众所周知，在C语言中，if语句条件中如果发生赋值行为，那么赋值语句的RHS将会作为这个语句的值，也会作为一个合法的布尔表达式，因此是一个合法的条件，能正常编译通过。但是，通常程序员希望的其实是进行等于关系的检查（`==`)，而并不想进行赋值操作。这样的错误往往很难检查出来，需要花费大量不必要的时间精力，大幅降低程序员的生产效率。因此我们希望设计一个checker，能够自动检查并提醒程序员在if语句中发生的赋值。   
在GNU C Coding Standards的5.3节[5.3 Clean Use of C Constructs](https://www.gnu.org/prep/standards/html_node/Syntactic-Conventions.html#Syntactic-Conventions)中也建议了尽量不要在if的条件中进行赋值：
> Try to avoid assignments inside if-conditions (assignments inside while-conditions are ok). For example, don’t write this:
```
if ((foo = (char *) malloc (sizeof *foo)) == NULL)
  fatal ("virtual memory exhausted");
```
> instead, write this:
```
foo = (char *) malloc (sizeof *foo);
if (foo == NULL)
  fatal ("virtual memory exhausted");
```
## 程序设计
我们采用在clang的StaticAnalyzer的基础上添加一个checker的方法，这样具备极高的灵活性，我们可以结合CSA中已有的checker，灵活地设定我们希望运行哪些checkers来分析我们的程序。这样使得我们可以在进行if语句赋值检查的同时顺带运行其他的checkers（例如是否给指针赋予立即数的地址，数组下标是否越界）。   
我们在`lib/StaticAnalyzer/Checkers`目录下建立一个`IfStmtAssignChecker.cpp`实现我们自己的checker。我们继承`Checker<check::BranchCondition>`类并且判断当前的条件是否处在一个if语句中，如果处在if语句中就检查是否发生赋值行为。如果发生了if语句中的赋值，那么就发出一条warning（`NonFatalErrorNode`）提醒程序员进行相应的修改。   
值得注意的是，`if语句中的赋值是不提倡的，但在while这样的语句条件中进行赋值是合法的。有时这也正是我们所期望的行为，即在while的条件中更新循环变量。

## 实现过程中遇到的问题
本次实验中遇到的最大的障碍可能就是学习如何使用clang front end提供的API接口在clang自带的Static Analyzer的基础上增加一个Checker，并集成到Clang Static Analyzer中。LLVM官方在这方面的文档较少，很多API仅有简单的doxygen生成的简略文档，没有具体的例子。   
在调研之后发现，CSA的Checker是通过继承`clang::ento::check<>`实现。[CheckerDocumentation.cpp](https://clang.llvm.org/doxygen/CheckerDocumentation_8cpp_source.html)中有各个checker类提供的详细功能的介绍。既然是希望检查所有的`if`语句，我起初使用`check::PreStmt<IfStmt>`的方式实现，在clang分析器处理每个`if`语句之前调用`void checkPreStmt(const ReturnStmt *DS, CheckerContext &C) const`进行处理。但是实现后进行测试时才发现`PreStmt`和`PostStmt`会跳过所有控制流语句，包括`if`语句，因为这些语句是不包含在`CFGElement`中。`CFGElement`是个顶层的基本块的表示，而控制流语句不包含在其中。`PreStmt`和`PostStmt`是对所有暴露的`CFGElement`进行处理的。   
在遇到了这个问题之后，在仔细阅读相关文档后，决定采用`check::BranchCondition`，检查每一个分支条件，并且判断其是否处在一个`if`语句的条件中，如果是，则进行相应的检查。

## 使用
为了正常使用我们所编写的checker，我们需要将我们的checker加入clang的源代码中，并且重新编译clang。    
首先，将`IfStmtAssignChecker.cpp`拷贝到clang项目的`lib/StaticAnalyzer/Checkers`目录下（clang项目通常已经被放到llvm的源码目录中了，例如位于`llvm/tools/clang`）。其次，我们需要打开该目录（`lib/StaticAnalyzer/Checkers`）下的`CMakeLists.txt`，在`add_clang_library`的末尾添加上`IfStmtAssignChecker.cpp`。此后进入`include/clang/StaticAnalyzer/Checkers`下打开`Checkers.td`，在`let ParentPackage = CoreAlpha in`的末尾加入：
```
def IfStmtAssignChecker : Checker<"IfStmtAssign">,
  HelpText<"Check whether assignment statements occur in if-conditions">,
  DescFile<"IfStmtAssignChecker.cpp">;
```
即将该checker放置在了`alpha.core`分组中   
此后重新编译并安装clang即可。

在此，我们便将我们的checker整合进了clang中。现在为了使用checker，我们可以使用如下命令：
```
clang --analyze -Xclang -analyzer-checker=alpha.core.IfStmtAssign [source file]
```
其中`[source file]`为要检查的源代码文件

例如，使用`clang --analyze -Xclang -analyzer-checker=alpha.core.IfStmtAssign pt_2.c`检查`test_cases/pt_2.c`会得到如下的输出：
```
pt_2.c:6:57: warning: Try to avoid assignments inside if-conditions (assignments inside
      while-conditions are ok)
    if (((foo < 2) && ((foo > 3) || ((foo = (5 + 2))))) == 1) {
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~
1 warning generated.
```