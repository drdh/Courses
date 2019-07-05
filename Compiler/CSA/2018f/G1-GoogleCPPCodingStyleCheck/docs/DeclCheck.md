# 声明检查

检查所有在声明时能确定的检查内容。

第一部分进行函数声明相关检查，对应规则第五章1-3，进行引用参数、函数重载和缺省参数的检查；规则1.4内联函数。

第二部分进行变量声明相关检查，对应规则第五章14、19、20，检查整型、auto检查和列表初始化检查。

## 总体设计

用一个类完成所有检查。

```c++
class DeclChecker : public Checker<
	check::ASTDecl<VarDecl>, check::ASTDecl<FunctionDecl>
	> {
  public:
    void checkASTDecl(const VarDecl *D, AnalysisManager &Mgr, BugReporter &BR) const;
    void checkASTDecl(const FunctionDecl *D, AnalysisManager &Mgr, BugReporter &BR) const;

  private:
    void referencecheck(clang::ParmVarDecl *param, clang::DiagnosticsEngine &DE)const;
    void defaultcheck(clang::ParmVarDecl *param, clang::DiagnosticsEngine &DE)const;
    void intcheck(const VarDecl *decl, clang::DiagnosticsEngine &DE)const;
    void autocheck(const VarDecl *decl, clang::DiagnosticsEngine &DE)const;
    void inlinestmtcheck(const Stmt *stmt, clang::DiagnosticsEngine &DE)const;
    int getline(const clang::SourceLocation SL, const clang::SourceManager &SM)const;
};
```

通过重载`checkASTDecl`在`Checker`处于`check::ASTDecl<VarDecl>, check::ASTDecl<FunctionDecl>`，即变量定义和函数定义时分别进行不同的检查。

## 函数定义相关

### 在语法树中获取函数参数

首先函数节点的类型为`clang::FunctionDecl`，对于此类型的节点`decl`，通过`decl->parameters()`可以获得其所有参数，返回值类型为`ArrayRef<ParmVarDecl *>`。通过迭代器或`for`循环很容易实现参数的遍历。得到的每一个参数的类型均为`ParmVarDecl*`。

### 引用参数

> 定义:
>
> > 在 C 语言中, 如果函数需要修改变量的值, 参数必须为指针, 如 `int foo(int *pval)`. 在 C++ 中, 函数还可以声明引用参数: `int foo(int &val)`.
>
> 优点:
>
> > 定义引用参数防止出现 `(*pval)++` 这样丑陋的代码. 像拷贝构造函数这样的应用也是必需的. 而且更明确, 不接受 `NULL` 指针.
>
> 缺点:
>
> > 容易引起误解, 因为引用在语法上是值变量却拥有指针的语义.
>
> 结论:
>
> > 函数参数列表中, 所有引用参数都必须是 `const`:
> >
> > > ```
> > > void Foo(const string &in, string *out);
> > > ```
> >
> > 事实上这在 Google Code 是一个硬性约定: 输入参数是值参或 `const` 引用, 输出参数为指针. 输入参数可以是 `const` 指针, 但决不能是非 `const` 的引用参数，除非用于交换，比如 `swap()`.
> >
> > 有时候，在输入形参中用 `const T*` 指针比 `const T&` 更明智。比如：
> >
> > - 您会传 null 指针。
> > - 函数要把指针或对地址的引用赋值给输入形参。
> >
> > 总之大多时候输入形参往往是 `const T&`. 若用 `const T*` 说明输入另有处理。所以若您要用 `const T*`, 则应有理有据，否则会害得读者误解。

对于获取到的类型为`ParmVarDecl*`的参数`param`，通过`param->getType()->isReferenceType()&&!param->getType().isConstQualified()`判断是否为引用类型且不为`const`属性。

### 缺省参数

> 禁止使用缺省函数参数。
>
> 优点:
>
> > 经常用到一个函数带有大量缺省值,偶尔会重写一下这些值,缺省参数为很少涉及的例外情况提供了少定义一些函数的方便。
>
> 缺点:
>
> > 大家经常会通过查看现有代码确定如何使用 API,缺省参数使得复制粘贴以前的代码难以呈现所有参数,当缺省参数不适用于新代码时可能导致重大问题。
>
> 结论:
>
> > 所有参数必须明确指定,强制程序员考虑 API 和传入的各参数值,避免使用可能不为程序员所知的缺省参数。

通过`decl->getMinRequiredArguments() < decl->getNumParams()`可以判断是否有缺省参数。

根据G3组的评测，在拥有可变长参数时会出现误报，例子如下：

```C++
template<class... T> void test1(T... args){}
```

因此改变实现方法，通过检查每个参数是否拥有初始化`getInit()`进行判断。

### 函数重载检查

> 定义:
>
> > 你可以编写一个参数类型为 `const string&` 的函数, 然后用另一个参数类型为 `const char*` 的函数重载它:
> >
> > > ```
> > > class MyClass {
> > >     public:
> > >     void Analyze(const string &text);
> > >     void Analyze(const char *text, size_t textlen);
> > > };
> > > ```
>
> 优点:
>
> > 通过重载参数不同的同名函数, 令代码更加直观. 模板化代码需要重载, 同时为使用者带来便利.
>
> 缺点:
>
> > 如果函数单单靠不同的参数类型而重载（acgtyrant 注：这意味着参数数量不变），读者就得十分熟悉 C++ 五花八门的匹配规则，以了解匹配过程具体到底如何。另外，当派生类只重载了某个函数的部分变体，继承语义容易令人困惑。
>
> 结论:
>
> > 如果您打算重载一个函数, 可以试试改在函数名里加上参数信息。例如，用 `AppendString()` 和 `AppendInt()` 等， 而不是一口气重载多个 `Append()`.

不同于前面，函数重载需要在全局范围内检查。通过`lookup()`函数可以在上下文中查找指定名字的定义，找出其中同名函数定义的数量即可判断。

### 内联函数

> **定义:**
>
> > 当函数被声明为内联函数之后, 编译器会将其内联展开, 而不是按通常的函数调用机制进行调用.
>
> **优点:**
>
> > 只要内联的函数体较小, 内联该函数可以令目标代码更加高效. 对于存取函数以及其它函数体比较短, 性能关键的函数, 鼓励使用内联.
>
> **缺点:**
>
> > 滥用内联将导致程序变得更慢. 内联可能使目标代码量或增或减, 这取决于内联函数的大小. 内联非常短小的存取函数通常会减少代码大小, 但内联一个相当大的函数将戏剧性的增加代码大小. 现代处理器由于更好的利用了指令缓存, 小巧的代码往往执行更快。
>
> **结论:**
>
> > 一个较为合理的经验准则是, 不要内联超过 10 行的函数. 谨慎对待析构函数, 析构函数往往比其表面看起来要更长, 因为有隐含的成员和基类析构函数被调用!
> >
> > 另一个实用的经验准则: 内联那些包含循环或 `switch` 语句的函数常常是得不偿失 (除非在大多数情况下, 这些循环或 `switch` 语句从不被执行).
> >
> > 有些函数即使声明为内联的也不一定会被编译器内联, 这点很重要; 比如虚函数和递归函数就不会被正常内联. 通常, 递归函数不应该声明成内联函数.（YuleFox 注: 递归调用堆栈的展开并不像循环那么简单, 比如递归层数在编译时可能是未知的, 大多数编译器都不支持内联递归函数). 虚函数内联的主要原因则是想把它的函数体放在类定义内, 为了图个方便, 抑或是当作文档描述其行为, 比如精短的存取函数.

检查是否在10行之内且不使用`for``while``do-while``switch`。

**问题：没有找到提供行号的接口**

解决：

通过`clang::SourceLocation`类可以得到详细的位置信息，可以通过`printToString`方法打印出一个标准格式的字符串，然后通过正则匹配获取其中的行号。

循环和`switch`检查通过搜索AST节点，判断其是否为对应类型`SwitchStmt`，`ForStmt`，`WhileStmt`和`DoStmt`。

## 变量定义相关

### 整型

> 定义:
>
> > C++ 没有指定整型的大小. 通常人们假定 `short` 是 16 位, `int` 是 32 位, `long` 是 32 位, `long long` 是 64 位.
>
> 优点:
>
> > 保持声明统一.
>
> 缺点:
>
> > C++ 中整型大小因编译器和体系结构的不同而不同.
>
> 结论:
>
> > `<stdint.h>` 定义了 `int16_t`, `uint32_t`, `int64_t` 等整型, 在需要确保整型大小时可以使用它们代替 `short`, `unsigned long long` 等. 在 C 整型中, 只使用 `int`. 在合适的情况下, 推荐使用标准类型如 `size_t` 和 `ptrdiff_t`.
> >
> > 如果已知整数不会太大, 我们常常会使用 `int`, 如循环计数. 在类似的情况下使用原生类型 `int`. 你可以认为 `int` 至少为 32 位, 但不要认为它会多于 `32` 位. 如果需要 64 位整型, 用 `int64_t` 或 `uint64_t`.
> >
> > 对于大整数, 使用 `int64_t`.
> >
> > 不要使用 `uint32_t` 等无符号整型, 除非你是在表示一个位组而不是一个数值, 或是你需要定义二进制补码溢出. 尤其是不要为了指出数值永不会为负, 而使用无符号类型. 相反, 你应该使用断言来保护数据.
> >
> > 如果您的代码涉及容器返回的大小（size），确保其类型足以应付容器各种可能的用法。拿不准时，类型越大越好。
> >
> > 小心整型类型转换和整型提升（acgtyrant 注：integer promotions, 比如 `int` 与 `unsigned int`运算时，前者被提升为 `unsigned int` 而有可能溢出），总有意想不到的后果。

实现对`unsigned`型警告。需要判断是否为无符号整数`isUnsignedIntegerType()`且不是枚举类型`isEnumeralType()`。

### auto

> 定义：
>
> > C++11 中，若变量被声明成 `auto`, 那它的类型就会被自动匹配成初始化表达式的类型。您可以用 `auto` 来复制初始化或绑定引用。
> >
> > ```
> > vector<string> v;
> > ...
> > auto s1 = v[0];  // 创建一份 v[0] 的拷贝。
> > const auto& s2 = v[0];  // s2 是 v[0] 的一个引用。
> > ```
>
> 优点：
>
> > C++ 类型名有时又长又臭，特别是涉及模板或命名空间的时候。就像：
> >
> > ```
> > sparse_hash_map<string, int>::iterator iter = m.find(val);
> > ```
> >
> > 返回类型好难读，代码目的也不够一目了然。重构其：
> >
> > ```
> > auto iter = m.find(val);
> > ```
> >
> > 好多了。
> >
> > 没有 `auto` 的话，我们不得不在同一个表达式里写同一个类型名两次，无谓的重复，就像：
> >
> > ```
> > diagnostics::ErrorStatus* status = new diagnostics::ErrorStatus("xyz");
> > ```
> >
> > 有了 auto, 可以更方便地用中间变量，显式编写它们的类型轻松点。
>
> 缺点：
>
> > 类型够明显时，特别是初始化变量时，代码才会够一目了然。但以下就不一样了：
> >
> > ```
> > auto i = x.Lookup(key);
> > ```
> >
> > 看不出其类型是啥，x 的类型声明恐怕远在几百行之外了。
> >
> > 程序员必须会区分 `auto` 和 `const auto&` 的不同之处，否则会复制错东西。
> >
> > auto 和 C++11 列表初始化的合体令人摸不着头脑：
> >
> > ```
> > auto x(3);  // 圆括号。
> > auto y{3};  // 大括号。
> > ```
> >
> > 它们不是同一回事——`x` 是 `int`, `y` 则是 `std::initializer_list<int>`. 其它一般不可见的代理类型（acgtyrant 注：normally-invisible proxy types, 它涉及到 C++ 鲜为人知的坑：[Why is vector not a STL container?](http://stackoverflow.com/a/17794965/1546088)）也有大同小异的陷阱。
> >
> > 如果在接口里用 `auto`, 比如声明头文件里的一个常量，那么只要仅仅因为程序员一时修改其值而导致类型变化的话——API 要翻天覆地了。
>
> 结论：
>
> > `auto` 只能用在局部变量里用。别用在文件作用域变量，命名空间作用域变量和类数据成员里。永远别列表初始化 `auto` 变量。
> >
> > `auto` 还可以和 C++11 特性「尾置返回类型（trailing return type）」一起用，不过后者只能用在 lambda 表达式里。

检查声明的`auto`是否为局部的。`auto`类型可以通过`getTypeClass()`获取类型并和`clang::Type::TypeClass::Auto`判断，变量的局部性通过`isLocalVarDecl()`判断。

列表初始化检查在下一项中检查。

### 列表初始化

不能列表初始化`auto`变量。

**问题：如何发现列表初始化？**

先看例子程序

```c++
#include <initializer_list>
struct test{
int ta;
};
int main(){
auto e = {1.23,1.1};
struct test test1={1,"12"};
    return 0;
}
```

生成的语法树如下

```
`-FunctionDecl 0x561e9feb91e8 <line:5:1, line:9:1> line:5:5 main 'int ()'
  `-CompoundStmt 0x561e9feba568 <col:11, line:9:1>
    |-DeclStmt 0x561e9feba398 <line:6:1, col:20>
    | `-VarDecl 0x561e9feb9330 <col:1, col:19> col:6 e 'std::initializer_list<double>':'std::initializer_list<double>' cinit
    |   `-ExprWithCleanups 0x561e9feba380 <col:10, col:19> 'std::initializer_list<double>':'std::initializer_list<double>'
    |     `-CXXStdInitializerListExpr 0x561e9feba350 <col:10, col:19> 'std::initializer_list<double>':'std::initializer_list<double>'
    |       `-MaterializeTemporaryExpr 0x561e9feba338 <col:10, col:19> 'const double [2]' xvalue extended by Var 0x561e9feb9330 'e' 'std::initializer_list<double>':'std::initializer_list<double>'
    |         `-InitListExpr 0x561e9feba2e8 <col:10, col:19> 'const double [2]'
    |           |-FloatingLiteral 0x561e9feb9390 <col:11> 'double' 1.230000e+00
    |           `-FloatingLiteral 0x561e9feb93b0 <col:16> 'double' 1.100000e+00
    |-DeclStmt 0x561e9feba518 <line:7:1, col:22>
    | `-VarDecl 0x561e9feba408 <col:1, col:21> col:13 test1 'struct test':'test' cinit
    |   `-InitListExpr 0x561e9feba4d0 <col:19, col:21> 'struct test':'test'
    |     `-IntegerLiteral 0x561e9feba468 <col:20> 'int' 1
    `-ReturnStmt 0x561e9feba550 <line:8:5, col:12>
      `-IntegerLiteral 0x561e9feba530 <col:12> 'int' 0
```

以上为`main`函数的语法树。

对于正常使用的列表初始化变量`test1`，其节点类型为`VarDecl`，其初始化变量类型为`InitListExpr`（13行），为一个正常的列表初始化表达式类型。但是如果使用`auto`的列表初始化（变量`e`），其初始化变量类型为`ExprWithCleanups`（5行）。

现在的判断方案是判断变量的初始化节点类型是否为`ExprWithCleanups`，若是则进一步判断其子表达式类型是否为`CXXStdInitializerListExpr`（对应第6行）。