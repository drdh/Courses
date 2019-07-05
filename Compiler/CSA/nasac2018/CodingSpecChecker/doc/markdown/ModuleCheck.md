## ModuleCheck

### 需求描述

对于模块外部传入的参数，必须进行合法性检查，保护程序免遭非法输入数据的破坏。模块内部函数调用，缺省由调用者负责保证参数的合法性，如果都由被调用者来检查参数合法性，可能会出现同一个参数，被检查多次，产生冗余代码，很不简洁。由调用者保证入参的合法性，这种契约式编程能让代码逻辑更简洁，可读性更好。

### 功能实现

#### 模块定义

C项目中模块的表现形式定义如下：

- 一个模块所涉及的源文件（.c/.h文件）存放在同一目录下，目录名即为模块名
- 一个模块必须提供一个与目录同名的头文件用于声明该模块对外暴露的接口

#### 错误指示符的定义

错误指示符用于表示函数调用过程中可能发生异常，表现形式定义如下

- 整数类型的返回值
- 布尔类型的返回值
- 指针类型的返回值
- 用户自定义类型返回值（尚未支持）

#### 存在异常处理的函数定义

一个函数被视作存在异常处理的表现形式定义如下:

- 函数返回错误指示符

#### 形参检查函数定义

一个被调函数被视作主调函数形参检查函数的表现形式定义如下：

- 被调函数只有一个输入参数
- 被调函数返回错误指示符
- 被调函数的实参为主调函数形参（取地址？？？）

#### 参数类型定义

函数的参数分为输入参数和输出参数两类，分别定义如下

- 输入参数：非指针类型参数或指向常量的指针类型参数（数组？？？）
- 输出参数：指向变量的指针类型参数

#### 检查定义

C函数中检查目的是确认左值表达式（代表内存对象）的右值是否合法（可能与领域知识有关），在ModuleChecker中将右值检查的表现形式定义如下：

- 在函数调用表达式中读取右值
  - 所在函数为形参检查函数
  - 所在函数为存在异常处理的函数
- 在分支条件表达式中读取右值
- 无外层语句包裹直接读取右值

---

程序状态定义
$$
S_1:Variable \rightarrow (2^{\{Tainted, Untained\}}, \subseteq)\\
S_2:Variable \rightarrow (2^{VariableSet}, \subseteq)
$$
程序状态$S_1$转换规则
$$
[[Entry]] = \bot
$$

$$
[[ParmDecl\ X]]=\mathbb{S}[X\mapsto Tainted]
$$

$$
[[VarDecl\ X]]=\mathbb{S}[X\mapsto Untainted]
$$

$$
[[X=E]]=\mathbb{S}[X\mapsto eval(\mathbb{S},E)]
$$

$$
[[condition\ E]]=\mathbb{S}[\forall X\in Vars(E),X\mapsto Untainted]
$$

---

$eval$函数定义如下：
$$
eval(\mathbb{S},Op(E_1,E_2,...E_n))=\bigsqcup\{eval(\mathbb{S},E_1),eval(\mathbb{S},E_2),...eval(\mathbb{S},E_n)\}
$$

$$
eval(\mathbb{S},X)=\mathbb{S}(X)
$$

$$
eval(\mathbb{S},E(E_1,E_2,...E_n))=\bigsqcup\{eval(\mathbb{S},E_1),eval(\mathbb{S},E_2),...eval(\mathbb{S},E_n)\}
$$

$$
eval(\mathbb{S},Member(E,M))=eval(\mathbb{S},E)
$$

$$
eval(\mathbb{S},E_1[E_2])=\bigsqcup\{eval(\mathbb{S},E_1), eval(\mathbb{S},E_2)\}
$$

$$
...
$$

$Vars$函数定义如下：
$$
Vars(Op(E_1,E_2,...,E_n))=Vars(E_1)\cup Vars(E_2)\cup...\cup Vars(E_n)
$$

$$
Vars(X) = \{X\}
$$

$$
Vars(E(E_1,E_2,...,E_n))=Vars(E_1)\cup Vars(E_2)\cup ...\cup Vars(E_n)
$$

$$
Vars(Member(E,M))=Vars(E)
$$

$$
Vars(E_1[E_2])=Vars(E_1)\cup Vars(E_2)
$$

---

程序状态$S_2$转换规则：
$$
[[Entry]] = \bot
$$

$$
[[X=E]]=\mathbb{S}[X\mapsto Vars(X)]
$$

---

检查动作$A_{pre}$:
$$
[[]]
$$

$$
[[condition\ E]]\rightarrow \{\}
$$

$$
[[read\ X]]\rightarrow \{if\ eval(S_1,X)=Tainted, then\ warning !\}
$$
