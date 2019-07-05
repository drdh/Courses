# 1. Usage

```bash
# in the project directory
mkdir build ;  cd build
cmake ..
make
# the binary file is in build/src/ directory
cd src
./google-cpp-coding-style-checker <directory>|<source-list> [<checker-selected-option>]
#the checker-select-option:
# -no-class-check: do not run the ClassChecker
# -no-decl-check: do not run the DeclChecker
# -no-stmt-check: do not run the StmtChecker
# -no-pp-check: do not run the PPChekcer
```



# 2.已完成的规范检查

[TOC]

## 1. 预处理宏

规范说明：

> - Don't define macros in a `.h` file.
> - `#define` macros right before you use them, and `#undef` them right after.
> - Do not just `#undef` an existing macro before replacing it with your own; instead, pick a name that's likely to be unique.
> - Try not to use macros that expand to unbalanced C++ constructs, or at least document that behavior well.
> - Prefer not using `##` to generate function/class/variable names.

完成1, 2, 3, 5, 第四点没有准确的定义，难以实现。

相关checker: `PPChecker`

## 2.include顺序

规范说明：

> Use standard order for readability and to avoid hidden dependencies: Related header, C library, C++ library, other libraries' `.h`, your project's `.h`.

由于Related header 没有准确定义，且回调函数提供的关于include的文件的信息很少，所以此处checker检查的顺序为：

 1.user_module
 2.system_module
 3.user header
 4.extern c header
 5.c header

相关checker: `PPCHecker`.

## 3. 构造函数的指责

规范说明：

> Avoid virtual method calls in constructors, and avoid initialization that can fail if you can't signal an error.

完成前面部分，后面那个难以确定初始化是否会失败。

相关checker:`ClassChecker`.

## 4.隐式转换

规范说明：

>Do not define implicit conversions. Use the `explicit` keyword for conversion operators and single-argument constructors.

完成所有要求

相关checker: `ClassChecker`.

## 5. 可拷贝和可移动类型

规范说明：

> A class's public API should make explicit whether the class is copyable, move-only, or neither copyable nor movable. Support copying and/or moving if these operations are clear and meaningful for your type.

此处检查：

1. 只允许构造函数和以及相对应的赋值操作符重载是否被同时定义；
2. 如果拷贝/移动构造函数没有显示定义，是否将其声明为`delete`.

相关checker: `ClassChecker`.

## 6. 结构体和类

规范说明：

> Use a `struct` only for passive objects that carry data; everything else is a `class`.

此处检查：

1. 是否定义了不必要的相关域的访问和修改；
2. 是否定义含参数的成员函数。

相关checker: `ClassChecker`.

## 7.动态分配堆栈内存

规范说明：

> We do not use `alloca()`.

检查了代码中（所有类中定义的方法、所有函数中的代码，不包括库文件）不能使用 `alloca()`

相关checker: `StmtCheckASTConsumer`.

## 8.异常

规范说明：

> We do not use C++ exceptions.

检查了代码中不使用 C++ 异常，包括了 `try`、`throw`、`catch`

相关checker: `StmtCheckASTConsumer`.

## 9.运行时类型识别

规范说明：

> Avoid using Run Time Type Information (RTTI).

- 检查了代码中不使用 `typeid`
- 检查了代码中不使用 `dynamic_cast`

相关checker: `StmtCheckASTConsumer`.

## 10.类型转换

规范说明：

> Use C++-style casts like `static_cast<float>(double_value)`, or brace initialization for conversion of arithmetic types like `int64 y = int64{1} << 42`. Do not use cast formats like `int y = (int)x` or `int y = int(x)`

检查了代码中不使用 C 风格的强制类型转换

相关checker: `StmtCheckASTConsumer`.

## 11.前置自增和自减

规范说明：

> Use prefix form (`++i`) of the increment and decrement operators with iterators and other template objects. When a variable is incremented (`++i` or `i++`) or decremented (`--i` or `i--`) and the value of the expression is not used, one must decide whether to preincrement (decrement) or postincrement (decrement).

- 由于迭代器类型重载的自增 / 自减运算符不出现在 AST 结点中，没有对迭代器的自增 / 自减进行检测
- 对于绝大多数表达式的值可被忽略的（无用的）基本类型的后缀自增 / 自减，提示修改为前缀自增 / 自减
- 对于自定义类重载的自增 / 自减，代码中使用对象的后缀自增 / 自减会提示尽量修改为前缀自增 / 自减

相关checker: `StmtCheckASTConsumer`.

## 12.Lambda 表达式

规范说明：

> Use lambda expressions where appropriate. Prefer explicit captures when the lambda will escape the current scope.
>
> - Prefer explicit captures if the lambda may escape the current scope.
>
> - Use default capture by reference ([&]) only when the lifetime of the lambda is obviously shorter than any potential captures. 
>
> - Use default capture by value ([=]) only as a means of binding a few variables for a short lambda, where the set of captured variables is obvious at a glance. Prefer not to write long or complex lambdas with default capture by value. 
> - Specify the return type of the lambda explicitly if that will make it more obvious to readers, as with `auto`.

- 禁用默认捕获，捕获都要显式写出来。打比方，比起 `[=](int x) {return x + n;}`, 该写成 `[n](int x) {return x + n;}` 。

  检测 lambda 表达式的捕获列表，如果有默认捕获会提示尽可能改为显示

- 匿名函数始终要简短，如果函数体超过了五行，那么还不如起名

  这个没有进行判断，需要额外的状态记录 lambda 表达式所在环境，如是否赋值给对象还是作为函数参数。

- 如果可读性更好，就显式写出 lambda 的尾置返回类型，就像auto.

  检测如果返回类型没有显式给出，则提示在可读的情况下尽量显式写出返回类型

- 匿名函数始终要简短，如果函数体超过了五行，那么还不如起名（acgtyrant 注：即把 lambda 表达式赋值给对象），或改用函数。

  检查了函数体中的行数，超过五行，则在如下两种情况中报错：

  - lambda 不在一个变量声明中包围，即没有赋值给对象
  - lambda 在一个变量声明中包围，但是它被更内层的函数调用包围，因此是出现为函数参数。

  其他情况下忽略。

相关checker: `StmtCheckASTConsumer`.

## 13.内联函数

规范说明：

> Define functions inline only when they are small, say, 10 lines or fewer.
>
> Another useful rule of thumb: it's typically not cost effective to inline functions with loops or switch statements (unless, in the common case, the loop or switch statement is never executed).

* 检查超过10行的内联函数并警告
* 检查内联函数中使用的`for``while``do-while``switch`并警告

相关checker:`DeclChecker`.

## 14.引用参数

规范说明：

> All parameters passed by lvalue reference must be labeled `const`.

* 检查没有被标记为`const`的引用类型参数并警告

相关checker:`DeclChecker`.

## 15.函数重载

规范说明：

> Use overloaded functions (including constructors) only if a reader looking at a call site can get a good idea of what is happening without having to first figure out exactly which overload is being called.

* 对重载的函数进行提示

相关checker:`DeclChecker`.

## 16.缺省参数

规范说明：

> Default arguments are allowed on non-virtual functions when the default is guaranteed to always have the same value. 
>
> Default arguments are banned on virtual functions, where they don't work properly, and in cases where the specified default might not evaluate to the same value depending on when it was evaluated. 

* 对虚函数的缺省参数警告
* 对其他情况的缺省参数提示

相关checker:`DeclChecker`.

## 17.整型

规范说明：

> Of the built-in C++ integer types, the only one used is `int`. If a program needs a variable of a different size, use a precise-width integer type from `<stdint.h>`, such as `int16_t`. If your variable represents a value that could ever be greater than or equal to 2^31 (2GiB), use a 64-bit type such as`int64_t`. Keep in mind that even if your value won't ever be too large for an `int`, it may be used in intermediate calculations which may require a larger type. When in doubt, choose a larger type.
>
> You should not use the unsigned integer types such as `uint32_t`, unless there is a valid reason such as representing a bit pattern rather than a number, or you need defined overflow modulo 2^N. In particular, do not use unsigned types to say a number will never be negative. Instead, use assertions for this.

* 对声明为无符号整数的变量进行警告

相关checker:`DeclChecker`.

## 18.`auto`

规范说明：

> Use `auto` to avoid type names that are noisy, obvious, or unimportant - cases where the type doesn't aid in clarity for the reader. Continue to use manifest type declarations when it helps readability.
>
> If an `auto` variable is used as part of an interface, e.g. as a constant in a header, then a programmer might change its type while only intending to change its value, leading to a more radical API change than intended.
>
> `auto` is permitted when it increases readability. Never initialize an `auto`-typed variable with a braced initializer list.

* `auto`类型只允许在局部变量使用，对非局部使用警告
* 对列表初始化的`auto`类型警告

相关checker:`DeclChecker`.

## 19.列表初始化

规范说明：

> You may use braced initializer lists.
>
> Never assign a *braced-init-list* to an auto local variable.

* 同`auto`一节规范一致，对列表初始化的`auto`类型警告

相关checker:`DeclChecker`.

## 20.继承
规范说明：
> All inheritance should be public. If you want to do private inheritance, you should be including an instance of the base class as a member instead.  
> 
> Do not overuse implementation inheritance. Composition is often more appropriate. Try to restrict use of inheritance to the "is-a" case: Bar subclasses Foo if it can reasonably be said that Bar "is a kind of" Foo.  
> 
> Limit the use of protected to those member functions that might need to be accessed from subclasses. Note that data members should be private.  
> 
> Explicitly annotate overrides of virtual functions or virtual destructors with exactly one of an override or (less frequently) final specifier. Do not use virtual when declaring an override. Rationale: A function or destructor marked override or final that is not an override of a base class virtual function will not compile, and this helps catch common errors. The specifiers serve as documentation; if no specifier is present, the reader has to check all ancestors of the class in question to determine if the function or destructor is virtual or not.  
> 
> Multiple inheritance is permitted, but multiple implementation inheritance is strongly discouraged.

* 继承必须使用public限定符来继承
* 尽量使用组合，数据成员必须为private
* 若函数是复写基类的虚函数，虚函数指定符vitual必须显式使用
## 21.操作符重载

规范说明：
> Define overloaded operators only if their meaning is obvious, unsurprising, and consistent with the corresponding built-in operators. For example, use | as a bitwise- or logical-or, not as a shell-style pipe.
> 
> Define operators only on your own types. More precisely, define them in the same headers, .cc files, and namespaces as the types they operate on. That way, the operators are available wherever the type is, minimizing the risk of multiple definitions. If possible, avoid defining operators as templates, because they must satisfy this rule for any possible template arguments. If you define an operator, also define any related operators that make sense, and make sure they are defined consistently. For example, if you overload <, overload all the comparison operators, and make sure < and > never return true for the same arguments.
> 
> Prefer to define non-modifying binary operators as non-member functions. If a binary operator is defined as a class member, implicit conversions will apply to the right-hand argument, but not the left-hand one. It will confuse your users if a < b compiles but b < a doesn't.
> 
> Don't go out of your way to avoid defining operator overloads. For example, prefer to define ==, =, and <<, rather than Equals(), CopyFrom(), and PrintTo(). Conversely, don't define operator overloads just because other libraries expect them. For example, if your type doesn't have a natural ordering, but you want to store it in a std::set, use a custom comparator rather than overloading <.
> 
> Do not overload &&, ||, , (comma), or unary &. Do not overload operator"", i.e. do not introduce user-defined literals.
> 
> Type conversion operators are covered in the section on implicit conversions. The = operator is covered in the section on copy constructors. Overloading << for use with streams is covered in the section on streams. See also the rules on function overloading, which apply to operator overloading as well.

* 不要使用操作符重载，转而使用函数
* 不推荐重载==，若必须重载，提示编写清晰的document
* 其他重载一律不推荐

## 22.访问控制

规范说明：
> Make classes' data members private, unless they are static const (and follow the naming convention for constants).
> 
> For technical reasons, we allow data members of a test fixture class in a .cc file to be protected when using Google Test).

* 所有类的数据成员使用private
* 对于所有数据成员，都需要get/set函数，来取得，保存其值

## 23.声明顺序

规范说明：
> Group similar declarations together, placing public parts earlier.
> 
> A class definition should usually start with a public: section, followed by protected:, then private:. Omit sections that would be empty.
> 
> Within each section, generally prefer grouping similar kinds of declarations together, and generally prefer the following order: types (including typedef, using, and nested structs and classes), constants, factory functions, constructors, assignment operators, destructor, all other methods, data members.
> 
> Do not put large method definitions inline in the class definition. Usually, only trivial or performance-critical, and very short, methods may be defined inline. See Inline Functions for more details.

* 声明主顺序：public，protect，private
* 声明副顺序：typedef/enum，constructor，destructor，member function，data field
* 按照主副顺序检测并给予警告

## 24. 0, `nullptr` 和 `NULL`

规范说明：

> Use `0` for integers, `0.0` for reals, `nullptr` for pointers, and `'\0'` for chars.
>
> Use `0` for integers and `0.0` for reals.
>
> For pointers (address values), use `nullptr`, as this provides type-safety.
>
> For C++03 projects, prefer `NULL` to `0`. While the values are equivalent, `NULL` looks more like a pointer to the reader, and some C++ compilers provide special definitions of `NULL` which enable them to give useful warnings.
>
> Use `'\0'` for the null character. Using the correct type makes the code more readable.

- 在 AST 层次中检查了 0，`NULL` 和 `nullptr` 的出现，对所有编译器不会报错的情况给出了 warning，如使用 `int *p = 0`

