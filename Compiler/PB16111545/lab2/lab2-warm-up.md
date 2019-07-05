# Lab2-1预热试验

将`llvm-install/bin`加入到环境变量PATH中

```bash
export PATH=$PATH:/home/drdh/lx/Compiler/LLVM/llvm-install/bin
```

## LLVM IR 的人工翻译

中间代码类似汇编，有一些语法问题可以直接在[LangRef](http://llvm.org/docs/LangRef.html)中通过搜索来查看。

1. 每个函数里面的`%1,%2...`出现的次序必须依次。
2. `br`是分支指令，在每个基本块的末尾。可以无条件跳转，如` br label %17`也可以有条件跳转`br i1 %9, label %10, label %11``
3. `icmp`是比较判断指令，后面加上`eq` 表示相等时返回`true`, 后面加上`slt`表示有符号数的小于
4. `load` `store` `alloca`都是对指针类型的地址进行操作。
5. `sub` `add`根据`clang -S -emit-llvm `输出的文件可以仿照

主要的相比于机器的翻译结果，人工主要要做的是去掉冗余的部分。比如某个存于memory的值`i`，每次使用它的时候都重新`load`了一次，没有必要。其他内容与机器翻译几乎一致。。。

## IRBuilder构建LLVM IR

按照[LLVM IRGen Example](https://github.com/ustc-compiler/2017fall/tree/master/llvm-irgen-example)例子来学习。

当按照以下命令时

```bash
c++ `llvm-config --cxxflags --ldflags --libs` llvm-irgen-example.cpp -o llvm-irgen-example
```

出现大量的`undefined reference to llvm::***`错误。搜索看到[Stackoverflow](https://askubuntu.com/questions/732326/undefined-references-compiling-clang-tutorial-on-14-04-2)同样的错误，说是cpp文件和clang lib的次序会影响，于是改为如下

```bash
c++ `llvm-config --cxxflags --ldflags` llvm-irgen-example.cpp `llvm-config --libs --system-libs` -o llvm-irgen-example 
```

或者

```bash
c++  llvm-irgen-example.cpp `llvm-config --cxxflags --ldflags --libs --system-libs` -o llvm-irgen-example
```

运行`./llvm-irgen-example`

输出

```
; ModuleID = 'foo_dead_recursive_loop'
source_filename = "foo_dead_recursive_loop"

define void @foo() {
entry:
  call void @foo()
  ret void
}
```
