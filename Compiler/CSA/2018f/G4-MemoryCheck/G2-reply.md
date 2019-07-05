# 第四组测评报告

## 程序主要功能

* 用CSA编译c/c++程序时部分访问内存的函数的参数检查，在参数不符合要求时报一个warning。

* 对strcpy和memcpy函数的两个参数的长度是否相等进行检查，对malloc和new分配空间时空间长度是否为0进行检查

## 测试方法

1. 将目录切到clang的项目目录下，目录为`<llvm path>/tools/clang/`

2. 将源文件和部分配置文件进行改动：
   
   - 修改`include/clang/StaticAnalyzer/Checkers/checkers.td`，在`let ParentPackage = CoreAlpha in`块中中添加如下内容：
   ```c
   def BufferChecker : Checker<"BufferSize">,
     	HelpText<"Check if the src buffer size is not larger than the dst buffer size">,
     	DescFile<"BufferChecker.cpp">;
     	
   def AllocSizeChecker : Checker<"AllocSize">,
  		HelpText<"Check if the Malloc size is zero">,
  		DescFile<"AllocSizeChecker.cpp">;
  		
   def AllocationZeroChecker : Checker<"AllocationZero">,
   		HelpText<"Check if allocated array size is zero">,
   		DescFile<"AllocationZeroChecker.cpp">;
   ```
   
   - 将源文件（`BufferChecker.cpp`，`AllocSizeChecker.cpp`，`AllocationZero.cpp`）添加到对应目录（`lib/StaticAnalyzer/Checkers`）下
   
   - 修改`CMakeList.txt`，在其中添加：
   ```c
   AllocSizeChecker.cpp
   BufferChecker.cpp
   AllocationZero.cpp
   ```
   
3. 重新编译clang

4. 使用`clang -cc1 -analyzer-checker-help`、`clang --analyzer -analyze-checker=xxx xxx.c`进行测试

## 测试例子（一）

```c
#include <string.h>
#include <stdlib.h>

int main()
{
  int a = 1;
  a -= a;
  const char* s1 = "Hello World!";
  char *s2 = (char *)malloc(a * sizeof(char));
  strcpy(s2, s1); // warn
  
  return 0;
}
```

```c
#include <iostream>
#include <string.h>

using namespace std;
int main()
{
  int* p1 = new int[3];
  int* p2 = new int;
  memcpy(p2, p1, 3); // warn
  return 0;
}
```
结果显示，会对strcpy、memcpy、malloc、new参数进行检查并报出warning。

## 测试例子（二）

```c
#include<stdlib.h>
#include<memory.h>
#include<string.h>

int main(){
	char *str = "Hello World!";
	int x = 10000;
	for(int i = 10000;i>=0;i--)
		x--;
	x=0;
	char *a = (char *)malloc(x);
	strcpy(a,str);
	return 0;
}
```
结果为，并不会产生warning，即使明显这里malloc参数为0，且进行了将一个不同长字符串的拷贝，说明检查还不够强大。

## 主要优点

1. 比较实用：写程序时，经常会用到使用malloc、new进行变长数组的声明，也常常使用strcpy或memcpy进行拷贝。但当参数为变量时，有时候忘记考虑可能参数为0或两字符串长度不一样，就会导致在内存访问时产生段错误。这个checker检查了这些常用函数的使用，检测这些低级错误，有效减少段错误的可能。

2. 使用灵活：可以对三个checker进行独立的检查，例如只检查程序中malloc函数参数是否为0，则使用`clang --analyzer -analyze-checker=alpha.core.AllocSize xxx.c`

## 不足之处

1. 功能比较简单：只能检测部分函数的使用，但实际上有很多函数会对内存进行不加检查的访问，最好是检查访问，而不是检查参数；并且没有对其他访问内存的语句、表达式等进行检查，段错误也有很大部分在这些过程中（例如数组越界读写，不过CSA已经有许多关于这部分的检查）

2. 一些隐藏的错误无法检测：当涉及比较复杂的计算时，有时将计算结果作为参数时检测不出来。这可能与clang静态检查的水平有限有关，但的确是可改进的地方。

2. 需要重新编译clang：需要对clang配置文件更改然后重新编译，比较麻烦（整个clang的编译过程时间相当长）
