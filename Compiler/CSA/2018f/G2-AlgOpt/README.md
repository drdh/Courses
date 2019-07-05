# Algebraic optimization

## Preparation
* LLVM 7.0.x
* clang 7.0.x
* cmake 3.12 above

## build and Run

`Algebraic` contains one transformation pass for LLVM 7.0.x.

1. Put this directory(that is Algebraic directory) under `your llvm-src-code/lib/Transforms`. Actually using soft link the file maybe better.And then open `/lib/Transforms/CMakeLists.txt`,and write `add_subdirectory(Algebraic)`at the end of the file.
2. Build LLVM from the top-level build directory.The command lines are as follows:
   ```
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 
    make -jn 
    ```
    and it may take you 0.5-1 hour to build at first time.
3. Generate the optimized file by clang -emit-llvm -O2, and use opt by `opt -load ./lib/LLVMAlgebraic.so -algebraic ...`.For example,if you have a C source code file called `SquareExpand.c`,just use clang by `clang -O2 SquareExpand.c -emit-llvm -c -o SquareExpand.bc` to generate `SquareExpand.bc` file, which has been optimized by clang.Then use opt tool and load the `LLVMAlgebraic.so` file.The command line is `opt -load -S ./lib/LLVMAlgebraic.so -algebraic < SquareExpand.bc` in this example.

## Example and Explanation
In ./Example, you can find a C source code file:
```C
#include <stdio.h>

int fun(int x)
{
    int a=(x+1)*(x+1);
    int b=x*x+2*x+3;
    int sum = a*3-b*3;
    return sum;
}

int main()
{
    int sum = 0, x;
    scanf("%d",&x);
    fun(x);
    return 0;
}
```
When we use `clang -O2 -emit-llvm -S` to generate a LLVM IR code
```LLVM
; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @fun(i32) local_unnamed_addr #0 {
  %2 = add nsw i32 %0, 1
  %3 = add i32 %0, 2
  %4 = mul i32 %2, %2
  %5 = mul i32 %3, %0
  %6 = sub i32 %4, %5
  %7 = mul i32 %6, 3
  %8 = add i32 %7, -3
  ret i32 %8
}
```

Just like we have mentioned in [ProjectInfo.md](./ProjectInfo.md), `clang -O2` cannot find out that the return value `sum` is always -6, clang choose the conservative method, just calculate the answer step by step, and finally get the return value.Of course we can always get a correct answer in that way, but obviously it wastes too much time.

When we use our `Algebraic` project, we can simply change the IR code like that:
```LLVM
After:
; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @fun(i32) local_unnamed_addr #0 {
Opt:
  ret i32 -6
                                                  ; No predecessors!
  %2 = add nsw i32 %0, 1
  %3 = add i32 %0, 2
  %4 = mul i32 %2, %2
  %5 = mul i32 %3, %0
  %6 = sub i32 %4, %5
  %7 = mul i32 %6, 3
  %8 = add i32 %7, -9
  ret i32 %8
}
```

As you can see, at the beginning of the function, we just add a `BasicBlock` called "Opt", and it just do one thing, return -6 directly! So it is definitly much quicker.

