# Julia性能分析
## Introduction to [PLBG](https://benchmarksgame-team.pages.debian.net/benchmarksgame/)
### Why toy programs
+ Attempts at running programs that are much simpler than a real application have led to performance pitfalls
+ But... you don't have time to inspect the source code of real applications to check that different implementations are kind-of comparable
+ You do have time to inspect 100-line programs. You do have time to write 100-line programs
### [How programes are measured](https://benchmarksgame-team.pages.debian.net/benchmarksgame/how-programs-are-measured.html)
> Measured on a quad-core 2.4Ghz Intel® Q6600® with 4GB of RAM and 250GB SATA II disk drive; using Ubuntu™ 18.10 Linux x64 4.18.0-10-generic.
+ How programs are timed
+ How program memory use is measured
+ How source code size is measured
+ How CPU load is measured

### Julia vs Javascript vs C on [binary-trees](https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/binarytrees.html#binarytrees)
| ×    | source     | secs  | gz      | cpu  | cpu load        |
| ---- | ---------- | ----- | ------- | ---- | --------------- |
| 1.0  | C gcc #3   | 3.73  | 116,600 | 836  | 11.74           |
| 6.5  | Node js #2 | 23.78 |         |      | 46% 35% 28% 58% |
| 11   | C gcc      | 39.77 |         |      | 0% 1% 100% 0%   |
| 23   | Python 3   | 83.95 |         |      | 88% 87% 87% 97% |
| 26   | Julia      | 96.71 |         |      | 0% 0% 0% 100%   |

> The GTop cpu idle and GTop cpu total are taken before forking the child-process and after the child-process exits. The percentages represent the proportion of cpu not-idle to cpu total for each core.
> On win32: GetSystemTimes UserTime IdleTime are taken before forking the child-process and after the child-process exits. The percentage represents the proportion of TotalUserTime to UserTime + IdleTime (because that's like the percentage you'll see in Task Manager).

> We started with PLBG programs written by the Julia developers and  xed some performance anomalies. The benchmarks are written in an idiomatic style, using the same algorithms as the C benchmarks.

[Julia srouce code](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-julia-1.html)
```julia
using Printf

abstract type BTree end

mutable struct Empty <: BTree
end

mutable struct Node <: BTree
    left::BTree
    right::BTree
end

function make(d)
    if d == 0
        Node(Empty(), Empty())
        else
        Node(make(d-1), make(d-1))
    end
end

check(t::Empty) = 0
check(t::Node) = 1 + check(t.left) + check(t.right)

function loop_depths(d, min_depth, max_depth)
    for i = 0:div(max_depth - d, 2)
        niter = 1 << (max_depth - d + min_depth)
        c = 0
        for j = 1:niter
            c += check(make(d)) 
        end
        @printf("%i\t trees of depth %i\t check: %i\n", niter, d, c)
        d += 2
    end
end

function perf_binary_trees(N::Int=10)
    min_depth = 4
    max_depth = N
    stretch_depth = max_depth + 1

    # create and check stretch tree
    let c = check(make(stretch_depth))
    @printf("stretch tree of depth %i\t check: %i\n", stretch_depth, c)
end

long_lived_tree = make(max_depth)

loop_depths(min_depth, min_depth, max_depth)
@printf("long lived tree of depth %i\t check: %i\n", max_depth, check(long_lived_tree))

end

n = parse(Int,ARGS[1])
perf_binary_trees(n)
```

[Javascript surce code](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-node-2.html)
```javascript
const TreeNode = function(left, right) {
    this.left  = left;
    this.right = right;
};

const itemCheck = function(node){
    if (node===null) return 1;
    return 1 + itemCheck(node.left) + itemCheck(node.right);
};

function bottomUpTree(depth){
    return depth>0 ? new TreeNode(
        bottomUpTree(depth-1),
        bottomUpTree(depth-1)
    ) : null;
};

const maxDepth = Math.max(6, parseInt(process.argv[2]));
const stretchDepth = maxDepth + 1;

let check = itemCheck(bottomUpTree(stretchDepth));
console.log("stretch tree of depth "+ stretchDepth+ "\t check: "+ check);

const longLivedTree = new TreeNode(
    bottomUpTree(maxDepth-1),
    bottomUpTree(maxDepth-1)
);

for (let depth=4; depth<=maxDepth; depth+=2){
    const iterations = 1 << maxDepth - depth + 4;
    check = 0;
    for (let i=1; i<=iterations; i++){
        check += itemCheck(bottomUpTree(depth));
    }
    console.log(iterations+"\t trees of depth "+ depth +"\t check: " + check);
}

console.log("long lived tree of depth "+ maxDepth
+ "\t check: "+ itemCheck(longLivedTree));
```

[C](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-gcc-1.html) source code
```c
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct tn {
struct tn*    left;
struct tn*    right;
} treeNode;


treeNode* NewTreeNode(treeNode* left, treeNode* right)
{
    treeNode*    new;
    new = (treeNode*)malloc(sizeof(treeNode));
    new->left = left;
    new->right = right;
    return new;
} /* NewTreeNode() */


long ItemCheck(treeNode* tree)
{
    if (tree->left == NULL)
    return 1;
    else
    return 1 + ItemCheck(tree->left) + ItemCheck(tree->right);
} /* ItemCheck() */


treeNode* BottomUpTree(unsigned depth)
{
    if (depth > 0)
    return NewTreeNode
    (
        BottomUpTree(depth - 1),
        BottomUpTree(depth - 1)
    );
    else
    return NewTreeNode(NULL, NULL);
} /* BottomUpTree() */


void DeleteTree(treeNode* tree)
{
    if (tree->left != NULL)
    {
        DeleteTree(tree->left);
        DeleteTree(tree->right);
    }
    free(tree);
} /* DeleteTree() */


int main(int argc, char* argv[])
{
unsigned   N, depth, minDepth, maxDepth, stretchDepth;
treeNode   *stretchTree, *longLivedTree, *tempTree;

N = atol(argv[1]);

minDepth = 4;

if ((minDepth + 2) > N)
    maxDepth = minDepth + 2;
else
    maxDepth = N;

stretchDepth = maxDepth + 1;

stretchTree = BottomUpTree(stretchDepth);
printf
(
    "stretch tree of depth %u\t check: %li\n",
    stretchDepth,
    ItemCheck(stretchTree)
);

DeleteTree(stretchTree);

longLivedTree = BottomUpTree(maxDepth);

for (depth = minDepth; depth <= maxDepth; depth += 2)
{
long    i, iterations, check;

iterations = pow(2, maxDepth - depth + minDepth);

check = 0;

for (i = 1; i <= iterations; i++)
{
    tempTree = BottomUpTree(depth);
    check += ItemCheck(tempTree);
    DeleteTree(tempTree);
} /* for(i = 1...) */

printf
(
    "%li\t trees of depth %u\t check: %li\n",
    iterations,
    depth,
    check
);
} /* for(depth = minDepth...) */

printf
(
    "long lived tree of depth %u\t check: %li\n",
    maxDepth,
    ItemCheck(longLivedTree)
);

return 0;
} /* main() */
```

## EVALUATING RELATIVE PERFORMANCE
为了测试语言的相对性能，选择了10个在编程语言基准游戏(PLBG)中的小程序，它们可以在C、JavaScript和Python中实现。作者从开发人员用Julia写的PLBG程序开始，基准测试采用惯用风格编写，使用与c基准测试相同的算法，代码基本上是无类型的，类型注释只出现在结构字段上。

下面的图片展示了以C程序运行时间为标准，其他语言编写的程序的运行时间，各语言使用的版本为
+ Julia v0.6.2
+ CPython 3.5.3
+ V8/Node.js v8.11.1
+ GCC 6.3.0 -O2
+ 运行环境为Debian 9.4 on a Intel i7-950 at 3.07GHz with 10GB of RAM
  所有测试都是单线程的，不使用任何其他优化参数(optimization flags)
  ![oop3.png](oop3.png)

## 关于汇编
一开始我想直接比较Julia和C的binary-tree汇编代码，但是我发现C的勉强能看懂个大概，Julia的就很混乱，有意思的是Julia自动启用了内联，因此我想看看Julia编译的过程中每一步它做了什么。
来看下面这段相对代码，使用@code_native查看在我的电脑中的生成的汇编结果
Julia source code
```julia
function vsum(x)
    sum = 0
    for i = 1:length(x)
        @inbounds v = x[i]
        if !is_na(v)
            sum += v
        end
    end
    sum
end
```

```
@code_native vsum(1)
...... // 省略
```

发现已经自动进行了inline，那么来看它的上一层次的代码，即llvm ir

```
@code_llvm vsum(1)
...... // 省略
```
通过llvm ir的代码可以看到在生成llvm ir的时候已经完成了inline，这个时候想起论文Julia: Dynamism and Performance Reconciled by Design上的一张图片:
![oop4.png](oop4.png)
看来在更早之前就完成了特化、类型推断、内联、拆箱的工作，经过查找发现有几个指令可以得到对应的阶段代码。

> The @code_lowered macro displays code in a format that is the closest to Python byte code, but rather than being intended for execution by an interpreter, it's intended for further transformation by a compiler. This format is largely internal and not intended for human consumption. The code is transformed into "single static assignment" form in which "each variable is assigned exactly once, and every variable is defined before it is used". Loops and conditionals are transformed into gotos and labels using a single unless/goto construct (this is not exposed in user-level Julia). 

```asm
julia> @code_lowered vsum(1)  // like python byte code
CodeInfo(
2 1 ─       sum = 0                                                         │
3 │   %2  = (Main.length)(x)                                                │
  │   %3  = 1:%2                                                            │
  │         #temp# = (Base.iterate)(%3)                                     │
  │   %5  = #temp# === nothing                                              │
  │   %6  = (Base.not_int)(%5)                                              │
  └──       goto #6 if not %6                                               │
  2 ┄ %8  = #temp#                                                          │
  │         i = (Core.getfield)(%8, 1)                                      │
  │   %10 = (Core.getfield)(%8, 2)                                          │
4 │         $(Expr(:inbounds, true))                                        │
  │   %12 = (Base.getindex)(x, i)                                           │
  │         v = %12                                                         │
  │         val = %12                                                       │
  │         $(Expr(:inbounds, :pop))                                        │
  │         val                                                             │
5 │   %17 = (Main.is_na)(v)                                                 │
  │   %18 = !%17                                                            │
  └──       goto #4 if not %18                                              │
6 3 ─       sum = sum + v                                                   │
  4 ─       #temp# = (Base.iterate)(%3, %10)                                │
  │   %22 = #temp# === nothing                                              │
  │   %23 = (Base.not_int)(%22)                                             │
  └──       goto #6 if not %23                                              │
5 ─       goto #2                                                         │
9 6 ─       return sum                                                      │
)

```

> Typed code. The @code_typed macro presents a method implementation for a particular set of argument types after type inference and inlining. This incarnation of the code is similar to the lowered form, but with expressions annotated with type information and some generic function calls replaced with their implementations.

```asm
julia> @code_typed vsum(1)
CodeInfo(
3 1 ──       (Base.ifelse)(true, 1, 0)::Int64                  │╻╷╷  Colon
  │    %2  = (Base.slt_int)(1, 1)::Bool                        ││╻╷╷  isempty
  └───       goto #3 if not %2                                 ││   
  2 ──       goto #4                                           ││   
  3 ──       goto #4                                           ││   
  4 ┄─ %6  = φ (#2 => true, #3 => false)::Bool                 │    
  │    %7  = φ (#3 => 1)::Int64                                │    
  │    %8  = φ (#3 => 1)::Int64                                │    
  │    %9  = (Base.not_int)(%6)::Bool                          │    
  └───       goto #22 if not %9                                │    
  5 ┄─ %11 = φ (#4 => 0, #21 => %36)::Int64                    │    
  │    %12 = φ (#4 => %7, #21 => %42)::Int64                   │    
  │    %13 = φ (#4 => %8, #21 => %43)::Int64                   │    
4 └───       goto #9 if not false                              │╻    getindex
  6 ── %15 = (%12 === 1)::Bool                                 ││╻    ==
  └───       goto #8 if not %15                                ││   
  7 ──       goto #9                                           ││   
  8 ── %18 = %new(Core.BoundsError)::BoundsError               ││╻    Type
  │          (Base.throw)(%18)::Union{}                        ││   
  └───       $(Expr(:unreachable))::Union{}                    ││   
  9 ┄─       goto #10                                          ││   
  5 10 ─ %22 = (Main.is_na)(x)::Any                              │    
  │    %23 = (isa)(%22, Missing)::Bool                         │    
  └───       goto #12 if not %23                               │    
  11 ─       goto #15                                          │    
  12 ─ %26 = (isa)(%22, Bool)::Bool                            │    
  └───       goto #14 if not %26                               │    
  13 ─ %28 = π (%22, Bool)                                     │    
  │    %29 = (Base.not_int)(%28)::Bool                         │╻    !
  └───       goto #15                                          │    
  14 ─ %31 = !%22::Union{Missing, Bool, ##54#55{_1} where _1}  │    
  └───       goto #15                                          │    
  15 ┄ %33 = φ (#11 => $(QuoteNode(missing)), #13 => %29, #14 => %31)::Union{Missing, Bool, ##54#55{_1} where _1}
  └───       goto #17 if not %33                               │    
6 16 ─ %35 = (Base.add_int)(%11, x)::Int64                     │╻    +
  17 ─ %36 = φ (#16 => %35, #15 => %11)::Int64                 │    
  │    %37 = (%13 === 1)::Bool                                 ││╻    == 
  └───       goto #19 if not %37                               ││   
  18 ─       goto #20                                          ││   
  19 ─ %40 = (Base.add_int)(%13, 1)::Int64                     ││╻    +
  └───       goto #20                                          │╻    iterate
  20 ┄ %42 = φ (#19 => %40)::Int64                             │    
  │    %43 = φ (#19 => %40)::Int64                             │    
  │    %44 = φ (#18 => true, #19 => false)::Bool               │    
  │    %45 = (Base.not_int)(%44)::Bool                         │    
  └───       goto #22 if not %45                               │    
  21 ─       goto #5                                           │    
9 22 ─ %48 = φ (#20 => %36, #4 => 0)::Int64                    │    
  └───       return %48                                        │    
) => Int64
```

## 与C语言的简单比较
对于下面的代码(C语言使用类似的代码)来比较Julia和C生成的汇编语言的区别
julia source code
```julia
function vsum()
    sum = 0
    for i = 1:2
        sum += i
    end
    sum
end
```

可见汇编码中的值已经计算完成
```asm
julia> @code_native vsum()
.section    __TEXT,__text,regular,pure_instructions
; Function vsum {
; Location: REPL[15]:2
    movl    $3, %eax
    retl
    nopw    %cs:(%eax,%eax)
;}
```

实际上在生成llvm ir的时候已经完成了
```asm
julia> @code_llvm vsum()

; Function vsum
; Location: REPL[15]:2
define i64 @julia_vsum_35219() {
top:
; Location: REPL[15]:6
ret i64 3
}
```

C source code
```
int vsum(){
    int sum = 0;
    for(int i=1;i<=2;++i){
        sum += i;
    }
    return sum;
}
```

执行得到结果
```shell
gcc -S small_test.c -o small_test.S
```

```asm
    .section    __TEXT,__text,regular,pure_instructions
    .macosx_version_min 10, 13
    .globl    _vsum                   ## -- Begin function vsum
    .p2align    4, 0x90
    _vsum:                                  ## @vsum
    .cfi_startproc
## BB#0:
    pushq    %rbp
Lcfi0:
    .cfi_def_cfa_offset 16
Lcfi1:
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
Lcfi2:
    .cfi_def_cfa_register %rbp
    movl    $0, -4(%rbp)
    movl    $1, -8(%rbp)
    LBB0_1:                                 ## =>This Inner Loop Header: Depth=1
    cmpl    $2, -8(%rbp)
    jg    LBB0_4
## BB#2:                                ##   in Loop: Header=BB0_1 Depth=1
    movl    -8(%rbp), %eax
    addl    -4(%rbp), %eax
    movl    %eax, -4(%rbp)
## BB#3:                                ##   in Loop: Header=BB0_1 Depth=1
    movl    -8(%rbp), %eax
    addl    $1, %eax
    movl    %eax, -8(%rbp)
    jmp    LBB0_1
LBB0_4:
    movl    -4(%rbp), %eax
    popq    %rbp
    retq
    .cfi_endproc
## -- End function
    .subsections_via_symbols
```

可见Julia的编译形式在某种程度上有优越性。

我想对julia的编译过程有更多的了解，所以从complier.jl出发，粗略地了解了一下各个include进来的文件在编译过程中的作用

