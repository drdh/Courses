# Julia 问题探究

何纪言

## 1. 函数默认参数出现在非结尾位置

例子：

```julia
> A = zeros(3, 3) # 默认类型为 Float64
3×3 Array{Float64,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
> A = zeros(Float64, 3) # 显式指定类型为 Float64
3-element Array{Float64,1}:
 0.0
 0.0
 0.0
```

### 采用默认参数的方法

多数语言中，函数参数缺省值只能位于参数列表的最后：

> C++: Default arguments (N3242=11-0012 8.3.6)
>
> If an initializer-clause is specified in a parameter-declaration this initializer-clause is used as a default argument. **Default arguments** will be used in calls **where trailing arguments are missing**. 
>
> In a given function declaration, **each parameter subsequent to a parameter with a default argument shall have a default argument** supplied in this or a previous declaration or shall be a function parameter pack. 

一个例子：

```c++
void func(int a = 1, int b);
```

编译会得到如下错误：

```shell
error: missing default argument on parameter 'b'
```

这一条限制可能的原因是为了避免二义，比如面临 `func` 的一次调用实例 `func(3)`，可能会有以下两种解释方法：

```c++
func a=1 b=3
func a=3
```

在 C++，Python 等一些常用语言中都没有支持“默认参数”出现在非尾部。

事实上，在 Julia 中也不支持这种写法：

```julia
> function func(a = 1, b) end
ERROR: syntax: optional positional arguments must occur at end
```

### 采用 Multiple Dispatch（多重派发） 方法

首先我们确认 `zeros` 的类型确实为函数类型，并且可以发现 `zeros` 存在 6 个不同的方法：

```julia
> methods(zeros)
# 6 methods for generic function "zeros":
[1] zeros(dims::Union{Integer, AbstractUnitRange}...) in Base at array.jl:463
[2] zeros(::Type{T}, dims::Union{Integer, AbstractUnitRange}...) where T in Base at array.jl:464
[3] zeros(dims::Tuple{Vararg{Union{Integer, AbstractUnitRange},N} where N}) in Base at array.jl:465
[4] zeros(::Type{T}, dims::Tuple{}) where T in Base at array.jl:468
[5] zeros(::Type{T}, dims::Tuple{Vararg{Integer,N}}) where {T, N} in Base at array.jl:467
[6] zeros(::Type{T}, dims::Tuple{Vararg{Union{Integer, OneTo},N}}) where {T, N} in Base at array.jl:466
```

在调用 `zeros(3, 3)` 和 `zeros(Float64, 3)` 时，根据参数类型的不同（**运行时**），分别匹配了第 1, 2 种情况，进而执行了不同的逻辑代码。

> Julia allows the **dispatch** process to choose which of a function's methods to call based on **the number of arguments given,** and on **the types of all of the function's arguments**.
>
> Using **all of a function's arguments** to choose which method should be invoked, **rather than just the first**, is known as multiple dispatch.
>
> *(from `doc/src/manual/methods.md`)*

### 扩展讨论

#### dispatch 的实现机制

多个方法的 dispatch 是由 `jl_apply_generic` 处理的，声明如下：

```c
// src/julia.h L1564
JL_DLLEXPORT jl_value_t *jl_apply_generic(jl_value_t **args, uint32_t nargs);
```

相关函数定义如下，根据：

```c
// src/gf.c: Generic Functions
// L2203
// 执行函数
JL_DLLEXPORT jl_value_t *jl_apply_generic(jl_value_t **args, uint32_t nargs)
{
    // 查找 method
    jl_method_instance_t *mfunc = jl_lookup_generic_(args, nargs,
                                                     jl_int32hash_fast(jl_return_address()),
                                                     jl_get_ptls_states()->world_age);
    JL_GC_PROMISE_ROOTED(mfunc);
    // 使用参数调用方法
    jl_value_t *res = mfunc->invoke(mfunc, args, nargs);
    return verify_type(res);
}

// L2111
STATIC_INLINE jl_method_instance_t *jl_lookup_generic_(jl_value_t **args, uint32_t nargs,
                                                       uint32_t callsite, size_t world)
{
    // 主要逻辑：
    // 两层缓存机制：
    //  - 相联缓存（通过 hash 确定相联缓存位置）
    //  - 完整缓存
    // 返回一个具体的 method
#ifdef JL_GF_PROFILE
    ncalls++;
#endif
#ifdef JL_TRACE
    int traceen = trace_en; //&& ((char*)&mt < jl_stack_hi-6000000);
    if (traceen)
        show_call(args[0], &args[1], nargs-1);
#endif

    /*
      search order:
      check associative hash based on callsite address for leafsig match
      look at concrete signatures
      if there is an exact match, return it
      otherwise look for a matching generic signature
      if no concrete or generic match, raise error
      if no generic match, use the concrete one even if inexact
      otherwise instantiate the generic method and use it
    */
    // compute the entry hashes
    // use different parts of the value
    // so that a collision across all of
    // them is less likely
    // 计算 hash，以便于快速判断
    uint32_t cache_idx[4] = {
        (callsite) & (N_CALL_CACHE - 1),
        (callsite >> 8) & (N_CALL_CACHE - 1),
        (callsite >> 16) & (N_CALL_CACHE - 1),
        (callsite >> 24 | callsite << 8) & (N_CALL_CACHE - 1)};
    jl_typemap_entry_t *entry = NULL;
    jl_methtable_t *mt = NULL;
    int i;
    // /在相联缓存中搜索并判断匹配
    // check each cache entry to see if it matches
    for (i = 0; i < 4; i++) {
        entry = call_cache[cache_idx[i]];
        if (entry && nargs == jl_svec_len(entry->sig->parameters) &&
            sig_match_fast(args, jl_svec_data(entry->sig->parameters), 0, nargs) &&
            world >= entry->min_world && world <= entry->max_world) {
            break;
        }
    }
    // 在相联缓存中 miss，在完整 cache 中继续搜索
    // if no method was found in the associative cache, check the full cache
    if (i == 4) {
        JL_TIMING(METHOD_LOOKUP_FAST);
        jl_value_t *F = args[0];
        mt = jl_gf_mtable(F);
        entry = jl_typemap_assoc_exact(mt->cache, args, nargs, jl_cachearg_offset(mt), world);
        if (entry && entry->isleafsig && entry->simplesig == (void*)jl_nothing && entry->guardsigs == jl_emptysvec) {
            // put the entry into the cache if it's valid for a leafsig lookup,
            // using pick_which to slightly randomize where it ends up
            // 如果找到了，加入相联缓存
            call_cache[cache_idx[++pick_which[cache_idx[0]] & 3]] = entry;
        }
    }

    jl_method_instance_t *mfunc = NULL;
    if (entry) {
        mfunc = entry->func.linfo;
    }
    else {
        JL_LOCK(&mt->writelock);
        // cache miss case
        // 缓存中没有找到
        JL_TIMING(METHOD_LOOKUP_SLOW);
        jl_tupletype_t *tt = arg_type_tuple(args, nargs);
        JL_GC_PUSH1(&tt);
        mfunc = jl_mt_assoc_by_type(mt, tt, /*cache*/1, /*allow_exec*/1, world);
        JL_GC_POP();
        JL_UNLOCK(&mt->writelock);
        if (mfunc == NULL) {
#ifdef JL_TRACE
            if (error_en)
                show_call(args[0], args, nargs);
#endif
            jl_method_error((jl_function_t*)args[0], args, nargs, world);
            // unreachable
        }
    }

#ifdef JL_TRACE
    if (traceen)
        jl_printf(JL_STDOUT, " at %s:%d\n", jl_symbol_name(mfunc->def.method->file), mfunc->def.method->line);
#endif
    return mfunc;
}
}
```

#### methods() 的实现机制

`methods()` 本身是一个 generic function：

```julia
> methods(methods)
# 3 methods for generic function "methods":
[1] methods(f::Core.Builtin) in Base at reflection.jl:757
[2] methods(f) in Base at reflection.jl:769
[3] methods(f, t) in Base at reflection.jl:749
```

定义在 `reflection.jl`：

```julia
function methods(@nospecialize(f), @nospecialize(t))
    if isa(f, Core.Builtin)
        throw(ArgumentError("argument is not a generic function"))
    end
    t = to_tuple_type(t)
    world = typemax(UInt)
    return MethodList(Method[m[3] for m in _methods(f, t, -1, world)], \
        typeof(f).name.mt) # <---- 这里
end
```

能获取关于符号信息的关键在于在于调用的 `typeof` 这个 builtin：

```c
// src/builtins.c
jl_value_t *jl_typeof(jl_value_t *v);
```

这涉及到 Julia 中的任何一个 object 都是 `jl_datatype_t` 的一个实例，可以通过 `jl_typeof` 来查询（ref: doc/src/devdocs/object.md)。 

```c
// src/julia.h L399
typedef struct _jl_datatype_t {
    JL_DATA_TYPE
    jl_typename_t *name;
    struct _jl_datatype_t *super;
    jl_svec_t *parameters;
    jl_svec_t *types;
    jl_svec_t *names;
    jl_value_t *instance;  // for singletons
    const jl_datatype_layout_t *layout;
    int32_t size; // TODO: move to _jl_datatype_layout_t
    int32_t ninitialized;
    uint32_t uid;
    uint8_t abstract;
    uint8_t mutabl;
    // memoized properties
    uint8_t hasfreetypevars; // majority part of isconcrete computation
    uint8_t isconcretetype; // whether this type can have instances
    uint8_t isdispatchtuple; // aka isleaftupletype
    uint8_t isbitstype; // relevant query for C-api and type-parameters
    uint8_t zeroinit; // if one or more fields requires zero-initialization
    uint8_t isinlinealloc; // if this is allocated inline
    void *struct_decl;  //llvm::Type*
    void *ditype; // llvm::MDNode* to be used as llvm::DIType(ditype)
} jl_datatype_t;
```

### 结论

文档中（https://docs.julialang.org/en/v1/base/arrays/#Base.zeros）`zeros([T=Float64,] dims...)` 以及其他类似的文档写法**并不代表 Julia 支持声明时函数默认参数出现在非结尾！**这是通过 multiple dispatch 的语言特性，在运行时实现的。

Multiple dispatch 也是 Julia 中最重要的特性之一，几乎所有的内置函数都有一大堆针对不同类型的 methods，比如运算符 `+` 就有 163 个可选的方法。这使得 Julia 非常适合在高层次描述算法，并且程序非常优雅简洁。

反过来说，在一定程度上，这也使得 Julia 运行时开销变大（时间和空间），可能导致程序运行缓慢，甚至也会使得编译时间受到较大影响。在 Julia 文档（doc/src/manual/performance-tips.md）中有 *The dangers of abusing multiple dispatch* 一节，说明了滥用 multiple dispatch 带来的危害，另外文档还指出了能和 multiple dispatch 等效替代的其他几种更高效的方法，如：`switch`, `dict lookup` 等（虽然看起来都不是没有这么优雅）。可以看出这 Julia 的一机制还存在一定的优化空间。

Julia 社区邮件列表（https://groups.google.com/forum/#!msg/julia-users/jUMu9A3QKQQ/qjgVWr7vAwAJ）中对此也有一些 benchmark。但邮件列表中的语法不适合 Julia 1.0，整理后的代码在这里：https://jl.hejiyan.com/notebooks/dispatch_benchmark.ipynb 。

## 2. Broadcast 的机制

首先看几个例子：

一些计算函数的 broadcast，可以将函数应用在每一个元素上，保持原来的数组尺寸：

```julia
julia> sin(1.0)
0.8414709848078965

julia> a = [1.0 2.0 3.0]
1×3 Array{Float64,2}:
 1.0  2.0  3.0

julia> sin.(a)
1×3 Array{Float64,2}:
 0.841471  0.909297  0.14112
```

这里要注意的是，`sin` 并没有提供针对 `Array{T}` 的 method，不能直接 `sin(A)`，点号（`.`)使得我们使用了 broadcast 机制，在 array 上逐元素应用 `sin`。（之所以提这个是因为在我们熟悉的 numpy 中的类似函数可以直接作用于 array，但技术上来说不是靠 broadcast 机制实现的）

一些二元函数的 broadcast，可以将相等尺寸的两个数组对应元素应用到函数上，得到结果：

```julia
julia> [1 2 3 4] .+ [4 3 2 1]
1×4 Array{Int64,2}:
 5  5  5  5
```

而在数组的尺寸不一样时，有的会引发错误，有的会采取一些自动补全，如：

```julia
# 错误：大小不匹配
julia> [1 2 3 4] .+ [4 3 2]
ERROR: DimensionMismatch("arrays could not be broadcast to a common size")

# 等价于每一行 +1
julia> [1 2 3 4] .+ [1]
1×4 Array{Int64,2}:
 2  3  4  5

# 等价于第一行 +1， 第二行 +2
julia> [1 2 3 4; 5 6 7 8] .+ [1; 2]
2×4 Array{Int64,2}:
 2  3  4   5
 7  8  9  10

# 等价于原来的数组复制变为两行，然后第一行 +1， 第二行 +2
julia> [1 2 3 4] .+ [1; 2]
2×4 Array{Int64,2}:
 2  3  4  5
 3  4  5  6

# 错误：大小不匹配
julia> [1 2 3 4; 5 6 7 8] .+ [1 1; 2 2]
ERROR: DimensionMismatch("arrays could not be broadcast to a common size")
```

### 语法糖

结合文档从上面的使用我们可以知道 `f.(args...)` 是 `broadcast(f, args...)` 的语法糖，之后不再作区分。

### 实现机制

broadcast 中对于不同大小的数组处理使得这个语法特性应用很方便，根据源代码我们可以看到 Julia 中的 broadcast 是一个运行时递归处理的过程，会根据 broadcast 的列表参数，每次将前头两个参数：`shape` 和 `shape1` 进行合并，以减少了一个参数，不断迭代直到只剩下一个最终的 output shape。

相关的函数如下：

```julia
# base/broadcast.jl
# Indices utilities

# 合并每一个参数的入口函数，处理只有一个的边界情况
@inline combine_axes(A, B...) = broadcast_shape(axes(A), combine_axes(B...))
combine_axes(A) = axes(A)

# shape (i.e., tuple-of-indices) inputs
# 合并前两个参数的 shape 为一个，并递归直至剩下一个
broadcast_shape(shape::Tuple) = shape
broadcast_shape(shape::Tuple, shape1::Tuple, shapes::Tuple...) = broadcast_shape(_bcs(shape, shape1), shapes...)

# _bcs consolidates two shapes into a single output shape
# 合并两个参数的 shape 为一个
_bcs(::Tuple{}, ::Tuple{}) = ()
_bcs(::Tuple{}, newshape::Tuple) = (newshape[1], _bcs((), tail(newshape))...)
_bcs(shape::Tuple, ::Tuple{}) = (shape[1], _bcs(tail(shape), ())...)
function _bcs(shape::Tuple, newshape::Tuple)
    return (_bcs1(shape[1], newshape[1]), _bcs(tail(shape), tail(newshape))...)
end

# _bcs1 handles the logic for a single dimension
# 特殊处理了 single dimension 的情况
_bcs1(a::Integer, b::Integer) = a == 1 ? b : (b == 1 ? a : (a == b ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size"))))
_bcs1(a::Integer, b) = a == 1 ? b : (first(b) == 1 && last(b) == a ? b : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
_bcs1(a, b::Integer) = _bcs1(b, a)
_bcs1(a, b) = _bcsm(b, a) ? _sametype(b, a) : (_bcsm(a, b) ? _sametype(a, b) : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
```

### 结论

Julia 的 broadcast 在运行时完成各个参数维度大小的检查和补全，使得我们可以写出诸如：

```julia
[1 2 3 4] .+ [1; 2]
```

这样方便的代码。

顺带一提，上面提到的对数组的 broadcast 也适用于元组。

## 3. 一些性能测试

这里收集并讨论一些针对 Julia 的性能分析。

### Julia Micro-Benchmarks

micro-benchmarks 是 Julia 官方进行的一些性能测试。

这个测试使用的各个语言版本如下：

```
Julia v1.0.0
SciLua v1.0.0-b12
Rust 1.27.0
Go 1.9
Java 1.8.0_17
Javascript V8 6.2.414.54
Matlab R2018a
Anaconda Python 3.6.3
R 3.5.0
Octave 4.2.2.
C (gcc 7.3.1)
Fortran (gcc 7.3.1)
```

在矩阵运算库的选择上，C, Fortran, Go, Julia, Lua, Python, Octave 使用 OpenBLAS v0.2.20，Mathematica 使用 Intel MKL。测试的环境是：a single core (serial execution) on an Intel(R) Core(TM) i7-3960X 3.30GHz CPU with 64GB of 1600MHz DDR3 RAM, running openSUSE LEAP 15.0 Linux.

![img](https://julialang.org/images/benchmarks.svg)

可以看出大多数的测试中 Julia 的性能排在前列，并且 Julia 有着更高层次的描述能力，写出来的代码会更友好。

在线性代数运算方面，即使是使用了和 numpy 使用相同的后端（OpenBLAS），Julia 也要快于 Python，因为相对于 Python，Julia 代码中的一些预处理工作在编译时已经被优化掉。这体现了 Julia 在引入抽象的同时，步引入太多开销的优点。

（有人在天河二号上运行了这个 benchmark，得到的结果和官方数据基本一致）

### LU Factorization

https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization

IBM 社区上一篇 LU 分解的测试显示 Julia 在小矩阵表现甚至优于 C，CPython，在矩阵大小变大时性能接近。

![image](https://www.ibm.com/developerworks/community/blogs/jfp/resource/BLOGS_UPLOADED_IMAGES/runtimes_210.png)

```julia
function det_by_lu(y, x, N)
    y[1] = 1.
    @inbounds for k = 1:N
        y[1] *= x[k,k]
        @simd for i = k+1:N # 这个优化类似于 C 中的 OpenMP，不过是 Julia 语言自带的（experimental）
            x[i,k] /= x[k,k]
        end
        for j = k+1:N
            @simd for i = k+1:N
                x[i,j] -= x[i,k] * x[k,j]
            end
        end
    end
end
```

### Merge sort

https://hackernoon.com/performance-analysis-julia-python-c-dd09f03282a3

n = 1,000

![img](https://cdn-images-1.medium.com/max/1600/1*HkGBMMHqa6FjJ5Jk7oieAQ.jpeg)

n = 1,00,000

![img](https://cdn-images-1.medium.com/max/1600/1*hh9qXw2FiPossRtUIjjyKA.jpeg)

n = 1,00,00,000 (C 语言因为崩溃运行失败)

![img](https://cdn-images-1.medium.com/max/1600/1*vZDQZFVG6SQ4oiDb3pVLVQ.jpeg)

根据文章的分析，性能差别的一个很大的原因是内存分配时，Python 会分配 `n*1.125+6` 的空间，而 Julia 会分配 `2*n` 的空间。

### NASA: Basic Comparison of Python, Julia, Matlab, IDL and Java (2018 Edition)

https://modelingguru.nasa.gov/docs/DOC-2676

结论：Julia 在很多问题的算法上运行效率仅次于 C。