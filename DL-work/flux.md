# Flux

[Flux](https://github.com/FluxML/Flux.jl) is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.

## Flux 的 AD

Back-propagation, or reverse-mode automatic differentiation, is handled by the`Flux.Tracker` module.

目前 Flux 的 AD 是 [Flux.Tracker](https://github.com/FluxML/Tracker.jl), 之后会被 [Zygote](https://github.com/FluxML/Zygote.jl) 换掉.

### 使用方法

`gradient` 接受一个函数, 和函数的参数, 输出关于这些参数的导函数.

```julia
julia> using Flux.Tracker

julia> f(x) = 3x^2 + 2x + 1;

julia> df(x) = Tracker.gradient(f, x; nest = true)[1]; # df/dx = 6x + 2

julia> df(2)
14.0 (tracked)
```

When a function has many parameters, we can pass them all in explicitly:

```julia
julia> f(W, b, x) = W * x + b;

julia> Tracker.gradient(f, 2, 3, 4)
(4.0 (tracked), 1.0 (tracked), 2.0 (tracked))
```

### 实现

`gradient` 是 `forward` 函数的简单封装.  `forward` 叫做 "backpropagator-based interface".

`param` 把 julia array 变成特殊的 object(`TrackedArray`, `TrackedReal` etc.), tracks extra information

结构体定义: [Tracker.jl](<https://github.com/FluxML/Tracker.jl/blob/master/src/Tracker.jl>)
```julia
mutable struct Tracked{T}
  ref::UInt32
  f::Call
  isleaf::Bool
  grad::T
  Tracked{T}(f::Call) where T = new(0, f, false)
  Tracked{T}(f::Call, grad::T) where T = new(0, f, false, grad)
  Tracked{T}(f::Call{Nothing}, grad::T) where T = new(0, f, true, grad)
end

isleaf(x::Tracked) = x.f == Call() # 不是一个函数则是一个叶子.

# 根据 nest 的值调用不同函数, nest 是什么?
gradient(f, xs...; nest = false) =
  nest ? gradient_nested(f, xs...) : gradient_(f, xs...)

function gradient_(f, xs...)
  xs = param.(data.(xs))
    # param 把参数数据变成 Tracked{} 类型
    # data.(xs)
    #  如果 xs 是变量, 则返回本身.
    #  如果 xs 是一个 Tracked 变量, 则返回其内部 data
  l = f(xs...)
    # Tracked 类型在计算后得到的结果仍然是 Tracked 类型.
    # 下一节介绍如何对 julia 里的所有操作进行重载.
  losscheck(l)  # 检测 l 的类型, 必须是标量.
  @interrupts back!(l) # 反向求导
  extract_grad!.(xs)
end
```

#### 计算重载

`Tracked` 类型在经过计算后仍是 `Tracked` 的类型.

例如:  `xs = param(2); xs + 2`  , 在执行加法时, 使用 `@which` 宏可知调用的是下面的函数.

```
+(a::Tracker.TrackedReal, b::Real) in Tracker at /Users/guoxing/project/julia/Tracker.jl/src/lib/real.jl:94
```

这个函数在[ `real.jl`](<https://github.com/FluxML/Tracker.jl/blob/master/src/lib/real.jl#L97>) 源文件中没有, 是自动生成的.

```julia
for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue   # 这个 for 循环处理 f 的 arity 为 2 的函数.
                           # 另有一个 for 循环处理 f 的 arity 为 1 的函数
  da, db = DiffRules.diffrule(M, f, :a, :b)
  # da, db 为 函数 f 的两个偏导数
  f = :($M.$f)
  @eval begin
    # 为 f 定义在 Tracked 变量上的运算
    # grad 和 track 介绍见下一节.
    @grad $f(a::TrackedReal, b::TrackedReal) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
    @grad $f(a::TrackedReal, b::Real) = $f(data(a), b), Δ -> (Δ * $da, _zero(b))
    @grad $f(a::Real, b::TrackedReal) = $f(a, data(b)), Δ -> (_zero(a), Δ * $db)
    $f(a::TrackedReal, b::TrackedReal)  = track($f, a, b)
    $f(a::TrackedReal, b::Real) = track($f, a, b)
    $f(a::Real, b::TrackedReal) = track($f, a, b)
  end
end
```

这里使用了 [DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) ,  from http://www.juliadiff.org/  , 其中包含以及定义好的 diffrules.  后面会介绍.

#### @grad

```julia
macro grad(ex)
    # @capture 是 MacroTools 提供的宏. 用来进行 Expr 模式匹配
    # 这里匹配一个函数定义
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  isexpr(name, :(::)) || (name = :(::typeof($name)))
  insert!(args, 1+isexpr(args[1], :parameters) , name)
  @q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end


# 函数调用
track(f::Call, x) = Tracked{typeof(x)}(f)
#
function track(f::F, xs...; kw...) where F
  y, back = _forward(f, xs...; kw...)
  track(Call(back, tracker.(xs)), y)
end


```



#### back

反向求导

```julia
function back!(x::TrackedReal; once = true)
    isinf(x) && error("Loss is Inf")
    isnan(x) && error("Loss is NaN")
    return back!(x, 1, once = once)
end

function back!(x, Δ; once = true)
  istracked(x) || return
  scan(x)
  back(tracker(x), Δ, once)
  return
end

# 这里对递归扫描 x, 将 x.grad 清零
scan(c::Call) = foreach(scan, c.args)

function scan(x::Tracked)
  x.isleaf && return
  ref = x.ref += 1 # 避免二次扫描
  if ref == 1  # 如果第一次
    scan(x.f)  # 扫描每个参数
    isdefined(x, :grad) && (x.grad = zero_grad!(x.grad))
  end
  return
end

function back(x::Tracked, Δ, once)
  x.isleaf && (x.grad = accum!(x.grad, Δ); return)
  ref = x.ref -= 1
  grad = if isdefined(x, :grad)
    x.grad = accum!(x.grad, Δ)
  elseif ref > 0
    x.grad = Δ
  else
    Δ
  end
  if ref == 0
    back_(x.f, grad, once)
    once && !x.isleaf && (x.f = Call(missing, ()))
  end
  return
end
```



####  DiffRules

[DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) 为常见的基本函数或算符定义了其导函数.

```julia
    @define_diffrule M.f(x) = :(df_dx(\$x))
    @define_diffrule M.f(x, y) = :(df_dx(\$x, \$y)), :(df_dy(\$x, \$y))
```

每个 `@define_diffrule`  定义一条规则, 一元函数给出其全导数, 二元函数给出其对两个变元的偏导数. 导函数的前面有一个`:` , 表示这是一个 `Expr`,   ?

例如:

```julia
    @define_diffrule Base.cos(x)          = :(-sin(\$x))
    @define_diffrule Base.:/(x, y)        = :(inv(\$y)), :(-\$x / (\$y^2))
    @define_diffrule Base.polygamma(m, x) = :NaN,       :(polygamma(\$m + 1, \$x))
```


乘法的导数

```julia
@grad a * b = data(a)*data(b), Δ -> (Δ*b, a*Δ)
```



## Julia 元编程

参考

* [Introducing Julia/Metaprogramming, wiki](https://en.wikibooks.org/wiki/Introducing_Julia/Metaprogramming)
* [Metaprogramming, Julia docs](<https://docs.julialang.org/en/v1/manual/metaprogramming/index.html>)

元编程即代码可以处理, 修改代码自身. Julia 的代码本身使用 Julia 的数据结构保存, 程序可以在程序抽象语法树(Abstract Syntac Tree, AST)层次上对代码自身进行修改. 可以实现反射, 自动代码生成等.

### 数据结构 Quoted expressions

**Quoted expressions** 用于存储已经 parse, 还没有执行的表达式. 以冒号`:`开头. 可以从字符串编译得到, `Meta.parse("1 + 1")` , 也可作为字面值在程序中出现. 主要类型为 `Symbol` 和 `Expr`

```julia
:x # Symbol 类型
:(2 + 2) # Expr 类型
# 也可写成 quote .. end 形式:
quote
   2 + 2
end
```

**执行**: 使用 `eval()` 即可执行 `Symbol` 或 `Expr`

**检查** :`Expr` 有两个 members : `(:head, :args)`  (可使用 `filednames(Expr)` 查看)

```julia
P = quote
   a = 2
   b = 3
   c = 4
   d = 5
   e = sum([a,b,c,d])
end

P.head # 是 :block
P.args # 是一个 Array, 含有该 Expr 中的子表达式和注释
P.args[2] # 第二个元素是 :(a = 2)
```

**修改**: 像修改 Array 一样直接修改即可

```julia
eval(P) # 结果为 14
P.args[end] = :( prod([a,b,c,d]) ) # 将 sum 改为 prod
eval(P) # 结果为  120
```

**AST** : 可以直接使用 `dump()`  打印 AST.  (`dump` 可以用于任何结构体)

```julia
dump(:(1 * sin(pi/2)))
#:
Expr
  head: Symbol call
  args: Array{Any}((3,))
    1: Symbol *
    2: Int64 1
    3: Expr
      head: Symbol call
      args: Array{Any}((2,))
        1: Symbol sin
        2: Expr
          head: Symbol call
          args: Array{Any}((3,))
            1: Symbol /
            2: Symbol pi
            3: Int64 2
```

**Expression interpolation** 将指向代码获得的结果插入 quoted expressions.

```julia
quote s = $(sin(1) + cos(1)) end
# $(sin(1) + cos(1)) 会被其执行结果替换掉
# 等价于:
quote s = 1.3817732906760363 end
```

### 宏 Macros

宏将输入的 表达式作为 Quoted expressions , 经过特定变换后得到 新的表达式 并执行.

**宏的定义与使用**,  定义时使用关键词 `macro`, 使用时使用 `@` 加上宏的名字.

```julia
macro p(n)
    if typeof(n) == Expr
       println(n.args)
    end
    return n
end
@p 3 # 返回 3
@p(1 + 1) # 打印 Any[:+, 1, 1], 返回 2.

```



```
typeof(x) # 类型信息
fieldnames(typeof(X)) # 查看一个类的类型
dump(x) # dump x 的所有成员
@time # 测时间
@which # 检测调用的是哪个 method.
```



### MacroTools

Flux.Tracker 依赖 julia 的 MacroTools 库. 这个库是宏的编写更为方便.

```julia
ex = quote
  struct Foo
    x::Int
    y
  end
end
```

If you know what you're doing, you can pull out the name and fields via:

```julia
julia> if isexpr(ex.args[2], :struct)
         (ex.args[2].args[2], ex.args[2].args[3].args)
       end
(:Foo,{:( # line 3:),:(x::Int),:( # line 4:),:y})
```

Enter MacroTools:

```julia
julia> using MacroTools

julia> @capture(ex, struct T_ fields__ end)
true

julia> T, fields
(:Foo, [:(x::Int), :y])
```

Symbols like `T_` underscore are treated as catchalls which match any expression, and the expression they match is bound to the (underscore-less) variable, as above.

Symbols like `f__` (double underscored) are similar, but slurp a sequence of arguments into an array.





<https://julialang.org/blog/2018/12/ml-language-compiler>

