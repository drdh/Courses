## Julia语言特征
主要是通过老师提供的文章Julia: A Fresh Approach to Numerical Computing以及查看官方文档总结出Julia的一些特征，如果是Python、C、C++、Java里面有的特征就不列举了，主要列举一些比较少见的特征。

### Julia哲学
一言以蔽之，Julia的设计目标是创建出一个足够“贪心”的语言。它的运行性能要像C和Fortran一样优秀，又能在数据科学应用上像Matlab、Python、R语言一样强大，还要吸收比如Perl、Lisp语言的一些优秀的地方。

### 数值计算十分方便

矩阵
```
A = rand(3,3) // 生成随机的3*3矩阵 
inv(A) // 求逆
```

方便的代数系统
```
julia> 2x^2 - 3x + 1
```

链式比较
```
julia> 1 < 2 <= 2 < 3 == 3 > 2 >= 1 == 1 < 3 != 5
true
```

分数
```
julia> 3//2
3//2
```

复数
```
(1+2im)*(1+3im)
-5 + 5im
```


### 数值准确性
在精度上，使用eps(f)函数来检查浮点数f和下一个可表示浮点数之间的绝对差
```
julia> eps(2.0)
4.440892098500626e-16
```
多种舍入模型
+ RoundDown
+ RoundUp
+ RoundNearest



### 无类型变量
变量是没又类型的，变量等于是一个指向类型的引用，也就是动态类型，但是这样往往会降低性能，那么Julia又是怎么实现解决这个问题的呢？我觉得还要深入底层去了解。

其中一个原因是Julia 的编译器使用类型推断，并为标量数组索引生成优化的代码，而不牺牲性能，并且时常会减少内存使用
```
Array{Int64,2} // 二维矩阵 所有值都是Int64类型
```

### 循环很快

### 方便的分配模式提高编写效率
```
methods(+)  // 查看+重载了多少方法
```

很简单地定义一些符号运算：
```
*(f::Function,g::Function)= x->f(g(x))  
*(f::Function,t::Number) = x->f(t*x)
```

### 函数
基本定义方式、多参数、可选参数类似python，箭头函数类似javascript

函数简化的写法：
```
function g(x,y)
x + y
end
```

等价
```
f(x,y) = x + y
```

自动返回:
```
julia> f(x,y) = x+y
f (generic function with 1 method)

julia> f(2,2)
4
```

按照约定，以感叹号结尾的函数会改变输入参数的内容。
```
julia> a = [3,2,1]
3-element Array{Int64,1}:
3
2
1

julia> sort(a)
3-element Array{Int64,1}:
1
2
3

julia> a
3-element Array{Int64,1}:
3
2
1

julia> sort!(a)
3-element Array{Int64,1}:
1
2
3

julia> a
3-element Array{Int64,1}:
1
2
3

```

### 短路求值
if <cond> <statement> end 等价 <cond> && <statement>
```
julia> function factorial(n::Int)
n >= 0 || error("n must be non-negative")
n == 0 && return 1
n * factorial(n-1)
end
```

### 内插
借鉴了Perl
```
julia> greet = "Hello"
"Hello"

julia> whom = "world"
"world"

julia> "$greet, $whom.\n"
"Hello, world.\n"
```

### 循环
```
julia> for i = 1:2, j = 3:4
println((i, j))
end
(1,3)
(1,4)
(2,3)
(2,4)
```

### 广播
借鉴matlab，类似map函数，但是更方便
```
julia> a = [1,2,3]
3-element Array{Int64,1}:
1
2
3

julia> sin.(a)
3-element Array{Float64,1}:
0.8414709848078965
0.9092974268256817
0.1411200080598672
```

### 元编程
继承自Lisp。Julia 把自己的代码表示为语言中的数据结构。既然代码被表示为了可以在语言中创建和操作的对象，程序就可以变换和生成自己的代码。
```
julia> exp = Meta.parse("1*3+2")
:(1 * 3 + 2)

julia> dump(exp)
Expr
head: Symbol call
args: Array{Any}((3,))
1: Symbol +
2: Expr
head: Symbol call
args: Array{Any}((3,))
1: Symbol *
2: Int64 1
3: Int64 3
3: Int64 2

julia> exp = Meta.parse("1+3*2")
:(1 + 3 * 2)

julia> dump(exp)
Expr
head: Symbol call
args: Array{Any}((3,))
1: Symbol +
2: Int64 1
3: Expr
head: Symbol call
args: Array{Any}((3,))
1: Symbol *
2: Int64 3
3: Int64 2
```

### 数组推导
```
julia> [x*x for x=1:10]
10-element Array{Int64,1}:
1
4
9
16
25
36
49
64
81
100
```

### 协程
可以理解为一种轻量级线程，线程还在实验阶段

### 运行外部程序
可以像shell、Perl、Ruby一样运行外部程序
```
julia> ls = `ls`
`ls`

julia> run(ls)
Desktop              Public               gcc_test
Documents            WebstormProjects     leon
Downloads            compli               node_modules
Library              compli.pub           node_test
Movies               eclipse              package-lock.json
Music                eclipse-workspace
Pictures             eclipse-workspace-ee
Process(`ls`, ProcessExited(0))
```

### 一些有用的链接
Julia1.0中文文档(https://juliacn.gitlab.io/JuliaZH.jl/)
Julia1.01英文文档https://docs.julialang.org/en/v1/manual/metaprogramming/)
Julia包(https://pkg.julialang.org/)
中文社区(http://discourse.juliacn.com/c/general/performance)
英文社区(https://discourse.julialang.org/)
数据集(http://archive.ics.uci.edu/ml/index.php)

### 关于IDE
Juno(http://docs.junolab.org/latest/man/installation.html)
IJulia(https://zhuanlan.zhihu.com/p/42812662)
