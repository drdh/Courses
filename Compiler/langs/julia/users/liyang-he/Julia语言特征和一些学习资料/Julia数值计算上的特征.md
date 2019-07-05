## Julia在数值计算上的特征
+ 在Julia原生库中已重载了丰富的数学运算符
一一列举各种运算符的用法显然是枯燥的，所以只举一个例子，以此来证明Julia要定义一种运算是很简单而且常见的：
定义复数结构(当然原生库已经定义过了)：
```
struct Mcomplex
    real::Float64
    imag::Float64
end
```
查看类型
```
julia> a = Mcomplex(3,4)
Mcomplex(3.0, 4.0)

julia> typeof(a)
Mcomplex
```
接着重新定义+、-、*、/、inv等运算符(以*为例)
```
julia> import Base.*

julia> *(a::Mcomplex,b::Mcomplex) = Mcomplex(a.real*b.real - a.real*b.imag,a.real*b.imag+a.imag*b.real)
* (generic function with 344 methods)

julia> b = Mcomplex(1,2)
Mcomplex(1.0, 2.0)

julia> a*b
Mcomplex(-3.0, 10.0)
```
重载show方法修改打印出来的字符串
```
import Base.show

julia> show(io::IO,::MIME"text/plain",x::Mcomplex) = print(io,"$(x.real)+$(x.imag)im")
show (generic function with 297 methods)

julia> a*b
-3.0+10.0im
```
其他符号类似
+ 数值计算的精确性（上次已经介绍过了一些函数）
+ 丰富的数学函数库
  + DataFrames.jl：在Python语言中，有一个专门用于处理行列式或表格式（tabular）数据的结构，名为DataFrame，是科学计算库Pandas的重要组成部分，已成为很多Python第三方包支持的基本数据操作框架。类似地，Julia语言的DataFrames库，正是意图建立这样一个基础的数据结构：
```
julia> df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"])
4×2 DataFrames.DataFrame

│ Row │ A │ B │

├───┼──┼──┤

│ 1   │ 1 │ M │

│ 2   │ 2 │ F │

│ 3   │ 3 │ F │

│ 4   │ 4 │ M │
```
暂时无法在v1.0以上版本使用
  + StatsFuns.jl：统计相关的数学函数，比如二项分布的概率密度函数：
```
help?> binompdf(1,2,3)
No documentation found.

StatsFuns.RFunctions.binompdf is a Function.

# 2 methods for generic function "binompdf":
[1] binompdf(n::Real, p::Real, x::Union{Float64, Int64}) in StatsFuns.RFunctions at /Users/heliyang/.julia/packages/StatsFuns/0W2sM/src/rmath.jl:55
[2] binompdf(n::Real, p::Real, k::Real) in StatsFuns at /Users/heliyang/.julia/packages/StatsFuns/0W2sM/src/distrs/binom.jl:16

julia> binompdf(10,0.5,3)
0.11718750000000014
```
还有这些
  + Lora.jl：蒙特卡罗方法
  + PDMats.jl：正定矩阵各种结构统一接口
  + ConjugatePriors.jl：共轭先验分布的Julia支持包
  + GLM.jl：广义线性模型
  + Distances.jl：向量之间距离的评估包

更多请参考(https://wenku.baidu.com/view/635521c04b35eefdc9d3336c.html)







