
# Julia数值计算有关语言特性

## 数组

类似于其它科学计算语言，Julia语言提供了内置的数组。Julia并不特别地对待数组。由编译器本身进行性能优化。通过继承 `AbstractArray` 来定制数组成为可能。 

数组是一个存在多维网格中的对象集合。通常，数组包含的对象的类型为 `Any` 。对大多数计算而言，数组对象一般更具体为 `Float64` 或 `Int32` 。

Julia不需要为了获得高性能而将程序被写成向量化的形式。Julia的编译器使用类型推断生成优化的代码来进行数组索引，这样的编程风格在没有牺牲性能的同时，可读性更好，编写起来更方便，有时候还会使用更少的内存。

有一些科学计算语言会通过值来传递数组，这在很多情况下很方便，而在 Julia 中，参数将通过引用传递给函数，这使得函数中对于一个数组输入的修改在函数外部是可见的。Julia 的库函数不会修改传递给它的输入。

指定类型为``T``的数组可以使用``T[A, B, C, ...]``来初始化. 这将会创建一个元素类型为``T``，元素初始化为``A``, `B`, [``](https://julia-cn.readthedocs.io/zh_CN/latest/manual/arrays/#id11)C``等的一维数组。比如``Any[x, y, z]``将创建一个包含任何类型的混合数组。

类似地，连接语句也能通过加前缀来指定元素类型

```julia
julia> [[1 2] [3 4]]
1×4 Array{Int64,2}:
 1  2  3  4

julia> Int8[[1 2] [3 4]]
1×4 Array{Int8,2}:
 1  2  3  4
```

列表推导为构造数组提供了一种更加一般，更加强大的方法。它的语法类似于数学中的集合标记法：

```julia
A = [ F(x,y,...) for x=rx, y=ry, ... ]
```

`F(x,y,...)` 根据变量 `x`, `y` 等来求值。这些变量的值可以是任何迭代对象，但大多数情况下，都使用类似于 `1:n` 或 `2:(n-1)` 的范围对象，或显式指明为类似 `[1.2, 3.4, 5.7]` 的数组。它的结果是一个 N 维稠密数组。

下例计算在维度 1 上，当前元素及左右邻居元素的加权平均数：

```julia
julia> x = rand(8)
8-element Array{Float64,1}:
 0.843025
 0.869052
 0.365105
 0.699456
 0.977653
 0.994953
 0.41084
 0.809411

julia> [ 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1] for i=2:length(x)-1 ]
6-element Array{Float64,1}:
 0.736559
 0.57468
 0.685417
 0.912429
 0.8446
 0.656511
```

### 生成器表达式

列表推导也可以被用不闭合的方括号写出，从而产生一个称为生成器的对象。这个对象可以通过迭代来产生所需的值，而不需要提前为一个数组分配内存。 （参见 man-interfaces-iteration）。 例如下面的表达式会对一列没有分配内存的数求和

```julia
julia> sum(1/n^2 for n=1:1000)
1.6439345666815615
```

在生成器参数列表中有多个维度的时候，需要通过括号来分割各个参数:

所有在 `for` 之后通过逗号分割的表达式将被解释成范围。通过增加括号能够使得我们给 `map` 增加第三个参数：

```julia
julia> map(tuple, (1/(i+j) for i=1:2, j=1:2), [1 3; 2 4])
2×2 Array{Tuple{Float64,Int64},2}:
 (0.5,1)       (0.333333,3)
 (0.333333,2)  (0.25,4)
```

## 稀疏矩阵

[稀疏矩阵](http://zh.wikipedia.org/zh-cn/%E7%A8%80%E7%96%8F%E7%9F%A9%E9%98%B5) 是其元素大部分为 0 ，并以特殊的形式来节省空间和执行时间的存储数据的矩阵。稀疏矩阵适用于当使用这些稀疏矩阵的表示方式能够获得明显优于稠密矩阵的情况。

### 列压缩（CSC）存储

Julia 中，稀疏矩阵使用 [列压缩（CSC）格式](http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29) 。Julia 稀疏矩阵的类型为 `SparseMatrixCSC{Tv,Ti}` ，其中 `Tv` 是非零元素的类型， `Ti` 是整数类型，存储列指针和行索引：

```julia
type SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::Vector{Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::Vector{Ti}      # Row values of nonzeros
    nzval::Vector{Tv}       # Nonzero values
end
```

列压缩存储便于按列简单快速地存取稀疏矩阵的元素，但按行存取则较慢。把非零值插入 CSC 结构等运算，都比较慢，这是因为稀疏矩阵中，在所插入元素后面的元素，都要逐一移位。



## 矩阵分解

[矩阵分解](http://zh.wikipedia.org/zh-cn/%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3) 是将一个矩阵分解为数个矩阵的乘积，是线性代数中的一个核心概念。

下面的表格总结了在 Julia 中实现的几种矩阵分解方式。具体的函数可以参考标准库文档的 [Linear Algebra](https://julia-cn.readthedocs.io/zh_CN/latest/stdlib/linalg/#stdlib-linalg) 章节。

 - List item

| `Cholesky`        | [Cholesky 分解](http://en.wikipedia.org/wiki/Cholesky_decomposition) |
| ----------------- | ------------------------------------------------------------ |
| `CholeskyPivoted` | [主元](http://zh.wikipedia.org/zh-cn/%E4%B8%BB%E5%85%83) Cholesky 分解 |
| `LU`              | [LU 分解](http://zh.wikipedia.org/zh-cn/LU%E5%88%86%E8%A7%A3) |
| `LUTridiagonal`   | LU factorization for Tridiagonal matrices                    |
| `UmfpackLU`       | LU factorization for sparse matrices (computed by UMFPack)   |
| `QR`              | [QR factorization](http://en.wikipedia.org/wiki/QR_decomposition) |
| `QRCompactWY`     | Compact WY form of the QR factorization                      |
| `QRPivoted`       | 主元 [QR 分解](http://zh.wikipedia.org/zh-cn/QR%E5%88%86%E8%A7%A3) |
| `Hessenberg`      | [Hessenberg 分解](http://mathworld.wolfram.com/HessenbergDecomposition.html) |
| `Eigen`           | [特征分解](http://zh.wikipedia.org/zh-cn/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3) |
| `SVD`             | [奇异值分解](http://zh.wikipedia.org/zh-cn/%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3) |
| `GeneralizedSVD`  | [广义奇异值分解](http://en.wikipedia.org/wiki/Generalized_singular_value_decomposition#Higher_order_version) |

