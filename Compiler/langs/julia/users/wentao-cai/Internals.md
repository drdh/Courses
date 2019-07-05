# Julia's Internals: AST

Julia 有两种代码的表现形式。 第一种是解析器返回的表面语法 AST （例如 [`Meta.parse`](https://juliacn.github.io/JuliaZH.jl/latest/base/base/#Base.Meta.parse-Tuple{AbstractString,Int64}) 函数），由宏来操控。是代码编写时的结构化表示，由字符流中的 `julia-parser.scm` 构造。 另一种则是低级的形式，或者 IR（中间表示），用来类型推导和代码生成。在低级形式上有少部分结点的类型，所有的宏都会被展开，所有的控制流会被转化成显式的分支和语句的序列。底层的形式由 `julia-syntax.scm` 构建。



### Quote expressions

Julia source syntax forms for code quoting (`quote` and `:( )`) support interpolation with `$`. In Lisp terminology, this means they are actually "backquote" or "quasiquote" forms. Internally, there is also a need for code quoting without interpolation. In Julia's scheme code, non-interpolating quote is represented with the expression head `inert`.

`inert` expressions are converted to Julia `QuoteNode` objects. These objects wrap a single value of any type, and when evaluated simply return that value.

A `quote` expression whose argument is an atom also gets converted to a `QuoteNode`.





## Module bindings

The exported names for a `Module` are available using [`names(m::Module)`](https://docs.julialang.org/en/v1/base/base/#Base.names), which will return an array of [`Symbol`](https://docs.julialang.org/en/v1/base/strings/#Core.Symbol) elements representing the exported bindings. `names(m::Module, all = true)` returns symbols for all bindings in `m`, regardless of export status.

## DataType fields

The names of `DataType` fields may be interrogated using [`fieldnames`](https://docs.julialang.org/en/v1/base/base/#Base.fieldnames). For example, given the following type, `fieldnames(Point)` returns a tuple of [`Symbol`](https://docs.julialang.org/en/v1/base/strings/#Core.Symbol)s representing the field names:

```julia-repl
julia> struct Point
           x::Int
           y
       end

julia> fieldnames(Point)
(:x, :y)
```

The type of each field in a `Point` object is stored in the `types` field of the `Point` variable itself:

```julia-repl
julia> Point.types
svec(Int64, Any)
```

While `x` is annotated as an `Int`, `y` was unannotated in the type definition, therefore `y` defaults to the `Any` type.

Types are themselves represented as a structure called `DataType`:

```julia-repl
julia> typeof(Point)
DataType
```

Note that `fieldnames(DataType)` gives the names for each field of `DataType` itself, and one of these fields is the `types` field observed in the example above.

## Subtypes

The *direct* subtypes of any `DataType` may be listed using [`subtypes`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.subtypes). For example, the abstract `DataType`[`AbstractFloat`](https://docs.julialang.org/en/v1/base/numbers/#Core.AbstractFloat) has four (concrete) subtypes:

```julia-repl
julia> subtypes(AbstractFloat)
4-element Array{Any,1}:
 BigFloat
 Float16
 Float32
 Float64
```

Any abstract subtype will also be included in this list, but further subtypes thereof will not; recursive application of [`subtypes`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.subtypes) may be used to inspect the full type tree.

## DataType layout

The internal representation of a `DataType` is critically important when interfacing with C code and several functions are available to inspect these details. [`isbits(T::DataType)`](https://docs.julialang.org/en/v1/base/base/#Base.isbits) returns true if `T` is stored with C-compatible alignment. [`fieldoffset(T::DataType, i::Integer)`](https://docs.julialang.org/en/v1/base/base/#Base.fieldoffset) returns the (byte) offset for field *i* relative to the start of the type.

## Function methods

The methods of any generic function may be listed using [`methods`](https://docs.julialang.org/en/v1/base/base/#Base.methods). The method dispatch table may be searched for methods accepting a given type using [`methodswith`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.methodswith).

## Expansion and lowering

As discussed in the [Metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/#Metaprogramming-1) section, the [`macroexpand`](https://docs.julialang.org/en/v1/base/base/#Base.macroexpand) function gives the unquoted and interpolated expression (`Expr`) form for a given macro. To use `macroexpand`, `quote` the expression block itself (otherwise, the macro will be evaluated and the result will be passed instead!). For example:

```julia-repl
julia> macroexpand(@__MODULE__, :(@edit println("")) )
:((InteractiveUtils.edit)(println, (Base.typesof)("")))
```

The functions `Base.Meta.show_sexpr` and [`dump`](https://docs.julialang.org/en/v1/base/io-network/#Base.dump) are used to display S-expr style views and depth-nested detail views for any expression.

Finally, the [`Meta.lower`](https://docs.julialang.org/en/v1/base/base/#Base.Meta.lower) function gives the `lowered` form of any expression and is of particular interest for understanding how language constructs map to primitive operations such as assignments, branches, and calls:

```julia-repl
julia> Meta.lower(@__MODULE__, :([1+2, sin(0.5)]) )
:($(Expr(:thunk, CodeInfo(
 1 ─ %1 = 1 + 2
 │   %2 = sin(0.5)
 │   %3 = (Base.vect)(%1, %2)
 └──      return %3
))))
```







## Layers		

Inspecting the lowered form for functions requires selection of the specific method to display, because generic functions may have many methods with different type signatures. For this purpose, method-specific code-lowering is available using [`code_lowered`](https://docs.julialang.org/en/v1/base/base/#Base.code_lowered), and the type-inferred form is available using [`code_typed`](https://docs.julialang.org/en/v1/base/base/#Base.code_typed). [`code_warntype`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.code_warntype) adds highlighting to the output of [`code_typed`](https://docs.julialang.org/en/v1/base/base/#Base.code_typed).

Closer to the machine, the LLVM intermediate representation of a function may be printed using by [`code_llvm`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.code_llvm), and finally the compiled machine code is available using [`code_native`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.code_native) (this will trigger JIT compilation/code generation for any function which has not previously been called).

For convenience, there are macro versions of the above functions which take standard function calls and expand argument types automatically:

```julia-repl
julia> @code_llvm +(1,1)

; Function Attrs: sspreq
define i64 @"julia_+_130862"(i64, i64) #0 {
top:
    %2 = add i64 %1, %0, !dbg !8
    ret i64 %2, !dbg !8
}
```

See [`@code_lowered`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_lowered), [`@code_typed`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_typed), [`@code_warntype`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_warntype), [`@code_llvm`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_llvm), and [`@code_native`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_native).												

### Layer 1: The AST

When the parser takes your code in (as a `String`), it will produce an AST (Abstract Syntax Tree). The AST is the compiler’s representation of your code. This representation is not saved, so if we want to see it, we’ll need to quote the expression.

```julia
julia> :(2 + 2)
:(2 + 2)
```

### Layer 2: The Lowered AST

```.jl
code_lowered(generic_function, (types_arg_list,))
```

`code_lowered` will return the lowered AST for any method of a generic function. Lowering in general is the process of moving from surface syntax (highest) to machine code (lowest). Here, lowering involves transforming the AST in ways that make it simpler. This includes unnesting some expressions and desugaring some syntax into the function calls indicated.

The lowered AST is stored for every generic function. This will work on methods you write and on ones in packages and on ones from the base libraries. `code_lowered` is a normal generic function: it will work from the [REPL](https://zh.wikipedia.org/zh-hans/%E8%AF%BB%E5%8F%96%EF%B9%A3%E6%B1%82%E5%80%BC%EF%B9%A3%E8%BE%93%E5%87%BA%E5%BE%AA%E7%8E%AF) and from any Julia code you write.

#### Examples

You can call it on one of the simple functions we defined earlier:

```julia
function linear_foo()
  x = 4
  y = 5
end
```

```.jl
code_lowered(linear_foo, ())
1-element Array{Core.CodeInfo,1}:
 CodeInfo(
2 1 ─     x = 4                                                             │
3 │       y = 5                                                             │
  └──     return 5 
```

The value you get back is a one-dimensional array of `Core.CodeInfo` is the type used to represent an expression in the AST; you also use them when writing macros.

#### control flow 

```.jl
function myloop(x::Int)
  result = 0  
  for i=1:x
    result += x
  end
  result
end
```

You can see a loop in the lowered code:

```.jl
julia> code_lowered(myloop, (Int, ))
1-element Array{Core.CodeInfo,1}:
 CodeInfo(
2 1 ─       result = 0                                                      │
3 │   %2  = 1:x                                                             │
  │         #temp# = (Base.iterate)(%2)                                     │
  │   %4  = #temp# === nothing                                              │
  │   %5  = (Base.not_int)(%4)                                              │
  └──       goto #4 if not %5                                               │
  2 ┄ %7  = #temp#                                                          │
  │         i = (Core.getfield)(%7, 1)                                      │
  │   %9  = (Core.getfield)(%7, 2)                                          │
4 │         result = result + x                                             │
  │         #temp# = (Base.iterate)(%2, %9)                                 │
  │   %12 = #temp# === nothing                                              │
  │   %13 = (Base.not_int)(%12)                                             │
  └──       goto #4 if not %13                                              │
  3 ─       goto #2                                                         │
6 4 ─       return result                                                   │
)

```

If you want to see what happens to an if-statment, you could use this example:

```.jl
julia> code_lowered(lessthan5, (Int, ))
1-element Array{Core.CodeInfo,1}:
 CodeInfo(
2 1 ─ %1 = x < 5                                                            │
  └──      goto #3 if not %1                                                │
3 2 ─      return true                                                      │
5 3 ─      return false                                                     │
)

```

You can see that, like the loop, this is also lowered into an `unless` and a `goto`.

```julia
julia> code_lowered(lessthan5,(Int,))
1-element Any Array:
 :($(Expr(:lambda, {:x}, {{},{{:x,:Any,0}},{}}, quote  # none, line 2:
        unless <(x,5) goto 0 # line 3:
        return true
        goto 1
        0:  # none, line 5:
        return false
        1: 
    end)))
```

### Layer 3: The Type-inferred, optimized AST

```julia
code_typed(generic_function, (types_arg_list,))
```

`code_typed` returns the type-inferred and optimized version of the Julia AST. It is the last layer that is internal to Julia.

```Julia
julia> code_typed(lessthan5, (Int, ))
1-element Array{Any,1}:
 CodeInfo(
2 1 ─ %1 = (Base.slt_int)(x, 5)::Bool                                    │╻ <
  └──      goto #3 if not %1                                             │ 
3 2 ─      return true                                                   │ 
5 3 ─      return false                                                  │ 
) => Bool

```

#### Structure of the Return Value

You should be getting an Array of `Expr`s back. It has three fields: `head`, `args`, and `typ`. (You can find this out by calling the function `names` on it.)

- `head` is a `Symbol` that tells you what kind of expression this is. For `Expr`s returned by `code_typed`, this will always be `:lambda`.
- `typ` is a `DataType`. Currently, it will always be `Any` for `Expr`s returned by `code_typed`.
- `args` is a 1-dimensional `Any` Array (`Array{Any,1}`). It’s the interesting part: it contains information about the body of the function and the variables used there.

#### Examples: 0-args, just assigning to local vars

The function:

```julia
function foo()
  x = 4
  y = 5
end
```

The result of `code_typed(foo,())`:

```julia
julia> c1 = code_typed(foo, ())
1-element Array{Any,1}:
 CodeInfo(
3 1 ─     return 5                                                          │
) => Int64

```

### Layer 4: LLVM IR

```.jl
code_llvm(generic_function, (types_arg_list,))
```

Calling `code_llvm` prints the LLVM IR for the function. This is going to be more unfamiliar-looking that the previous layers, since it looks like a complicated kind of assembly code, rather than being Julia-specific. It also differs in that it prints out the code, not returning a maniputable value to you.

#### Usage Examples

```julia
code_llvm(linear_foo, ())

; Function linear_foo
; Location: REPL[7]:2
define i64 @julia_linear_foo_35325() {
top:
  ret i64 5
}
```

### Layer 5: Assembly Code

```.jl
code_native(generic_function, (types_arg_list,))
```

Calling `code_native` prints the native assembly code for the specified method.





## Surface syntax AST

Front end ASTs consist almost entirely of `Expr`s and atoms (e.g. symbols, numbers). There is generally a different expression head for each visually distinct syntactic form. Examples will be given in s-expression syntax. Each parenthesized list corresponds to an Expr, where the first element is the head. For example `(call f x)` corresponds to `Expr(:call, :f, :x)` in Julia.

### Calls

| Input            | AST                                |
| ---------------- | ---------------------------------- |
| `f(x)`           | `(call f x)`                       |
| `f(x, y=1, z=2)` | `(call f x (kw y 1) (kw z 2))`     |
| `f(x; y=1)`      | `(call f (parameters (kw y 1)) x)` |
| `f(x...)`        | `(call f (... x))`                 |

`do` syntax:

```julia
f(x) do a,b
    body
end
```

parses as `(do (call f x) (-> (tuple a b) (block body)))`.

### Operators

Most uses of operators are just function calls, so they are parsed with the head `call`. However some operators are special forms (not necessarily function calls), and in those cases the operator itself is the expression head. In julia-parser.scm these are referred to as "syntactic operators". Some operators (`+` and `*`) use N-ary parsing; chained calls are parsed as a single N-argument call. Finally, chains of comparisons have their own special expression structure.

| Input       | AST                       |
| ----------- | ------------------------- |
| `x+y`       | `(call + x y)`            |
| `a+b+c+d`   | `(call + a b c d)`        |
| `2x`        | `(call * 2 x)`            |
| `a&&b`      | `(&& a b)`                |
| `x += 1`    | `(+= x 1)`                |
| `a ? 1 : 2` | `(if a 1 2)`              |
| `a:b`       | `(: a b)`                 |
| `a:b:c`     | `(: a b c)`               |
| `a,b`       | `(tuple a b)`             |
| `a==b`      | `(call == a b)`           |
| `1<i<=n`    | `(comparison 1 < i <= n)` |
| `a.b`       | `(. a (quote b))`         |
| `a.(b)`     | `(. a b)`                 |

### Bracketed forms

| Input                    | AST                                  |
| ------------------------ | ------------------------------------ |
| `a[i]`                   | `(ref a i)`                          |
| `t[i;j]`                 | `(typed_vcat t i j)`                 |
| `t[i j]`                 | `(typed_hcat t i j)`                 |
| `t[a b; c d]`            | `(typed_vcat t (row a b) (row c d))` |
| `a{b}`                   | `(curly a b)`                        |
| `a{b;c}`                 | `(curly a (parameters c) b)`         |
| `[x]`                    | `(vect x)`                           |
| `[x,y]`                  | `(vect x y)`                         |
| `[x;y]`                  | `(vcat x y)`                         |
| `[x y]`                  | `(hcat x y)`                         |
| `[x y; z t]`             | `(vcat (row x y) (row z t))`         |
| `[x for y in z, a in b]` | `(comprehension x (= y z) (= a b))`  |
| `T[x for y in z]`        | `(typed_comprehension T x (= y z))`  |
| `(a, b, c)`              | `(tuple a b c)`                      |
| `(a; b; c)`              | `(block a (block b c))`              |

### Macros

| Input         | AST                                          |
| ------------- | -------------------------------------------- |
| `@m x y`      | `(macrocall @m (line) x y)`                  |
| `Base.@m x y` | `(macrocall (. Base (quote @m)) (line) x y)` |
| `@Base.m x y` | `(macrocall (. Base (quote @m)) (line) x y)` |

### Strings

| Input      | AST                                 |
| ---------- | ----------------------------------- |
| `"a"`      | `"a"`                               |
| `x"y"`     | `(macrocall @x_str (line) "y")`     |
| `x"y"z`    | `(macrocall @x_str (line) "y" "z")` |
| `"x = $x"` | `(string "x = " x)`                 |
| ``a b c``  | `(macrocall @cmd (line) "a b c")`   |

Doc string syntax:

```julia
"some docs"
f(x) = x
```

parses as `(macrocall (|.| Core '@doc) (line) "some docs" (= (call f x) (block x)))`.

### Imports and such

| Input               | AST                                 |
| ------------------- | ----------------------------------- |
| `import a`          | `(import (. a))`                    |
| `import a.b.c`      | `(import (. a b c))`                |
| `import ...a`       | `(import (. . . . a))`              |
| `import a.b, c.d`   | `(import (. a b) (. c d))`          |
| `import Base: x`    | `(import (: (. Base) (. x)))`       |
| `import Base: x, y` | `(import (: (. Base) (. x) (. y)))` |
| `export a, b`       | `(export a b)`                      |

### Numbers

Julia supports more number types than many scheme implementations, so not all numbers are represented directly as scheme numbers in the AST.

| Input                   | AST                                                     |
| ----------------------- | ------------------------------------------------------- |
| `11111111111111111111`  | `(macrocall @int128_str (null) "11111111111111111111")` |
| `0xfffffffffffffffff`   | `(macrocall @uint128_str (null) "0xfffffffffffffffff")` |
| `1111...many digits...` | `(macrocall @big_str (null) "1111....")`                |

### Block forms

A block of statements is parsed as `(block stmt1 stmt2 ...)`.

If statement:

```julia
if a
    b
elseif c
    d
else
    e
end
```

parses as:

```none
(if a (block (line 2) b)
    (elseif (block (line 3) c) (block (line 4) d)
            (block (line 5 e))))
```

A `while` loop parses as `(while condition body)`.

A `for` loop parses as `(for (= var iter) body)`. If there is more than one iteration specification, they are parsed as a block: `(for (block (= v1 iter1) (= v2 iter2)) body)`.

`break` and `continue` are parsed as 0-argument expressions `(break)` and `(continue)`.

`let` is parsed as `(let (= var val) body)` or `(let (block (= var1 val1) (= var2 val2) ...) body)`, like `for` loops.

A basic function definition is parsed as `(function (call f x) body)`. A more complex example:

```julia
function f(x::T; k = 1) where T
    return x+1
end
```

parses as:

```none
(function (where (call f (parameters (kw k 1))
                       (:: x T))
                 T)
          (block (line 2) (return (call + x 1))))
```

Type definition:

```julia
mutable struct Foo{T<:S}
    x::T
end
```

parses as:

```none
(struct true (curly Foo (<: T S))
        (block (line 2) (:: x T)))
```

The first argument is a boolean telling whether the type is mutable.

`try` blocks parse as `(try try_block var catch_block finally_block)`. If no variable is present after `catch`, `var` is `#f`. If there is no `finally` clause, then the last argument is not present.



### Line numbers

Source location information is represented as `(line line_num file_name)` where the third component is optional (and omitted when the current line number, but not file name, changes).

These expressions are represented as `LineNumberNode`s in Julia.

### Macros

Macro hygiene is represented through the expression head pair `escape` and `hygienic-scope`. The result of a macro expansion is automatically wrapped in `(hygienic-scope block module)`, to represent the result of the new scope. The user can insert `(escape block)` inside to interpolate code from the caller.