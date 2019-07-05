# Parallel Computing

## Overview

分为三个等级的并行状态：

1. Julia Coroutines (Green Threading)
2. Multi-Threading
3. Multi-Core or Distributed Processing



## Corountines

###[Tasks (aka Coroutines)](https://docs.julialang.org/en/v1/manual/control-flow/#man-tasks-1)

Tasks 是一个控制流特性，可以允许任务的推迟和恢复，也被称为 cooperative multitasking。在一个任务中可以随时进行暂停切换至另一个任务，看起来类似于函数调用，但是有以下两点关键区别：

1. 切换任务不需要占用空间，无论切换多少任务都不会消耗调用堆栈的空间
2. 可以以任何顺序进行任务的切换，无需在任务结束后返回至调用的任务处

### Channel

提供了 Channel 结构进行多任务并行的实现，Channel 是一个先进先出的队列，类似于 socket 的缓冲区。最简单的例子即为生产者消费者问题

一些基本操作：

- `take!(c)` ：出队

- `put!(c, item)` ：入队

- `fetch(c)` ：读取队头数据但是不弹出

- `Channel{Type}(sz)` ：定义一个 Channel ，类型为 Type，缓冲区大小为 sz
- `@async` ：非阻塞宏

示例程序 `multitask.jl` 



## Multi-Threading (Experimental)

仍在实验中，未来接口可能会有改变

### Setup

通过环境变量规定启用线程数：

```shell
export JULIA_NUM_THREADS=4
```

可在命令行窗口中确认总线程数与当前线程 id：

```
julia> Threads.nthreads()
4
julia> Threads.threadid()
1
```

### @threads

通过 `Threads.@threads` 宏可以使 for 循环由多个线程进行操作：

```julia
julia> Threads.@threads for i = 1:10
		   a[i] = Threads.threadid()
	   end
```

结果如下：

```
julia> a
10-element Array{Float64,1}:
 1.0
 1.0
 1.0
 2.0
 2.0
 2.0
 3.0
 3.0
 4.0
 4.0
```

### Atomic Operation

Juia 提供原子操作避免出现 race condition，如 `Threads.atomic_add!()` 、`Threads.atomic_sub!()` 等，例如：

```julia
julia> using Base.Threads

julia> nthreads()
4

julia> acc = Ref(0)
Base.RefValue{Int64}(0)

julia> @threads for i in 1:1000
          acc[] += 1
       end

julia> acc[]
926

julia> acc = Atomic{Int64}(0)
Atomic{Int64}(0)

julia> @threads for i in 1:1000
          atomic_add!(acc, 1)
       end

julia> acc[]
1000
```

###[@threadcall (Experimental)](https://docs.julialang.org/en/v1/manual/parallel-computing/index.html#@threadcall-(Experimental)-1)



## Multi-Core or Distributed Processing

Julia 提供了多进程环境，可以使多个进程拥有独立分隔的内存。

Julia 分布式编程主要由以下两个原语构成：*remote references* 和 *remote calls* 。remote reference 是可以由任何进程使用的来引用存储在特定进程上的对象的对象。remote call 是由一个进程向另外一个进程发出特定的函数调用请求。

Remote references 有两种主要形式： [`Future`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.Future) 和 [`RemoteChannel`](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.RemoteChannel)

可以由 `julia -p N/auto` 启动多个进程，在主进程中指定 id 进行任务调度，如 `@spawnat id` 、`remotecall(f, id, args...)` ，其返回值为 `Future` 类型，通过 `fetch` 得到结果，若计算结果仍未得到则会等待至产生结果。例如：

```julia
$ ./julia -p 2

julia> r = remotecall(rand, 2, 2, 2)
Future(2, 1, 4, nothing)

julia> s = @spawnat 2 1 .+ fetch(r)
Future(2, 1, 5, nothing)

julia> fetch(s)
2×2 Array{Float64,2}:
 1.18526  1.50912
 1.16296  1.60607
```

`@spawn` 也可以进行远程执行，并且会自动分配进程执行 

```julia
julia> r = @spawn rand(2,2)
Future(2, 1, 4, nothing)

julia> s = @spawn 1 .+ fetch(r)
Future(3, 1, 5, nothing)

julia> fetch(s)
2×2 Array{Float64,2}:
 1.38854  1.9098
 1.20939  1.57158
```

定义函数和调用模块前要加 `@everywhere` 使其全局有效，如`@everywhere foo()` 、 `@everywhere include("MyModule.jl")` 



**后续待施工**



## Lacks

1. 多线程由 `Base.Threads` 模块进行支持，因为 Julia 仍不是完全的线程安全的，所以多线程仍在实验中。
2. 在 I\O 操作或任务切换过程中可能会出现特定的 segfaults。
3. 对于Parallel Computing 仍有很多问题： [issues](https://github.com/JuliaLang/julia/issues?q=is%3Aopen+is%3Aissue+label%3Amultithreading)
4. Future versions of Julia may support scheduling of tasks on multiple threads

## Packages

`MPI.jl` and `DistributedArrays.jl`

