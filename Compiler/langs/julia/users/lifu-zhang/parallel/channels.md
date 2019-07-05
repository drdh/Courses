# channels & task

## channels.jl

抽象类型

```julia
abstract type AbstractChannel{T} end
```

Channel 结构体定义

```julia
mutable struct Channel{T} <: AbstractChannel{T}
    cond_take::Condition                 # waiting for data to become available
    cond_put::Condition                  # waiting for a writeable slot
    state::Symbol
    excp::Union{Exception, Nothing}         # exception to be thrown when state != :open

    data::Vector{T}
    sz_max::Int                          # maximum size of channel

    # Used when sz_max == 0, i.e., an unbuffered channel.
    waiters::Int
    takers::Vector{Task}
    putters::Vector{Task}

    function Channel{T}(sz::Float64) where T
        if sz == Inf
            Channel{T}(typemax(Int))
        else
            Channel{T}(convert(Int, sz))
        end
    end
    function Channel{T}(sz::Integer) where T
        if sz < 0
            throw(ArgumentError("Channel size must be either 0, a positive integer or Inf"))
        end
        ch = new(Condition(), Condition(), :open, nothing, Vector{T}(), sz, 0)
        if sz == 0
            ch.takers = Vector{Task}()
            ch.putters = Vector{Task}()
        end
        return ch
    end
end
```

####成员变量

1. `cond_take::Condition` & `cond_put::Condition` ：对于从 Channel 中读写数据的等待信号；
2. `state::Symbol` ：标示当前 Channel 状态，`:open` & `:close` ；
3. `excp::Union` ：标示异常；
4. `data::Vector{T}` ：数据内容；
5. `sz_max::Int` ：Channel 大小；
6. `waiters::Int` ：等待操作的个数；
7. `takers::Vector{Task}` ：等待从 Channel 中获取数据的 Tasks；
8. `putters::Vector{Task}` ：等待向 Channel 中写入数据的 Tasks。



#### 特殊构造函数

```julia
function Channel(func::Function; ctype=Any, csize=0, taskref=nothing)
```

通过一个调用，直接将一个新的 Task 即 func 与一个新建的 Channel 进行关联，并调度 func。

```julia
function Channel(func::Function; ctype=Any, csize=0, taskref=nothing)
    chnl = Channel{ctype}(csize)
    task = Task(() -> func(chnl))
    bind(chnl, task)
    yield(task) # immediately start it

    isa(taskref, Ref{Task}) && (taskref[] = task)
    return chnl
end
```



#### 其他相关函数

**`put!(c::Channel, v)`**

向 Channel 中放入一个数据 v，首先检查 Channel 是否为开启状态，之后判断该 Channel 是否具有 buffer：

```julia
isbuffered(c) ? put_buffered(c,v) : put_unbuffered(c,v)
```

对于 `put_buffered(c, v)` ：

```julia
function put_buffered(c::Channel, v)
    while length(c.data) == c.sz_max
        wait(c.cond_put)	# wait for notify on a condition
    end
    push!(c.data, v)

    # notify all, since some of the waiters may be on a "fetch" call.
    notify(c.cond_take, nothing, true, false)
    v
end
```

其中：`notify(condition, val=nothing; all=true, error=false)` 

对于 `put_unbuffered(c, v)` ：

```julia
function put_unbuffered(c::Channel, v)
    if length(c.takers) == 0
        push!(c.putters, current_task())
        c.waiters > 0 && notify(c.cond_take, nothing, false, false)

        try
            wait()
        catch ex
            filter!(x->x!=current_task(), c.putters)
            rethrow(ex)
        end
    end
    taker = popfirst!(c.takers)
    yield(taker, v) # immediately give taker a chance to run, but don't block the current task
    return v
end
```

如果没有在等待的 `c.takers` 将当前任务放入 `c.putters` 并等待，捕获到 ex 会将当前任务弹出，更新 `c.putters` 。



**`take!(c::Channel)`**

和 put! 同样先进行 Channel 的类型判断，然后执行对应的 take 操作：

```julia
take!(c::Channel) = isbuffered(c) ? take_buffered(c) : take_unbuffered(c)
function take_buffered(c::Channel)
    wait(c)
    v = popfirst!(c.data)
    notify(c.cond_put, nothing, false, false) # notify only one, since only one slot has become available for a put!.
    v
end

popfirst!(c::Channel) = take!(c)

# 0-size channel
function take_unbuffered(c::Channel{T}) where T
    check_channel_state(c)
    push!(c.takers, current_task())
    try
        if length(c.putters) > 0
            let refputter = Ref(popfirst!(c.putters))
                return Base.try_yieldto(refputter) do putter
                    # if we fail to start putter, put it back in the queue
                    putter === current_task || pushfirst!(c.putters, putter)
                end::T
            end
        else
            return wait()::T
        end
    catch ex
        filter!(x->x!=current_task(), c.takers)
        rethrow(ex)
    end
end
```



**`fetch!(c::Channel)`**

fetch 只支持对具有 buffer 的 Channel 操作：

```julia
function fetch_buffered(c::Channel)
    wait(c)
    c.data[1]
end
```



**`close(c::Channel)` **

关闭一个 Channel。



**`isopen(c::Channel) = (c.state == :open)`**



**`bind(chnl::Channel, task::Task)` **

将一个 Channel 和一个 task 关联起来，当 task 完成时 Channel 会自动关闭。一个 task 可以关联多个 Channel，多个 task 也可以关联一个 Channel，当第一个 task 终结时会关闭 Channel。

```julia
julia> c = Channel(0);

julia> task = @async foreach(i->put!(c, i), 1:4);

julia> bind(c,task);

julia> for i in c
           @show i
       end;
i = 1
i = 2
i = 3
i = 4

julia> isopen(c)
false
```



**`channeled_tasks(n::Int, funcs...; ctypes=fill(Any,n), csizes=fill(0,n))`**

一次调用创建 n 个 Channel，和 length(funcs) 个 task，并将每一个 task 和每一个 Channel 进行关联，之后进行调度。

```julia
foreach(t -> foreach(c -> bind(c, t), chnls), tasks)
foreach(schedule, tasks)
yield()
```


