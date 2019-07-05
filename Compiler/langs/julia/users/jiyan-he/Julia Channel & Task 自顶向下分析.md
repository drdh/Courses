# Julia Channel & Task 自顶向下分析

何纪言

## 简介

本文是对 Julia 中Channel 和 Task 实现的技术细节的一个自顶向下分析，并解释了 Julia 事件通知模型在不同操作系统的底层技术实现，下面大致分为几个层次分析，每一层的实现几乎都是依赖于下一层提供的模型：

- Channel (base/channel.jl)
- Event (base/event.jl)
- libuv.jl (base/libuv.jl)
- jl_uv.c (src/jl_uv.c)
- libuv (extern libraray)

此外附录中有对 libuv 的一些简单介绍和例子，最后附有参考资料。

## Channel (base/channel.jl)

首先是 Julia 中 Channel 提供的主要接口有 `put!()` 和 `take!()`。

忽略各种 multi-dispatch 和一些特殊情况，这两个函数的主要逻辑是调用 `wait` 和 `notify` 两个接口实现的：

```julia
# 生产者调用
function put_buffered(c::Channel, v)
    # Chanel 满了，阻塞，直到有可写的位置
    while length(c.data) == c.sz_max
        wait(c.cond_put)
    end
    # 添加一个数据
    push!(c.data, v)

    # notify all, since some of the waiters may be on a "fetch" call.
    # 通知消费者，可以拿数据了
    notify(c.cond_take, nothing, true, false)
    v
end

# 消费者调用
function take_buffered(c::Channel)
    # 阻塞，直到有数据可读
    wait(c)
    
    # 注意！这个 wait 不是按条件 wait
    
    # 取出一个数据
    v = popfirst!(c.data)
    # 通知想放置数据的生产者，有位置可写了
    notify(c.cond_put, nothing, false, false) # notify only one, since only one slot has become available for a put!.
    v
end


```

值得一提的是，`take` 中使用的 `wait` 并不是 `event` 标准库中提供的 `wait`，而是：

```julia
wait(c::Channel) = isbuffered(c) ? wait_impl(c) : wait_unbuffered(c)
function wait_impl(c::Channel)
    while !isready(c)
        check_channel_state(c) # 检验 c.state 状态
        wait(c.cond_take) # 等生产者放数据
    end
    nothing
end
```

总结：

这一层依靠 Event 提供的事件阻塞和通知模型，提供了一个多任务可读可写的 Channel。

## Event (base/event.jl)

由源代码可以看出， `wait` 和 `notify` 两个接口为我们提供了一个阻塞和唤醒的模型。

其中：

`notify` 这个函数的作用是唤醒一些等待某些条件的 Task。

```julia
notify(condition, val=nothing; all=true, error=false)

function notify(c::Condition, arg, all, error)
    cnt = 0
    if all
        # 通知排队队列中的所有
        cnt = length(c.waitq)
        for t in c.waitq
            # 安排任务执行
            error ? schedule(t, arg, error=error) : schedule(t, arg)
        end
        # 清空排队队列
        empty!(c.waitq)
    elseif !isempty(c.waitq)
        # 只通知排队队首的那一个
        cnt = 1
        t = popfirst!(c.waitq)
        # 安排这个任务执行
        error ? schedule(t, arg, error=error) : schedule(t, arg)
    end
    cnt
end
```

`wait` 这个函数的作用是阻塞某些 Task，直到某些事件发生。

```julia
wait([x])

function wait(c::Condition)
    ct = current_task()
    
    # 将当前任务加入排队队列
    push!(c.waitq, ct)

    try
        # 等待
        return wait() # 注意这里不是调用自身，是下面的函数
    catch
        filter!(x->x!==ct, c.waitq)
        rethrow()
    end
end

function wait()
    while true
        if isempty(Workqueue)
            # 工作队列是空的
            c = process_events(true)
            if c == 0 && eventloop() != C_NULL && isempty(Workqueue)
                # 没有活跃的 handles，等待信号
                pause()
                # 注意：pause() = ccall(:pause, Cvoid, ())
            end
        else
            # 取得一个任务
            reftask = poptask()
            if reftask !== nothing
                result = try_yieldto(ensure_rescheduled, reftask)
                process_events(false)
                # return when we come out of the queue
                return result
            end
        end
    end
    # unreachable
end
```

这里的 `schedule` 本质上是包装 `eventloop` 的一个函数：

```julia
schedule(t::Task) = enq_work(t)
function enq_work(t::Task)
    t.state == :runnable || error("schedule: Task not runnable")
    ccall(:uv_stop, Cvoid, (Ptr{Cvoid},), eventloop())
    # 将任务放入工作队列，并且更改更改任务的状态：正在排队
    push!(Workqueue, t)
    t.state = :queued
    return t
end
```

`schedule` 实现的功能是：将任务添加到工作队列异步执行（排队），不阻塞，实现所谓的并发，即“计划执行”的意思，并不阻塞住立刻生效。



总结：这一层利用 `uv_stop` 、 `process_events` 以及 `eventloop`  实现了任务的事件阻塞和通知模型。

（PS. 这里 `uv_stop` 跨层调用了，我的看法是应该在 `libuv.jl` 中添加一层封装更优雅。）

# libuv.jl (base/libuv.jl)

`eventloop` 这个函数只是对外部函数调用的一个简单封装： 

```julia
eventloop() = uv_eventloop::Ptr{Cvoid}

global uv_eventloop = ccall(:jl_global_event_loop, Ptr{Cvoid}, ())
```
`process_events` 同理：

```julia
function process_events(block::Bool)
    loop = eventloop()
    if block
        return ccall(:jl_run_once,Int32,(Ptr{Cvoid},),loop)
    else
        return ccall(:jl_process_events,Int32,(Ptr{Cvoid},),loop)
    end
end
```

总结：这一层作为 Julia 内部（语言层面的函数）和外部（libuv）的连接，对 libuv 的接口进行了封装。

## jl_uv.c (src/jl_uv.c)

在这一层中我们关注 libuv 究竟为 Julia 提供了怎样的接口：

首先 `jl_global_event_loop` 是一个返回 `jl_io_loop` 的函数：

```c
JL_DLLEXPORT uv_loop_t *jl_global_event_loop(void)
{
    return jl_io_loop;
}
```

这里的 `jl_io_loop` 其实就是 `uv_default_loop()` （见：https://github.com/JuliaLang/julia/blob/b55b85cd9952ce39246d1bcb6c4b0417ac457531/src/jl_uv.c#L391 ），是 libuv 提供的默认 loop。



这个函数是 `process_events(block=false)` 时实际调用的函数，使用的 `uv_run` 参数是 `UV_RUN_NOWAIT`。

```c
JL_DLLEXPORT int jl_process_events(uv_loop_t *loop)
{
    jl_ptls_t ptls = jl_get_ptls_states(); // 还没太搞懂这是什么，和 LLVM, GC 有关
    if (loop) {
        loop->stop_flag = 0;
        jl_gc_safepoint_(ptls);
        // 调用 uv_run
        return uv_run(loop,UV_RUN_NOWAIT); // UV_RUN_NOWAIT
    }
    else return 0;
}
```

这个函数是 `process_events(block=true)` 时实际调用的函数，使用的 `uv_run` 参数是 `UV_RUN_ONCE`。
```c
JL_DLLEXPORT int jl_run_once(uv_loop_t *loop)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    if (loop) {
        loop->stop_flag = 0;
        jl_gc_safepoint_(ptls);
        // 调用 uv_run
        return uv_run(loop,UV_RUN_ONCE); // UV_RUN_ONCE
    }
    else return 0;
}
```

总结：这一层主要是调用外部库 libuv 中的 `uv_run`，当然别忘了还有在 `schedule` 中调用的 `uv_stop` 。

## libuv (extern libraray)

libuv 采用事件循环的方式来完成各种异步操作，例如发射一个 I/O 请求，然后等待返回数据这一行为，就很适合异步完成，在数据返回前可以去做别的任务，程序由阻塞变为非阻塞会大大提供效率。

libuv 在不同功能操作系统上使用不同的高并发异步模型：

- Linux: epoll

- FreeBSD: kqueue

- Windows: iocp

libuv 可以用于很多地方，Julia 使用的是 `uv_async_init` 这种 handle（见附录）， 实现线程间各种信号的通信。

## Appendix A: libuv 基础

libuv 强制使用异步的，事件驱动的编程风格。它的核心工作是提供一个 event-loop，还有基于 I/O 和其它事件通知的回调函数。libuv 还提供了一些核心工具，例如定时器，非阻塞的网络支持，异步文件系统访问，子进程等。

libuv 的工作建立在用户表达对特定事件的兴趣。这通常通过创造对应 I/O 设备，定时器，进程等的 handle 来实现。handle 是不透明的数据结构，其中对应的类型 `uv_TYPE_t` 中的type指定了 handle 的使用目的。

(Julia 使用的是 `uv_async_init`)

libuv 的一些简单例子：

1. 什么都不做例子

```c
int main() {
    # 创建一个循环并且初始化
    uv_loop_t *loop = malloc(sizeof(uv_loop_t));
    uv_loop_init(loop);
    
    # 循环是空的，所以一下子就跑完了
    uv_run(loop, UV_RUN_DEFAULT);
    
    # 关闭循环
    uv_loop_close(loop);
    free(loop);
    return 0;
}
```

2. 空转 handle 例子

```c
int64_t counter = 0;

void wait_for_a_while(uv_idle_t* handle) {
    counter++;

    if (counter >= 10e6)
        uv_idle_stop(handle);
}

int main() {
    uv_idle_t idler; // uv_idle_t 是 Handle 的一种类型

    uv_idle_init(uv_default_loop(), &idler);
    uv_idle_start(&idler, wait_for_a_while);

    printf("Idling...\n");
    uv_run(uv_default_loop(), UV_RUN_DEFAULT);

    uv_loop_close(uv_default_loop());
    return 0;
}
```

这个例子中，程序开始运行后几乎立刻运行到 `uv_run` 处，然后直到 `counter` 增加到足够大前都阻塞在这里，一旦 `counter`  足够大，这个 handle 停止，没有其他 handle 存活，程序随之退出。

## Appendix B: libuv Reference

```
 * uv_async_t is a subclass of uv_handle_t.
 *
 * uv_async_send wakes up the event loop and calls the async handle's callback.
 * There is no guarantee that every uv_async_send call leads to exactly one
 * invocation of the callback; the only guarantee is that the callback function
 * is called at least once after the call to async_send. Unlike all other
 * libuv functions, uv_async_send can be called from another thread.
```

- `uv_run(uv_loop_t*)`
  - UV_RUN_DEFAULT: Runs the event loop until there are no more active and referenced handles or requests. Returns non-zero if [`uv_stop()`](http://docs.libuv.org/en/v1.x/loop.html#c.uv_stop) was called and there are still active handles or requests. Returns zero in all other cases.
  - UV_RUN_ONCE: Poll for i/o once. Note that this function blocks if there are no pending callbacks. Returns zero when done (no active handles or requests left), or non-zero if more callbacks are expected (meaning you should run the event loop again sometime in the future).
  - UV_RUN_NOWAIT: Poll for i/o once but don’t block if there are no pending callbacks. Returns zero if done (no active handles or requests left), or non-zero if more callbacks are expected (meaning you should run the event loop again sometime in the future).
- `uv_stop(uv_loop_t*)`
  - Stop the event loop, causing [`uv_run()`](http://docs.libuv.org/en/v1.x/loop.html#c.uv_run) to end as soon as possible. This will happen not sooner than the next loop iteration. If this function was called before blocking for i/o, the loop won’t block for i/o on this iteration.

## Ref

http://luohaha.github.io/Chinese-uvbook/source/basics_of_libuv.html