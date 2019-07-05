# To do

###11.8 

并发：

1. 看 Task 源码了解实现机制 src 目录 task.c 、base 目录 channels.jl & task.jl
2. ray 中现有的并发使用 Julia 如何表达，表达能力如何，是否需要对一些表达进行加强？
3. 如数据共享，数据通信，其中 Julia 能提供什么，缺少的是什么？

ray：

1. 以 ray 为主调用 Julia 或以 Julia 为主表达 ray，能否像 ray 一样把现有成熟的模型整合进来调度？
2. ray 和 Julia 进行对接，ray 的 model 中调用 Julia 的 flux.jl。

RL：

1. 强化学习可以利用哪些 Julia 的动态交互特性，相同模型不同语言的 benchmark

在小的方向上做解决方案，写出一些小的例子