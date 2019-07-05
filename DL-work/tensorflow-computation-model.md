# TensorFlow 的计算模型

参考  A computational model for TensorFlow: an introduction, from google <https://ai.google/research/pubs/pub46196>

本文介绍 Tensorflow(TF) 的计算模型, 及其操作语义. Tensorflow 有使用 [TLA 语
言, Lamport,2002](https://lamport.azurewebsites.net/tla/book-02-08-08.pdf)(描述
的逻辑规范, 本文是对其的解释.

TensorFlow 的理论基础是 dataflow system. 在很早就已经有相关的理论研究了, 比如
The semantics of a simple language for parallel programming.North Holland,
Amsterdam, 1974.

TensorFlow 将计算用 dataflow graphs(数据流图)表示. 尽管被用作机器学习的框架,  该
模型不关心上层应用具体是什么.

使用TensorFlow 时,  client 一般使用前端语言, 比如 python 构建计算图,  client 代
码指定图的输入输出, 并调用这个图, TensorFlow 根据输入的值不断执行图上的操作
(operation), 知道 no more nodes can fire.

TensorFlow 计算结果和计算顺序可能有关, 所以计算结果可能不唯一, 参考
[例子](https://blog.csdn.net/LoseInVain/article/details/78780020).  这样的设计可
以允许 TF 优化计算顺序和并行化计算.

## 数据流图模型

数据流图包含 **nodes** (节点)和 **edges** (边).  nodes: an instantiation of an
**operation**, **value**s flow along the edges.   operation 使用 "kernel" 实现,
可以在 CPU 或 GPU 上运行.

**value** 一般是 tensor, 即多维数组.

变量(**variable**) 是一种特殊的 operation.  执行后返回  **handle** to a tensor, 我们可以看做 tensor 保存在变量中.

除了传递 tensor 的边, 还有一种特殊的边: 控制边(**control edge**) , 用来控制图的执行顺序. 执行顺序的不同会导致在存在可变状态的计机图产生不同的可观察到的结果.


值可以是tensors , 变量(即tensor 的handle) , GO 一个触发常量, EMPTY 表示没有产生,
或者已经消费掉的值.

    value = tensors | vars (handles of tensors) | GO | EMPTY

三种边:

    edge = tensor edge | variable edge | control edge(用于 GO)

操作:

    operation = tensor 函数 | Var(x) | Read | Assign-f

tensor 函数是一个输入输出都是 tensor(s) 的函数,  Var(x) 在 x是变量是输出 x .  Read 读取一个变量 x 的值.  Assign-f 中 f 是一个类型为 (Tensor x Tensor) $\to​$ Tensor 的函数, Assign-f 输入一个变量 x 和一个和一个 tensor v,  它读取变量 x 的当前值, 和 v 一起输入函数 f, 将 f 输出的值存到变量 x 中.  可以用 Assign-f 实现写变量的功能.  这里显然要比单独写一个变量要强, 个人认为这样是为了将计算和写变量做为一个原子操作.

一个程序(图)的状态包括所有变量的当前的值, 所有边的当前值.  程序(图)的初始状态为: 所有输入边非 EMPTY, 其他边为 EMPTY. 每个变量有初始值.

程序的执行即节点的执行(firing) , 当一个节点的所有输入边准备好时则 firing. 图状态
发生一步变化, 消费输入边的值, 产生输出边的值.

一般一个图在执行多次时,  "mutable state"(比如变量) 中的 tensor 会保留, 中间计算结果 tensor 会丢弃. 在一个典型应用中, 一个图表示训练机器学习模型的一步.  模型参数保存在变量中.
