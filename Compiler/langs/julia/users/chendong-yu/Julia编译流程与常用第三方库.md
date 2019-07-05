# Julia编译流程与常用第三方库
[TOC]


## 代码执行过程
- 用户启动Julia
- 调用`ui/repl.c`中的`main()`函数，处理命令行参数，初始化`Julia`，将控制权转移给`Base._start()`
- `_start`如果提供文件名，则执行文件，否则启动交互式`REPL`
- 如果要运行的代码块在文件中，则调用`jl_load(char *filename)`加载文件并解析它。然后传递每个代码片段给`eval`执行
- 代码（或AST）的每个片段都被移交给eval()执行并得到结果
- `jl_toplevel_eval_flex()`决定代码是否是“顶级”动作（例如`using`或`module`），这在动作中是无效的。如果是这样，它会将代码传递给顶层解释器
- `jl_toplevel_eval_flex()`然后扩展代码消除宏并“降低”AST以使其更容易执行
- `jl_toplevel_eval_flex() `使用一些简单的启发式方法来决定是对`JIT`编译`AST`还是直接解释它
- 解释代码的大部分工作由`evalininterpreter.c`处理
- 编译代码的工作由`codegen.cpp`处理，第一次调用Julia函数时，对其进行类型推断生成更快的代码
- 退出REPL或者执行到程序末尾，`_start()`返回
- 退出前会对清理所有的`libuv handler`

##编译过程
Parser——>扩展宏——>类型推断——>JIT代码生成

### Parser
`julia-parser.scm`处理Julia代码标记并转换成AST，并使用`julia-syntax.scm`处理将复杂AST表示转换为更简单的AST表示，更适合于分析和执行。

#### 两种AST
`Julia`的代码表示分为两种，第一种是`Surface Syntax AST`，是代码的结构化表示；第二种是`IR`（中间表示），用于类型推断和代码生成
- `Surface Syntax AST`：由Exprs和原子组成（例如符号，数字），比如：`(call f x)`对应`Expr(:call, :f, :x)`
| 输入| AST|
|:--:|:--:|
|`f(x)`|	`(call f x)`|
|`f(x, y=1, z=2)`|	`(call f x (kw y 1) (kw z 2))`|
|`f(x; y=1)`	|`(call f (parameters (kw y 1)) x)`|
|`f(x...)`|	`(call f (... x))`|
- `IR`：节点类型少，扩展了宏，并且将控制流转化为显示分支和语句序列

### 扩展宏
当`eval()`处理到宏的时候，会尝试扩展对应的AST节点

### 类型推断
在`conpiler/typeinfer.jl`中实现类型推断，检查Julia函数并确定其每个变量类型的边界的过程，以及函数返回值类型的界限，可用于进一步优化函数，具体可以看`《Julia： A Fast Dynamic Language for Technical Computing》`


### JIT代码生成
`Codegen`将`Julia AST`转化为本机机器代码

具体过程：
`Julia JIT`编译器是一个名为`libLLVM`的库。`Julia`中的`Codegen`指`Julia AST`转换为`LLVM指令`的过程，以及`LLVM`优化并将其转换为本机汇编指令

使用`jl_init_codegen in codegen.cpp`初始化JIT环境
通过函数`emit_function(jl_method_instance_t*)`将Julia方法转化为一个函数

其余的帮助文件：
- `debuginfo.cpp`
处理JIT函数的回溯
- `ccall.cpp`
处理`ccall`和`llvmcall FFI`以及各种`abi_*.cpp`文件
- `intrinsics.cpp`
处理底层函数调用

### 系统映像
系统映像是Julia文件的预编译存档，通过执行`sys.jl`生成`sysimg.jl`，并将生成的环境（包括类型，函数，模块和所有其他定义的值）序列化到文件中，作为未来Julia运行的起始点


## 常用第三方库
### 文档与注释
- `Documenter.jl`
  用于将`Julia`代码中的注释系统及其中的`Markdown`文件生成完整的说明文档

### 科学计算与数据处理
- `Distributions.jl`：为概率分布计算提供支持，还支持多元统计分布、混合模型、假设检验及分布拟合等计算
- `Distances.jl`：提供了各种相似性度量函数的支持

- `DataFrames.jl`：一个专门用于处理行列式或表格式（tabular）的库
- `CSV.jl`：专为`CSV`文件的读写提供支持的库
- `JSON.jl`：提供对`JSON`格式数据的处理
- `Taro.jl`：用于从`Word`、`Excel`或`PDF`文件中提取内容的库

### 图形可视化
- `Gadfly.jl`：绘图与可视化库，图形渲染质量高，直管一致的接口设计，支持大量绘图类型
- `ECharts.jl`：基于Javascript的数据可视化图表库，提供直观，生动，可交互，可个性化定制的数据可视化图表
- `VegaLite.jl`：一种高级可视化语法。它支持简洁的 `JSON` 语法，用于支持快速生成可视化以支持分析。`Vega-Lite` 支持交互式多视图图形，可以编译成 Vega
- `QML.jl`：用于`GUI`编程，通过`CxxWrap.jl`包提供了`Qt5 QML`的编程支持

### 数据库操作
- `JDBC.jl`：基于`JavaCall.jl`包的功能，通过`Java`的接口访问`JDBC`驱动。提供的`API`包括两种核心组件，一种是访问`JDBC`的直接接口，另一种是支持`DataStreams.jl`的`Julia`接口

### 机器学习与深度学习
- `TensorFlow.jl`：是对`TensorFlow`库的封装
- `Mocha.jl`：一个高效的深度学习框架，包含了通用的随机梯度求解器，可以它构建层训练深、浅(卷积）网络。
- `Flux.jl`：一个机器学习工具包，可以实现各种基本模型（如线性回归）到复杂模型（如神经网络）的搭建、优化和使用。 
- `MXNet.jl`：提供`MXNet`相关支持
- `ScikitLearn.jl`：`ScikitLearn`的`Julia`封装
- `SVM.jl`：支持向量机库
- `KNN.jl`：对机器学习中的`KNN`算法提供支持

其余的还有`MachineLearning.jl`、`DecisionTree.jl`、`Boltzmann.jl`、`BayesNets.jl`等