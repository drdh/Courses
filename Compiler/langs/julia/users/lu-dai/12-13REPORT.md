# JULIA REPORT





## Fashionable Modeling with Flux

There are three pillars that set Flux apart among ML systems: simplicity, hackability, and underlying compiler technology.

有两个支柱将Flux与ML系统区分开来：简单性，可破解性和底层编译器技术。



简单性，可破解性在之前的讨论中已经介绍过。这里主要介绍编译器技术和机器学习中计算图的结合。

由于机器学习模型的复杂性和性能要求越来越高，研究将越来越多地受到语言和编译器技术功能的支持或限制。

本文认为，通过使用SSA形式的IR，以极低的运行时开销来进行微分，同时为将来的更多优化开辟了机会。由于SSA被许多编译器用作中间表示，因此可以将微分作为许多现代编译语言的第一类语言特性添加，从而实现真正的可微分编程。

> 以下解释来源于机器之心：

基于”JULIA的语法就是计算图”的想法，论文构建了 Zygote，它直接在 SSA 形式的中间表征（IR）上工作，支持控制流、递归、数据结构和宏等语言功能。然后可以通过 LLVM 之类的编译器生成 SSA 形式的伴随代码，并将传统编译器优化的所有优势应用于前向和后向传播。此外，这种方法还为扩展该编译器基础结构提供了可能，可以使用更高级和特定领域的优化，例如用于 TPU 等加速器的内核融合和编译。




## Julia IR 与 GPU code之间的交互



## 传统方案的问题

![1544689762251](C:\Users\dldai\AppData\Roaming\Typora\typora-user-images\1544689762251.png)

传统方案通过生成通用IR,再从头生成GPU兼容的IR, 因此代码的复用性很小。

此外，处理**某些问题**仅仅使用IR是不够的，还需要对源代码，AST, 机器代码的接口。



比如：

在为不支持Julia运行时库的环境编译代码时，编译器需要避免对julia runtime库进行调用。 比如exception，其依赖runtime进行堆栈展开，错误报告。 在主要的Julia编译器中，这些对运行时的调用是作为代码生成过程的一部分生成的。 要生成不需要运行时库而不改变代码生成过程的代码，编译器需要从IR中删除对运行时的调用。而这样的方法不够一般化。

论文中给出的解决方案：

引入CodegenParam，CodegenHook，从而对代码生成的过程进行更为细致的，更为自动化的控制。因此，不同的语言和compiler可以拥有不同的参数和hook，从而定制代码生成过程。此外，这种方法还有大量对main compiler代码的重用，因此效率和功能性都更好。



### 论文内容

前端IR接口（high-level IR)

提供生成JULIA IR, LLVM IR的接口



后端IR接口（low-level IR）

作者用JULIA语言对LLVM C API进行了封装。





