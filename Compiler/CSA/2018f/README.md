# G1 谷歌C++代码规范检查
- 队员
  - PB16030684-刘奕品：[ypliu88@mail.ustc.edu.cn](mailto:ypliu88@mail.ustc.edu.cn)
  - PB16060674-归舒睿：[agnesgsr@mail.ustc.edu.cn](mailto:agnesgsr@mail.ustc.edu.cn)
  - PB16001707-王梓涵：[wzh99@mail.ustc.edu.cn](mailto:wzh99@mail.ustc.edu.cn)
  - PB16001749-伊昕宇：[yixinyu@mail.ustc.edu.cn](mailto:yixinyu@mail.ustc.edu.cn)

- 参考链接：<https://google.github.io/styleguide/cppguide.html>

- 目录名：G1-GoogleCPPCodingStyleCheck

- 项目说明：

  C++是一门极复杂的语言，导致C++极难debug，所以要求程序员需要按照一定的编码规范来进行编程以减少后期debug的难度。在团队合作中，遵照一定的代码规范也是实现有效的交流，提高合作效率的有力保证。谷歌C++代码规范是少数几个公开的且有较多C++程序员遵循的代码规范之一，并且在实际项目中也有应用。实现此代码规范检查工具既可以在未来实际应用中使用，也是了解和熟悉代码规范的一个很好的机会。

# G2 

- 队员(请参照G1的队员信息更新)
  - PB16110386-丁峰： [df12138@mail.ustc.edu.cn](mailto:df12138@mail.ustc.edu.cn)
  - PB16110846-于颖奇：[yu971207@mai.ustc.edu.cn](mailto:yu971207@mai.ustc.edu.cn)
  - PB16111428-马凯：[ksqsf@mail.ustc.edu.cn](mailto:ksqsf@mail.ustc.edu.cn)
  - PB16060117-曾明亮： [zml2016@mail.ustc.edu.cn](mailto:zml2016@mail.ustc.edu.cn)
- 目录名：(请G3队补充以'G3-'开头的名称)
- 我们小组想调研：在llvm IR中，对一个复杂的多项式计算进行数学上的优化。
  -  通过调研，我们发现clang里面用for循环对一个多项式求和会直接优化成一个公式，但gcc做不到,只能向量化。icc甚至可以推断每个分支被执行的概率，从而进行进一步的优化。我们的想法是让llvm IR也做到对一个复杂的多项式进行代数上的优化
- Q：
  - clang里面用for循环对一个多项式求和直接优化成公式的原理是什么？
  - 在 llvm IR上做对一个复杂的多项式进行代数优化的依据是哪些？ 计划处理哪些复杂的多项式的patterns？识别和优化的主要原理是否有考虑？llvm 中类似的处理是哪些？打算利用LLVM的哪些开展你们的工作 ？

# G3 两开花队 - DefineMagic

- 队员
  - PB16001676-吴永基：[wyj317@mail.ustc.edu.cn](mailto:wyj317@mail.ustc.edu.cn)
  - PB16110428-王浩宇：[tzwhy@mail.ustc.edu.cn](mailto:tzwhy@mail.ustc.edu.cn)
  - PB16120156-邵军阳：[sjygd@mail.ustc.edu.cn](mailto:sjygd@mail.ustc.edu.cn)
  - PB116110766-陆万航：[rg1336@mail.ustc.edu.cn](mailto:rg1336@mail.ustc.edu.cn)
- 目录名：G3-DefineMagic
- 选题： c语言变量定义的优化：包括c语言变量的命名规范及检查、结构体存储空间的极小化（对对齐问题的优化处理）、较长布尔表达式的换行处理。
- 说明
  1. c语言变量命名的规范有助于增加代码的可读性和安全性。主要优化方向有：对作用域内同名的变量进行重新命名和对变量进行分词重构，增强变量的规范命名。
  2. 由于c语言的对齐方式问题，结构体变量的存储空间可能未达到极小化，通过调整结构体内部的变量命名顺序，可以降低结构体变量的存储开销。
  3. 进行条件判断时，由于判断的条件可能会较多，if语句中的布尔表达式可能很多，因此可以对较长的布尔表达式进行合理的换行，使得条件判断的内容能以一个较为清晰的模式呈现。

# G4 （补充中文名）

- 队员(请参照G1的队员信息更新)
  - PB16110989-杨子奇：[pb160989@mail.ustc.edu.cn](mailto:pb160989@mail.ustc.edu.cn)
  - PB15010419-牛田：[niutian@mail.ustc.edu.cn](mailto:niutian@mail.ustc.edu.cn)
  - PB16110365-连家诚：[lianjc@mail.ustc.edu.cn](mailto:lianjc@mail.ustc.edu.cn)
  - PB16110775-李楠：[ln2016@mail.ustc.edu.cn](mailto:ln2016@mail.ustc.edu.cn)
- 目录名：G4-MemoryCheck
- 项目说明
  - c/c++语言中对内存分配和使用的检查。包括部分函数的参数检查和new关键字使用的检查等。

项目名字为MemoryCheck，下面有各个子目录代表各个具体的Checkers。

# G5 (补充中文名)
- 队员(请参照G1的队员信息更新)
  - 金孜达（PB15000129）
  - 庞茂林（PB15000159）
  - 李文睿（PB16111360）
- 目录名（请G5队补充以'G5-'开头的名称）
- 项目内容：编写LLVM IR代码优化器(请G5队补充具体的优化内容和描述)
- 描述：我们将编写一个中间代码优化器以优化生成的LLVM IR中间代码。我们会先把传统的方法应用上去，随后再应用上自己发掘的创新思路。

# G6 CodingSpecChecker++
- 队员
  - PB15000256-胡清泳
  - PB15111595-刘伟森
  - PB16001778-吴豫章
  - PB15020554-张益博：[zyb233@mail.ustc.edu.cn](mailto:zyb233@mail.ustc.edu.cn)

- 目录名：G6-CodingSpecChecker++

- 项目说明：

	1. 根据张老师放出来的ppt比赛汇报，目前的 GoogleCPPCodingStyleCheck 仍然存在一些问题。如在函数参数检查、函数头注释检查、按需初始化检查以及函数返回值检查等项目均没有做到百分之百。
	2. 因此，我们希望能够通过研习源代码，然后进一步优化它。

# Julia 科研探索

- 队员

  - PB15081576-蔡文韬：[elisa@mail.ustc.edu.cn](mailto:elisa@mail.ustc.edu.cn)
  - PB16080210-戴路：[dldaisy@mail.ustc.edu.cn](mailto:dldaisy@mail.ustc.edu.cn)
  - PB16111545-董恒：[drdh@mail.ustc.edu.cn](mailto:drdh@mail.ustc.edu.cn)
  - PB16111447-何纪言：[hejiyan@mail.ustc.edu.cn](mailto:hejiyan@mail.ustc.edu.cn)
  - PB16110264-何理扬：[heliyang@mail.ustc.edu.cn](mailto:heliyang@mail.ustc.edu.cn)
  - PB15000135-俞晨东：[ycdxsb@mail.ustc.edu.cn](mailto:ycdxsb@mail.ustc.edu.cn)
  - PB15020718-张立夫：[zlf123@mail.ustc.edu.cn](mailto:zlf123@mail.ustc.edu.cn)

- 目录名：julia

- 项目说明：

  [Julia](https://julialang.org/) ([github](https://github.com/JuliaLang)) 是2009年开始由MIT 开发的、全球热度上升最快的编程语言之一。2018年8月发布了Julia 1.0。该语言的研制目标是**high-level**和**fast**，你可以阅读 [Why We Created Julia](https://julialang.org/blog/2012/02/why-we-created-julia) 来了解Julia语言的研制动机。目前Julia已经用于自动驾驶汽车、机器人、基准医疗、增强现实等领域。它通过LLVM为多个平台编译**高效**的本地代码；它使用**多分派**，使得表达函数式等的编程模式变得容易；它提供**动态**类型，支持交互式使用；它擅长**数值计算**，支持多种数值数据类型并具有良好的并行性；它拥有丰富的**描述性数据类型**，使用这些类型声明可以使程序条理清晰且稳定。

  **本专题探索的问题**包括但不限于：调研Julia的编译和运行机制，调研和使用Julia的一些包，例如，用于机器学习和AI的[Flux.jl](https://github.com/FluxML/Flux.jl)、[Merlin.jl](https://github.com/hshindo/Merlin.jl)、[Tensorflow.jl](https://github.com/malmaud/TensorFlow.jl) ，理解包的运转机制，尝试写自己的扩展包，等等。

# Probability Programming 概率编程

- 队员

  - PB16110957-任正行：[zh2016@mail.ustc.edu.cn](mailto:zh2016@mail.ustc.edu.cn)
  - PB16110881-唐珑涛：[tlongtao@mail.ustc.edu.cn](mailto:tlongtao@mail.ustc.edu.cn)

- 目录名：probability

- 项目说明：

  概率编程的概念：基于贝叶斯推断和概率图模型，提供随机变量和贝叶斯推断过程的高层抽象，根据观测数据进行后验分布的计算和推理决策。
  概率编程的好处：用简洁的接口和高效的底层实现让任何一个开发者可以轻松构建扩展性很强，结构非常清晰的应用。
  本项目探索的内容：理解概率编程的底层实现原理，学习如何用概率编程方法解决实际问题。比较概率编程和现有的机器学习框架解决方案的差异，分析概率编程的优势和劣势。
  结合机器人应用场景(导航、地图构建等)思考如何使用概率编程的思想改善建模过程。
# Quantum 量子编程科研探索
- 队员

  - PB16020878-李权熹：[crazylqx@mail.ustc.edu.cn](mailto:crazylqx@mail.ustc.edu.cn)
  - PB16080377-聂雷海：[nlh@mail.ustc.edu.cn](mailto:nlh@mail.ustc.edu.cn)
  - PB16120412-宋昊泽：[shz666@mail.ustc.edu.cn](mailto:shz666@mail.ustc.edu.cn)
  - PB16020582-邓皓巍：[jackdhw@mail.ustc.edu.cn](mailto:jackdhw@mail.ustc.edu.cn)

- 目录名：Quantum

- 项目说明：

  自从量子计算机的诞生起，量子计算就一直活跃于人们的视线中。由于量子的特殊性质，量子计算机可以在多项式时间内解决许多NP难问题。其中最为经典的便是可以在多项式时间分解质因数的[Shor算法](https://en.wikipedia.org/wiki/Shor%27s_algorithm),其可以有效地破解RSA公钥而受到广泛关注。算法科学家们也致力于研究各种量子算法。同时计算机界的科研人员也在编程语言、编译器、微指令集、底层架构等等各个层次完善量子计算机的体系结构。<br><br>
  在量子计算中，量子比特是量子信息的基本单元，量子门用于操作量子信息。量子计算系统在现实应用中面临**量子比特的退相干**(decoherence)、噪声引起的**量子门的精度问题**、**量子纠错**(error correction)等诸多挑
  战。目前的量子计算机处于[嘈杂中型量子](https://arxiv.org/abs/1801.00862)（Noisy Intermediate-Scale
Quantum，NISQ）时代,量子比特数量不足以实现纠错功能。
  纠错的另一种方法是避免错误或消除它们的影响，即**错误减轻**（error mitigation）。
  <br><br>
  本项目组目前致力于探索优化量子程序以降低错误率的方法，同时调研量子电路评测方法，据此研制出量子程序的评测软件。

# 智能家居科研探索

- 队员

  - PB16111487-邓胜亮：[](mailto:dengsl@mail.ustc.edu.cn)
  - PB16020870-王博：[](mailto:wbhalo@mail.ustc.edu.cn)
- 目录名：SmartHome
- 项目说明：
  随着物联网的发展、社会生产力水平的提高，智能家居逐渐融入人们的生活，Android 作为现在广泛使用的移动端操作系统，也逐渐成为连接用户和智能家居设备的桥梁。SmartThings是智能家居的主要平台之一。资源过度使用是该领域常见问题，比如设备耗电过快、消耗过多流量和带宽等问题屡见不鲜，一方面是开发者的经验和水平有限，另一方面，应用环境相对于传统的桌面环境更加灵活多变也使得缺陷容易出现，使得开发者可能会在应用中频繁申请系统资源、占用资源不释放，进而影响用户的使用。本课题以智能家居为应用场景，研究SmartThings API + Android App在其中的应用案例，调研、分析其中的导致过度资源使用的软件缺陷，研究相应的检测理论和方法，并开发出可用的缺陷检查工具。 