# 概率编程Team work

概率编程的主要目的是能够**快速高效**地构建一个能够处理非确定性因素的系统，来帮助人们进行分析预测，做出决断，
目前来看对机器学习系统非常有用。微软的Infer.net已经被整合进入了Microsoft.ML的机器学习工具库中。

概率编程的目的：

- 能够降低对专业能力的要求
- 缩短开发时间，提高代码可读性
- 优化运行的效率，减少开销

要注意普通的编程语言在一些比较权威的包或者库的帮助下也一样可以构建类似的系统，但是往往对编程人员要求比较高。

（以上来自John Bronskill 2017 presentation@cambridge）

## 概率编程中的概念

先验概率分布(Prior Distribution)：概率编程是从对一系列变量的分布进行一个估计开始的，这样的估计往往基于人的经验。这也是贝叶斯推断的特点。

检查/观察(Observation)：对某个定义的随机变量进行“观察”，可以使其变成固定的量，这会对需要推断的分布产生影响（变成某个条件概率分布）。

后验概率分布(Posterior Distribution)：在经过观察后，相当于有了额外的经验，这时结合先验概率分布和观察得到的随机变量的分布就称为后验概率分布。先验概率分布和后验概率分布都是随机变量的分布，并没有本质的区别，最重要的区别是有没有进行观察(Observation)。

## infer.net 介绍

infer.net是微软设计的基于C#的概率编程工具，严格来讲是C#的一个包或者库，并不是一个真正的编程语言（区别于fidaro)。在C#中使用

```c#
using Mircrosoft.ML.Probabilistic
```

就可以使用infer.net。

### 如何构建模型

Infer.net提供了一系列的API，用来建立随机变量，为随机变量给定先验分布，为模型中的变量设置逻辑关系，或者根据观察值进行变量分布的预测推断。

#### 模型API举例

Infer.net中可以构建具有某个分布的随机变量，也可以是常量，某个随机变量还可以在“观察”后影响系统的后验概率分布。

以最简单的扔硬币的模型举例，我们知道理想情况下扔硬币符合伯努利分布。我们可以用一个简单的模型推断两枚硬币都是正的概率分布，主要代码：

```C#
Variable<bool> firstCoin = Variable.Bernoulli(0.5);
Variable<bool> secondCoin = Variable.Bernoulli(0.5);

Variable<bool> bothHeads = firstCoin & secondCoin;

InferenceEngine engine = new InferenceEngine();
Console.WriteLine("Probability both coins are heads: " + engine.Infer(bothHeads)
```

这样可以得到输出：

Probability both coins are heads: Bernoulli(0.25)

对于 Observation 的应用，假设我们得到了 bothHeads 为 False，则我们可以推断两个硬币变量的概率分布。

```C#
bothHeads.ObservedValue=false;
Console.WriteLine("Probability distribution over firstCoin: " + engine.Infer(firstCoin)
```

得到输出

Probability distribution over firstCoin: Bernoulli(0.3333)

比较简单的应用构建大概就是经过：模型构建，设置ObservedValue，进行推断。更复杂的模型还可以实现预测分析，分类，推荐，聚类等应用。

更多的例子在 Infer.Net 的教程中还有介绍。

### Infer.Net是如何工作的

因为是被整合进入了.net平台，所以Infer.net的一个优势就在于能够比较方便地被其他.net平台地语言调用，而且自身地编译器也比较容易设计，大致地工作过程如下：

![howwork](./how_work.png)

1. 首先开发者需要定义好模型和根据模型所需要进行的推断。
2. 将模型定义和推断内容交给模型的编译器，编译器会将其编译成标准的代码。
3. 这些代码被交给C#编译器，编译成一个标准的.Net程序。
4. 这个程序可以根据Inference Settings和Observed values 进行推断，得到最后的分布。
5. 之后的应用程序可以根据这个分布进行分析和决策。

### Infer.Net目前被用于哪些地方

- XBOX Live的Ranking System，根据玩家的战绩进行分类，在线匹配玩家的时候将水平类似的玩家匹配在一起。
- Azure 的机器学习系统。
- Office 365的垃圾邮件识别和邮件分类。
- 在线商城的推荐系统。

可以发现这些应用其实不用Infer.Net也能做出来，也就是说概率编程并没有提供新的功能，而主要在于简化模型的程序表示，缩短开发周期，提高扩展性。

### 团队项目的初步规划、资料汇总

目前还需要调研一些比较活跃的概率编程工具，除了Infer.Net之外还有Figaro，这是一本比较系统介绍概率编程语言的书《概率编程实战》中介绍的语言，edward，Google开发的基于Python的概率编程工具包。

除此之外还需要具备一些机器学习的模型，从概率模型的角度理解机器学习方法。这里有一本基于模型介绍机器学习的书：[Model-Based Machine Learning](http://mbmlbook.com/)

目前想到的几个问题：

1. 概率编程是做成一个单独的编程语言好还是作为一个语言的库比较好？
2. 如果设计成一个单独的编程语言，如何设计其文法让它表示一个概率模型？
3. 如果设计成一个语言的库，拿Python的为例子，如何设计其API用来表示一个概率模型？

2种可能的小目标，2个选一个

1. 用某个概率编程模型开发一个机器学习应用并比较和其他方式构建的区别。
2. 设计一个基于某个语言的概率编程工具。

#### 目前的资料

[Google Edward](http://edwardlib.org/)
[Microsoft Infer.Net](https://dotnet.github.io/)
[Model Based Machine Learning](http://mbmlbook.com/)
[Practical Probabilistic Programming O'RELLY Media](https://www.oreilly.com/library/view/practical-probabilistic-programming/9781617292330/)

微软的Infer.Net资料和API文档非常详细可靠。