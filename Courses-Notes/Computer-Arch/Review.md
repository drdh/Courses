# 体系结构复习

[对应章节](<http://home.ustc.edu.cn/~candrol/>)

| Topic        | **Readings 5th Edition** | PPT                                 |
| ------------ | ------------------------ | ----------------------------------- |
| Introduction | Ch. 1                    | 01                                  |
| ISA          | 附录A                    | 02                                  |
| 流水线       | 附录C                    | 03                                  |
| 存储器       | Ch. 2 && 附录B           | 04                                  |
| ILP          | Ch. 3                    | 05                                  |
| DLP          | Ch. 4                    | 06 (vector-I,II; GPU-I,II )         |
| TLP          | Ch. 5                    | 07(Coherence-I,II,III; Consistency) |

## 1. Intro

计算机组成与实现

### 1.1 引言

#### 计算机体系结构的定义

##### 动态功耗

晶体管能耗
$$
\text{Dynamic Energy}\propto \text{Capacitive Load}\times \text{Voltage}^2
$$
(Capacitive Load: 容性负载)

表示0-1-0或者1-0-1的能耗，如果是0-1或1-0就乘以1/2. 功率(功耗)则需要乘以转换频率
$$
\text{Dynamic Power}\propto \text{Capacitive Load}\times \text{Voltage}^2 \times \text{Frequency Switched}
$$

降低频率可以降低功耗

降低频率导致执行时间增加 ==> 不能降低能耗

降低电压可有效降低功耗和能耗

##### 减少动态功耗的技术

- 关闭不活动模块或处理器核的时钟 (Do nothing well) 比如没有浮点指令时关闭浮点计算单元
- 动态电压-频率调整(Dynamic Voltage-Frequency Scaling, DVFS)
  在有些场景不需要CPU全力运行; 降低电压和频率可降低功耗
- 针对典型场景特殊设计 (Design for the typical case)
  电池供电的设备常处于idle状态,DRAM和外部存储采用低功耗模式工作以降低能耗
- 超频(Overclocking)
  - 当在较高频率运行安全时,先以较高频率运行一段时间,直到温度开始上升至不安全区域
  - 一个core以较高频率运行,同时关闭其他核

##### 静态功耗

$$
\text{Static Power}=\text{Static Current}\times \text{Voltage}
$$



#### ~~计算机的分类~~

#### ~~现代计算机系统发展趋势~~

### 1.2 定量分析基础

#### 性能的含义

性能度量

- 响应时间 (response time)

- 吞吐率 (Throughput)

  

#### CPU性能度量

$$
\text{CPU time}=\text{Instruction Count(IC)}\times \text{Cycles Per Instruction(CPI)} 
\times \text{Clock Cycle time(T)）}\\
\text{CPI}=\sum_{i=i}^n(\text{CPI}_i \times \text{Freq}_i)
$$

Amdahl定律
$$
\text{Performance Increasing Ratio}=\frac{1}{x+\frac{1-x}{N}}
$$
MIPS (Millions of Instructions Per Second)
$$
\text{MIPS}=\frac{1}{\text{CPI}\times \text{T}\times 10^6}
$$




#### 计算机系统性能度量

SPEC性能综合
$$
\text{SPEC Ratio}=\frac{\text{Time on Reference Computer}}{\text{Time on Computer Being Rated}}\\
\frac{\text{SPEC Ratio}_A}{\text{SPEC Ratio}_B}=\frac{\frac{\text{ExecutionTime}_{\text{Ref}}}{\text{ExecutionTime}_A}}{\frac{\text{ExecutionTime}_{\text{Ref}}}{\text{ExecutionTime}_B}}=\frac{\text{ExecutionTime}_B}{\text{ExecutionTime}_A}=\frac{\text{Performance}_A}{\text{Performence}_B}\\
\text{Geometric Mean of SPEC Ratios}=\sqrt[n]{\prod_{i=1}^n \text{SPEC Ratio}_i}
$$
特性(除一下就证明了)

- SPEC Ratio几何平均的比率 = SPEC Ratio 比率的几何平均 = 性能比率的几何平均



## 2. ISA

### 2.1 ISA分类

### 2.2 存储器寻址

尾端问题

- 大小尾端

对齐问题

寻址方式

- 寄存器寻址
- 立即数寻址
- 位移量寻址

### 2.3 操作数的类型与大小

### 2.4 指令集中的操作

- CISC(Complex Instruction Set Computer)
  - 目标:强化指令功能,减少运行的指令条数,提高系统性能
  - 方法:面向目标程序的优化,面向高级语言和编译器的优化
- RISC(Reduced Instruction Set Computer)
  - 目标:通过简化指令系统,用高效的方法实现最常用的指令
  - 方法:充分发挥流水线的效率,降低(优化)CPI

### 2.5 控制流指令

**条件分支**( Conditional branch) 、跳转(Jump)、过程调用(Procedure calls)、过程返回(Procedure returns)

### 2.6 指令集编码

变长、定长、混合

### 2.7 MIPS体系结构



## 3. 流水线

### 3.1 流水线性能

基本度量参数:吞吐率,加速比,效率

#### 吞吐率(TP)

在单位时间内流水线所完成的任务数量或输出结果的数量。
$$
\text{TP}=\frac{n}{(k+n-1)\Delta t}\to \frac{1}{\Delta t}=\text{TP}_{\text{max}}\qquad n\to \infty	\\
\text{TP}=\frac{n}{(k+n-1)}\text{TP}_{\text{max}}
$$
考虑到不同段的时间不同，
$$
\text{TP}=\frac{n}{\sum_{i=1}^k \Delta t_i +(n-1)\max(\Delta t_1,\Delta t_2,\cdots,\Delta t_k)}\to \frac{1}{\max(\Delta t_1,\Delta t_2,\cdots,\Delta t_k)}=\text{TP}_{\text{max}}
$$


#### 加速比(S)

完成同样一批任务,不使用流水线所用的时间与使用流水线所用的时间之比

$k$段流水线，在阶段$S_i$的延迟为$\tau_i$, 则时钟周期定为最大的延迟$\tau=\max(\tau_i)$, 时钟频率为$f=1/\tau$

对于$n$个任务，第一个任务需要$k$个时钟周期，其他$n-1$需要$n-1$个，总的为$k+n-1$个时钟周期，理想加速比(想对于非流水)为
$$
S_k=\frac{nk}{k+n-1}\to k\qquad n \to \infty
$$
即其潜在的加速比=流水线的级数

各段时间不等时
$$
S=\frac{n\sum_{i=1}^k \Delta t_i }{\sum_{i=1}^k \Delta t_i +(n-1)\max(\Delta t_1,\Delta t_2,\cdots,\Delta t_k)}
$$
流水线的加速比计算
$$
\text{CPI}_{\text{pipelined}}=\text{Ideal CPI}+\text{Average Stall Cycles per Inst}\\
\text{Speedup}=\frac{\text{Ideal CPI}\times \text{Pipeline depth}}{\text{Ideal CPI}+\text{Pipeline stall CPI}}\times \frac{\text{Cycle time}_{\text{unpipelined}}}{\text{Cycle time}_{\text{pipelined}}}
$$
对于简单的RISC流水线，上述的Ideal CPI=1



#### 效率(E)

流水线中的设备实际使用时间与整个运行时间的比值,即流水线设备的利用率。

各段时间相等时，各段的效率$e_i$相同
$$
e_i=\frac{n\Delta t}{T_k}=\frac{n}{k+n-1}
$$
从而整条流水线的效率为
$$
E=\frac{ke_i}{k}=\frac{n}{k+n-1}\to 1 \qquad n\to\infty
$$
每段时间不同时
$$
E=\frac{n\sum_{i=1}^k \Delta t_i}{k[\sum_{i=1}^k \Delta t_i +(n-1)\Delta t_{\text{max}}]}=\text{TP}\frac{\sum_{i=1}^k \Delta t_i}{k}
$$


### 3.2 流水线冒险

- 结构冒险
  - 延迟冲突指令的执行
  - 增加新的资源

- 数据冒险
  - 类型
    - RAW	
    - WAR: 反相关
    - WAW: 输出相关
  - 策略
    - 互锁(Interlock): 等待相关的解除
    - 旁路转发(Bypass/Forwarding)
    - 推测(Speculate):  猜测一个值继续,如果猜测错了再更正
    - 采用软件改变指令执行顺序
- 分支冒险
  - Stall 直到分支方向确定
  - 预测分支失败
  - 预测分支成功
  -  延迟转移技术

### 3.3 异常、陷阱和中断





## 4. 存储器

### 4.1 存储层次结构

利用caches**缓解**微处理器与存储器性能上的差异

存储层次工作原理：Temporal Locality (时间局部性) + Spatial Locality (空间局部性)

### 4.2 Cache基本知识

映象规则：全相联、组相联、直接映射

查找方法

替换算法：LRU、FIFO、随机

写策略：写直达(Write-Through)+写不分配(no-write allocate)、写回(Write-Back)+写分配(write allocate); 通常配有写入缓冲区

缓存公式P420

>$$
>2^{\text{索引}}=\frac{\text{缓存大小}}{\text{块大小}\times \text{组相联度}}\\
>\text{CPU 执行时间}=(\text{CPU 时钟周期}+\text{存储器停顿周期})\times \text{时钟周期时间}\\
>\text{存储器停顿周期}=\text{缺失数}\times \text{缺失代价}\\
>\text{存储器停顿周期}=\text{IC}\times \text{每条指令缺失数} \times \text{缺失代价}\\
>\text{每条指令缺失数}=\text{缺失率}\times \text{每条指令访存数}\\
>\text{存储器平均访存时间}=\text{命中时间}+\text{缺失率}\times\text{缺失代价}\\
>\text{CPU 执行时间}= \text{IC}\times (\text{CPI}_{\text{执行}}+\text{每条指令的存储器停顿数})\times \text{时钟周期时间}	\\
>\text{CPU 执行时间}= \text{IC}\times (\text{CPI}_{\text{执行}}+ \text{每条指令缺失数} \times \text{缺失代价})\times \text{时钟周期时间}	\\
>\text{CPU 执行时间}= \text{IC}\times (\text{CPI}_{\text{执行}}+\text{缺失率}\times \text{每条指令访存数}\times \text{缺失代价})\times \text{时钟周期时间}	\\
>\text{每条指令存储器停顿周期}=\text{每条指令缺失数}\times (\text{总缺失延迟}-\text{重叠缺失延迟})\\
>\text{存储器平均访存时间}=\text{命中时间}_{L1}+ \text{缺失率}_{L1}\times (\text{命中时间}_{L2}+\text{缺失率}_{L2}\times\text{缺失代价}_{L2})\\
>\text{每条指令存储器停顿周期}=\text{每条指令缺失数}_{L1}\times \text{命中时间}_{L2} + \text{每条指令缺失数}_{L2}\times \text{缺失代价}_{L2}
>$$



### 4.3 基本的Cache优化方法

$\text{存储器平均访存时间}=\text{命中时间}+\text{缺失率}\times\text{缺失代价}$

3C: 强制性失效 (Compulsory), 容量失效(Capacity), 冲突失效(Conflict (collision))

2:1 Cache经验规则: 即大小为N的直接映象Cache的失效率约等于大小为N/2的两路组相联的Cache失效率。

- 降低失效率
  - 1、增加Cache块的大小
  - 2、增大Cache容量
  - 3、提高相联度
- 减少失效开销
  - 4、多级Cache: 多级包含/多级互斥
  - 5、使读失效优先于写失效：写缓冲
- 缩短命中时间
  - 6、避免在索引缓存期间进行地址转换

### 4.4 高级的Cache优化方法

- 缩短命中时间
  - 1、小而简单的第一级Cache
  - 2、路预测方法
- 增加Cache带宽
  - 3、Cache访问流水化
  - 4、无阻塞Cache:  允许在Cache失效下继续命中
  - 5、多体Cache
- 减小失效开销
  - 6、关键字优先和提前重启
  - 7、合并写
- 降低失效率
  - 8、编译优化
- 通过并行降低失效开销或失效率
  - 9、硬件预取
  - 10、编译器控制的预取

### 4.5 存储器技术与优化



### 4.6 虚拟存储器-基本原理



## 5. ILP

### 5.1 指令集并行的基本概念及挑战

### 5.2 软件方法挖掘指令集并行

循环展开

### 5.3 硬件方法挖掘指令集并行

#### Scoreboard

#### Tomasulo

### 5.4 基本块的指令集并行

### 5.5 基于硬件的推测执行

### 5.6 以多发射和静态调度来挖掘指令集并行

### 6.7 以动态调度、多发射和推测执行来挖掘指令集并行





## 6. DLP

### 6.1 数据级并行的研究动机

#### 传统指令级并行技术的问题

传统方法(挖掘ILP)的主要缺陷

- 程序内在的**并行性**

- 提高流水线的时钟频率: 提高时钟频率,有时导致CPI**随着增加** (branches, other hazards)

- 指令预取和译码: 有时在每个时钟周期很难预取和译码多条指令

- 提高Cache命中率 : 在有些计算量较大的应用中(科学计算)需要大量的数据,其局部性较差,有些程序处理的是连续的媒体流(multimedia),其局部性也较差。

#### SIMD结构的优势

- SIMD 结构可有效地挖掘**数据级并行**:
  - 基于**矩阵运算**的科学计算
  - **图像**和**声音**处理
- SIMD比MIMD更节能
  - 针对每组数据操作仅需要**取指一次**
  - SIMD对PMD( personal mobile devices)更具吸引力

- SIMD 允许程序员继续以**串行模式思维**

#### 数据级并行的种类

- 向量体系结构

- 多媒体SIMD指令集扩展

- Graphics Processor Units (GPUs)

### 6.2 向量体系结构

#### 性能评估及优化

Initiation rate: 功能部件消耗向量元素的速率

Convoy: 可在同一时钟周期开始执行的指令集合 (PPT: 无结构/数据冲突; 课本: 仅无结构冲突, 数据冲突使用链接解决)

Chime: 执行一个convoy所花费的大致时间; m convoys 需要 m chimes; m x n clock cycles 

$R_\infty$ : 当**向量长度为无穷大**时的向量流水线的最大性能

$R_n$ 表示向量长度为n时的向量流水线的性能

$N_{1/2}$: 达到$R_\infty$ 一半的值所需的**向量长度**,是评价向量流水线**start-up 时间**对性能的影响。

$N_V$:向量流水线方式的工作速度**优于标量串行方式**工作时所需的**向量长度临界值**



向量长度问题: Strip Mining(分段开采)
$$
T_n=\left\lceil\frac{n}{\text{MVL}} \right\rceil\times(T_\text{loop}+T_\text{start-up})+n\times N_\text{chime}
$$

链接技术: 前一条指令的第一个结果出来后,就可以启动下一条相关指令的执行

条件执行(VM: Vector Mask)

```
SNEVS.D V1,F0 		;V1[i]!=F0, 则VM[i]=1
SUBVV.D V1,V1,V2 	;仅计算为VM[i]==1的
```

稀疏矩阵(集中-分散)

```
LVI V1,(R1+V2)
SVI (R1+V2),V1
```

运算密度:定义为运行程序时执行的**浮点运算数**除以**主存储器中访问的字节数**

### 6.3 面向多媒体应用的SIMD指令集扩展

在已有的ISA中添加一些向量长度很短的向量操作指令

### 6.4 GPU

#### GPU简介

多处理器, 每个处理器为多线程的, 每个线程为SIMD(Warp-based SIMD)

GPUs 使用 SIMT(Single Instruction, Multiple Thread)模型, 每个CUDA线程的标量指令流汇聚在一起**在硬件上**以SIMD方式执行 (Nvidia groups 32 CUDA threads into a **warp**)

Warp: 一组执行相同指令的**线程**($\mu$threads作用于不同的数据元素)

#### GPU的编程模型

SPMD on SIMT Machine, 不是用SIMD指令编程

#### GPU的存储系统



## 7. TLP

### 7.1 引言

存储一致性(Coherence):

- 不同处理器访问**相同存储单元**时的**访问顺序**问题
- 访问每个Cache块的**局部**序问题
- 如果对某个数据项的任何读操作均可得到其最新写入的值,则认为这个存储系统是一致的(非正式定义)

存储同一性(Consistency):

- **不同处理器发出的所有存储器操作的顺序问题**(即针对不同存储单元或相同存储单元)
- 访问所有存储单元的**全序**问题
  

### 7.2 集中式共享存储器体系结构

MSI

MESI



### 7.3 分布式共享存储器体系结构

### 7.4 存储同一性

### 7.5 同步与通信



