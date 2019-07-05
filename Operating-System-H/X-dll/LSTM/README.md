### 数据的选择
目前科研界对于file access patterns的研究非常少，相关的数据集也非常非常少，在查询了很多相关的论文之后，一共看到了如下几种解决办法。
- 使用[DFSTrace](http://coda.cs.cmu.edu/DFSTrace/)的，这个界面上提供了相关论文以及使用的工具，但是，这个数据集是在２０年前搜集的，而且使用的解析工具以及不能使用，如果重新构建的话，需要耗费大量时间。
- 使用另一个搜集的数据集，具体论文为《Characteristics of File System Workloads》,这个数据集也是将近２０年前搜集的，而且我没有找到相关的地址，无法使用。
- 使用[Traces and Snapshots Public Archive](http://tracer.filesystems.org/),这个是比较近期搜集的，但是这个是对文件系统的一个Snapshot，每一天都有一个，而且文件非常大，比较不容易解析出文件的变化，同样无法使用。
- 自行构建数据集。这是最后一条路，也比较难以反应真实场景。因此需要仔细构建。

### 构建数据前说明
使用自行构建的数据进行训练和测试，本质上是完全不可行的，这种方式会导致测试信息的泄露。泛化能力表现成虚高，而且容易过拟合。

### 构建细节
** 注 ** 以下没有被罗列出来全部忽略。
- name
  - 2D tensor (10,36) one-hot encodding
  - example: data3 [[0,0,0,1,0,0,...],[1,0,0,...],[0,...0,1,0...],...]
  - a-z (same as A-Z)
  - 0-9
- id
  - 0D tensor (1,) normalize to 0-1 float ?
  - example: 36
  - 0-999
- extension
  - 1D tensor (20,) one-hot encodding
  - example: pdf [0,1,0,0,...]
  - UNK:0 doc:1 pdf:2 jpg:3 docx:4 html:5 txt:6 xls:7 png:8 gif:9 avi:10 md:11 c:12 cc:13 java:14 py:15 tex:16 rar:17 tar:18 zip:19
- directory
  - 1D tensor (5,)
  - example: /1/2/3, [1,2,3]
  - 0-9
    - 0-9
      - 0-9
- size
  - 0D tensor
  - example 36
  - small files <=1024 bytes
  - 1-1024
- protection
  - 0D tensor
  - example: 0
  - executing 0
  - reading 1
  - writting 2
- owner
  - 0D tensor
  - example: 4
  - 0-4
- created time
  - 1D tensor (4,)
  - example: 2018/6/2 17:07 [18,6,2,17]
- modified time
  - 2D tensor (10,4)
  - example: 2018/6/2 17:07, 2018/6/3 16:09 ... [[18,6,2,17],[18,6,3,16],...]
- access time
  - 2D tensor (10,4)
  - the same as above
- operation
  - 2D tensor (10,2)
  - example: [[0,1],[0,1],[0,1],...]
  - read 0
  - write 1

### 测试说明
- 请使用test.py 与network.py 进行接口测试
- predict.py 暂时未完成

### 模型的构建
鉴于以上的数据不同构，而且各自的权重也是不同的，为了充分利用这些信息，首先单独为每一个元素构建一个简单的神经网络。
