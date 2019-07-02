# Lab2 实验报告

**注**: 本实验使用`Jupyter Notebook`完成,而非单独的`.py`文件,所提交的文件是分割后的内容,并未进行测试.完整的实验代码见`overall_ipynb/Lab2.ipynb`

## Part I

### 数据预处理

需要处理的主要内容是将类别的字符串,换成数字.然后将数据存成`.npy`便于下次使用.

代码见`data_preprocessing.py`

### 常用函数

为了便于下面的计算,设计两个函数.

`classification_evaluation`是给预测分类和正确的结果,然后返回`Accuracy,Macro_F1,Micro_F1`

`split_data_5_fold`是用于分割验证数据的.

代码在`utils.py`

使用时需要加上

```python
from utils import classification_evaluation,split_data_5_fold
```

### kNN

#### 解决思路

先进行交叉验证,然后选择最优的`k`在测试集上测试

#### 伪代码



#### 结果



### Decision Tree

#### 解决思路



#### 伪代码



#### 结果



### SVM

#### 解决思路



#### 伪代码



#### 结果



## Part II

### 数据预处理

需要处理的主要内容是将类别的字符串,换成数字.然后将数据存成`.npy`便于下次使用.

代码见`data_preprocessing.py`

### 常用函数

`cluster_evaluation(label,C)`返回`purity,RI`

`cluster_save(C,file,k)`将聚类结果存在`.csv`文件总

见`utils.py`

使用时,加上如下代码

```python
from utils import cluster_evaluation,cluster_save
```

### kMeans

#### 解决思路



#### 伪代码



#### 结果



### PCA

#### 解决思路



#### 伪代码



#### 结果





### Hierachical Clustering

#### 解决思路



#### 伪代码



#### 结果







