### 利用docker搭建HDFS集群

#### 搭建环境

- 安装docker

- 拉取镜像

  这一步，我在网上找到了一个配置地比较好的具有hadoop环境的镜像

  ```docker pull registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop```

- 创建容器
  
  创建四个容器，分别用作一个master节点、两个slave节点和一个client
  
  ```docker run -i -t --name Master -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

  ```docker run -i -t --name Slave1 -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

  ```docker run -i -t --name Slave2 -h Master registry.cn-hangzhou.aliyuncs.com/kaibb/hadoop /bin/bash```

  (client稍后再说)

- 配置Java环境
  
  由于该镜像中已经集成了JDK，所以不需要进行这一步操作，这也是选择这个镜像的好处。

- 配置SSH
  
  启动SSH```/etc/init.d/ssh start```
  
  生成秘钥```ssh-keygen -t rsa```

  将公钥互相添加到~/.ssh/authorized_keys中
  
  将IP地址互相添加到/etc/hosts中
  
#### 配置hadoop

  在Master节点进行配置，然后通过scp命令分发到各节点。总共有四个文件需要配置(在/opt/tools/hadoop/etc/hadoop目录下)。

- core-site.xml

  (指定namenode的地址和使用hadoop时产生的文件存放目录)
  
  ```xml
  <configuration>
    <property>
      <name>fs.defaultFS</name>
      <value>hdfs://Master:9000</value>
    </property>
    <property>
      <name>hadoop.tmp.dir</name>
      <value>/hadoop/data</value>
    </property>
  </configuration>
  ```

- hdfs-site.xml

  (指定保存的副本的数量、namenode的存储位置和datanode的存储位置)

  ```xml
  <configuration>
    <property>
      <name>dfs.replication</name>
      <value>1</value>
    </property>
    <property>
      <name>dfs.datanode.data.dir</name>
      <value>/hadoop/data</value>
    </property>
    <property>
      <name>dfs.namenode.name.dir</name>
      <value>/hadoop/name</value>
    </property>
  </configuration>
  ```
  
- mapred-site.xml
  
  ```xml
  <configuration>
    <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
    </property>
  </configuration>
  ```
  
- yarn-site.xml

  ```xml
  <configuration>
    <property>
      <name>yarn.resourcemanager.address</name>
      <value>Master:8032</value>
    </property>
    <property>
      <name>yarn.resourcemanager.scheduler.address</name>
      <value>Master:8030</value> </property> <property>
      <name>yarn.resourcemanager.resource-tracker.address</name>
      <value>Master:8031</value>
    </property>
    <property>
      <name>yarn.resourcemanager.admin.address</name>
      <value>Master:8033</value>
    </property>
    <property>
      <name>yarn.resourcemanager.webapp.address</name>
      <value>Master:8088</value>
    </property>
    <property>
       <name>yarn.nodemanager.aux-services</name>
       <value>mapreduce_shuffle</value>
    </property>
    <property>
      <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
      <value>org.apache.hadoop.mapred.ShuffleHandler</value>
    </property>
  </configuration>
  ```
  
- 修改slave文件

  将/opt/tools/hadoop/etc/hadoop目录下的slave文件修改为
  
  ```
  Slave1
  Slave2
  ```
  
**注：由于使用的镜像不同，hadoop的配置文件所在的目录也可能不尽相同，但具体配置应该是大同小异的。**

#### 运行hadoop

  进行格式化```hadoop namenode -format```
  然后在```/opt/tools/hadoop/sbin```目录下启动```./start-all.sh```

  启动成功应该如下图所示，各个节点应该有图中的几个进程：

![docker启动](/home/linan/X-dll/Report4/DFS_img/docker启动.png)

#### 客户机

由于我使用的镜像比较精简，很多命令都没有，所以如果用该镜像创建一个client的话，安装神经网络预测所需要的各种包比较麻烦，所以我直接将宿主机作为client访问HDFS集群。

需要在宿主机上安装JDK和hadoop，其中hadoop的配置方法和master、slave节点的配置方法一样。主要是要把宿主机的IP地址和ssh公钥添加到其他节点，其他节点的IP地址和ssh公钥也要添加到宿主机中。

#### 遇到的问题

  一个问题：第一次集群启动成功，第二次就失败了，大概是我不小心改了什么配置。如果始终无法解决的话，就直接在实体机上搭建集群,步骤也差不太多。

错误已经得到解决：一方面是因为docker镜像关闭后，保存的IP地址会消失，尽管我已经保存了对镜像的修改；另一方面是因为由于多次格式化，造成namdenode的namespaceID与datanode的namespaceID不一致，从而导致namenode和datanode的断连，slave节点的datanode不能启动(详情参考这个[博客](https://blog.csdn.net/love666666shen/article/details/74350358))。

### 添加神经网络预测模块

#### 概述

![图片1](/home/linan/X-dll/DFS/图片1.png)

图片中的接口函数层就是用来添加模块的。

有两种添加方案：①利用HDFS提供java接口添加模块，HDFS与用户交互的程序不变(shell)；②利用其他编程语言(shell，python等等)，在HDFS之外连接预测模块和HDFS，HDFS与用户交互的程序为该接口。方案一相当于在HDFS内部增加一部分代码，性能可能很好，但比较麻烦；方案二相当于在两个函数（进程）之间再写一个函数，用于参数的传递，但联系的层数变多了，性能可能有所下降。

由于时间有限，我采用的是方案二，并且使用的编程语言是python，这是因为用python写起来比较方便，调用终端的命令也比较简单，再加上预测模块也是用python写的。

#### 相关实现

处理过程如下：

1. 用户输入命令，如果是上传/下载命令，则先检查HDFS/本地临时文件夹，如果要上传/下载的文件已经存在，则直接移动，否则令HDFS执行下载/上传。
2. 在上传/下载完文件后，调用预测模块，根据文件名得到进行预取/预存的文件ID，然后进行预存/预取，保存到HDFS/本地临时文件夹。

此处的文件ID不同于linux下的文件ID，这里的ID是由于预测模块接口的要求而自定义的ID，我设置为从1开始。由此，还需要维护一个name2id的字典和一个id2name的字典。

具体实现过程请参阅github仓库中的源代码interface.py。

#### 问题及改进方案

目前的实现还存在如下几个问题：

- 随着存取文件的增多，要维护的字典也越来越大，时间开销和空间开销
- 每存取一次文件就进行一次预存取，当一个client连续存取大量文件时，性能会下降

可能的对应解决方案：

- 可以定期清除一部分name和ID
- 将预测和上传/下载单独作为一个/多个进程，设置两个队列来存放要预存取的文件名，在执行用户的上传/下载命令时同时进行预存取(该项已基本实现，详见github仓库中的源代码interface_optimized.py)

#### 实际测试

实际测试的结果并不好。

主要可能原因是：我是使用的docker容器来模拟HDFS集群，client是宿主机，但不知为何，宿主机和容器的通信很慢，即使是一条简单的hadoop fs -ls /命令都会用上一秒左右，显然通信的时间已经远远大于文件真正读写的时间，所以即便在存取方面做了很大的优化，总体时间并没有什么变化。

但通过对神经网络的本地测试(见董恒的工作)，可以认为在真正的HDFS分布式集群中，存取的效率是有较好的改善的。

### 参考资料

1. [使用Docker搭建hadoop集群](https://blog.csdn.net/qq_33530388/article/details/72811705)
  
