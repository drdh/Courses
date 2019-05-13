> 2019/5/13下午与陈广大博士交流记录
>
> 参与人员：董恒、任正行、张宇翔

1. 环境配置

   Ubuntu: 16.04

   ROS: Kinetic

   硬件: 单机()

2. 仿真环境

   - 用的是gazebo
     - 优点：对ROS封装好，直接用service或者topic传递信息
     - 缺点：过于“真”，一些在导航里不需要的物理参数，比如摩擦、重力等等，无法去掉，消耗计算量
   - 其他基于ROS的：Stage
     - 几乎是二维仿真，没有上面的多余物理参数
     - 可能没有gazebo那么好的封装
   - 不基于ROS的: V-REP
     - 学生版和教育版免费
     - 具体没用过，但是看到过一些用的，效果很好

3. DH做的是否有价值？

   - 问题回顾

     近距离导航、使用imitation learning + reinforcement learning + state representation learning

   - 有价值

     我们目前的效果也不是很好，泛化性不好。所以多做一些探索也可以