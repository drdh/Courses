#### 项目说明
DFS File Access Predict Based on Machine Learning

- 详细说明请见[Report4](./Report4)

#### 成员组成
- 董恒:DH
- 李楠:LN
- 连家诚:LJC

#### 目录组成
- Report1 初期报告:DH
- Report2 可行性报告，概要设计:LN
- Report3 中期汇报:LJC
- Report4 最终报告与文档
  - LSTM.md: 预测文档
  - DFS.md: 添加模块文档
- Plan.md 计划安排以及完成情况
- Explanation.md FAQ :DH
- LSTM 预测模块:DH
  - model_for_generate: 静态预测
    - data: 人工生成数据
    - img: 图片
    - model: 模型存储
    - model_src: 模型代码
  - model_for_strace: 动态预测
    - data: strace跟踪数据
    - img: 图片
    - model: 模型存储
    - model_src: 模型代码
- DFS 添加模块：LN DH LJC
    - predict_1: 预测模块
      - model: 模型存储
      - src
        - dynamic_predict.py: 动态预测
        - predcit.py: 静态预测
        - interface.py: 接口
        - interface_optimization.py: 优化后的接口，具有多进程（该python脚本代替shell与用户进行交互）
    - Add_module: 添加模块
    - JAVA_API.md
