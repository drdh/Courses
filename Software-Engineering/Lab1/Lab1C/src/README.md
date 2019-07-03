# 使用说明

- `make`			生成测试程序`WordChainTest`
- `make html`           生成html格式的测试覆盖率报表，需要在运行过测试程序`WordListTest`之后才能执行
- `make clean`         清除无关文件



**注：**

1. `make html`命令执行输出的总体覆盖率是综合各个头文件如`isotream`的，应当看`WordChain.cpp`的测试覆盖率。
2. 单元测试是在google test的框架下进行的，缺少相关库可能无法运行。