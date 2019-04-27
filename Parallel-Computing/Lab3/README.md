# Lab 3 GPU Cuda

## 算法设计





## 实验环境

SSH Tunnel

```bash
ssh -N -L localhost:5901:localhost:5901 drdh@222.195.92.204 -p 5555
```

`Ubuntu`中安装`remmina` 

![1556336875446](README.assets/1556336875446.png)



```bash
CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHZ
GPU: GeForce GTX 1080
内存: 125G
操作系统: CentOS 9
软件平台: cuda10
#在实验室服务器中~/lx/lab使用cuda
#设置环境变量：
export PATH=/usr/local/cuda/bin:$PATH 
make
```

## 结果截图

![1555578528463](README.assets/1555578528463.png)

## Ref

[StackoverFlow Matrix Vector Multiplication](<https://stackoverflow.com/questions/26417475/matrix-vector-multiplication-in-cuda-benchmarking-performance>)

[Cuda Example](<https://people.cs.pitt.edu/~melhem/courses/xx45p/cuda_examples.pdf>)