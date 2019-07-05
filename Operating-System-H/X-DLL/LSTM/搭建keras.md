### 概述
鉴于keras的易用性和高效，首先使用它来进行神经网络的搭建

### 安装tensorflow
keras 基础可以使用不同的backend， 官方推荐使用tensorflow， 又由于实际上构建神经网络的最多使用的也是tensorflow， 其次才是keras

安装独立的python3环境，由于处在试探期。这里只写我的环境，其他搭建方式可见[tensorflow documemnt](https://www.tensorflow.org/install/install_linux)

```
$ sudo apt-get install python3-pip python3-dev python-virtualenv
$ virtualenv --system-site-packages -p python3 targetDirectory
$ source targetDirectory/bin/activate
(tensorflow)$ easy_install -U pip
(tensorflow)$ pip3 install --upgrade tensorflow
```
插一句，激活独立环境和退出独立环境的方式分别是(视创建的目录而定)
```
$ source ~tensorflow/bin/activate
(tensorflow)$ deactivate
```
以下内容均是在独立环境中进行，为了简便，不再标示前缀。

### 安装keras 及其他可能用到的包
```
pip3 install numpy jupyter keras matplotlib 
```

[numpy user guide](https://docs.scipy.org/doc/numpy/user/)

### 常见问题
- 有可能出现jupyter不在目录环境中,在~/.zshrc中添加
```
export PATH=$PATH:~/.local/bin
```
- 下载很慢
```
pip3 install tensorflow -i http://mirrors.aliyun.com/pypi/simple  --trusted-host mirrors.aliyun.com
```

### 训练尝试
[keras mnist tutorial](keras-mnist-tutorial)

[iris hello world](https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb)
