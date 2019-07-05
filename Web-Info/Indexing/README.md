# 倒排索引

### 使用

###### 编译生成可执行文件

```bash
make
```

###### 测试

```bash
make test n=4
```

需要指定有多少文件（默认为4）1.txt, 2.txt ...文件放在./doc目录

会自动输出基本版和升级版的文本内容

###### 删除生成的文件

```bash
make clean
```

### 实现内容

同时实现了基本版和升级版。以下为输出文件。

- 基本版：index.txt
- 升级版：dict.txt    list.txt

### 目录说明

```
.
├── dict.txt	//升级版
├── doc		//文本文件
│   ├── 1.txt
│   ├── 2.txt
│   ├── 3.txt
│   ├── 4.txt
│   └── ...
├── index	//可执行文件
├── index.cpp	//源码
├── index.txt	//基本版
├── Makefile	//make
├── list.txt	//升级版
└── README.md	
```

### 其他

目前暂支持`,.?!; \n\t\\<>{}[]()|/`其中的符号去除，需要添加的，请在源码的main中调用split函数传参处添加或修改。