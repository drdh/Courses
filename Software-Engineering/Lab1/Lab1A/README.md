# 单词链说明文档

## 文件内容

```c
|── GUI/
│   ├── lib/			//均为Qt程序运行所需的依赖库
│   ├── main.cpp			//用户交互界面主程序
│   ├── qss/					//Qt样式表资源文件，用于完善用户交互界面的外观
│   │   └── style.qss
│   ├── style.qrc			//描述Qt资源文件
│   ├── widget.cpp			//用户交互界面的相关函数的实现
│   ├── widget.h			//用户交互界面的头文件
│   ├── WordList_GUI.pro	//Qt工程文件
├── README.md　//使用方法
├── src/
│   ├── LongestWordChain.cpp　//主要函数头文件
│   ├── LongestWordChain.h //主要函数的实现
│   ├── main.cpp	//命令行方式的主函数
│   └── Makefile	
└── test/　//测试文件
    ├── test_1.txt 
    ...
```

## 使用方式

### 编译使用

在`src`目录中`make`

```bash
./main -f ../test/test_1.txt -w 
cat solution.txt
rm solution.txt
```

**参数使用说明**

1. 参数的位置随意
2. 指定文件需要加上`-f`
3. `-w`   `-c`有且只有一个
4. `-h -t -n`与上面的参数可任意搭配

**输出的内容解释**

1. 只标示`-w`时，会输出单词链单词的数量，以及相应的单词链。
2. 只标示`-c`时，会输出单词链字符的数量，以及相应的单词链。
3. 标示了`-n`时，会输出满足要求的单词链的个数，然后分别输出相应的单词链，使用空行间隔。

### 清除

`make clean`