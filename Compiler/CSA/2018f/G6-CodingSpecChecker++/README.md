## 项目说明
对NASAC原型竞赛的部分代码进行改进，主要改进了函数参数检查、函数头注释检查和函数返回值检查部分，并修复了几个编译错误。

## 版本要求
- llvm version 7.0.0
- clang version 7.0.0
- cmake version 3.4
- c++ version 7.3.0

## 编译
- 将src文件夹替换到CSA/nasac2018/CodingSpecChecker/src
- 进入到CSA/nasac2018/CodingSpecChecker目录下
```bash
mkdir build
cd build
cmake ..
make
```
编译将在src目录下生成code-spec-checker可执行文件

## 运行
测试文件在CSA/nasac2018/CodingSpecChecker/test目录下

testfile可以为单个文件，也可以为一个文件夹

### 函数参数检查
```bash
./src/code-spec-checker -no-error-handling-check -no-init-in-need-check -no-header-check -no-full-comment-check -no-naming-check ../test/testfile -b=/
```
函数参数检查测试文件目录为CSA/nasac2018/CodingSpecCheckertest/NASAC-testcase/2/

### 函数头注释检查
```bash
./src/code-spec-checker -no-error-handling-check -no-init-in-need-check -no-header-check -no-module-check -no-naming-check ../test/testfile -b=/
```
函数头注释检查测试文件目录为CSA/nasac2018/CodingSpecCheckertest/NASAC-testcase/5/

### 函数返回值检查
```bash
./src/code-spec-checker -no-full-comment-check -no-init-in-need-check -no-header-check -no-module-check -no-naming-check ../test/testfile -b=/
```
函数头注释检查测试文件目录为CSA/nasac2018/CodingSpecCheckertest/NASAC-testcase/3/

## 运行结果
![result](result.png)

## 注意事项
检查工具报出找不到头文件stddef.h

解决方案：将llvm安装目录下的lib文件夹复制到build目录下，或者将lib文件夹软链接到build目录下
