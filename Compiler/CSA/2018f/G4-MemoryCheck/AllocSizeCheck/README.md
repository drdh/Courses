## 功能

AllocSizeCheck是一个检查分配空间的Checker，在调用malloc函数时检查分配空间是否为0，如果为0则会报一个warning。

## 使用说明

1. 修改`checkers.td`

	在`llvm/tools/include/clang/StaticAnalyzer/Checkers`目录下在文件`Checkers.td`中添加以下代码：
	```c
	let ParentPackage = CoreAlpha in {
		...
		//从这里开始添加
		def AllocSizeChecker : Checker<"AllocSize">,
  			HelpText<"Check if the Malloc size is zero">,
  			DescFile<"AllocSizeChecker.cpp">;
  		//结束
  		...
  	}
	```
	
2. 将AllocSizeCheck.cpp加到目录`llvm/tools/lib/StaticAnalyzer/Checkers`下，并修改该目录下的CMakeList.txt，向其中添加AllocSizeCheck.cpp。

3. 重新编译clang

4. 测试

	执行`clang -cc1 -analyzer-checker-help`列出所有checker就会发现定义的AllocSizeChecker已经在其中：

	![](images/img1.png)
	
	使用clang编译test目录下的Malloctest1.c，会产生如下warning：

	![](images/test.png)


