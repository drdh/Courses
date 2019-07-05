### 功能

BufferCheck用于检查执行strcpy和memcpy等内存拷贝函数时，目的内存的大小是否小于源内存的大小。



### 如何使用

1. 修改`checkers.td`

   在clang的项目目录下（通常位于`llvm/tools/`下），在`include/clang/StaticAnalyzer/Checkers`

   `/Checkers.td`中的`let ParentPackage = CoreAlpha in`块中中添加如下内容：

   ```c
   def BufferChecker : Checker<"BufferSize">,
     	HelpText<"Check if the src buffer size is not larger than the dst buffer size">,
     	DescFile<"BufferChecker.cpp">;
   ```

2. 添加BufferChecker的源文件

   将源文件`BufferChecker.cpp`放到`lib/StaticAnalyzer/Checkers`目录下，并在该目录中的`CMakeList.txt`中添加`BufferChecker.cpp`。

3. 编译clang

4. 测试

   执行`clang -cc1 -analyzer-checker-help`列出所有checker就会发现定义的BufferChecker已经在其中：

   ![](/home/linan/Documents/compiler/CSA/2018f/G4-MemoryCheck/BufferCheck/images/checkers.png)

   调用该checker，进行检查，如果存在错误，则会产生如下warning：

   ![](/home/linan/Documents/compiler/CSA/2018f/G4-MemoryCheck/BufferCheck/images/check_result.png)



### 其他

还看到一种将自己定义的checker编译成动态库的方法，这样就不用每次修改一堆文件并且重新编译clang了，但是我遇到一点问题，就没有采用。