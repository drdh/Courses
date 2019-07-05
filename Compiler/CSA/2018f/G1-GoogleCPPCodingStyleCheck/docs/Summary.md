# 总结
[TOC]
## 实验过程中遇到的问题
### 编译环境问题
* 1.首先对于如何构建项目：调研到的两个方法，1是把我们写的checker加入到clang中，再差异编译clang来完成，2是根据助教的驱动，把clang的库调过来自己编译成一个可执行文件来解决，最后是采取第二种方案。
* 在根据助教的驱动和架构的过程中，在编译阶段，由于LLVM和Clang的版本不同，有些助教代码编译不过，因此我们统一装了Clang7.0.0版本。
* 即使版本相同，但是各组员在编译的过程中还是出现了It only works on my machine的情况（实际情况是会出现类似`llvm::sys::Process::FileDescriptorHasColors(int)': (.text+0x85b):
  undefined reference tosetupterm' /usr/lib/llvm-3.5/lib/libLLVMSupport.a(Process.o): In function  llvm::sys::Process::FileDescriptorHasColors(int)': (.text+0x87a):
  undefined reference totigetnum' /usr/lib/llvm-3.5/lib/libLLVMSupport.a(Process.o): In function llvm::sys::Process::FileDescriptorHasColors(int)': (.text+0x888):
  undefined reference toset_curterm' /usr/lib/llvm-3.5/lib/libLLVMSupport.a(Process.o): In function  llvm::sys::Process::FileDescriptorHasColors(int)': (.text+0x890):
  undefined reference todel_curterm'`的问题）。最后发现加上tinfo库就可以编译通过了（各成员要安装这个库），原因是llvm-config 没有为Terminfo库加上link选项。
* 之后各成员的编译问题解决

### 关于各部分交互进行的问题
* 从ClassChecker这边来说，我们分为了几部分Checkers：1.checkEndAnalysis 2.cxxConstrucotrDeclCheck 3.cxxDestructorDeclCheck 4.cxxMethodDeclCheck 5.baseClassCheck 6.friendCheck 7.fieldDeclCheck. 于是很明显的，我们不是按照规范条目来区分Check的，因此在比如继承规范的检查中，直接的理解是只用到baseClassCheck来检查基类继承的限定符，但实际上由于还要检查继承的虚函数等，于是这个检查还拆分了一部分到cxxMethodDeclCheck中。
* 比较幸运的是，我们找到了很好clang的类的文档，因此我们共享文档，很少会有一个人需要用到另一名成员的成果或者方法的情况，对于那些特殊情况，我们只需要在线上沟通即可。