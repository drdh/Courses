* 针对 llibevent 项目进行检查时，虽然已使用 cmake 正确配置且生成了 compile_commands.json 并且可以构建，但检查工具报出找不到头文件 stddef.h。

  向命令行选项中加入 -v 看到其搜索路径如下：

  >  /home/shengliangd/CCheck/libevent/cmake-build-Debug/include
  >  /home/shengliangd/CCheck/libevent/include
  >  /home/shengliangd/CCheck/libevent/compat
  >  /home/shengliangd/CCheck/libevent
  >  /usr/local/include
  >  /home/shengliangd/CCheck/llvm-work/nasac/2018/CodingSpecChecker/cmake-build-Debug/src/../lib/clang/7.0.0/include
  >  /usr/include/x86_64-linux-gnu
  >  /usr/include

  注意到倒数第三项，可以看出 clang 是希望相关头文件放在二进制文件所在目录的相对路径 `../lib/clang/7.0.0/include` 的。

  **临时解决方案：**在我的系统中（Debian sid），相关文件在 `/usr/local/lib/clang/7.0.0/include` 下。软链接 `/usr/local/lib` 到 `src/../`，就没有问题了。

* FullCommentCheck 给出了没有名字的参数报错，待解决。

  assign to: DSL