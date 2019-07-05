# Lab 2-3



## 项目简介

~~Lab 2-3 是以中国共产党最高纲领为指导，中国科学技术大学校训 “红专并进，理实交融” 为宗旨，传承 “编译原理与技术(H)” 的实验思想，在张老师和两位优秀助教的引领下，由第 5 小组设计的实验项目。~~

在课程实验项目 Lab 2-2 后，我们得到了 C1 语言生成的 LLVM IR 代码，但这份代码是未经优化的原始版本，因此我们队希望做一些基于 LLVM IR 代码的优化。

在各种优化计划中，我们最后选择了实现**读取合并**。读取合并优化通过删除冗余的读取指令，减少了机器码中潜在的 MEM I/O 次数，从而加速代码的运行。如果程序本身含有很多次迭代，那么通常这种优化带来的效果更加显著。



## 使用方法

### 方法一：使用编译好的优化模块

* 此方法需要你拥有一份已经编译好的 LLVM 6.X 或者 7.X （手动编译或者通过包管理器下载均可）。

* 将 `run.sh` 和 `LoadOpt.so` 放置在同一目录下，在该目录下执行如下操作：

  ```
  $ ./run.sh /path-to/src
  ```

  `src` 的扩展名必须为以下选项之一：`.c`、 `.ll` 和 `.bc`。

* 生成的优化后的 LLVM IR 代码文件会放在源文件同目录下，名为 `src-opt.ll`。

​	

### 方法二：自行编译优化模块的源代码

* 你必须有一份 LLVM 6.X 或者 7.X 的源代码才能使用此方法。

* 将 `LoadOpt` 目录放置到 `/path-to-llvm-src/lib/Transforms/` 目录下。

* 修改 `/path-to-llvm-src/lib/Transforms/CMakeLists.txt` ，追加一行：

  ```
  add_subdirectory(LoadOpt)
  ```

* 新建 `/path-to-llvm-build` ，进入该目录下执行如下操作：

  ```
  make install
  ```

  * 注意：如果你之前已经编译过，那么如此执行便可；否则，你需要先执行合适的 `cmake` 操作才能再执行 `make` 操作，第一次编译 LLVM 会耗费数小时，但重新编译 LLVM 的耗时通常在一分钟之内。

* 编译完成后，进入 `/path-to-llvm-build/bin` 下，执行：

  ```
  opt -load ../lib/LoadOpt.so -labopt - disable-output /path-to/src.bc 2> /path-to/dest.ll
  ```

  * `src.bc` 必须是 LLVM bitcode 文件，如果你手中的文件是 C 文件，你需要先执行

    ```
    clang -S -emit-llvm /path-to/src.c -c -o /path-to/src.bc
    ```

  * 优化后的 LLVM IR 文件为 `/path-to/dest.ll` 。
