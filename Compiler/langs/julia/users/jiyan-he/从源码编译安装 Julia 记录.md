# 从源码编译安装 Julia 记录

何纪言

## 准备编译环境

1. 下载 ubuntu 16.04.5 环境并解压

```shell
wget http://mirrors.ustc.edu.cn/ubuntu-cdimage/ubuntu-base/releases/16.04.5/release/ubuntu-base-16.04.5-base-amd64.tar.gz

sudo tar xpzf ../ubuntu-base-16.04.5-base-amd64.tar.gz
```

2. 准备 rootfs 环境

```shell
sudo cp -v /etc/resolv.conf etc/resolv.conf # DNS Server
sudo wget -O etc/apt/sources.list https://mirrors.ustc.edu.cn/repogen/conf/ubuntu-http-4-xenial # 换源加速

# 一些必要的环境
sudo mount -t proc proc proc/
sudo mount -t sysfs sys sys/
sudo mount -o bind /dev dev/

# Julia 编译的最后阶段需要用到 pty
sudo mount -t devpts none "dev/pts" -o ptmxmode=0666,newinstance
sudo ln -fs "pts/ptmx" "dev/ptmx"
```

3. 进入编译环境

```shell
sudo chroot ubuntu /bin/su
```

4. 系统初始化

```shell
apt update
apt install sudo vim git curl # 装一些常用的软件
```

## 安装依赖

之所以选 ubuntu 就是因为绝大多数依赖的版本都符合要求，如果依赖的程序版本太低需要自己编译安装：

```shell
sudo apt-get install build-essential libatomic1 python gfortran perl wget m4 cmake pkg-config
```

## 获取 Julia 源代码

（仓库目前约 189M）

```shell
cd /opt
git clone https://github.com/JuliaLang/julia.git
```

切换到一个稳定的版本：

```shell
cd julia
git checkout v1.0.2 # 选择一个发布版本，使用仓库中的最新版不一定能编译成功
```

## 开始编译

进入刚才的源代码目录，即可开始编译：

```shell
make -j <cpu_num>
```

编译过程中会自动下载很多第三方库（可选支持 Intel MKL）：

- **LLVM** (3.9 + patches) — compiler infrastructure (see [note below](https://github.com/JuliaLang/julia/tree/v1.0.2#llvm)).
- **FemtoLisp** — packaged with Julia source, and used to implement the compiler front-end.
- **libuv** (custom fork) — portable, high-performance event-based I/O library.
- **OpenLibm** — portable libm library containing elementary math functions.
- **DSFMT** — fast Mersenne Twister pseudorandom number generator library.
- **OpenBLAS** — fast, open, and maintained [basic linear algebra subprograms (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library, based on [Kazushige Goto's](https://en.wikipedia.org/wiki/Kazushige_Goto)famous [GotoBLAS](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2) (see [note below](https://github.com/JuliaLang/julia/tree/v1.0.2#blas-and-lapack)).
- **LAPACK** (>= 3.5) — library of linear algebra routines for solving systems of simultaneous linear equations, least-squares solutions of linear systems of equations, eigenvalue problems, and singular value problems.
- **MKL** (optional) – OpenBLAS and LAPACK may be replaced by Intel's MKL library.
- **SuiteSparse** (>= 4.1) — library of linear algebra routines for sparse matrices (see [note below](https://github.com/JuliaLang/julia/tree/v1.0.2#suitesparse)).
- **PCRE** (>= 10.00) — Perl-compatible regular expressions library.
- **GMP** (>= 5.0) — GNU multiple precision arithmetic library, needed for `BigInt` support.
- **MPFR** (>= 4.0) — GNU multiple precision floating point library, needed for arbitrary precision floating point (`BigFloat`) support.
- **libgit2** (>= 0.23) — Git linkable library, used by Julia's package manager.
- **curl** (>= 7.50) — libcurl provides download and proxy support for Julia's package manager.
- **libssh2** (>= 1.7) — library for SSH transport, used by libgit2 for packages with SSH remotes.
- **mbedtls** (>= 2.2) — library used for cryptography and transport layer security, used by libssh2
- **utf8proc** (>= 2.1) — a library for processing UTF-8 encoded Unicode strings.
- **libosxunwind** — clone of [libunwind](http://www.nongnu.org/libunwind), a library that determines the call-chain of a program.

编译的前面阶段是对依赖库的编译，对这些库进行编译需要花较长的时间。

后一阶段是 Julia bootstrap，对自身进行编译，这时候会看到正在编译大量的 `.jl` 文件。

编译的最后还会输出一些 Julia 模块的的编译时间，比如：

```
Sysimage built. Summary:
Total ───────  91.567419 seconds
Base: ───────  28.806127 seconds 31.4589%
Stdlibs: ────  62.759492 seconds 68.5391%
```

正确编译的话，最后会看到以下输出：

```shell
    JULIA usr/lib/julia/sys-o.a
Generating precompile statements... 1090 generated in  83.666091 seconds (overhead  52.346493 seconds)
    LINK usr/lib/julia/sys.so
```

这代表编译正确结束。

## 测试运行

编译好的目标二进制文件就在源代码目录下：

```
./julia
```

即可运行。

源代码中还提供了测试用的程序，可以用：

```shell
make test
```

测试运行，如：

```
Test                           (Worker) | Time (s) | GC (s) | GC % | Alloc (MB) | RSS (MB)
triplequote                        (18) |     1.28 |   0.03 |  2.2 |     132.25 |   189.40
unicode/utf8                       (11) |     1.72 |   0.10 |  6.1 |     135.73 |   189.40
intrinsics                         (19) |     1.96 |   0.07 |  3.8 |     156.71 |   189.40
compiler/validation                 (4) |     2.29 |   0.03 |  1.4 |     151.79 |   189.40
strings/search                      (7) |     3.83 |   0.17 |  4.3 |     245.31 |   189.40
```

## 遇到的问题

- 编译时出错 `g++: internal compiler error: Killed (program cc1plus)`。

  主要是 LLVM 编译时内存不够，`make -j 31` 内存峰值约 10G，建议根据内存情况调小 make -j 的线程数量，否则使用 swap 会更慢。

- 类似于 `fatal error:(io-error "file: could not open \"flisp.boot\"")` 的错误。

  第一个原因可能是之前的残留文件没有删除干净。

  除了 `make clean` 以外，还需要看看 `git status --ignored` 有没有残留的中间文件（`*.o, *.a`），删除之后再重新编译。

  ```shell
  ./src -name '*.o' -type f -print -exec rm -rf {} \;
  ./src -name '*.a' -type f -print -exec rm -rf {} \;
  ```

- `./usr/tools/llvm-config` 报错：`Error: cannot open shared object file`

  这个问题花了我很长时间解决，解决方案是添加了上面的 `sudo mount -t proc proc proc/` ，有人在 [PythonAnywhere](https://www.pythonanywhere.com/) 上面编译遇到同样的问题，猜想这个平台也是使用了 chroot 但是没有提供 proc 的挂载（因为存在安全问题）。

  参考：

  - https://groups.google.com/forum/#!topic/julia-users/tJphAtWb854

- 编译快结束时：`Generating precompile statements...ERROR: LoadError: Failed to open PTY master`

  也是 chroot 的问题，用独立的电脑或者虚拟机不会遇到。

  解决方案是添加虚拟的 pts 设备。