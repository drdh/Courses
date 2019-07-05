# 追踪linux内核启动过程
### 搭建内核调试环境
环境 ubuntu 17.10

采用的是qemu加上gdb的方式来追踪，期间碰到了很多坑，具体的太多记不太清，下面先写可用的环境搭建方式以及追踪内核记录。


安装 qemu
```
sudo apt install qemu
```
下载内核
```
wget http://www.kernel.org/pub/linux/kernel/v4.x/linux-4.14.tar.xz
```
解压
```
xz -d linux-4.14.tar.xz
tar -xvf linux-4.14.tar
```
编译内核
```
make menuconfig
```
此时需要进行一些设置，不然无法使用断点
```
--> kerel hacking
--> compile-time checks and compiler option
--> compile the kernel with debug info
```
使用调试信息
```
--> processor type and features
--> randomize the address of the kernel image
```
取消这一项，不然设置的断点不起作用

之后再继续编译
```
make
```
需要的时间非常长

利用本系统的initrd文件
```
cp /boot/initrd.img-4.13.0-37-generic .
```
启动qemu
```
qemu-system-x86_64 -s -kernel arch/x86/boot/bzImage -initrd initrd.img-4.13.0-37-generic  -S
```
-s表示 -gdb tcp::1234
-S表示 加载后立即暂停，否则会一直执行

打开一个新的终端
```
gdb vmlinux
```
再连上远程系统
```
target remote:1234
```

但是使用这种方式只能运行到rest_init()之后出现挂载失败的情况，此时可借助[BusyBox](https://www.busybox.net/)构建极简的initramfs
```
cd busybox-1.28.0
make menuconfig
```

```
Settings -->
    [*]Build static binary(no shared libs)
```
静态编译版，可以不依赖动态链接库，独立运行，方便构建initamfs
```
make
make install
```
再创建initramfs，其中包含BusyBox可执行程序、必要的设备文件、启动脚本init.
init脚本只挂载了虚拟文件系统procfs和sysfs，没有挂载磁盘根文件系统，所有的调试都只在内存中进行，不落磁盘。

```
$ mkdir initramfs
$ cd initramfs
$ cp ../_install/* -rf ./
$ mkdir dev proc sys
$ sudo cp -a /dev/{null,console,tty,tty1,tty2,tty3,tty4} dev/
$ rm linuxrc
$ vim init
$ chmod a+x init
$ ls
$ bin   dev  init  proc  sbin  sys   usr
```

init文件的内容
```
#!/bin/busybox sh         
mount -t proc none /proc  
mount -t sysfs none /sys  

exec /sbin/init
```
打包initramfs
```
find . -print0 | cpio --null -ov --format=newc |gzip -9 > ../initramfs.cpio.gz
```
再启动内核方式为
```
qemu-system-x86_64 -s -kernel arch/x86/boot/bzImage -initrd ../initramfs.cpio.gz -S
````


### 追踪过程
使用atom对照看init/main.c源码
```
b start_kernel
c
```
之后单步执行，可以观察到一系列函数的运行,直到最后一个rest_init()后qemu进入console模式
期间观察到的有，trap_init()中断向量的初始化，mm_init()内存管理的初始化，sched_init()模块调度的初始化
