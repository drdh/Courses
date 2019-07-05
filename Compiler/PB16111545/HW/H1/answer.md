## 编译测试

#### 说明

源码位置`/c/test.c`

##### 编译选项详解

###### reference

[GCC 参数详解](http://www.cppblog.com/SEMAN/archive/2005/11/30/1440.html)

`-c` 只激活预处理,编译,和汇编,也就是他只把程序做成obj文件 

`-S` 只激活预处理和编译，就是指把文件编译成为汇编代码

`-E`只激活预处理,这个不生成文件,需要把它重定向到一个输出文件里面

`-o`　制定目标名称

`-m32` `-m64` 　指定32或64位

##### 预处理中的变换

- 头文件的展开
- 对数据类型的`typedef`
- 定义各类结构体
- `extern`变量、函数的说明

##### 汇编代码注释

`gcc test.c -m32 -S`

```assembly
	.file	"test.c"
	.text
	.section	.rodata
.LC0:
	.string	"%d\n"
	.text
	.globl	main 
	.type	main, @function
main:
.LFB0:
	.cfi_startproc 
	leal	4(%esp), %ecx　　;ecx=&esp+4
	.cfi_def_cfa 1, 0
	andl	$-16, %esp	;esp+=-16
	pushl	-4(%ecx)	;ecx-4 -> esp
	pushl	%ebp	;ebp ->esp
	.cfi_escape 0x10,0x5,0x2,0x75,0
	movl	%esp, %ebp 	;本指令与上一条指令建立函数的堆栈框架
	pushl	%ebx
	pushl	%ecx
	.cfi_escape 0xf,0x3,0x75,0x78,0x6
	.cfi_escape 0x10,0x3,0x2,0x75,0x7c
	subl	$16, %esp
	call	__x86.get_pc_thunk.bx
	addl	$_GLOBAL_OFFSET_TABLE_, %ebx
	movl	$0, -12(%ebp)
	jmp	.L2
.L3:
	subl	$8, %esp
	pushl	-12(%ebp)
	leal	.LC0@GOTOFF(%ebx), %eax
	pushl	%eax
	call	printf@PLT ;调用printf函数
	addl	$16, %esp
	addl	$1, -12(%ebp)
.L2:
	cmpl	$4, -12(%ebp) ;计数器，为了判断是否循环结束
	jle	.L3 ;跳转到.L3
	movl	$0, %eax
	leal	-8(%ebp), %esp
	popl	%ecx
	.cfi_restore 1
	.cfi_def_cfa 1, 0
	popl	%ebx
	.cfi_restore 3
	popl	%ebp
	.cfi_restore 5
	leal	-4(%ecx), %esp
	.cfi_def_cfa 4, 4
	ret ;return
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.text.__x86.get_pc_thunk.bx,"axG",@progbits,__x86.get_pc_thunk.bx,comdat
	.globl	__x86.get_pc_thunk.bx
	.hidden	__x86.get_pc_thunk.bx
	.type	__x86.get_pc_thunk.bx, @function
__x86.get_pc_thunk.bx:
.LFB1:
	.cfi_startproc
	movl	(%esp), %ebx
	ret
	.cfi_endproc
.LFE1:
	.ident	"GCC: (Debian 7.3.0-19) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
```

`gcc test.c -m64 -S`

```assembly
	.file	"test.c"
	.text
	.section	.rodata
.LC0:
	.string	"%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	jmp	.L2
.L3:
	movl	-4(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rdi
	movl	$0, %eax
	call	printf@PLT    ;调用printf函数
	addl	$1, -4(%rbp)
.L2:
	cmpl	$4, -4(%rbp) ;计数器，为了判断是否循环结束
	jle	.L3	 ;跳转到.L3
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret ;return
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.ident	"GCC: (Debian 7.3.0-19) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
```

##### 汇编代码说明

1. 32位中都是`addl`后面加上`l`表示32位，而64位指令中部分是`addq`表示64位
2. 寄存器，32位是`%eax`等等`%e`开头的，而64位是以`%rax`以`%r`开头的居多
3. 基本结构类似：
   - 开头定义诸多元数据
   - `.LC0`开始`main`函数的定义
   - `.LEB0`初始化一些值，32位中构造堆栈，64位中并未显式构造
   - `L3`均是循环内容
   - `.L2`是跳出循环后的结束部分
   - `.LFE0`是最后的信息，在32位中同时还存在`.LFB1``LFE1`

##### Reference

[几种基本汇编指令详解](http://blog.luoyuanhang.com/2015/07/07/%E5%87%A0%E7%A7%8D%E5%9F%BA%E6%9C%AC%E6%B1%87%E7%BC%96%E6%8C%87%E4%BB%A4%E8%AF%A6%E8%A7%A3/)

[AT&T 汇编指令](http://ted.is-programmer.com/posts/5262.html)

## lexer 的说明

位置`HW/H1/lexer`

### 主要考虑

- 能够读入空格，使用`scanf("%[^\n],s")`
- 记录记号的起止位置，使用一个数组
- 识别，按照NFA图即可

### 问题

`<==> a<=b`的识别过程

首先将它全部读入字符串，然后从头开始分析。

- 读到`<`继续到`=`形成`LE`
- `=`直接是`EQ`
- 读到`>`下一个是空格，形成`GT`
- 读到`a`，下一个是`<`，表示为`OT`
- 读到`<`与`=`形成`LE`
- 最后为`b`与`\0`为`OT`结束