# 回答问题

## a)

`sizeof(a)`的值为0, 因为这里计算的是数组a所占的字节数，但是由于a的第一维为0, 所以数组所占的字节数为0

## b)

`a[0][0]=4`

活动记录

```
[%ebp ]
[	  ]
[     ]
[     ]
[  i  ](&a)
[  j  ]
```

*[reply] 不是4哟,活动记录也不对哟*

## c)

` .cfi_startproc`  

> 用在每个函数的开始，用于初始化一些内部数据结构，与`cfi_endproc`配合使用



`.cfi_endproc`  

>  在函数结束的时候使用与`.cfi_startproc`相配套使用



`.cfi_def_cfa 1, 0`  

> `.cfi_def_cfa REGISTER, OFFSET`
>
> 定义一个计算`CFA`的规则，也就是从REGISTER处获得地址，然后加上OFFSET



`.cfi_escape 0x10,0x5,0x2,0x75,0`

> 应该是添加某种信息



` .cfi_restore 5`

> 表示这之后寄存器的值与初始化(`.cfi_startproc`)之前的值一样



# 设计问题

>OS: Arch Linux
>
>Kernel: x86_64 Linux 4.19.4-arch1-1-ARCH
>
>gcc version 8.2.1 20180831 (GCC)

## 1.

###  Q:

下面的Ｃ语言程序的结构大小都是24, 解释为什么

```c
#include<stdio.h>

typedef struct a{
    short d1_1;
    int d2;
    long d3;
    short d1_2;
}a;

typedef struct b{
    short d1;
    long d3;
    int d2;
}b;

typedef struct c{
    short d1_1,d1_2,d2_3,d1_4;
    long d3;
    int d2_1,d2_2;
}c;


int main(){
    printf("short=%d,int=%d,long=%d\n",sizeof(short),sizeof(int),sizeof(long));
    printf("a=%d,b=%d,c=%d\n",sizeof(a),sizeof(b),sizeof(c));
}
```

### A:

考虑现代机器的对齐方式。

## 2. 

### Q:

下列C语言代码运行时输出的`i`值为`123`为什么？

```c
#include<stdio.h>
int *p;
void func1(){
    int k;
    p=&k;
}

void func2(){
    int i;
    printf("%p,%p\n",p,&i);
    printf("%d\n",i);
}

int main(){
    func1();
     *p=123;
    func2();
}
```

### A:

这是因为Ｃ语言采用栈式分配内存，所以`func1`中的`k`与`func2`中的`i`所分配的内存一样。在`func1`中，全局变量指向`k(i)`所分配的空间。