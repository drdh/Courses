# 7.9

```asm
        .file   "ex7-9.c"
        .text
        .globl  main
        .type   main, @function
main:
.LFB0:
        pushl   %ebp			
        movl    %esp, %ebp
        subl    $16, %esp
        jmp     .L2
.L5:
        movl    -4(%ebp), %eax	;取出j
        movl    %eax, -8(%ebp)	;i=j
.L2:	
        cmpl    $0, -8(%ebp)	;比较i与0
        jne     .L3				;i!=0 则判断下一个必要的条件
        cmpl    $0, -4(%ebp)	;比较j与0
        jne     .L3				;j!=0 则判断下一个必要的条件
        cmpl    $0, -4(%ebp)	;比较j与0
        je      .L4				;j为0,则直接跳转到结束
.L3:
        cmpl    $5, -4(%ebp)	;比较j与5
        jg      .L5				; 此时第一个必要条件已经满足了，只要第二个条件满足就执行
.L4:
        movl    $0, %eax	;结束程序
        leave
        ret
.LFE0:
        .size   main, .-main
        .ident  "GCC: (Ubuntu 5.4.0-6ubuntu1~16.04.5) 5.4.0 20160609"
```

其判断分支主要在`.L2 .L3`， 其中的`je      .L4`表示`&&`前面的条件不满足，就无需计算后面的，即为短路计算。



# 7.10

(a) 有多个标号是因为程序会从不同的地方跳转到相同的代码段，每个跳转的程序段都会指定一个标号。

(b) `.L1`表示直接结束该程序，并且返回，相当于`return`，本程序中没有是因为没有显式的`return`语句

(c) 

`.L3`  设置`return`为0并且返回。

`.L4` 将`i`与0比较，如果不为0就跳转

`.L5` `j++`



# 7.17

(a) 
$$
A[i_i][i_2]...[i_k] \\
addr =base+i_k\times w +i_{k-1}\times n_k \times w +...+i_1\times n_2 \times n_3\times ... \times n_k\times w     \\
$$
同时，定义$s_m=n_k\times n_{k-1}\times ... \times n_{m+1}$且$s_k=1$

则
$$
addr=base+i_k \times s_k \times w+...+i_1\times s_1 \times w
$$


(b)

```
L--> id {L.place=id.place;
		 L.m=newTemp();
		 emit(L.m,'=',1)
		 emit(L.offset,'=',L.offset,'+',invariant(id));}
		 
L-->L1[E] {L.place=L1.place;
		   L1.m=newTemp();
		   emit(L1.m,'=',L.m,'*',limit(L1.array))
		   t=newTemp();
		   emit(t,'=',E.place,'*',L1.m)
		   t1=newTemp();
		   emit(t1,'=',t,'*',width(L1.array))
		   L.offset=newTemp();
		   emit(L.offset,'=',L1.offset,'+',t1)}
```



