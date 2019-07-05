Compiler Directive的基本格式如下：

```cpp
#pragma omp directive-name [clause[ [,] clause]...]
```

其中`[]`表示可选，每个Compiler Directive作用于其后的语句（C++中“{}”括起来部分是一个复合语句）。

directive-name可以为：（共11个，只有前4个有可选的clause）

```cpp
parallel, for, sections, single, atomic, barrier, critical, flush, master, ordered, threadprivate
```

clause（子句）相当于是Directive的修饰，定义一些Directive的参数什么的。clause可以为：（共13个）

```
copyin(variable-list), copyprivate(variable-list), default(shared | none), firstprivate(variable-list), if(expression), lastprivate(variable-list), nowait, num_threads(num), ordered, private(variable-list), reduction(operation: variable-list), schedule(type[,size]), shared(variable-list)
```

[OpenMP编程总结表](https://www.cnblogs.com/liangliangh/p/3565136.html)

[OpenMP共享内存并行编程详解](https://www.cnblogs.com/liangliangh/p/3565234.html)