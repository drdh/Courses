# G2组实验评测

## 主要功能
对程序中的多项式相关的计算进行模式识别及优化，使得生成的IR代码能够直接基于最终的多项式进行运算，省去了中间可能出现的较为复杂的计算方法。

## 测试方法
通过对由clang直接生成的llvm IR以及优化后的IR进行比较，人工分析，直观了解优化的效率。之后通过比较使用 clang 产生的可执行文件与由中间代码生成的结果进行结果正误的比较以及运行时间的比较。

## 测试通过样例

1. 测试样例一
    ```c
    include <stdio.h>
    
    int fun(int x)
    {
        int a = (x+1)*(x+1)*(x+1)*(x+1);
        int b = x*x*x*x+4*x*x*x+6*x*x+4*x+8;
        int sum = a-b;
        return sum;
    }
    
    int main()
    {
        int sum = 0, x;
        scanf("%d",&x);
        int y =fun(x);
        printf("%d",y);
        return 0;
    }
    
    ```

2. 测试样例二（以下仅改变fun函数中的取值）
    ```c
    include <stdio.h>
    
    int fun(int x)
    {
        int a = (x+1)*(x+2);
        int b = (x+3)*(x+5);
        int c = (x+4)*x;
        int d = (x+2)*(x+5);
        int sum = a*b-c*d;
        return sum;
    
    }
    int main()
    {
        int sum = 0, x;
        scanf("%d",&x);
        int y =fun(x);
        printf("%d",y);
        return 0;
    }
    
    ```


## 测试反例

```c
#include <stdio.h>
int fun(int x)
{
    int a=(x+1)*(x+1);
    int b=x*x+2*x+3;
    int sum = a*3-b;
    return sum;
}

int main()
{
    int sum = 0, x;
    scanf("%d",&x);
    int y =fun(x);
    printf("%d",y);
    return 0;
}
```

上述测试样例当出现减法操作时，就会使得优化无法进行

## 不足之处

1. 当前能支持的运算操作较少
2. 当前代码优化效率不明显

## 可改进的一些建议

1. 使用快速幂优化整系数多项式的计算效率
2. 可以考虑basic block间变量依赖关系的分析，进而实现进一步对计算过程的优化。
