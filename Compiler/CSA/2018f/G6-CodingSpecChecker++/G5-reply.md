# 第五组测评报告

## 程序主要功能
对NASAC原型竞赛的部分代码进行改进，主要改进了函数参数检查、函数头注释检查和函数返回值检查部分，并修复了几个编译错误。

## 运行环境
- llvm version 7.0.0
- clang version 7.0.0
- cmake version 3.10.2
- gcc version 7.3.0

## 测试方法
- 将src文件夹替换到CSA/nasac2018/CodingSpecChecker/src
- 进入到CSA/nasac2018/CodingSpecChecker目录下
```bash
mkdir build
cd build
cmake ..
make
```
编译将在src目录下生成code-spec-checker可执行文件

```bash
./src/code-spec-checker -no-error-handling-check -no-init-in-need-check -no-header-check -no-naming-check -no-full-comment-check ../test/NASAC-testcase/2/2_2.c -b=/
```

测试代码(其一):
```c
#include <stdio.h>
#include <assert.h>

int checkPrimeNumber(int number);
int main()
{
    int start, end, number, flag;

    printf("Enter two positive integers: ");
    if (scanf("%d %d", &start, &end) != 2)
    {   
      printf("input error.\n"); 
      return 0;
    }
    printf("Prime numbers between %d and %d are: ", start, end);

    for(number=start+1; number<end; ++number)
    {
        // i is a prime number, flag will be equal to 1
        flag = checkPrimeNumber(number);

        if(flag == 1)
            printf("%d ",number);
		else
            printf("none ");
    }
    return 0;
}

// user-defined function to check prime number
// Input: input number be checked
// Output: 0:successful; 1:failure
int checkPrimeNumber(int number)
{
    int divisor, flag = 1;
    
    assert(number > 0);  //bad: the callee should not verify parameter
    for(divisor=2; divisor <= number/2; ++divisor)
    {
        if (number%divisor == 0)
        {
            flag =0;
            break;
        }
    }
    return flag;
}
```

Warning:
```
[ParseCommandLineOptions]: No compilations database is found, running without flags
[ParseCommandLineOptions]: Receive source file list from command line
/home/ubuntu/nasac2018/CodingSpecChecker/build/../test/NASAC-testcase/2/2_2.c:20:16: warning: pass unchecked parameter
        flag = checkPrimeNumber(number);
               ^~~~~~~~~~~~~~~~~~~~~~~~
/home/ubuntu/nasac2018/CodingSpecChecker/build/../test/NASAC-testcase/2/2_2.c:37:19: warning: reduntant parameter check
    assert(number > 0);  //bad: the callee should not verify parameter
           ~~~~~~~^~~
/usr/include/assert.h:109:11: note: expanded from macro 'assert'
      if (expr)                                                         \
          ^~~~
2 warnings generated.
```

测试结果：程序运行行为符合预期(其他样例也符合预期)