/*
 *lab2 By LiJiaWei
 *
 *该程序仅是占用CPU  可用于测试进程调度
*/
#include "syscall.h"

//向控制台输出字符串。
void printStr(char *str)
{
    int i = 0;
    while (str[i] != '\0')
        i++;
    Write(str, i, 1);
}

int main()
{
    int i = 1000, j = 1000, total = 0;
    for (; i > 0; i--)
    {
        for (; j > 0; j--)
        {
            total = total + j;
        }
    }
    printStr("i am testshell\n");
}