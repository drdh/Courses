/*
 *lab2 By LiJiaWei
 *
 *该程序仅能测试exec，不能测试fork和join
 *但它能够正确执行的前提是：系统调用Fork能正确执行，其次是Exec正确.
 *它和Join没有任何关系
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
    int i = 100000, j = 10000, t = 0;
    int childID;
    childID = Fork();
    if (childID == 0)
    {
        printStr("\n1. i am child. i am runing 'add'.\n");
        printStr("   ");
        Exec("add");
        printStr("ERROR: when execute add. child finished\n"); //not reached
    }
    else
    {
        for (; i > 0; i--)
        {
            for (; j > 0; j--)
            {
                t = i + j;
            }
        }
        printStr("2. i am parent. i finished after my child\n\n");
    }
}