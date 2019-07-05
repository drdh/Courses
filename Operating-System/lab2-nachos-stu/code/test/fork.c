/*
 *lab2 By LiJiaWei
 *
 *该程序仅能测试fork，不能测试exec和join，且和Exec Join没有关系
*/

#include "syscall.h"

void writeDigitR(int num, int size, int id)
{
    if (size == 1)
    {
        int t = num + '0';
        Write((char *)(&t), 1, id);
    }
    else
    {
        writeDigitR(num / 10, size - 1, id);
        writeDigitR(num % 10, 1, id);
    }
}
int getLength(int num)
{
    int i = 0, n = num;
    if (n == 0)
        return 1;
    while (n > 0)
    {
        n = n / 10;
        i++;
    }
    return i;
}

void printDigit(int num)
{
    writeDigitR(num, getLength(num), 1);
}

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
    int i;
    char *var = "parent";
    int childID;

    printStr("\n1. init var = \"");
    printStr(var);
    printStr("\"\n");

    childID = Fork();
    if (childID == 0)
    {
        var = "child";
        printStr("2. i am child. i change var = \"");
        printStr(var);
        printStr("\"\n");
    }
    else
    {
        for (i = 0; i < 30000; i++)
            ; //do nothing
        printStr("3. i am parent. my var = \"");
        printStr(var);
        printStr("\"\n\n");
    }
}