/*
 *lab2 By LiJiaWei
 *
 *该程序仅能测试join，不能测试fork和exec
 *但它能够正确执行的前提是：Fork和Exec这两个调用能正确执行，其次是Join正确
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
    int i = 100000, j = 50000, t = 0;
    int childID;
    childID = Fork();
    if (childID == 0)
    {
        for (; i > 0; i--)
        {
            for (; j > 0; j--)
            {
                t = i + j;
            }
        }
        printStr("2. i am child. i am runing 'add'. please wait...\n");
        printStr("   ");
        Exec("add");
        printStr("ERROR: child finished\n"); //not reached
    }
    else
    {
        printStr("\n1. i am parent. i am waitting my childID=");
        printDigit(childID);
        printStr("\n");

        Join(childID);
        for (i = 0; i < 100; i++)
            ;
        printStr("3. parent finished\n\n");
    }
}