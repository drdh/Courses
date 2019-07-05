#include "syscall.h"
#include "func.h"

int main()
{
    int i;
    int childID;
    int a=1;
    childID = Fork();
    if (childID == 0)
    {
        a = 2;
        //for (i = 0; i < 10000; i++); //do nothing
        printStr("\ni am child. a = ");
        printNum(a);
    }
    else
    {
        for (i = 0; i < 10000; i++);
        printStr("\ni am parent. i finished after my child. a = ");
        printNum(a);
    }
}