#include "syscall.h"
#include "func.h"

int main()
{
    int i,childID;
    childID = Fork();
    if (childID == 0)
    {
        printStr("i am child. i am runing 'add'.\n");
        printStr("  ");
        Exec("add");
    }
    else
    {
        for (i=10000; i > 0; i--)
        {}
        printStr("i am parent. i finished after my child\n\n");
    }
}