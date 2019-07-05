#include "syscall.h"
#include "func.h"

int main()
{    
    int i, j;
    int childID;
    childID = Fork();
    if (childID == 0)
    {
        for (i = 10000; i > 0; i--)
        {}
        printStr("2. i am child. i am runing 'add'. please wait...\n");
        printStr("   ");
        Exec("add");
        printStr("ERROR: child finished\n"); //not reached
    }
    else
    {
        printStr("\n1. i am parent. i am waitting my childID=");
        printNum(childID);
        printStr("\n");

        Join(childID);
        for (i = 0; i < 100; i++)
            ;
        printStr("3. parent finished\n\n");
    }
}