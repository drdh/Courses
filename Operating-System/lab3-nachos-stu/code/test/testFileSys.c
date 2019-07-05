#include "syscall.h"
#include "func.h"

int main(){
    int i;
    int fd1;
    int fd2;
    char str[512];

    Create("hello");
    fd1=Open("prince.txt");
    fd2=Open("hello");

    //Read 512B from fd1, Write 5120B to fd2
    Read(str,512,fd1);
    for(i=0;i<10;i++)
    Write(str,512,fd2);
    Close(fd1);
    Close(fd2);

    //Print file system state
    Print();
    return 0;
}
