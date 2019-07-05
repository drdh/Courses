#include "syscall.h"
#include "func.h"

int main(){
    int fd1;
    int fd2;
    int result;
    char str[256];
    Create("hello");
    fd1=Open("h.txt");
    fd2=Open("hello");
    Write("1111111111111111111111111111111",16,fd1);
    Close(fd1);
    fd1=Open("h.txt");
    Read(str,256,fd1);
    Write(str,256,fd2); 
    Close(fd1);
    Close(fd2);
    Print();
    return 0;
}