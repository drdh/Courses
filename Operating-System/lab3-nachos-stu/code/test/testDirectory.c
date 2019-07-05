#include "syscall.h"
#include "func.h"

int main(){
    int i;
    int fd1;
    int flag;

    flag = CreateFolder("folder1");
    if (flag == -1) {
        printStr("create folder1 error!\n");
        return 0;
    }

    flag = CreateFolder("folder1/folder2");
    if (flag == -1) {
        printStr("create folder1/folder2 error!\n");
        return 0;
    }

    flag = CreateFolder("folder2/folder3");
    if (flag != -1) {
        printStr("folder2 doesn\'t exist, but create folder2/folder3 success!\n");
        return 0;
    }

    flag = Create("folder1/folder2/file");
    if (flag == -1) {
        printStr("create folder1/folder2/file error!\n");
        return 0;
    }

    fd1=Open("folder1/folder2/file");

    for(i=0;i<10;i++)
        Write("we write contents to folder1/folder2/file\n",42,fd1);
    Close(fd1);

    return 0;
}
