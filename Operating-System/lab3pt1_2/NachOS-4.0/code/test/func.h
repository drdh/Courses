#include "syscall.h"

void itoa(int n, char* line){
    int i=0;
    int j=0;
    int neg = -1;
    if(n<0) {
        n=-n;
        neg=1;
    }
    while(n){
        line[i++] = '0'+n%10;
        n/=10;
    }
    if(neg==1) line[i++] = '-';
    for(i=i-1;i>j;j++,i--){
        char t = line[i];
        line[i] = line[j];
        line[j] = t;
    }
}

void writeNum(int num,int fid){
    int cnt = 0;
    int t = num;
    char numStr[64];

    /*计算数字字符串长度*/
    if(t==0) {}
    else {
        if(t<0) cnt++;
        while(t){
            t/=10;
            cnt++;
        }
    }

    itoa(num,numStr);
    Write(numStr,cnt,fid);
    Write("\n",1,fid);
}

void printNum(int num){
    writeNum(num,1);
}

void printStr(char *str){
    int len=0;
    while(str[len]!='\0') len++;
    Write(str,len,1);
}