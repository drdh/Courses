#include<stdio.h>

int *p;

void func1(){
    int k;
    p=&k;
}

void func2(){
    int i;
    printf("%p,%p\n",p,&i);
    printf("%d\n",i);
}

int main(){
    func1();
     *p=123;
    func2();
}