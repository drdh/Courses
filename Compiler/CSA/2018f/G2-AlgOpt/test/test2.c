#include<stdio.h>

int fun(int x)
{
    int a = (x+3)*(x+2)*(x+1);
    int b = x*x*x+6*x*x+11*x;
    int sum = a-b;
    return sum;
}

int main()
{
    int x;
    scanf("%d",&x);
    printf("%d\n",fun(x));
    return 0;
}