#include <stdio.h>

int fun(int x)
{
    int a=(x+1)*(x+1);
    int b=x*x+2*x+3;
    int sum = a*3-b*3;
    return sum;
}

int main()
{
    int sum = 0, x;
    scanf("%d",&x);
    int y =fun(x);
    printf("%d",y);
    return 0;
}
