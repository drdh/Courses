#include <stdio.h>
#include <math.h>

#define N 100

int prime[N];

int main()
{
    int i = 1, j =2, c = 0, passed =0;

    for(i = 2; i < N; i++)
    {
        passed = 1;
        for(j = 2; j * j <= i; j++)
            if(i % j == 0)
            {
                passed = 0;
                break;
            }
        if(passed == 1)
            prime[c++] = i;
    }

    //for(i = 0; i < c; i++)printf("%d ", prime[i]);
    //printf("\n");

    return 0;
}
