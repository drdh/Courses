#include<stdio.h>
#define max 10
int main()
{
    int i, j, tmp, t;
    int s[max] = {0};
    for(i=0;i<max;i++)
        s[i] = max-i;
    for(i=0;i<max-1;i++)
    {
        tmp = s[i];
        t = i;
        for(j=i+1;j<max;j++)
        {
            if (s[j]<tmp)
            {
                tmp = s[j];
                t = j;
            }
        }
        tmp = s[i];
        s[i] = s[t];
        s[t] = tmp;
    }
    for(i=0;i<max;i++)
        printf("%d ", s[i]);
}
