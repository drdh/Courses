#include<stdio.h>
#include<stdlib.h>

void find(int *a,int k,int n)
{
    int i;
    if(k>n)
    {
        for(i=0;i<2*n;i++)
            printf("%d ",a[i]);
        printf("\n");
        return;
    }
    for(i=0;i<2*n-2;i++)
    {
        if(!a[i]&&i+k+1<2*n&&!a[i+k+1])
        {
            a[i]=k;
            a[i+k+1]=k;
            find(a,k+1,n);
            a[i]=0;
            a[i+k+1]=0;
        }
    }
    return;
}

int main()
{
    int n;
    scanf("%d",&n);
    int *a=(int *)malloc(2*n*sizeof(int));
    int i;
    for(i=0;i<2*n;i++)
        a[i]=0;
    find(a,1,n);
    free(a);
}













