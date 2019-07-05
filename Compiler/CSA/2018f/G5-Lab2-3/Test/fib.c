int main()
{
    int fib[4]={0,1};
    int i;
    for (i = 2; i < 2000000000; i++)
    {
        fib[i & 3] = fib[(i - 1) & 3] + fib[(i - 2) & 3];
    }
    return fib[(i - 1) & 3];
}