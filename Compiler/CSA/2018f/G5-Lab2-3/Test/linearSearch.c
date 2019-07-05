#define N 100

int main()
{
    int array[N], i, x = N/6;
    for(i = 0; i < N; i++)
    {
        array[i] = i * i;
    }
    for (i = 0; i < N; i++)
    {
        if (array[i] == x)
        {
            return i;
        }
    }
    return -1;
}