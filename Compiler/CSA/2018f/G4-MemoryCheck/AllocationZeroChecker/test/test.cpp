#include <string.h>
#include <stdlib.h>
int main()
{
    int i = 0;
    char a = '\0';
    int *p = new int[i+a];
    free(p);
}

