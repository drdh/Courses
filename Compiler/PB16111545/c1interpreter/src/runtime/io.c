
#include <stdio.h>
#include "io.h"

void inputInt(int *i)
{
    scanf("%d", i);
}

void inputFloat(double *f)
{
    scanf("%lf", f);
}

void outputInt(int *i)
{
    printf("%d\n", *i);
}

void outputFloat(double *f)
{
    printf("%lf\n", *f);
}
