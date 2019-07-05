#include <stdio.h>

int checkPrimeNumber(int number);
int main()
{
    int start, end, number, flag;

    printf("Enter two positive integers: ");
    // bad: lack of parameter verification
    if (scanf("%d %d", &start, &end) != 2) 
    {   
      printf("input error.\n"); 
      return 0;
    }
    printf("Prime numbers between %d and %d are: ", start, end);

    for(number=start+1; number<end; ++number)
    {
        // i is a prime number, flag will be equal to 1
        flag = checkPrimeNumber(number);

        if(flag == 1)
            printf("%d ",number);
        else
            printf("none ");
    }
    return 0;
}

// user-defined function to check prime number
// Input: input number be checked
// Output: 0:successful; 1:failure
int checkPrimeNumber(int number)
{
    int divisor, flag = 1;

    for(divisor=2; divisor <= number/2; ++divisor)
    {
        if (number%divisor == 0)
        {
            flag =0;
            break;
        }
    }
    return flag;
}
