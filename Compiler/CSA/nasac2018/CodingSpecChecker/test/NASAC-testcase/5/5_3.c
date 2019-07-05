#include <stdio.h>

/* 
 * Function to check prime number
 * Input: // bad: need necessary explanation
 * Output:
 */
int checkPrime(int number)
{
    int index, isPrime = 1;

    for(index = 2; index <= number/2; ++index)
    {
        if(number % index == 0)
        {
            isPrime = 0;
            break;
        }  
    }

    return isPrime;
}

/*
 * Function to check whether a number can be expressed as sum of two prime numbers
 * Input:  // bad: need necessary explanation
 * Output: 
 */
int main()
{
    int number, index, flag = 0;

    printf("Enter a positive integer: ");
    scanf("%d", &number);

    for(index = 2; index <= number/2; ++index)
    {
        // condition for index to be a prime number
        if (checkPrime(index) == 1)
        {
            // condition for number-index to be a prime number
            if (checkPrime(number-index) == 1)
            {
                // number = primeNumber1 + primeNumber2
                printf("%d = %d + %d\n", number, index, number - index);
                flag = 1;
            }

        }
    }

    if (flag == 0)
        printf("%d cannot be expressed as the sum of two prime numbers.", number);

    return 0;
}