#include <stdio.h>
#include <assert.h>

int divideNumbers(int dividend, int divisor);         

int main()
{
    int number1,number2,quotient;

    printf("Enters two numbers: ");
    if (scanf("%d %d",&number1,&number2) != 2)
    {   
      printf("input error.\n"); 
      return 0;
    }
    quotient = divideNumbers(number1, number2);        

    printf("quotient = %d",quotient);

    return 0;
}

// user-defined function to divide number from the other
// Input: divident; divisor
// Output: quotient: integer
int divideNumbers(int divident,int divisor)           
{
    int result;
    assert(divisor != 0);  // bad: callee should not verify parameter

    result = divident / divisor;
    return result;                  // return statement
}
