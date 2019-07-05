#include <stdio.h>
#define MAXNUMBER 10

/* print powers of 2 and -3 table
 * for exp = 0, 1, 2, ..., 9 */

typedef int ExponentType; 

int power(int,int);

void main()
{
  ExponentType exponent; 

  for (exponent = 0; exponent < MAXNUMBER; ++exponent)
    printf("%d %d %d\n", exponent, power(2, exponent), power(-3, exponent));
}

// function for the product of multiplying number's base
// input: base: integer; exponent: integer
// output: the product of multiplying number's base
int power(int base,int number)
{
  //good correct word concatenation
  int exponent, power_result;
  // correct word concatenation
  power_result = 1;
  for (exponent = 1; exponent <= number; ++exponent)
    // correct word concatenation
    power_result = power_result * base;
  // correct word concatenation
  return power_result;
}
