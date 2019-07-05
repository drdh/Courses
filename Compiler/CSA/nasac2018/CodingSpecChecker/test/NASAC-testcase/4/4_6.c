#include <stdio.h>
#define MAXNUMBER 10

/* print powers of 2 and -3 table
 * for exp = 0, 1, 2, ..., 9 */

typedef int ExponentType; 

int power(int,int);

void main()
{
  //bad: e is abbreviation
  ExponentType e; 
  
  for (e = 0; e < MAXNUMBER; ++e)
    printf("%d %d %d\n", e, power(2, e), power(-3, e));
}

// function for the product of multiplying number's base
// input: base: integer; exponent: integer
// output: the product of multiplying number's base
int power(int base,int number)
{
  int exponent, product;

  product = 1;
  for (exponent = 1; exponent <= number; ++exponent)
    product = product * base;
  return product;
}
