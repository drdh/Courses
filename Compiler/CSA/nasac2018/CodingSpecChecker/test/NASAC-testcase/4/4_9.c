#include <stdio.h>
#define MAXNUMBER 10

/* print powers of 2 and -3 table
 * for exp = 0, 1, 2, ..., 9 */

typedef int ExponentType; 

int power(int,int);

void main()
{
  ExponentType exponent; 
  //good: correct abbreviation
  int arg = 1;
  int clk = 1;
  int cmp = 1;
  int dev = 1;
  int hex = 1;
  int init = 1;
  int msg = 1;
  int para = 1;
  int reg = 1;
  int stat = 1;
  int tmp = 1;
  int buf = 1;
  int cmd = 1;
  int cfg = 1;
  int err = 1;
  int inc = 1;
  int max = 1;
  int min = 1;
  int prev = 1;
  int sem = 1;
  int sync = 1;

  for (exponent = 0; exponent < MAXNUMBER; ++exponent)
    printf("%d %d %d\n", exponent, power(2, exponent), power(-3, exponent));
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
