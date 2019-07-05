#include <stdio.h>
//zuidashu is Chinese Pinyin
#define zuidashu 10 

/* print powers of 2 and -3 table
 * for exp = 0, 1, 2, ..., 9 */

int power(int,int);

void main()
{
  int exponent;

  //bad:zuidashu is Chinese Pinyin
  for (exponent = 0; exponent < zuidashu; ++exponent)
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
