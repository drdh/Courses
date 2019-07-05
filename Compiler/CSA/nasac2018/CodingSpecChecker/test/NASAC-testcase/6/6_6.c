#include <stdio.h>

/* print Fahrenheit-celsius table
 * for f = 0, 20, ..., 300 */
struct 
{
  int x;
  int y;
} pointer = {0,0}; //bad: redundant initialization

void main()
{
  int lower, upper, step;
  float fahr, celsius;

  lower = 0; /* lower limit of temperature table */
  upper = 300;  /* upper limit */
  step = 20;  /* step size */
  pointer.x = 2;
  pointer.y = 3;

  fahr = lower;
  while (fahr <= upper) {
    celsius = (5.0/9.0) * (fahr-32.0);
    printf("%4.0f %6.1f\n", fahr, celsius);
    fahr = fahr + step;
  }
}
