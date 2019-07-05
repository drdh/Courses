#include <stdio.h>
#include <string.h>

/* print Fahrenheit-celsius table
 * for f = 0, 20, ..., 300 */

void main()
{
  int lower, upper, step;
  float fahr, celsius;
  char buf[10] = {0};

  lower = 0; /* lower limit of temperature table */
  upper = 300;  /* upper limit */
  step = 20;  /* step size */
  memset(buf, 0, sizeof(buf)); // bad: Redundant reset

  fahr = lower;
  while (fahr <= upper) {
    celsius = (5.0/9.0) * (fahr-32.0);
    printf("%4.0f %6.1f\n", fahr, celsius);
    fahr = fahr + step;
  }
}
