/* add.c
 *	Simple program to test whether the systemcall interface works.
 *	
 *	Just do a add syscall that adds two values and returns the result.
 *
 */

#include "syscall.h"
#include "func.h"

int
main()
{
  int result;
  Create("h.txt");
  result = Add(42, 23);
  printStr("42+23=");
  printNum(result);
  printStr("\n");

  return 0;
}
