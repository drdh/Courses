/*
 *	Simple program to test whether the systemcall interface works.
 *	
 *	Just do a sub syscall that subs two values and returns the result.
 *
 */

#include "syscall.h"

int main()
{
  int result;

  result = Sub(23, 42);

  //Halt();
  /* not reached */
}
