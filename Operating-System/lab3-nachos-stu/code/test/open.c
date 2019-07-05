/* open.c
 *	Simple program to test whether the systemcall interface works.
 *	
 *	Just do a open syscall that open a file in filesys and returns fileID.
 *
 */

#include "syscall.h"

int
main()
{
  int result=0;
  result=Open("012345");
  //fprintf(stdio,"in test openFunc, fileID: %d.",result);

  Halt();
  /* not reached */
}
