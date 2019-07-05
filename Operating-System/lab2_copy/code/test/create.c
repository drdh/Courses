/* create.c
 *	Simple program to test whether the systemcall interface works.
 *	
 *	Just do a creat syscall that create a file in filesys and returns whether file is create successfully.
 *
 */

#include "syscall.h"

int
main()
{
  //int result=0;
  Create("012345");

  Halt();
  /* not reached */
}
