#ifndef _BLURB_
#define _BLURB_
/*

    DFStrace: an Experimental File Reference Tracing Package

       Copyright (c) 1990-1995 Carnegie Mellon University
                      All Rights Reserved.

Permission  to use, copy, modify and distribute this software and
its documentation is hereby granted (including for commercial  or
for-profit use), provided that both the copyright notice and this
permission  notice  appear  in  all  copies  of   the   software,
derivative  works or modified versions, and any portions thereof,
and that both notices appear  in  supporting  documentation,  and
that  credit  is  given  to  Carnegie  Mellon  University  in all
publications reporting on direct or indirect use of this code  or
its derivatives.

DFSTRACE IS AN EXPERIMENTAL SOFTWARE PACKAGE AND IS KNOWN TO HAVE
BUGS, SOME OF WHICH MAY  HAVE  SERIOUS  CONSEQUENCES.    CARNEGIE
MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS" CONDITION.
CARNEGIE MELLON DISCLAIMS ANY  LIABILITY  OF  ANY  KIND  FOR  ANY
DAMAGES  WHATSOEVER RESULTING DIRECTLY OR INDIRECTLY FROM THE USE
OF THIS SOFTWARE OR OF ANY DERIVATIVE WORK.

Carnegie Mellon encourages (but does not require) users  of  this
software to return any improvements or extensions that they make,
and to grant Carnegie Mellon the  rights  to  redistribute  these
changes  without  encumbrance.   Such improvements and extensions
should be returned to Software.Distribution@cs.cmu.edu.

*/

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/pid.h,v 1.2 1998/10/06 19:40:30 tmkr Exp $";
#endif _BLURB_

#ifndef _PID_H_
#define _PID_H_

#include <sys/time.h>
#include <sys/types.h>
#include "tracelib.h"

extern void AddPid();
extern uid_t GetPidUser();
extern int IsAncestor();
extern void DeleteAllUsers();
extern void DoneWithPid();

#define USER_HASH_SIZE 32
#define PID_HASH_SIZE 64

typedef struct pid_struct {
	short pid;
	struct timeval forkTime;
	struct pid_struct *parentPtr;
	struct pid_struct *siblingPtr;
	struct pid_struct *childPtr;
	struct pid_struct *nextPidPtr;
	struct user_struct *userPtr;
	char *dataPtr;
} trace_pid_t;

typedef struct user_struct {
	uid_t 	uid;
	struct user_struct *nextUserPtr;
	trace_pid_t *pidHashTable[PID_HASH_SIZE];
	char *dataPtr;
} user_t;

#endif _PID_H_
