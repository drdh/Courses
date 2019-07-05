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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/pid.c,v 1.2 1998/10/06 19:40:30 tmkr Exp $";
*/

#endif _BLURB_


/*
 *  pid.c -- builds pid tree from a trace
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/errno.h>
#include <string.h>
#include "tracelib.h"
#include "trace.h"
#include "pid.h"

#define UNKNOWN_UID 65535

user_t *CreateUser(tfPtr, uid)
trace_file_t *tfPtr;
uid_t uid;
{
	user_t *newPtr;
	int bucket;

	newPtr = (user_t *) malloc(sizeof(user_t));
	bzero(newPtr, sizeof(user_t));
	newPtr->uid = uid;

	bucket = uid % USER_HASH_SIZE;
	newPtr->nextUserPtr = tfPtr->userHashTable[bucket];
	tfPtr->userHashTable[bucket] = newPtr;
	return(newPtr);
}

user_t *LookupUser(tfPtr, uid)
trace_file_t *tfPtr;
uid_t uid;
{
	int bucket;
	user_t *curUserPtr;

	bucket = uid % USER_HASH_SIZE;
	curUserPtr = tfPtr->userHashTable[bucket];
	while (curUserPtr) {
		if (curUserPtr->uid == uid) {
			break;
		}
		curUserPtr = curUserPtr->nextUserPtr;
	}
	return(curUserPtr);
}

/*
 * DeleteAllPids -- free storage used by pid structures
 */
void DeleteAllPids(userPtr)
user_t *userPtr;
{
	int bucket;
	trace_pid_t *curPidPtr, *nextPidPtr;

	for (bucket = 0; bucket < PID_HASH_SIZE; bucket++) {
		curPidPtr = userPtr->pidHashTable[bucket];
		while (curPidPtr) {
			nextPidPtr = curPidPtr->nextPidPtr;
			free(curPidPtr);
			curPidPtr = nextPidPtr;
		}
	}		
}

void DeleteUser(tfPtr, uid)
trace_file_t *tfPtr;
uid_t uid;
{
	int bucket;
	user_t **prevUserPtr, *curUserPtr;

	bucket = uid % USER_HASH_SIZE;
	prevUserPtr = &tfPtr->userHashTable[bucket];
	curUserPtr = *prevUserPtr;
	while (curUserPtr) {
		if (curUserPtr->uid == uid) {
			*prevUserPtr = curUserPtr->nextUserPtr;
			DeleteAllPids(curUserPtr);
			free(curUserPtr);
			return;
		}
		prevUserPtr = &curUserPtr->nextUserPtr;
		curUserPtr = *prevUserPtr;
	}
}

trace_pid_t *CreatePid(userPtr, parentPtr, pid)
user_t *userPtr;
trace_pid_t *parentPtr;
short pid;
{
	trace_pid_t *newPtr;
	int bucket;

	newPtr = (trace_pid_t *) malloc(sizeof(trace_pid_t));
	bzero(newPtr, sizeof(trace_pid_t));
	newPtr->pid = pid;
	newPtr->userPtr = userPtr;

	bucket = pid % PID_HASH_SIZE;
	newPtr->nextPidPtr = userPtr->pidHashTable[bucket];
	userPtr->pidHashTable[bucket] = newPtr;

	/* link with parent and siblings */
	newPtr->parentPtr = parentPtr;
	if (parentPtr) {
		if (parentPtr->childPtr) 
			newPtr->siblingPtr = parentPtr->childPtr;
		parentPtr->childPtr = newPtr;
	}
	return(newPtr);
}


trace_pid_t *LookupPid(userPtr, pid)
user_t *userPtr;
short pid;
{
	int bucket;
	trace_pid_t *curPidPtr;

	bucket = pid % PID_HASH_SIZE;
	curPidPtr = userPtr->pidHashTable[bucket];
	while (curPidPtr) {
		if (curPidPtr->pid == pid) {
			break;
		}
		curPidPtr = curPidPtr->nextPidPtr;
	}
	return(curPidPtr);
}

void DeletePid(userPtr, pid)
user_t *userPtr;
short pid;
{
	int bucket;
	trace_pid_t **prevPidPtrPtr, *curPidPtr, **prevSibPtrPtr, *curSibPtr;

	bucket = pid % PID_HASH_SIZE;
	prevPidPtrPtr = &userPtr->pidHashTable[bucket];
	curPidPtr = *prevPidPtrPtr;
	while (curPidPtr) {
		if (curPidPtr->pid == pid) {
			*prevPidPtrPtr = curPidPtr->nextPidPtr;
			
			/* 
			 * the following case can happen when the pid
			 * exits.  The children will close files and
			 * exit shortly anyway. We delete this pid's
			 * children here.  To make sure that their
			 * subsequent close and exit records appear
			 * under the correct user, we re-insert the
			 * children under this user as a top-level pid.
			 */
			if (curPidPtr->childPtr) {
				short orphanPid;
				trace_pid_t *curChildPtr, *nextChildPtr;
				
				curChildPtr = curPidPtr->childPtr;
				while (curChildPtr) {
					nextChildPtr = curChildPtr->siblingPtr;
					orphanPid = curChildPtr->pid;

					DeletePid(userPtr, orphanPid);
					(void) CreatePid(userPtr, NULL, orphanPid);

					curChildPtr = nextChildPtr;
				}
			}

			assert(curPidPtr->childPtr == NULL);
			/* if parent non-null, remove from sibling list */
			if (curPidPtr->parentPtr) {
				prevSibPtrPtr = &curPidPtr->parentPtr->childPtr;
				curSibPtr = *prevSibPtrPtr;
				while (curSibPtr) {
					if (curSibPtr->pid == pid) {
						*prevSibPtrPtr = curSibPtr->siblingPtr;
						break;
					}
					prevSibPtrPtr = &curSibPtr->siblingPtr;
					curSibPtr = *prevSibPtrPtr;
				}
			}
			free(curPidPtr);
			return;
		}
		prevPidPtrPtr = &curPidPtr->nextPidPtr;
		curPidPtr = *prevPidPtrPtr;
	}
}

/*
 * LookupPidUser -- looks up the user associated with process pid.
 * Searches all of the user tables and returns the user structure
 * if it's there, and NULL if it's not. The user structure might 
 * be the one for the "unknown" category.
 */
user_t *LookupPidUser(tfPtr, pid)
trace_file_t *tfPtr;
short pid;
{
	user_t *curUserPtr;
	int bucket;

	for (bucket = 0; bucket < USER_HASH_SIZE; bucket++) {
		curUserPtr = tfPtr->userHashTable[bucket];
		while (curUserPtr) {
			if (LookupPid(curUserPtr, pid))
				return(curUserPtr);
			curUserPtr = curUserPtr->nextUserPtr;
		}
	}
	return(NULL);
}

void PrintPids(pidPtr, level)
trace_pid_t *pidPtr;
int level;
{
	int i;
	trace_pid_t *curPidPtr;

	curPidPtr = pidPtr;
	while (curPidPtr != NULL) {
		for (i = 0; i < level; i++) printf("\t");  /* ugh */
		printf("pid %d, created %s", curPidPtr->pid, 
			(curPidPtr->forkTime.tv_sec == 0)?
		       "unknown\n":ctime(&curPidPtr->forkTime.tv_sec));
		if (curPidPtr->childPtr != NULL)
			PrintPids(curPidPtr->childPtr, level+1);
		curPidPtr = curPidPtr->siblingPtr;
	}
}
	
void Trace_UserPrintPids(tfPtr, uid)
trace_file_t *tfPtr;	
uid_t uid;
{
	int bucket;
	user_t *userPtr;
	trace_pid_t *curPidPtr;

	if ((userPtr = LookupUser(tfPtr, uid))) {
		printf("\nUser %d\n", userPtr->uid);
		for (bucket = 0; bucket < PID_HASH_SIZE; bucket++) {
			curPidPtr = userPtr->pidHashTable[bucket];
			while (curPidPtr) {
				if (curPidPtr->parentPtr == NULL)
					PrintPids(curPidPtr, 1);
				curPidPtr = curPidPtr->nextPidPtr;
			}
		}		
	}
}

void Trace_PidPrintPids(tfPtr, pid)
trace_file_t *tfPtr;	
short pid;
{
	trace_pid_t *pidPtr;
	user_t *userPtr;

	if ((userPtr = LookupPidUser(tfPtr, pid)) &&
	    (pidPtr = LookupPid(userPtr, pid)))
		PrintPids(pidPtr, 1);
}

/*
 * external shell to lookup a user.  Since we're returning
 * a uid, we lose some information from LookupPidUser.
 * Here we return the uid if the pid is known, and
 * the unknown uid if it is not. We don't reveal whether
 * or not the pid has been seen before.
 */
uid_t GetPidUser(tfPtr, pid)
trace_file_t *tfPtr;
short pid;
{
	user_t *userPtr;
	uid_t uid;

	userPtr = LookupPidUser(tfPtr, pid);
	if (userPtr == NULL)
		uid = UNKNOWN_UID;
	else 
		uid = userPtr->uid;
	return(uid);
}

/* 
 * DeleteAllUsers -- free storage used by user structures
 */
void DeleteAllUsers(tfPtr)
trace_file_t *tfPtr;	
{
	int bucket;
	user_t *curUserPtr, *nextUserPtr;

	for (bucket = 0; bucket < USER_HASH_SIZE; bucket++) {
		curUserPtr = tfPtr->userHashTable[bucket];
		while (curUserPtr) {
			DeleteAllPids(curUserPtr);
			nextUserPtr = curUserPtr->nextUserPtr;
			free(curUserPtr);
			curUserPtr = nextUserPtr;
		}
	}		
}


/* 
 * adds a node to the pid tree. Pids added for records other
 * than forks may be added under the "unknown" uid.  Pids in
 * this category never have any children -- as soon as there
 * is a fork, the user is known, and the parent is moved.
 */
void AddPid(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	trace_pid_t *pidPtr;
	user_t *userPtr;
	static user_t *unknownUserPtr;

	if ((userPtr = LookupPidUser(tfPtr, recordPtr->pid)) == NULL) {
		/* add to unknown list */
		if (verbose)
			printf("AddPid: adding pid %d to unknown list.\n",
			       recordPtr->pid);
		if ((unknownUserPtr = LookupUser(tfPtr, UNKNOWN_UID)) == NULL) {
			unknownUserPtr = CreateUser(tfPtr, UNKNOWN_UID);
		}
		pidPtr = CreatePid(unknownUserPtr, NULL, recordPtr->pid);
		userPtr = unknownUserPtr;
	} else {
		pidPtr = LookupPid(userPtr, recordPtr->pid);
	}
	assert(pidPtr);

	if (recordPtr->opcode == DFS_FORK && 
	    recordPtr->error == 0) {
		/* 
		 * we don't create these structures if the fork failed.
		 * if it fails with EAGAIN (no more processes), the 
		 * child pid will be 0 (whoa).
		 */
		trace_pid_t *newPtr;
		trace_pid_t *parentPtr;
		struct dfs_fork *forkPtr;

		forkPtr = (struct dfs_fork *) recordPtr;

		if ((userPtr = LookupUser(tfPtr, forkPtr->userId)) == NULL)
			userPtr = CreateUser(tfPtr, forkPtr->userId);

		if ((parentPtr = LookupPid(userPtr, recordPtr->pid)) == NULL) {
			if (verbose)
				printf("AddPid: moving pid %d to user %d.\n",
				       recordPtr->pid, forkPtr->userId);
			DeletePid(unknownUserPtr, recordPtr->pid);
			parentPtr = CreatePid(userPtr, NULL, recordPtr->pid);
		}			
		newPtr = CreatePid(userPtr, parentPtr, forkPtr->childPid);
		newPtr->forkTime = recordPtr->time;
	}
}

/*
 * DoneWithPid -- called when a process exits, so we can
 * get rid of the pid structure.  A user record is guaranteed
 * to be there, since we did an AddPid when we got the exit
 * record.
 */
void DoneWithPid(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	user_t *userPtr;

	assert(recordPtr->opcode == DFS_EXIT);

	userPtr = LookupPidUser(tfPtr, recordPtr->pid);
	assert(userPtr);

	DeletePid(userPtr, recordPtr->pid);
}

/*
 * IsPidAncestor -- climbs pid trees to determine if
 * pid1 is an ancestor of pid2.
 */

int IsPidAncestor(tfPtr, pid1, pid2)
trace_file_t *tfPtr;
short pid1, pid2;
{
	trace_pid_t *pidPtr, *curPidPtr;
	user_t *userPtr;

	userPtr = LookupPidUser(tfPtr, pid2);
	assert(userPtr);
	pidPtr = LookupPid(userPtr, pid2);
	assert(pidPtr);

	curPidPtr = pidPtr->parentPtr;
	while (curPidPtr) {
		if (pid1 == curPidPtr->pid)
			return(1);
		curPidPtr = curPidPtr->parentPtr;
	}
	return(0);
}
