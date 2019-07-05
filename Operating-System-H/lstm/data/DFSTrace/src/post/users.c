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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/users.c,v 1.3 1998/10/13 21:14:32 tmkr Exp $";
*/
#endif _BLURB_
/*
 *  user.c -- prints active uids in a trace
 */

#include <stdio.h>
#include <strings.h>
#include <sys/param.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "tracelib.h"


FILE *inFile;

/* user records */
typedef struct urec {
	uid_t uid;
	u_long recs;
	u_long pids;
	struct urec *urecLink;
} urec_t;

#define UNKNOWN_UID 65535

#define UREC_HASH_SIZE 101
urec_t *URecTable[UREC_HASH_SIZE];

urec_t **base;
int slot = 0;
int numUsers = 0;

/* pid records */
typedef struct prec {
	short pid;
	u_long recs;
	struct prec *pidLink;
} prec_t;

#define PREC_HASH_SIZE 101
prec_t *PRecTable[PREC_HASH_SIZE];


urec_t *URecLookup(key)
uid_t key;
{
       urec_t *current = URecTable[((unsigned int) key) % UREC_HASH_SIZE];
       while (current != (urec_t *) 0) {
	   if (key == current->uid)
	       return(current);
	   current = current->urecLink;
       }
       return((urec_t *) 0);
}


void  URecInsert(pRecord)
urec_t *pRecord;
{
       urec_t **bucketPtr =
	   &URecTable[((unsigned int) pRecord->uid) % UREC_HASH_SIZE];
       pRecord->urecLink = *bucketPtr;
       *bucketPtr = pRecord;
}


urec_t *URecDelete(key)
uid_t key;
{
    urec_t **linkPtr = &URecTable[((unsigned int) key) % UREC_HASH_SIZE];
    while (*linkPtr != (urec_t *) 0) {
	if (key ==(*linkPtr)->uid) {
	    urec_t *current = *linkPtr;
	    (*linkPtr) = current->urecLink;
	    return(current);
	}
	linkPtr = &(*linkPtr)->urecLink;
    }
    return((urec_t *) 0);
}

void URecForall(f)
void (*f)();
{
      unsigned int i;
      urec_t *current,*next;
      for (i=0; i< UREC_HASH_SIZE; i++) {
	  current = URecTable[i];
	  while (current != (urec_t *) 0) {
	      next = current->urecLink;
	      f(current);
	      current = next;
	  }
      }
}


prec_t *PRecLookup(key)
short key;
{
    prec_t *current = PRecTable[((unsigned int) key) % PREC_HASH_SIZE];
    while (current != (prec_t *) 0) {
	if (key == current->pid)
	    return(current);
	current = current-> pidLink;
    }
    return((prec_t *) 0);
}


prec_t *PRecDelete(key)
short key;
{
    prec_t **linkPtr = &PRecTable[((unsigned int) key) % PREC_HASH_SIZE];
    while (*linkPtr != (prec_t *) 0) {
	if (key == (*linkPtr)->pid) {
	    prec_t *current = *linkPtr;
	    (*linkPtr) = current-> pidLink;
	    return(current);
	}
	linkPtr = &(*linkPtr)-> pidLink;
    }
    return((prec_t *) 0);
}


void   PRecInsert(pRecord)
prec_t *pRecord;
{
    prec_t **bucketPtr =
	&PRecTable[((unsigned int) pRecord->pid) % PREC_HASH_SIZE];
    pRecord-> pidLink = *bucketPtr;
    *bucketPtr = pRecord;
}


void PRecForall(f)
void (*f)();
{
    unsigned int i;
    prec_t *current,*next;
    for (i=0; i< PREC_HASH_SIZE; i++) {
	current = PRecTable[i];
	while (current != (prec_t *) 0) {
	    next = current-> pidLink;
	    f(current);
	    current = next;
	}
    }
}


prec_t *MakeProcess(pid)
short pid;
{
	prec_t *precPtr;

	precPtr = (prec_t *) malloc(sizeof(prec_t));
	precPtr->pid = pid;
	precPtr->recs = 0;
	precPtr->pidLink = NULL;
	return(precPtr);
}

urec_t *MakeUser(uid)
uid_t uid;
{
	urec_t *urecPtr;

	urecPtr = (urec_t *) malloc(sizeof(urec_t));
	urecPtr->uid = uid;
	urecPtr->pids = 0;
	urecPtr->recs = 0;
	urecPtr->urecLink = NULL;
	return(urecPtr);
}

int URecCompare(u1, u2)
urec_t **u1, **u2;
{
	return((*u2)->recs - (*u1)->recs);
}

void CopyToBase(urecPtr)
urec_t *urecPtr;
{
	base[slot++] = urecPtr;
}

void PrintUsage()
{
	printf("Usage: users [-v] [-d] [-f filter] files\n");
	exit(1);
}

/* TallyUser -- add the pid info to the user record */
void TallyUser(pid)
short pid;
{
	urec_t *curUserPtr;
	prec_t *curProcessPtr;
	uid_t uid;

	curProcessPtr = PRecLookup(pid);
	assert(curProcessPtr);

	(void) Trace_GetUser(inFile, pid, &uid);
	if ((curUserPtr = URecLookup(uid)) == NULL) {
		curUserPtr = MakeUser(uid);
		URecInsert(curUserPtr);
		numUsers++;
	}
	curUserPtr->pids++;
	curUserPtr->recs += curProcessPtr->recs;
	(void) PRecDelete(pid);
}

/* TallyPid -- add a trace record to a pid record */
void TallyPid(pid)
short pid;
{
	prec_t *curProcessPtr;

	if ((curProcessPtr = PRecLookup(pid)) == NULL) {
		curProcessPtr = MakeProcess(pid);
		PRecInsert(curProcessPtr);
	}
	curProcessPtr->recs++;
}

void CleanUpPids(precPtr)
prec_t *precPtr;
{
	TallyUser(precPtr->pid);
}

int main(argc, argv)
int argc;
char *argv[];
{
	int i;
	extern int optind;
	extern char *optarg;
	char filterName[MAXPATHLEN];
	unsigned int totalRecords = 0;
	dfs_header_t *recPtr;
	trace_stat_t traceStats;

	filterName [0] = 0;

	if (argc < 2)
		PrintUsage();

	/* Obtain invocation options */
	while ((i = getopt(argc, argv, "vdf:")) != EOF)
		switch (i) {
		case 'v':
			verbose = 1;
			break;
		case 'd':
			debug = 1;
		    break;
		case 'f':
			(void) strcpy(filterName, optarg);
			break;
		default:
			PrintUsage();
		}

	for (; optind < argc; optind++) {
		if (((inFile = Trace_Open(argv[optind])) == NULL) ||
		    (filterName[0] && Trace_SetFilter(inFile, filterName)))
			printf("users: can't process file %s.\n", argv[optind]);
		else
			while ((recPtr = Trace_GetRecord(inFile))!=  NULL) {
				totalRecords++;
				TallyPid(recPtr->pid);
				/* if we don't know the user now, we never will. */
				if (recPtr->opcode == DFS_EXIT)
					TallyUser(recPtr->pid);
				Trace_FreeRecord(inFile, recPtr);
			}

		/* tally the processes that haven't exited */
		PRecForall(CleanUpPids);
		Trace_Stats(inFile, &traceStats);
		Trace_Close(inFile);

		/* sort the results */
		base = (urec_t **) malloc(numUsers * sizeof(urec_t *));
		URecForall(CopyToBase);

		qsort((char *) base, numUsers, sizeof(urec_t *), URecCompare);

		printf("uid\tprocesses\t     records  (%%)\n");
		for (i = 0; i < numUsers; i++)
			if (base[i]->uid == UNKNOWN_UID)
				printf("Unknown\t%9ld\t%12ld (%2.1f)\n",
				       base[i]->pids, base[i]->recs,
				       (float) base[i]->recs*100/
				       (float) traceStats.recordsUsed);
			else
				printf("%d\t%9ld\t%12ld (%2.1f)\n", base[i]->uid,
				       base[i]->pids, base[i]->recs,
				       (float) base[i]->recs*100/
				       (float) traceStats.recordsUsed);
	}
	 return 0;
}
