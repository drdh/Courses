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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/sessions.c,v 1.3 1998/10/13 21:14:31 tmkr Exp $";
*/
#endif _BLURB_


/*
 *    session.c -- identifies and prints "sessions" in a trace
 */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "tracelib.h"

#define	DFLT_INTERVAL	900			/* in seconds */
#define	DFLT_SESSION	16			/* in intervals */
#define	DFLT_MAXGAP	4			/* in intervals */
#define	DFLT_ACTIVITY	16			/* in operations */

int interval = DFLT_INTERVAL;
int session = DFLT_SESSION;
int maxgap = DFLT_MAXGAP;
int activity = DFLT_ACTIVITY;

unsigned long SimTime;
unsigned long SessionStart;
int ActiveIntervals;
int Gap;
int SessionMutations;
int SessionNonmutations;
unsigned long IntervalStart;
int IntervalMutations;
int IntervalNonmutations;

 void PrintUsage()
{
	printf("Usage: sessions file [-v] [-d] [-f filter] [-i interval] [-s session] [-g gap] [-a activity]\n");
	exit(1);
}




void BeginInterval() {
	IntervalStart = SimTime;
	IntervalMutations = 0;
	IntervalNonmutations = 0;
}



void BeginSession() {
/*      printf("BeginSession: %s", ctime(&IntervalStart));*/
	SessionStart = IntervalStart;
	ActiveIntervals = 1;
	Gap = 0;
	SessionMutations = IntervalMutations;
	SessionNonmutations = IntervalNonmutations;
}


void EndSession() {
/*    printf("EndSession: %s", ctime(&IntervalStart));*/
	if (ActiveIntervals >= session) {
		char StartTime[25];
		char EndTime[25];

		bcopy(ctime((const long*)&SessionStart), StartTime, 24);
		StartTime[24] = '\0';
		bcopy(ctime( (const long*)&SimTime), EndTime, 24);
		EndTime[24] = '\0';

		printf("Begin %s, end %s (%2.2f hours)\n",
		       StartTime, EndTime,
		       (((float) SimTime - (float) SessionStart) / 3600));
		printf("\tActiveIntervals = %d, Activity = %dm+n (%dm, %dn)\n",
		       ActiveIntervals, SessionMutations + SessionNonmutations,
		       SessionMutations, SessionNonmutations);
	}
	ActiveIntervals = 0;
}


void EndInterval() {
	if (IntervalMutations + IntervalNonmutations >= activity) {
	/*	printf("ActiveInterval: %s", ctime(&IntervalStart));*/
		if (ActiveIntervals == 0) {
			BeginSession();
		}
		else {
			ActiveIntervals++;
			Gap = 0;
			SessionMutations += IntervalMutations;
			SessionNonmutations += IntervalNonmutations;
		}
	}
	else {
/*	printf("InactiveInterval: %s", ctime(&IntervalStart));*/
		if (ActiveIntervals > 0) {
			Gap++;
			if (Gap > maxgap) {
				EndSession();
			}
		}
	}
}


void ProcessRecord(recPtr)
dfs_header_t *recPtr;
{
	if (SimTime == 0) {
		SimTime = recPtr->time.tv_sec;
		BeginInterval();
	}

	while (recPtr->time.tv_sec > SimTime) {
		SimTime++;

		if ((SimTime - IntervalStart) % interval == 0) {
			EndInterval();
			BeginInterval();
		}
	}

	switch(recPtr->opcode) {
	case DFS_OPEN:
		if (((struct dfs_open *)recPtr)->flags & DFS_FWRITE)
			IntervalMutations++;
		else
			IntervalNonmutations++;
		break;

	case DFS_CLOSE:
		if (((struct dfs_close *)recPtr)->flags & DFS_FWRITE)
			IntervalMutations++;
		else
			IntervalNonmutations++;
		break;

	case DFS_UNLINK:
	case DFS_CREAT:
	case DFS_CHMOD:
	case DFS_RENAME:
	case DFS_RMDIR:
	case DFS_LINK:
	case DFS_CHOWN:
	case DFS_MKDIR:
	case DFS_SYMLINK:
	case DFS_TRUNCATE:
	case DFS_UTIMES:
		IntervalMutations++;
		break;

	case DFS_STAT:
	case DFS_LSTAT:
	case DFS_ACCESS:
	case DFS_READLINK:
	case DFS_LOOKUP:
	case DFS_GETSYMLINK:
	case DFS_ROOT:
		IntervalNonmutations++;
		break;

	case DFS_EXECVE:
	case DFS_SEEK:
	case DFS_EXIT:
	case DFS_FORK:
	case DFS_CHDIR:
	case DFS_SETREUID:
	case DFS_SETTIMEOFDAY:
	case DFS_MOUNT:
	case DFS_UNMOUNT:
	case DFS_CHROOT:
	case DFS_MKNOD:
	case DFS_SYSCALLDUMP:
	case DFS_READ:
	case DFS_WRITE:
	case DFS_NOTE:
		break;

	case DFS_UNUSED:
	default:
		fprintf(stderr, "bogus opcode (%d)\n", recPtr->opcode);
		exit(-1);
	}
}

int main(argc, argv)
int argc;
char **argv;
{
	int i;
	FILE *inFile;
	extern int optind;
	extern char *optarg;
	char filterName[MAXPATHLEN];
	dfs_header_t *recPtr;
	trace_stat_t traceStats;

	filterName [0] = 0;

	if (argc < 2)
		PrintUsage();

	/* Obtain invocation options */
	while ((i = getopt(argc, argv, "vdf:i:s:g:a:")) != EOF)
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
		case 'i':
			interval = atoi(optarg);
			break;
		case 's':
			session = atoi(optarg);
			break;
		case 'g':
			maxgap = atoi(optarg);
			break;
		case 'a':
			activity = atoi(optarg);
			break;
		default:
			PrintUsage();
		}

	for (; optind < argc; optind++) {
		if (((inFile = Trace_Open(argv[optind])) == NULL) ||
		    (filterName[0] && Trace_SetFilter(inFile, filterName)))
			printf("sessions: can't process file %s.\n", argv[optind]);
		else {
			Trace_Stats(inFile, &traceStats);
			printf("Trace starts %s\n",
			       ctime(&traceStats.firstTraceRecordTime.tv_sec));
			while ((recPtr = Trace_GetRecord(inFile))!=NULL) {
				ProcessRecord(recPtr);
				Trace_FreeRecord(inFile, recPtr);
			}
			EndInterval();
			EndSession();

			Trace_Stats(inFile, &traceStats);
			Trace_Close(inFile);

			printf("\nTrace ends %s",
			       ctime(&traceStats.lastTraceRecordTime.tv_sec));
		}
	}
	return 0;
}
