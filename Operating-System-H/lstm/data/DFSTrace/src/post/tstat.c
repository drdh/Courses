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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/tstat.c,v 1.5 1998/11/06 03:38:35 tmkr Exp $";
*/
#endif _BLURB_


/*
 *  tstat.c -- accumulates statistics on a trace.
 */

#include <stdio.h>
#include <strings.h>
#include <sys/param.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include "tracelib.h"

struct tr_stat_struct {
	unsigned int records;
	unsigned int failures;
	unsigned int inodes[DFS_ITYPE_CFS+1];
} tr_stat[DFS_MAXOPCODE+1];

trace_stat_t traceStats;
char nonzero = 0;

 void PrintUsage()
{
	printf("Usage: tstat [-v] [-d] [-n] [-f filter] files\n");
	exit(1);
}

void PrintEntry(i)
int i;
{
    if (traceStats.recordsRead==0) {
        printf("%12s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
	       Trace_OpcodeToStr(i), tr_stat[i].records,
	       0,
	       tr_stat[i].failures,
	       tr_stat[i].inodes[DFS_ITYPE_UFS],
	       tr_stat[i].inodes[DFS_ITYPE_AFS],
	       tr_stat[i].inodes[DFS_ITYPE_CFS],
	       tr_stat[i].inodes[DFS_ITYPE_NFS]);
    }
    else
        printf("%12s\t%d\t%ld\t%d\t%d\t%d\t%d\t%d\n",
	       Trace_OpcodeToStr(i), tr_stat[i].records,
	       (tr_stat[i].records * 100)/traceStats.recordsRead,
	       tr_stat[i].failures,
	       tr_stat[i].inodes[DFS_ITYPE_UFS],
	       tr_stat[i].inodes[DFS_ITYPE_AFS],
	       tr_stat[i].inodes[DFS_ITYPE_CFS],
	       tr_stat[i].inodes[DFS_ITYPE_NFS]);

}

void ProcessFile(file)
FILE *file;
{
	int i, num;
	char st[25];
	dfs_header_t *recPtr;
	generic_fid_t *fidplist[DFS_MAXFIDS];
	double rate = 0.0;

	while ((recPtr = Trace_GetRecord(file))!=  NULL ) {
		tr_stat[recPtr->opcode].records++;
		if (recPtr->error)
			tr_stat[recPtr->opcode].failures++;
		Trace_GetFid(recPtr, fidplist, &num);
		for (i = 0; i < num; i++)
			if (fidplist[i]->tag != -1)
				tr_stat[recPtr->opcode].inodes[(int)fidplist[i]->tag]++;
		Trace_FreeRecord(file, recPtr);
	}

	Trace_PrintPreamble(file);
	Trace_Stats(file, &traceStats);
	Trace_Close(file);

	strcpy(st, ctime(&traceStats.firstTraceRecordTime.tv_sec));
	st[24] = '\0';

	printf("Trace starts %s, ends %s", st,
	       ctime(&traceStats.lastTraceRecordTime.tv_sec));
	rate = (double) traceStats.totalRecords/ (double)
	       (traceStats.lastTraceRecordTime.tv_sec -
		traceStats.firstTraceRecordTime.tv_sec);
	printf("%ld bytes, %ld raw records ( %5.2f /sec), %ld records, %ld returned\n",
	       traceStats.totalBytes, traceStats.totalRecords,
	       rate,  traceStats.recordsRead, traceStats.recordsUsed);

	printf("\n      Opcode\tnum\t%%\tfail\tufs\tafs\tcfs\tnfs\n");
	if (nonzero) {
		for (i = 1; i <= DFS_MAXSYSCALL; i++)
			if (tr_stat[i].records)
				PrintEntry(i);
	} else {
		for (i = 1; i <= DFS_MAXSYSCALL; i++)
			PrintEntry(i);
	}

	/* now print pseudo-ops if there are any */
	if (tr_stat[DFS_LOOKUP].records)
		PrintEntry(DFS_LOOKUP);
	if (tr_stat[DFS_GETSYMLINK].records)
		PrintEntry(DFS_GETSYMLINK);
	if (tr_stat[DFS_ROOT].records)
		PrintEntry(DFS_ROOT);
	/* don't care about syscall dumps */
	if (tr_stat[DFS_NOTE].records)
		PrintEntry(DFS_NOTE);
	if (tr_stat[DFS_SYNC].records)
		PrintEntry(DFS_SYNC);
	if (tr_stat[DFS_FSYNC].records)
		PrintEntry(DFS_FSYNC);
}

int main(argc, argv)
int argc;
char *argv[];
{
	int i;
	FILE *inFile;
	extern int optind;
	extern char *optarg;
	char filterName[MAXPATHLEN];

	filterName [0] = 0;

	if (argc < 2)
		PrintUsage();

	/* Obtain invocation options */
	while ((i = getopt(argc, argv, "df:nv")) != EOF)
		switch (i) {
		case 'd':
			debug = 1;
			break;
		case 'f':
			(void) strcpy(filterName, optarg);
			break;
		case 'n':
			nonzero = 1;
			break;
		case 'v':
			verbose = 1;
			break;
		default:
			PrintUsage();
		}

	for (; optind < argc; optind++) {
	    if (((inFile = Trace_Open(argv[optind])) == NULL) ||
	        (filterName[0] && Trace_SetFilter(inFile, filterName))) {
	        printf("tstat: can't process file %s.\n", argv[optind]);
		exit (-1);
	    }
		else
		    ProcessFile(inFile);
	}
	return (0);
}
