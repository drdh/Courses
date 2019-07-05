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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/patterns.c,v 1.4 1998/11/06 03:38:35 tmkr Exp $";
*/
#endif _BLURB_

/*
 *  patterns.c -- prints a table of statistics on file access patterns.
 */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include "tracelib.h"

void TallyClose();
void TallyPattern();
void PrintRow();
void Normalize();
char *GbToStr();
float Perc();

typedef struct {
	u_long gbytes;
	u_long bytes;
} u_gig;

u_gig patterns[4][2][4];

#define TOTAL 0

#define READONLY 1
#define WRITEONLY 2
#define READWRITE 3

#define ACCESSES 0
#define BYTES 1

#define WHOLEFILE 1
#define OTHERSEQ 2
#define RANDOM 3

 int main(argc, argv)
int argc;
char *argv[];
{
	int i;
	FILE *inFile;
	dfs_header_t *recPtr;
	extern int optind;
	extern char *optarg;
	char filterName[MAXPATHLEN];
	filterName [0] = 0;

	/* get filename */
	if (argc < 2) {
		printf("Usage: patterns [-v] [-d] [-f filter] files\n");
		exit(1);
	}

	/* Obtain invocation options */
	while ((i = getopt(argc, argv, "dlvf:")) != EOF)
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
			printf("Usage: patterns [-v] [-d] [-f filter] files\n");
			exit(-1);
		}

	for (; optind < argc; optind++) {
		if (((inFile = Trace_Open(argv[optind])) == NULL) ||
		    (filterName[0] && Trace_SetFilter(inFile, filterName)))
			printf("patterns: can't process file %s.\n", argv[optind]);
		else {
			while ((recPtr = Trace_GetRecord(inFile))!=NULL) {
				/* tally on close records w/ref 1,
				   since info in file table cumulative */
				if ((recPtr->opcode == DFS_CLOSE) &&
				    (((struct dfs_close *)recPtr)->refCount == 1)) {
					TallyClose(recPtr);
					Normalize();
				}
				Trace_FreeRecord(inFile, recPtr);
			}
			Trace_Close(inFile);
		}
	}

	printf("Access Type    Accesses (%%)            Bytes (%%)     Transfer Type  Accesses (%%)            Bytes (%%)\n");
	for (i = READONLY; i <= READWRITE; i++) {
		printf("\n");
		PrintRow(i);
	}
	printf("Total      %s         ",
	       GbToStr(&patterns[TOTAL][ACCESSES][TOTAL]));
	printf("%s\n",
	       GbToStr(&patterns[TOTAL][BYTES][TOTAL]));
	return 0;
}

void TallyClose(recPtr)
struct dfs_close *recPtr;
{
	struct dfs_close *closeRec;

	closeRec = (struct dfs_close *) recPtr;
	patterns[TOTAL][ACCESSES][TOTAL].bytes++;
	patterns[TOTAL][BYTES][TOTAL].bytes += closeRec->bytesRead + closeRec->bytesWritten;
	if (closeRec->numWrites == 0)
		TallyPattern(closeRec, READONLY, closeRec->bytesRead);
	else if (closeRec->numReads == 0)
		TallyPattern(closeRec, WRITEONLY, closeRec->bytesWritten);
	else
		TallyPattern(closeRec, READWRITE, closeRec->bytesRead+closeRec->bytesWritten);
}

void TallyPattern(closeRec, accessType, bytes)
struct dfs_close *closeRec;
int accessType;
int bytes;
{
	patterns[accessType][ACCESSES][TOTAL].bytes++;
	patterns[accessType][BYTES][TOTAL].bytes += bytes;
	if (closeRec->numSeeks == 0)
		if (closeRec->size == bytes) {
			patterns[accessType][ACCESSES][WHOLEFILE].bytes++;
			patterns[accessType][BYTES][WHOLEFILE].bytes += bytes;
		} else {
			patterns[accessType][ACCESSES][OTHERSEQ].bytes++;
			patterns[accessType][BYTES][OTHERSEQ].bytes += bytes;
		}
	else {
		patterns[accessType][ACCESSES][RANDOM].bytes++;
		patterns[accessType][BYTES][RANDOM].bytes += bytes;
	}
}

void Normalize()
{
	int i, j, k;

	for (i = 0; i < 4; i++)
		for (j = 0; j < 2; j++)
			for (k = 0; k < 4; k++) {
				if (patterns[i][j][k].bytes > 1000000000) {
					patterns[i][j][k].bytes -= 1000000000;
					patterns[i][j][k].gbytes++;
				}
			}
}

char *AccessTypeToStr(accessType)
int accessType;
{
	static char buf[20];

	switch (accessType) {
	case READONLY:
		(void) strcpy(buf, "Read-only");
		break;
	case WRITEONLY:
		(void) strcpy(buf, "Write-only");
		break;
	case READWRITE:
		(void) strcpy(buf, "Read-write");
		break;
	default:
		(void) strcpy(buf, "Unknown!");
		break;
	}
	return(buf);
}

char *GbToStr(gp)
u_gig *gp;
{
	static char buf[20];

	if (gp->gbytes)
		sprintf(buf, "%3ld%09ld", gp->gbytes, gp->bytes);
	else
		sprintf(buf, "%12ld", gp->bytes);

	return(buf);
}

float Perc(a, b)
u_gig a, b;
{
	float p, n;

       n= (float) b.gbytes * 1000000000.0 + (float) b.bytes;
       if (n==0.0 ) return 0.0;

	p = ((float) a.gbytes * 1000000000.0 + (float) a.bytes) * 100.0 /
		((float) b.gbytes * 1000000000.0 + (float) b.bytes);
	return(p);
}

void PrintRow(accessType)
{
	printf("%53sWhole-file %s (%5.1f) ",
	       "", GbToStr(&patterns[accessType][ACCESSES][WHOLEFILE]),
	       Perc(patterns[accessType][ACCESSES][WHOLEFILE],
		    patterns[accessType][ACCESSES][TOTAL]));
	printf("%s (%5.1f)\n", GbToStr(&patterns[accessType][BYTES][WHOLEFILE]),
	       Perc(patterns[accessType][BYTES][WHOLEFILE],
		    patterns[accessType][BYTES][TOTAL]));
	printf("%10s %s (%5.1f) ",
	       AccessTypeToStr(accessType),
	       GbToStr(&patterns[accessType][ACCESSES][TOTAL]),
	       Perc(patterns[accessType][ACCESSES][TOTAL],
		    patterns[TOTAL][ACCESSES][TOTAL]));
	printf("%s (%5.1f) ",
	       GbToStr(&patterns[accessType][BYTES][TOTAL]),
	       Perc(patterns[accessType][BYTES][TOTAL],
		    patterns[TOTAL][BYTES][TOTAL]));
	printf("Other Seq  %s (%5.1f) ",
	       GbToStr(&patterns[accessType][ACCESSES][OTHERSEQ]),
	       Perc(patterns[accessType][ACCESSES][OTHERSEQ],
		    patterns[accessType][ACCESSES][TOTAL]));
	printf("%s (%5.1f)\n",
	       GbToStr(&patterns[accessType][BYTES][OTHERSEQ]),
	       Perc(patterns[accessType][BYTES][OTHERSEQ],
		    patterns[accessType][BYTES][TOTAL]));
	printf("%53sRandom     %s (%5.1f) ",
	       "", GbToStr(&patterns[accessType][ACCESSES][RANDOM]),
	       Perc(patterns[accessType][ACCESSES][RANDOM],
		    patterns[accessType][ACCESSES][TOTAL]));
	printf("%s (%5.1f)\n",
	       GbToStr(&patterns[accessType][BYTES][RANDOM]),
	       Perc(patterns[accessType][BYTES][RANDOM],
		    patterns[accessType][BYTES][TOTAL]));
}
