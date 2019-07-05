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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/replay.c,v 1.3 1998/10/13 21:14:31 tmkr Exp $";
*/
#endif _BLURB_


/*
 *  replay.c -- prints trace data in readable form.
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

int main(argc, argv)
int argc;
char *argv[];
{
	int i;
	FILE *inFile;
	dfs_header_t *recPtr;
	extern int optind;
	extern char *optarg;
	char longForm = 0;
	char filterName[MAXPATHLEN];
	filterName [0] = 0;

	/* get filename */
	if (argc < 2) {
		printf("Usage: replay [-v] [-d] [-l] [-f filter] files\n");
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
		case 'l':
			longForm = 1;
			break;
		case 'f':
			(void) strcpy(filterName, optarg);
			break;
		default:
			printf("Usage: replay [-v] [-d] [-l] [-f filter] files\n");
			exit(-1);
		}

	for (; optind < argc; optind++) {
		if (((inFile = Trace_Open(argv[optind])) == NULL) ||
		    (filterName[0] && Trace_SetFilter(inFile, filterName)))
			printf("replay: can't process file %s.\n", argv[optind]);
		else {
			while ((recPtr = Trace_GetRecord(inFile))!=NULL) {
				if (longForm)
					Trace_DumpRecord(recPtr);
				else
					Trace_PrintRecord(recPtr);
				Trace_FreeRecord(inFile, recPtr);
			}
			Trace_Close(inFile);
		}
	}
	return 0;
}
