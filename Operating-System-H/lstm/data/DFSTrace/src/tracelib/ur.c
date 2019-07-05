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


static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/ur.c,v 1.2 1998/10/06 19:40:31 tmkr Exp $";

*/
#endif _BLURB_


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ur_private.h"
#include "trace.h"
#include "tracelib.h"

int URFilePreamble(buf)
char *buf;
{
	return(!(strncmp(buf, "SenecaRO", 8)));
}

int UnpackURFilePreamble(tfPtr)
trace_file_t *tfPtr;
{
	char *buf;
	dfs_file_preamble_ur_t *preamblePtr;

	buf = (char *) malloc(sizeof(dfs_file_preamble_ur_t));
	if (fread(buf, 1, sizeof(dfs_file_preamble_ur_t), tfPtr->fp) !=
	    sizeof(dfs_file_preamble_ur_t)) {
		printf("Couldn't read file preamble!\n");
		free(buf);
		return(TRACE_FILEREADERROR);
	}

	tfPtr->preamblePtr = (char *) malloc(sizeof(dfs_file_preamble_ur_t));
	tfPtr->chunkPreamblePtr = NULL;
	tfPtr->chunk = NULL;

	preamblePtr = (dfs_file_preamble_ur_t *) tfPtr->preamblePtr;
	strncpy(preamblePtr->hostName, buf, 8);

	if (verbose)
		printf("Rochester trace %s\n", preamblePtr->hostName);

	tfPtr->traceStats.totalBytes += sizeof(dfs_file_preamble_ur_t);
	free(buf);
	return(TRACE_SUCCESS);
}

void PrintURFilePreamble(tfPtr)
trace_file_t *tfPtr;
{
}

void UnpackFirstURRecordTime(tfPtr, buf)
trace_file_t *tfPtr;
char *buf;
{
}

void UnpackURRecord(tfPtr, recordPtrPtr)
trace_file_t *tfPtr;
dfs_header_t **recordPtrPtr;		 
{
}
