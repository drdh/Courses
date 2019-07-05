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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/ucb.c,v 1.3 1998/10/23 03:37:57 tmkr Exp $";

*/
#endif _BLURB_

#include <sys/param.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <stdlib.h>
#include "ucb_private.h"
#include "unpack_private.h"
#include "trace.h"
#include "tracelib.h"

/* 
 * CheckUCBFilePreamble -- returns whether or not the preamble makes sense.
 */
int CheckUCBFilePreamble(preamblePtr)
dfs_file_preamble_ucb_t *preamblePtr;
{
	int truth = 0;

	/* 
         * host name is either ucbernie, ucbarpa, or ucbcad
	 * trace number is from 1 to 5
	 * trace version is either 4 or 5
	 */
	if (((STREQ(preamblePtr->hostName, "ucbernie")) ||
	     (STREQ(preamblePtr->hostName, "ucbarpa")) ||
	     (STREQ(preamblePtr->hostName, "ucbcad"))) &&
	    ((preamblePtr->version == 4) ||
	     (preamblePtr->version == 5)) &&
	    ((preamblePtr->number >= 1) &&
	     (preamblePtr->number <= 5)))
		truth = 1;

	return(truth);
}

int UCBFilePreamble(buf)
char *buf;
{
	dfs_file_preamble_ucb_t preamble;
	int offset = 8;

	strncpy(preamble.hostName, buf, 8);
	DFS_LOG_UNPACK_TWO(preamble.number, buf, offset);
	DFS_LOG_UNPACK_TWO(preamble.version, buf, offset);

	return(CheckUCBFilePreamble(&preamble));
}

int UnpackUCBFilePreamble(tfPtr)
trace_file_t *tfPtr;
{
	char *buf;
	int offset = 8;
	dfs_file_preamble_ucb_t *preamblePtr;

	buf = (char *) malloc(sizeof(dfs_file_preamble_ucb_t));
	if (fread(buf, 1, sizeof(dfs_file_preamble_ucb_t), tfPtr->fp) !=
	    sizeof(dfs_file_preamble_ucb_t)) {
		printf("Couldn't read file preamble!\n");
		free(buf);
		return(TRACE_FILEREADERROR);
	}

	tfPtr->preamblePtr = (char *) malloc(sizeof(dfs_file_preamble_ucb_t));
	tfPtr->chunkPreamblePtr = NULL;
	tfPtr->chunk = NULL;

	preamblePtr = (dfs_file_preamble_ucb_t *) tfPtr->preamblePtr;
	strncpy(preamblePtr->hostName, buf, 8);
	DFS_LOG_UNPACK_TWO(preamblePtr->number, buf, offset);
	DFS_LOG_UNPACK_TWO(preamblePtr->version, buf, offset);
	
	tfPtr->traceStats.totalBytes += offset;
	free(buf);
	return(TRACE_SUCCESS);
}

void PrintUCBFilePreamble(tfPtr)
trace_file_t *tfPtr;
{
	dfs_file_preamble_ucb_t *preamblePtr = (dfs_file_preamble_ucb_t *) tfPtr->preamblePtr;

	printf("UCB trace %s%d (version %d)\n",
	       preamblePtr->hostName, preamblePtr->number,
	       preamblePtr->version);
}

void UnpackFirstUCBRecordTime(tfPtr, buf)
trace_file_t *tfPtr;
char *buf;
{
	int offset = 12;

	/* file header(12), rec header(9) */
	/* time is in first two longwords of rec header */
	DFS_LOG_UNPACK_FOUR(tfPtr->traceStats.firstTraceRecordTime.tv_sec,
			    buf, offset);
	DFS_LOG_UNPACK_FOUR(tfPtr->traceStats.firstTraceRecordTime.tv_usec,
			    buf, offset);
}

void UnpackUCBRecord(tfPtr, recordPtrPtr)
trace_file_t *tfPtr;
dfs_header_t **recordPtrPtr;		 
{
	int offset = 0;
	struct timeval time;
	u_char opcode;

	*recordPtrPtr = NULL;

	UNPACK4(time.tv_sec, offset);
	UNPACK4(time.tv_usec, offset);
	UNPACK1(opcode, offset);

	if (feof(tfPtr->fp))
		return;

	switch ((int) opcode) {
	case T_OPEN:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_open));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_open));
		(*recordPtrPtr)->opcode = DFS_OPEN;
		(*recordPtrPtr)->time = time;
		UNPACK2(((struct dfs_open *)*recordPtrPtr)->flags, offset);
		UNPACKFID(((struct dfs_open*)*recordPtrPtr)->fid, offset);
		UNPACK2(((struct dfs_open *)*recordPtrPtr)->findex, offset);
		UNPACK2(((struct dfs_open *)*recordPtrPtr)->uid, offset);
		UNPACK4(((struct dfs_open *)*recordPtrPtr)->size, offset);
		break;
	case T_CREATE:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_creat));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_creat));
		(*recordPtrPtr)->opcode = DFS_CREAT;
		(*recordPtrPtr)->time = time;
		UNPACK2(((struct dfs_creat *)*recordPtrPtr)->flags, offset);
		UNPACKFID(((struct dfs_creat*)*recordPtrPtr)->fid, offset);
		UNPACK2(((struct dfs_creat *)*recordPtrPtr)->findex, offset);
		UNPACK2(((struct dfs_creat *)*recordPtrPtr)->uid, offset);
		UNPACK4(((struct dfs_creat *)*recordPtrPtr)->oldSize, offset);
		break;
	case T_UNLINK:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_rmdir));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_rmdir));
		(*recordPtrPtr)->opcode = DFS_UNLINK;
		(*recordPtrPtr)->time = time;
		UNPACKFID(((struct dfs_rmdir *)*recordPtrPtr)->fid, offset);
		break;
	case T_TRUNC:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_truncate));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_truncate));
		(*recordPtrPtr)->opcode = DFS_TRUNCATE;
		(*recordPtrPtr)->time = time;
		UNPACKFID(((struct dfs_truncate *)*recordPtrPtr)->fid, offset);
		UNPACK4(((struct dfs_truncate *)*recordPtrPtr)->newSize, offset);
		break;
	case T_CLOSE:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_close));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_close));
		(*recordPtrPtr)->opcode = DFS_CLOSE;
		(*recordPtrPtr)->time = time;
		UNPACK2(((struct dfs_close *)*recordPtrPtr)->findex, offset);
		UNPACK4(((struct dfs_close *)*recordPtrPtr)->offset, offset);
		UNPACK4(((struct dfs_close *)*recordPtrPtr)->bytesRead, offset);
		UNPACK4(((struct dfs_close *)*recordPtrPtr)->bytesWritten, offset);
		((struct dfs_close *)*recordPtrPtr)->refCount = 1;   /* always the last one */
		break;
	case T_SEEK:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_seek));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_seek));
		(*recordPtrPtr)->opcode = DFS_SEEK;
		(*recordPtrPtr)->time = time;
		UNPACK2(((struct dfs_seek *)*recordPtrPtr)->findex, offset);
		UNPACK4(((struct dfs_seek *)*recordPtrPtr)->oldOffset, offset);
		UNPACK4(((struct dfs_seek *)*recordPtrPtr)->offset, offset);
		UNPACK4(((struct dfs_seek *)*recordPtrPtr)->bytesRead, offset);
		UNPACK4(((struct dfs_seek *)*recordPtrPtr)->bytesWritten, offset);
		break;
	case T_EXECVE:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_execve));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_execve));
		(*recordPtrPtr)->opcode = DFS_EXECVE;
		(*recordPtrPtr)->time = time;
		UNPACK2(((struct dfs_execve *)*recordPtrPtr)->euid, offset);
		UNPACK2(((struct dfs_execve *)*recordPtrPtr)->ruid, offset);
		UNPACKFID(((struct dfs_execve *)*recordPtrPtr)->fid, offset);
		UNPACK4(((struct dfs_execve *)*recordPtrPtr)->size, offset);
		break;
	case T_SYNC:  /* header only */
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_sync));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_sync));
		(*recordPtrPtr)->opcode = DFS_SYNC;
		(*recordPtrPtr)->time = time;
		break;
	case T_FSYNC:
		*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_fsync));
		bzero((char *)*recordPtrPtr, sizeof(struct dfs_fsync));
		(*recordPtrPtr)->opcode = DFS_FSYNC;
		(*recordPtrPtr)->time = time;
		UNPACK2(((struct dfs_fsync *)*recordPtrPtr)->findex, offset);
		break;
	default:
		printf("Bogus bytes! record at %ld, offset %d, opcode = 0x%x\n", 
		       tfPtr->traceStats.totalBytes, offset, opcode);
		break;
 	}

	/* check timestamps */
/*
	if (timercmp(&(*recordPtrPtr)->time, &tfPtr->traceStats.lastTraceRecordTime, <)) {
		printf("Warning: Decreasing timestamp: byte %d, opcode %s.\n\tLast timestamp was %s", 
		       tfPtr->traceStats.totalBytes, Trace_OpcodeToStr((*recordPtrPtr)->opcode),
		       ctime(&tfPtr->traceStats.lastTraceRecordTime.tv_sec));
		printf("\tCurrent time is %s", ctime(&(*recordPtrPtr)->time.tv_sec));
	}
*/
	LONGALIGN(offset);  /* read filler bytes, if any */
	tfPtr->traceStats.totalBytes += offset;

	return;
}


