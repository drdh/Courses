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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/ftable.c,v 1.2 1998/10/06 19:40:30 tmkr Exp $";

*/
#endif _BLURB_


/* 
 * ftable.c -- file table maintenance
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include "trace.h"
#include "tracelib.h"

/*
 * FileTableInsert  -- called on open or creat. Copies info from
 * the trace record into the file table. Filtering is off by default.
 */
void FileTableInsert(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	file_table_entry_t *entryPtr;
	short findex;

	assert((recordPtr->opcode == DFS_OPEN) ||
	       (recordPtr->opcode == DFS_CREAT));

	findex = Trace_GetFileIndex(recordPtr);
	assert((findex >= 0) && (findex < MAXFILES));
	
	entryPtr = &tfPtr->kernFileTable[findex];
	assert(entryPtr->allocated == 0);
	entryPtr->allocated = 1;

	if (recordPtr->opcode == DFS_OPEN) {
		entryPtr->fd = ((struct dfs_open *)recordPtr)->fd;
		entryPtr->flags = ((struct dfs_open *)recordPtr)->flags;
		entryPtr->mode = ((struct dfs_open *)recordPtr)->mode;
		entryPtr->fileType = ((struct dfs_open *)recordPtr)->fileType;
		entryPtr->uid = ((struct dfs_open *)recordPtr)->uid;
		entryPtr->oldSize = ((struct dfs_open *)recordPtr)->oldSize;
		entryPtr->size = ((struct dfs_open *)recordPtr)->size;
		entryPtr->fid = ((struct dfs_open *)recordPtr)->fid;
		entryPtr->dirFid = ((struct dfs_open *)recordPtr)->dirFid;
		if (((struct dfs_open *)recordPtr)->path) {
			entryPtr->path = (char *) 
				malloc(strlen(((struct dfs_open *)recordPtr)->path) + 1);
			(void) strcpy(entryPtr->path, ((struct dfs_open *)recordPtr)->path);
		}
	} else {
		entryPtr->fd = ((struct dfs_creat *)recordPtr)->fd;
		entryPtr->flags = DFS_FWRITE|DFS_FCREAT|DFS_FTRUNC;
		entryPtr->mode = ((struct dfs_creat *)recordPtr)->mode;
		entryPtr->fileType = DFS_IFREG | (entryPtr->mode&~DFS_IFMT);
		entryPtr->uid = ((struct dfs_creat *)recordPtr)->uid;
		entryPtr->oldSize = ((struct dfs_creat *)recordPtr)->oldSize;
		entryPtr->size = 0;
		entryPtr->fid = ((struct dfs_creat *)recordPtr)->fid;
		entryPtr->dirFid = ((struct dfs_creat *)recordPtr)->dirFid;
		if (((struct dfs_creat *)recordPtr)->path) {
			entryPtr->path = (char *) 
				malloc(strlen(((struct dfs_creat *)recordPtr)->path)+1);
			(void) strcpy(entryPtr->path, ((struct dfs_creat *)recordPtr)->path);
		}
	}
}		

/* 
 * FileTableUpdate -- updates the file table entry associated with 
 * the index in record recordPtr, assuming that an open or creat
 * has been seen for the record. File statistics are updated only
 * on seek -- reads and writes may not be in the trace.
 */
void FileTableUpdate(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	file_table_entry_t *entryPtr;
	short findex;

	assert(recordPtr->opcode == DFS_SEEK);

	findex = Trace_GetFileIndex(recordPtr);
	if (!tfPtr->kernFileTable[findex].allocated)
		return;

	entryPtr = &tfPtr->kernFileTable[findex];
	entryPtr->numSeeks++;
	entryPtr->numReads += ((struct dfs_seek *)recordPtr)->numReads;
	entryPtr->numWrites += ((struct dfs_seek *)recordPtr)->numWrites;
	entryPtr->bytesRead += ((struct dfs_seek *)recordPtr)->bytesRead;
	entryPtr->bytesWritten += ((struct dfs_seek *)recordPtr)->bytesWritten;
	entryPtr->offset = ((struct dfs_seek *)recordPtr)->offset;
}

/* 
 * FileTableExtract -- grabs information from the file table to
 * fill out close, seek, read and write records.
 */
void FileTableExtract(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	file_table_entry_t *entryPtr;
	short findex;

	assert((recordPtr->opcode == DFS_CLOSE) ||
	       (recordPtr->opcode == DFS_SEEK) ||
	       (recordPtr->opcode == DFS_WRITE) ||
	       (recordPtr->opcode == DFS_READ));

	findex = Trace_GetFileIndex(recordPtr);
	if (!tfPtr->kernFileTable[findex].allocated)
		return;

	entryPtr = &tfPtr->kernFileTable[findex];
	switch (recordPtr->opcode) {
	case DFS_CLOSE:
		((struct dfs_close *)recordPtr)->fid = entryPtr->fid;
		((struct dfs_close *)recordPtr)->fileType = entryPtr->fileType;
		((struct dfs_close *)recordPtr)->flags = entryPtr->flags;
		((struct dfs_close *)recordPtr)->oldSize = entryPtr->size;

		/* compare seek counts as a sanity check */
		if ((tfPtr->version == TRACE_VERSION_CMU3) &&
		    (((struct dfs_close *)recordPtr)->numSeeks != entryPtr->numSeeks))
			printf("FileTableUpdate: Warning: #seeks: %d table, %d close\n",
			       entryPtr->numSeeks, ((struct dfs_close *)recordPtr)->numSeeks);

		/* update stats, offset */
		((struct dfs_close *)recordPtr)->numReads += entryPtr->numReads;
		((struct dfs_close *)recordPtr)->numWrites += entryPtr->numWrites;
		((struct dfs_close *)recordPtr)->bytesRead += entryPtr->bytesRead;
		((struct dfs_close *)recordPtr)->bytesWritten += entryPtr->bytesWritten;
		((struct dfs_close *)recordPtr)->offset = entryPtr->offset +
			((struct dfs_close *)recordPtr)->bytesRead +
				((struct dfs_close *)recordPtr)->bytesWritten;

		if (entryPtr->path) {
			((struct dfs_close *)recordPtr)->path = (char *) 
				malloc(strlen(entryPtr->path)+1);
			(void) strcpy(((struct dfs_close *)recordPtr)->path, entryPtr->path);
		}
		break;
	case DFS_SEEK:
		((struct dfs_seek *)recordPtr)->fid = entryPtr->fid;
		((struct dfs_seek *)recordPtr)->oldOffset = entryPtr->offset +
			((struct dfs_seek *)recordPtr)->bytesRead +
				((struct dfs_seek *)recordPtr)->bytesWritten;
		if (entryPtr->path) {
			((struct dfs_seek *)recordPtr)->path = (char *) 
				malloc(strlen(entryPtr->path)+1);
			(void) strcpy(((struct dfs_seek *)recordPtr)->path, entryPtr->path);
		}
		break;
	case DFS_READ:
	case DFS_WRITE:
		((struct dfs_read *)recordPtr)->fid = entryPtr->fid;
		if (entryPtr->path) {
			((struct dfs_read *)recordPtr)->path = (char *) 
				malloc(strlen(entryPtr->path)+1);
			(void) strcpy(((struct dfs_read *)recordPtr)->path, entryPtr->path);
		}
		break;
	default:
		break;
	}
}

/* 
 * FileTableDelete -- clears the entry associated with the index in recordPtr.
 * Appropriate for close.
 */
void FileTableDelete(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	short findex;
	file_table_entry_t *entryPtr;

	assert((recordPtr->opcode == DFS_CLOSE) &&
	       (((struct dfs_close *)recordPtr)->refCount == 1));

	findex = Trace_GetFileIndex(recordPtr);
	entryPtr = &tfPtr->kernFileTable[findex];
	if (entryPtr->path)
		free(entryPtr->path);
	bzero(entryPtr, sizeof(file_table_entry_t));
	entryPtr->fid.tag = -1;     /* invalidate fid */
	entryPtr->dirFid.tag = -1;
}

/*
 * DoFileTable -- shell for stuff above, called mindlessly from 
 * GetRecord.  If the record is an open or creat, insert an entry
 * into the file table. If it's a read, write, or seek, or close, 
 * fill out the fields of those records. Update the table with
 * new seek statistics.  We do not clear the entry from the table
 * on close here if filtering is "on"; we do it after that check is done.
 */
void DoFileTable(tfPtr, recordPtr)
trace_file_t *tfPtr;
dfs_header_t *recordPtr;
{
	assert(recordPtr->error == 0);

	switch (recordPtr->opcode) {
	case DFS_OPEN:
	case DFS_CREAT:
		FileTableInsert(tfPtr, recordPtr);
		break;
	case DFS_SEEK:
		FileTableExtract(tfPtr, recordPtr);
		FileTableUpdate(tfPtr, recordPtr);
		break;
	case DFS_READ:
	case DFS_WRITE:
		FileTableExtract(tfPtr, recordPtr);
		break;
	case DFS_CLOSE:
		FileTableExtract(tfPtr, recordPtr);
		if (!tfPtr->filterPtr->matchfds &&
		    (((struct dfs_close *)recordPtr)->refCount == 1))
			FileTableDelete(tfPtr, recordPtr);
		break;
	}
}

