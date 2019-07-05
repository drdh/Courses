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


static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/split.c,v 1.2 1998/10/06 19:40:30 tmkr Exp $";
*/
#endif _BLURB_


/*
 * split.c -- code for handling split records
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "split.h"
#include "tracelib.h"

trace_merge_t *headPtr = NULL;

void StashSplitRecord(recPtr)
dfs_header_t *recPtr;
{
        trace_merge_t *newTracePtr;
	
	if (debug) 
		printf("Stashing pre record, opcode %s.\n", 
		        Trace_OpcodeToStr(recPtr->opcode));
	newTracePtr = (trace_merge_t *) malloc(sizeof(trace_merge_t));
	newTracePtr->recPtr = recPtr;
	newTracePtr->nextPtr = headPtr;
	headPtr = newTracePtr;
}

void GetSplitRecord(part2Ptr, part1PtrPtr)
dfs_header_t *part2Ptr;
dfs_header_t **part1PtrPtr;
{
        trace_merge_t *curMatchPtr, *prevMatchPtr;

	if (debug) 
		printf("Looking for pre record, opcode %s...", 
		       Trace_OpcodeToStr(part2Ptr->opcode));
	*part1PtrPtr = NULL;
	if (headPtr == NULL)
		return;

	curMatchPtr = headPtr;
	if (trace_headers_equal(curMatchPtr->recPtr, part2Ptr)) { /* common case */
		*part1PtrPtr = curMatchPtr->recPtr;
		headPtr = headPtr->nextPtr;
		free(curMatchPtr);
		if (debug) printf("found.\n");
		return;
	}
	prevMatchPtr = curMatchPtr;
	curMatchPtr = headPtr->nextPtr;
	while (curMatchPtr) {
		if (trace_headers_equal(curMatchPtr->recPtr, part2Ptr)) {
			*part1PtrPtr = curMatchPtr->recPtr;
			prevMatchPtr->nextPtr = curMatchPtr->nextPtr;
			free(curMatchPtr);
			if (debug) printf("found.\n");
			return;
		} else {
			prevMatchPtr = curMatchPtr;
			curMatchPtr = curMatchPtr->nextPtr;
		}
	}
}

/* 
 * MergeSplitRecords -- internal routine to merge pre and post records. 
 * A pointer to the merged record is returned.  The pre and post records
 * are freed. The header of the merged record is the post record header, since
 * that contains the time closest to the time the call actually finished,
 * and also contains the final error code.  Note the pre record ptr may be null.
 */
dfs_header_t *MergeSplitRecords(prePtr, postPtr)
dfs_header_t *prePtr;
dfs_header_t *postPtr;
{
	dfs_header_t *mergedPtr = NULL;

	switch (postPtr->opcode) {
	case DFS_OPEN:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_open));
		((struct dfs_open *)mergedPtr)->header = 
			((struct dfs_post_open *)postPtr)->header;
		((struct dfs_open *)mergedPtr)->flags = 
			((struct dfs_post_open *)postPtr)->flags;
		((struct dfs_open *)mergedPtr)->mode = 
			((struct dfs_post_open *)postPtr)->mode;
		((struct dfs_open *)mergedPtr)->fd = 
			((struct dfs_post_open *)postPtr)->fd;
		((struct dfs_open *)mergedPtr)->findex = 
			((struct dfs_post_open *)postPtr)->findex;
		((struct dfs_open *)mergedPtr)->fileType = 
			((struct dfs_post_open *)postPtr)->fileType;
		((struct dfs_open *)mergedPtr)->uid = 
			((struct dfs_post_open *)postPtr)->uid;
		if (prePtr) {
			((struct dfs_open *)mergedPtr)->oldSize = 
				((struct dfs_pre_open *)prePtr)->oldSize;
			((struct dfs_open *)mergedPtr)->dirFid = 
				((struct dfs_pre_open *)prePtr)->dirFid;
		} else {
			((struct dfs_open *)mergedPtr)->oldSize = -1;
			((struct dfs_open *)mergedPtr)->dirFid.tag = -1;
		}
		((struct dfs_open *)mergedPtr)->size = 
			((struct dfs_post_open *)postPtr)->size;
		((struct dfs_open *)mergedPtr)->fid = 
			((struct dfs_post_open *)postPtr)->fid;
		((struct dfs_open *)mergedPtr)->header = 
			((struct dfs_post_open *)postPtr)->header;
		((struct dfs_open *)mergedPtr)->pathLength = 
			((struct dfs_post_open *)postPtr)->pathLength;
		((struct dfs_open *)mergedPtr)->path = 
			((struct dfs_post_open *)postPtr)->path;
		break;
	case DFS_MKDIR:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_mkdir));
		((struct dfs_mkdir *)mergedPtr)->header = 
			((struct dfs_post_mkdir *)postPtr)->header;
		((struct dfs_mkdir *)mergedPtr)->fid = 
			((struct dfs_post_mkdir *)postPtr)->fid;
		if (prePtr)
			((struct dfs_mkdir *)mergedPtr)->dirFid = 
				((struct dfs_pre_mkdir *)prePtr)->dirFid;
		else
			((struct dfs_mkdir *)mergedPtr)->dirFid.tag = -1;
		((struct dfs_mkdir *)mergedPtr)->mode = 
			((struct dfs_post_mkdir *)postPtr)->mode;
		((struct dfs_mkdir *)mergedPtr)->pathLength = 
			((struct dfs_post_mkdir *)postPtr)->pathLength;
		((struct dfs_mkdir *)mergedPtr)->path = 
			((struct dfs_post_mkdir *)postPtr)->path;
		break;
	case DFS_CREAT:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_creat));
		((struct dfs_creat *)mergedPtr)->header = 
			((struct dfs_post_creat *)postPtr)->header;
		((struct dfs_creat *)mergedPtr)->fid = 
			((struct dfs_post_creat *)postPtr)->fid;
		if (prePtr) {
			((struct dfs_creat *)mergedPtr)->dirFid = 
				((struct dfs_pre_creat *)prePtr)->dirFid;
			((struct dfs_creat *)mergedPtr)->oldSize = 
				((struct dfs_pre_creat *)prePtr)->oldSize;
		} else {
			((struct dfs_creat *)mergedPtr)->dirFid.tag = -1;
			((struct dfs_creat *)mergedPtr)->oldSize = -1;
		}
		((struct dfs_creat *)mergedPtr)->fd = 
			((struct dfs_post_creat *)postPtr)->fd;
		((struct dfs_creat *)mergedPtr)->findex = 
			((struct dfs_post_creat *)postPtr)->findex;
		((struct dfs_creat *)mergedPtr)->mode = 
			((struct dfs_post_creat *)postPtr)->mode;
		((struct dfs_creat *)mergedPtr)->pathLength = 
			((struct dfs_post_creat *)postPtr)->pathLength;
		((struct dfs_creat *)mergedPtr)->path = 
			((struct dfs_post_creat *)postPtr)->path;
		break;
	case DFS_TRUNCATE:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_truncate));
		((struct dfs_truncate *)mergedPtr)->header = 
			((struct dfs_post_truncate *)postPtr)->header;
		if (prePtr)
			((struct dfs_truncate *)mergedPtr)->oldSize = 
				((struct dfs_pre_truncate *)prePtr)->oldSize;
		else
			((struct dfs_truncate *)mergedPtr)->oldSize = -1;
		((struct dfs_truncate *)mergedPtr)->newSize = 
			((struct dfs_post_truncate *)postPtr)->newSize;
		((struct dfs_truncate *)mergedPtr)->fid = 
			((struct dfs_post_truncate *)postPtr)->fid;
		((struct dfs_truncate *)mergedPtr)->pathLength = 
			((struct dfs_post_truncate *)postPtr)->pathLength;
		((struct dfs_truncate *)mergedPtr)->path = 
			((struct dfs_post_truncate *)postPtr)->path;
		break;
	case DFS_MKNOD:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_mknod));
		((struct dfs_mknod *)mergedPtr)->header = 
			((struct dfs_post_mknod *)postPtr)->header;
		((struct dfs_mknod *)mergedPtr)->dev = 
			((struct dfs_post_mknod *)postPtr)->dev;
		((struct dfs_mknod *)mergedPtr)->fid = 
			((struct dfs_post_mknod *)postPtr)->fid;
		if (prePtr)
			((struct dfs_mknod *)mergedPtr)->dirFid = 
				((struct dfs_pre_mknod *)prePtr)->dirFid;
		else
			((struct dfs_mknod *)mergedPtr)->dirFid.tag = -1;
		((struct dfs_mknod *)mergedPtr)->mode = 
			((struct dfs_post_mknod *)postPtr)->mode;
		((struct dfs_mknod *)mergedPtr)->pathLength = 
			((struct dfs_post_mknod *)postPtr)->pathLength;
		((struct dfs_mknod *)mergedPtr)->path = 
			((struct dfs_post_mknod *)postPtr)->path;
		break;
	case DFS_RENAME:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_rename));
		((struct dfs_rename *)mergedPtr)->header = 
			((struct dfs_post_rename *)postPtr)->header;
		((struct dfs_rename *)mergedPtr)->fromFid = 
			((struct dfs_post_rename *)postPtr)->fromFid;
		((struct dfs_rename *)mergedPtr)->fromDirFid = 
			((struct dfs_post_rename *)postPtr)->fromDirFid;
		((struct dfs_rename *)mergedPtr)->fileType = 
			((struct dfs_post_rename *)postPtr)->fileType;
		if (prePtr) {
			((struct dfs_rename *)mergedPtr)->toFid = 
				((struct dfs_pre_rename *)prePtr)->toFid;
			((struct dfs_rename *)mergedPtr)->size = 
				((struct dfs_pre_rename *)prePtr)->size;
			((struct dfs_rename *)mergedPtr)->numLinks = 
				((struct dfs_pre_rename *)prePtr)->numLinks;
		} else {
			((struct dfs_rename *)mergedPtr)->toFid.tag = -1;
			((struct dfs_rename *)mergedPtr)->size = -1;
			((struct dfs_rename *)mergedPtr)->numLinks = -1;
		}
		((struct dfs_rename *)mergedPtr)->toDirFid = 
			((struct dfs_post_rename *)postPtr)->toDirFid;
		((struct dfs_rename *)mergedPtr)->fromPathLength = 
			((struct dfs_post_rename *)postPtr)->fromPathLength;
		((struct dfs_rename *)mergedPtr)->toPathLength = 
			((struct dfs_post_rename *)postPtr)->toPathLength;
		((struct dfs_rename *)mergedPtr)->fromPath = 
			((struct dfs_post_rename *)postPtr)->fromPath;
		((struct dfs_rename *)mergedPtr)->toPath = 
			((struct dfs_post_rename *)postPtr)->toPath;
		break;
	case DFS_RMDIR:
	case DFS_UNLINK:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_rmdir));
		((struct dfs_rmdir *)mergedPtr)->header = 
			((struct dfs_post_rmdir *)postPtr)->header;
		if (prePtr) {
			((struct dfs_rmdir *)mergedPtr)->fid = 
				((struct dfs_pre_rmdir *)prePtr)->fid;
			((struct dfs_rmdir *)mergedPtr)->dirFid = 
				((struct dfs_pre_rmdir *)prePtr)->dirFid;
			((struct dfs_rmdir *)mergedPtr)->size = 
				((struct dfs_pre_rmdir *)prePtr)->size;
			((struct dfs_rmdir *)mergedPtr)->fileType = 
				((struct dfs_pre_rmdir *)prePtr)->fileType;
			((struct dfs_rmdir *)mergedPtr)->numLinks = 
				((struct dfs_pre_rmdir *)prePtr)->numLinks;
		} else {
			((struct dfs_rmdir *)mergedPtr)->fid.tag = -1;
			((struct dfs_rmdir *)mergedPtr)->dirFid.tag = -1;
			((struct dfs_rmdir *)mergedPtr)->size = -1;
			((struct dfs_rmdir *)mergedPtr)->fileType = -1;
			((struct dfs_rmdir *)mergedPtr)->numLinks = -1;
		}
		((struct dfs_rmdir *)mergedPtr)->pathLength = 
			((struct dfs_post_rmdir *)postPtr)->pathLength;
		((struct dfs_rmdir *)mergedPtr)->path = 
			((struct dfs_post_rmdir *)postPtr)->path;
		break;
	case DFS_UNMOUNT:
		mergedPtr = (dfs_header_t *) malloc(sizeof(struct dfs_unmount));
		if (prePtr) 
			((struct dfs_unmount *)mergedPtr)->fid = 
				((struct dfs_pre_unmount *)prePtr)->fid;
		else 
			((struct dfs_unmount *)mergedPtr)->fid.tag = -1;
		((struct dfs_unmount *)mergedPtr)->header = 
			((struct dfs_post_unmount *)postPtr)->header;
		((struct dfs_unmount *)mergedPtr)->pathLength = 
			((struct dfs_post_unmount *)postPtr)->pathLength;
		((struct dfs_unmount *)mergedPtr)->path = 
			((struct dfs_post_unmount *)postPtr)->path;
		break;
	default:
		printf("Warning: MergeRecords: Opcode %s is not split!\n",
		       Trace_OpcodeToStr(prePtr->opcode));
		break;
	}

	/* 
         * note we do not call Trace_FreeRecord, first because pre records do not
         * have paths (so it's not necessary), and second because we don't want 
         * the path freed anyway, since it's now part of the merged record.
         */
	if (prePtr) free(prePtr);
	free(postPtr);  
	mergedPtr->flags &= (u_char) ~DFS_POST;  /* don't make this distinction */
	return(mergedPtr);
}


