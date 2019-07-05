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

*/

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/split.h,v 1.1.1.1 1998/09/29 18:39:18 tmkr Exp $";
#endif _BLURB_

/*
 * split.h -- trace record structures used internally by the library
 * for dealing with split records.
 */

#ifndef _SPLIT_H_
#define _SPLIT_H_

#include "tracelib.h"

/* routines exported to trace.c */
extern void StashSplitRecord();
extern void GetSplitRecord();
extern dfs_header_t *MergeSplitRecords();

#define record_is_split(_opcode_) \
	(((_opcode_) == DFS_RMDIR) || ((_opcode_) == DFS_UNLINK) || \
	 ((_opcode_) == DFS_UNMOUNT) || ((_opcode_) == DFS_OPEN) || \
	 ((_opcode_) == DFS_MKDIR) || ((_opcode_) == DFS_CREAT) || \
	 ((_opcode_) == DFS_TRUNCATE) || ((_opcode_) == DFS_MKNOD) || \
	 ((_opcode_) == DFS_RENAME))

/* 
 * Some trace records are split -- that is, they are written in two parts, because
 * not all of the information desired was available at one time.  These are referred to
 * as "pre" and "post" records.  Post records are always recorded. Pre records may
 * or may not be recorded, depending on whether or not the call fails early. The
 * thread address is recorded with split records, to aid in matching.
 */

/* trace_merge_t -- for merging pre and post records. */
typedef struct trace_merge {
	struct trace_merge *nextPtr;
	dfs_header_t *recPtr;
} trace_merge_t;

/* pre and post records */
/* rmdir */
struct dfs_pre_rmdir {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t fid;
	generic_fid_t dirFid;
	u_long size;
	u_short fileType;
	short numLinks;
};

struct dfs_post_rmdir {
	dfs_header_t header;
	caddr_t threadAddr;
	u_short pathLength;
	char *path;
};

/* unmount */
struct dfs_pre_unmount {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t fid;
};

struct dfs_post_unmount {
	dfs_header_t header;
	caddr_t threadAddr;
	u_short pathLength;
	char *path;
};
	
/* open */
struct dfs_pre_open {
        dfs_header_t header;
	caddr_t threadAddr;
	long oldSize;   /* if file was already there */
	generic_fid_t dirFid;
};

struct dfs_post_open {
        dfs_header_t header;
	caddr_t threadAddr;
	u_short flags;  /* file mode */
	u_short mode;   /* create mode */
	short	fd; 
	u_short fileType;  /* file type (dir, link, etc.) and mode */
	uid_t	uid;   /* owner uid, a u_short */
	u_long	size;       
	generic_fid_t fid;
	short findex;
	u_short pathLength;
	char *path;
};			

/* creat */
struct dfs_pre_creat {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t dirFid;
	long    oldSize;   /* if object already there */
};

struct dfs_post_creat {	
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t fid;
	short	fd; 
	u_short mode;
	short findex;
	u_short pathLength;
	char *path;
};

/* mkdir */
struct dfs_pre_mkdir {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t dirFid;
};

struct dfs_post_mkdir {	
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t fid;
	u_short mode;
	u_short pathLength;
	char *path;
};

/* rename */
struct dfs_pre_rename {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t toFid;
	u_long size;
	short numLinks; 
};

struct dfs_post_rename {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t fromFid;
	generic_fid_t fromDirFid;
	generic_fid_t toDirFid;
	u_short fileType;  /* file type (dir, link, etc.) */
	u_short fromPathLength;
	u_short toPathLength;
	char *fromPath;
	char *toPath;
};

/* truncate */
struct dfs_pre_truncate { 
	dfs_header_t header;
	caddr_t threadAddr;
	long    oldSize;
};

struct dfs_post_truncate { 
	dfs_header_t header;
	caddr_t threadAddr;
	u_long	newSize;
	generic_fid_t fid;
	u_short	pathLength;
	char *path;
};

/* mknod */
struct dfs_pre_mknod {
	dfs_header_t header;
	caddr_t threadAddr;
	generic_fid_t dirFid;
};

struct dfs_post_mknod { 
	dfs_header_t header;
	caddr_t threadAddr;
	int dev;
	generic_fid_t fid;
	u_short mode;
	u_short pathLength;
	char *path;
};
	
#define trace_headers_equal(_prePtr_, _postPtr_) \
	((_prePtr_->opcode == _postPtr_->opcode) && \
	 (_prePtr_->pid == _postPtr_->pid) && \
	 (!(_prePtr_->flags & DFS_POST)) && \
	 (_postPtr_->flags & DFS_POST) && \
	 (*((int *)((char *)_prePtr_+sizeof(dfs_header_t))) == \
	     *((int *) ((char *)_postPtr_+sizeof(dfs_header_t)))) ) /* tid */

#endif _SPLIT_H_
