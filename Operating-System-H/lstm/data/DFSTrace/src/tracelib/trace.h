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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/trace.h,v 1.3 1998/10/06 19:40:31 tmkr Exp $";
#endif _BLURB_


/* 
 * trace.h.  Defines the trace file struct, which is visible
 * to other modules.
 */
#ifndef _TRACE_H_
#define _TRACE_H_

#include "tracelib.h"
#include "filter.h"
#include "pid.h"
#include "unpack.h"

/* file_table_t -- for simulating the kernel file table */
typedef struct file_table_entry {
	u_char        allocated; /* saw the open _in_this_trace_ */
	u_char        filtered;
	short         fd;
	u_short       flags;     /* file mode (-FOPEN) */
	u_short       mode;      /* create mode */
	u_short       fileType;  /* file type (dir, link, etc.) and mode */
	uid_t         uid;
	long          oldSize;   /* if file already there */
	u_long        size;
	generic_fid_t fid;
	generic_fid_t dirFid;
	u_short	      numReads;
	u_short	      numWrites;
	u_short	      numSeeks;
	u_long	      bytesRead;
	u_long	      bytesWritten;
	u_long        offset;
	char         *path;
} file_table_entry_t;

#define TRACE_HASH_SIZE 101

/* 
 * trace_file_t
 * information about a trace file and the trace contained therein.
 * The preamble and chunk fields are set and interpreted by 
 * the unpacking routines. 
 */
typedef struct trace_file {
	struct trace_file *traceFileLink;       /* for file hash table */
	FILE              *fp;                  /* the key */
	char              *fileName; 
        int                compressed;          /*indicate if the trace
						 * file  is compressed */
        char              *command;             /* the uncompress command */
	int                version;             /* tag for next three fields */
	char              *preamblePtr;         /* file preamble */
	char              *chunkPreamblePtr;    /* last chunk preamble */
	char              *chunk;               /* last chunk read */
	int                chunkOffset;         /* offset in current chunk */
        filter_t          *filterPtr;           /* set if filter defined */
	void             (*unpackRecordProc)();
	trace_stat_t       traceStats;          /* statistics so far */
	user_t            *userHashTable[USER_HASH_SIZE];  /* set of users for this trace */
	file_table_entry_t kernFileTable[MAXFILES]; /* simulated file table for this trace */
} trace_file_t;

extern char *VersionToStr();
extern void FreeTraceFile();
#endif _TRACE_H_


