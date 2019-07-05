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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/filter.h,v 1.1.1.1 1998/09/29 18:39:18 tmkr Exp $";
#endif _BLURB_


/*
 * filter.h  -- filter specific definitions
 */

#ifndef _FILTER_H_
#define _FILTER_H_

#include "tracelib.h"

/* functions exported to trace.c */
extern void ParseFilterFile();
extern void InitFilter();
extern int FilteredOut();
extern int TimeIsUp();

/* constants */
#define MAXCMDLEN 1024  /* command length in filter file */
#define MAXFILES 1024   /* guess at max number of entries in kernel file table */
#define MAXPIDARRAY 256 /* number of pids we allow to be specified for filter */
#define MAXUSERARRAY 50 /* number of users we allow to be specified for filter */
#define MAXPATHARRAY 50 /* number of paths we can filter at one time */
#define MAXERRARRAY 10  /* number of error codes specified in a single filter */
#define MAXWIDTH 2      /* width of opcode array. maxopcode/32 + 1. */

#define FILTER_INCLUDE  0x0  /* default */
#define FILTER_EXCLUDE  0x1 

/* pfil_t -- for filtering by pids and pid subtrees */
typedef struct pfil {
	short pid;
	char  kid;  /* if set, will filter pid and descendants */
} pfil_t;

/* filter_t -- for filtering traces in various ways. */
typedef struct filter {
	u_long   opcodes[2];         /* bit field -- see tracelib.h */
	u_short  fileTypes;          /* bit field: IFCHR,IFDIR,IFBLK,IFREG,IFLNK,IFSOCK */ 
	u_char   inodeTypes;         /* bit field, 0-5 */
	short    refCount;
	char     matchfds;           /* match file descriptor based ops w/opens */
	char     pidFlag;            /* including or excluding pids (+children) */
	char     userFlag;           /* including or excluding users */
	char     pathFlag;           /* including or excluding paths */
	int    **errorList;
	pfil_t **pidList;
	uid_t  **userList; 
	char   **pathList;           /* path name filtering */
	struct timeval startTime;
	struct timeval endTime;
} filter_t; 

/* for parsing the filter command file. */
#define	STREQ(a, b) (strcasecmp(a, b) == 0)

#endif _FILTER_H_
