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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/v1_private.h,v 1.1.1.1 1998/09/29 18:39:19 tmkr Exp $";
#endif _BLURB_


/* 
 * v1_private.h -- information specific to Version 1 traces.
 */

#ifndef _V1_PRIVATE_H_
#define _V1_PRIVATE_H_

#include <sys/types.h>
#include <sys/time.h>

#define V1_START   666939600   /* 2/19/91 00:00 in seconds */
#define V1_END     673070400   /* 5/1/91 00:00 in seconds */

#define DFS_TRACE_CHUNK_SIZE_V1 1024

/* 
 * opcodes DFS_LOOKUP through DFS_SYSCALLDUMP must be translated
 * to their appropriate values in v2-speak before being passed
 * on. Also DFS_MAXSYSCALL and DFS_MAXOPCODE are different internally.
 */
#define DFS_LOOKUP_V1      0x1d
#define DFS_GETSYMLINK_V1  0x1e
#define DFS_ROOT_V1        0x1f
#define DFS_SYSCALLDUMP_V1 0x20

#define DFS_MAXSYSCALL_V1  0x1c
#define DFS_MAXOPCODE_V1   0x20

/* 
 * trace file preamble -- written by collection server. 
 * Data fields in network order. 
 * host address(4), boot time(4), compile time(4)
 */
typedef struct dfs_file_preamble_v1 {
	u_int   hostAddress;
	time_t  bootTime;  /* seconds only */
	time_t  versionTime;
} dfs_file_preamble_v1_t;

/* 
 * chunk preamble -- written by agent. Data fields in network order.
 * agent birth time(4), chunk sequence number(4), server troubles(4), 
 * bytes lost(4)
 */
typedef struct dfs_chunk_preamble_v1 {
	time_t  agentBirthTime;  /* This and the next field */
	u_int   chunkSeq;        /*      yield a unique chunk id */
	u_int   serverTroubles;  /* a flag. Could be smaller if need be. */
	u_int   bytesLost;
} dfs_chunk_preamble_v1_t;

#endif _V1_PRIVATE_H_
