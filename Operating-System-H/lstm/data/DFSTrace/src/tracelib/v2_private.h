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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/v2_private.h,v 1.1.1.1 1998/09/29 18:39:19 tmkr Exp $";
#endif _BLURB_


/* 
 * v2_private.h -- Information specific to Version 2 traces.
 */

#ifndef _V2_PRIVATE_H_
#define _V2_PRIVATE_H_

#include <sys/types.h>
#include <sys/time.h>

#define V2_START   673070400   /* 5/1/91 00:00 in seconds */
#define V2_END     688971600   /* 11/1/91 00:00 in seconds */

#define DFS_TRACE_CHUNK_SIZE_V2 4096

#define DFS_MAXSYSCALL_V2  0x1e
#define DFS_MAXOPCODE_V2   0x22

#define KERNELVERSIONMAX_V2     8      /* kernel minor version */
#define AGENTVERSIONMAX_V2      6      /* agent minor version */
#define COLLECTORVERSIONMAX_V2  6      /* collector minor version */

/* 
 * trace file preamble -- written by collection server. 
 * Data fields in network order.
 */
typedef struct dfs_file_preamble_v2 {
	u_int   hostAddress;
	time_t  bootTime;  /* seconds only */
	time_t  agentBirthTime;
	u_short kernelVersion;
	u_short agentVersion;
	u_short collectorVersion;
} dfs_file_preamble_v2_t;

/* tracing levels */
#define DFS_TRACE_OFF 0x0  /* tracing off */
#define DFS_TRACE_ON 0x01   /* tracing on, default */
#define DFS_TRACE_NAMERES 0x02  /* trace name resolution data */
#define DFS_TRACE_READWRITE 0x04  /* trace read/write data */
#define DFS_TRACE_ALL 0x07  /* trace everything */

/* 
 * chunk preamble -- written by agent. Data fields in network order.
 */
typedef struct dfs_chunk_preamble_v2 {
	u_int   chunkSeq;
	u_short traceLevel; /* levels defined above */
	u_short serverTroubles;  /* a flag. Could be smaller if need be. */
	u_int   bytesLost;
} dfs_chunk_preamble_v2_t;

#endif _V2_PRIVATE_H_
