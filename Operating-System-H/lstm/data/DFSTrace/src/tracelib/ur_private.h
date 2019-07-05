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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/ur_private.h,v 1.1.1.1 1998/09/29 18:39:19 tmkr Exp $";
#endif _BLURB_


/* 
 * ur_private.h -- information specific to converted Floyd-style traces
 */
#ifndef _UR_PRIVATE_H_
#define _UR_PRIVATE_H_

#include <sys/types.h>
#include "tracelib.h"

typedef struct dfs_file_preamble_ur {
	char  hostName[8];
} dfs_file_preamble_ur_t;

/* unpacking macros */
#define SHORTALIGN(_offset_)					              \
do {                                                                          \
	char buf;							      \
	if (_offset_ & 0x1) {                                                 \
		(void) fread(&buf, 1, sizeof(char), tfPtr->fp);               \
		_offset_++;                                                   \
	}		                                                      \
} while (0)

#define LONGALIGN(_offset_)					              \
do {                                                                          \
	char buf[2];							      \
	SHORTALIGN(_offset_);                                                 \
	if (_offset_ & 0x2) {                                                 \
		(void) fread(buf, 1, sizeof(short), tfPtr->fp);               \
		_offset_ += 2;                                                \
	}                                                                     \
} while (0)

#define UNPACK1(_item_, _offset_)                                             \
do {                                                                          \
        (void) fread(&_item_, 1, sizeof(char), tfPtr->fp);                    \
	_offset_++;	                                                      \
} while (0)
	
#define UNPACK2(_item_, _offset_)                                             \
do {                                                                          \
	SHORTALIGN(_offset_);				                      \
	(void) fread(&_item_, 1, sizeof(short), tfPtr->fp);                   \
	*((u_short *) &(_item_)) = ntohs(*((u_short *) _item_));              \
	_offset_ += 2;	                                                      \
} while (0)

#define UNPACK4(_item_, _offset_)                                             \
do {                                                                          \
	LONGALIGN(_offset_);				                      \
	(void) fread(&_item_, 1, sizeof(long), tfPtr->fp);                    \
	*((u_long *) &(_item_)) = ntohl(*((u_long *) _item_));                \
	_offset_ += 4;	                                                      \
} while (0)

#define UNPACKFID(_fid_, _offset_)                                            \
do {                                                                          \
	_fid_.tag = DFS_ITYPE_UFS;                                            \
	UNPACK4(_fid_.value.local.device, _offset_);                          \
	UNPACK4(_fid_.value.local.number, _offset_);                          \
} while (0)
		
#endif _UR_PRIVATE_H_




