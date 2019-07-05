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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/unpack_private.h,v 1.2 1998/10/13 21:14:32 tmkr Exp $";
#endif _BLURB_


/*
 * unpack_private.h -- macros for unpacking and swapping things,
 * used by all unpacking modules.
 */

#ifndef _UNPACK_PRIVATE_H_
#define _UNPACK_PRIVATE_H_

#define DECODE_AMOUNT 48     /* number of bytes to read to figure out
                                what sort of a trace we have. length
                                of longest file header+chunk header+
				record header. */
#define DFS_LOG_UNPACK_TWO(_item_, _buf_, _offset_)                           \
    do {                                                                      \
	*((u_short *) &(_item_)) = ntohs(*((u_short *) &_buf_[_offset_]));    \
	_offset_ += 2;                                                        \
    } while (0)

#define DFS_LOG_UNPACK_FOUR(_item_, _buf_, _offset_)                          \
    do {                                                                      \
	*((u_long *) &(_item_)) = ntohl(*((u_long *) &_buf_[_offset_]));      \
	_offset_ += 4;                                                        \
    } while (0)

#define DFS_LOG_UNPACK_HEADER(_header_, _buf_, _offset_)                      \
    do {                                                                      \
	_header_.opcode = _buf_[_offset_++];                                  \
	_header_.flags = _buf_[_offset_++];                                   \
	_header_.error = _buf_[_offset_++];                                   \
	vntype = _buf_[_offset_++];                                           \
        DFS_LOG_UNPACK_TWO(_header_.pid, _buf_, _offset_);                    \
        DFS_LOG_UNPACK_FOUR(_header_.time.tv_sec, _buf_, _offset_);           \
        DFS_LOG_UNPACK_FOUR(_header_.time.tv_usec, _buf_, _offset_);          \
    } while (0)

#define DFS_LOG_UNPACK_STRING(_path_, _length_, _buf_, _offset_)              \
     if (_length_ > 0) {                                                      \
	      int _i_;							      \
	      _path_ = (char *) malloc(_length_+1);                           \
	      for (_i_ = 0; _i_ < (int) _length_; _i_++)                      \
		      _path_[_i_] = _buf_[_offset_++];                        \
	      _path_[_length_] = 0;   /* null terminate */                    \
      }			      
             
#define DFS_LOG_UNPACK_PATH(_path_, _length_, _flags_, _buf_, _offset_)       \
   if (!(_flags_ & DFS_NOPATH)) {                                             \
	      DFS_LOG_UNPACK_TWO(_length_, _buf_, _offset_);                  \
	      DFS_LOG_UNPACK_STRING(_path_, _length_, _buf_, _offset_);       \
    }

#define DFS_LOG_UNPACK_FID(_fid_, _vntype_, _bad_, _buf_, _offset_)           \
	if (_bad_) _fid_.tag = -1;                                            \
	else _fid_.tag = (short) _vntype_;                                    \
	switch (_fid_.tag) {                                                  \
	case DFS_ITYPE_UFS:                                                    \
	case DFS_ITYPE_NFS:                                                    \
	case DFS_ITYPE_SPEC:						      	\
		DFS_LOG_UNPACK_FOUR(_fid_.value.local.device, _buf_, _offset_);\
		DFS_LOG_UNPACK_FOUR(_fid_.value.local.number, _buf_, _offset_);\
		break;                                                        \
	case DFS_ITYPE_AFS:                                                   \
		DFS_LOG_UNPACK_FOUR(_fid_.value.afs.Cell, _buf_, _offset_);   \
		DFS_LOG_UNPACK_FOUR(_fid_.value.afs.Fid.Volume, _buf_, _offset_);\
		DFS_LOG_UNPACK_FOUR(_fid_.value.afs.Fid.Vnode, _buf_, _offset_);\
		DFS_LOG_UNPACK_FOUR(_fid_.value.afs.Fid.Unique, _buf_, _offset_);\
		break;                                                        \
	case DFS_ITYPE_CFS:                                                   \
		DFS_LOG_UNPACK_FOUR(_fid_.value.cfs.Volume, _buf_, _offset_); \
		DFS_LOG_UNPACK_FOUR(_fid_.value.cfs.Vnode, _buf_, _offset_);  \
		DFS_LOG_UNPACK_FOUR(_fid_.value.cfs.Unique, _buf_, _offset_); \
		break;                                                        \
	}                                                                     \
             
#endif _UNPACK_PRIVATE_H_
