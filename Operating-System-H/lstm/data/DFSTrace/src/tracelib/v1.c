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


static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/v1.c,v 1.4 1998/10/23 03:37:57 tmkr Exp $";
*/
#endif _BLURB_


/*
 * v1.c -- code for unpacking version 1 traces.
 */
#include <sys/param.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <time.h>
#include <sys/errno.h>
#include <sys/time.h>
#include "tracelib.h"
#include "trace.h"
#include "split.h"
#include "unpack_private.h"
#include "v1_private.h"

/*
 * Packing list for Version 1 traces.
 * All data fields in network order.
 *
 * trace file preamble        host address(4), boot time(4), version time(4)
 *
 * chunk preamble             agent birth time(4), chunk sequence number(4),
 *                            server troubles(4), bytes lost(4)
 *
 * trace record header        record length(2), opcode(1), flags(1),
 *                            error code(1), vnode types(1), pid(2), time(8)
 *
 * Vnode types are defined in sys/inode.h, and fit in a nybble. If the
 * system call concerns one object, the vnode type is in the lower
 * nybble. If the vnode type is 0xf, that means we couldn't get the
 * vnode type (i.e., the vnode pointer was NULL).  There are two
 * system calls that concern more than two objects -- link and rename.
 * For those, four bytes are used in the record body to store the type
 * of each object. (They could have fit in two, but that wouldn't have
 * helped.)
 *
 * local fid                  dev(4), inode(4)
 * coda fid                   volume(4), vnode(4), uniquifier(4)
 * afs fid                    cell(4), volume(4), vnode(4), uniquifier(4)
 *
 * Trace records. Path names do *not* necessarily start on a
 * word boundary. Records do.
 *
 * Syscall                    Items recorded, in addition to header
 * open (pre -- create only)  thread address(4),
 *                              [directory fid, unless DFS_NOFID],
 *                              old size(4), (!= -1 only if object exists)
 *  "   (post -- always)      thread address(4),
 *                              flags(2), mode(2), file desc(2), ft index(2),
 *                              [fid(see above), unless DFS_NOFID],
 *                              [size(4), owner uid(2), file type(2),
 *                                        unless DFS_NOATTR],
 *                              [path length(2), path, unless DFS_NOPATH]
 * stat, lstat                [fid, unless DFS_NOFID],
 *                              [file type(2), unless DFS_NOATTR],
 *                              [path length(2), path, unless DFS_NOPATH]
 *
 * chdir, chroot, readlink, getsymlink       [fid, unless DFS_NOFID],
 *                              [path length(2), path, unless DFS_NOPATH]
 * access, chmod              [fid, unless DFS_NOFID],
 *                              mode(2),
 *                              [file type(2), unless DFS_NOATTR],
 *                              [path length (2), path, unless DFS_NOPATH]
 * mkdir (pre)                thread address(4),
 *                              [directory fid, unless DFS_NOFID]
 *   "   (post)               thread address(4),
 *                              [fid, unless DFS_NOFID],
 *                              mode(2),
 *                              [path length (2), path, unless DFS_NOPATH]
 * creat (pre)                thread address(4),
 *                              [directory fid, unless DFS_NOFID]
 *                              old size(4), (!= -1 only if object exists)
 *   "   (post)               thread address(4),
 *                              mode(2), file descriptor(2),
 *                              [fid, unless DFS_NOFID],
 *                              [path length (2), path, unless DFS_NOPATH]
 * chown                      [fid, unless DFS_NOFID],
 *                              uid(2), gid(2),
 *                              [file type(2), unless DFS_NOATTR],
 *                              [path length(2), path, unless DFS_NOPATH]
 * truncate (pre)             thread address(4),
 *                              [old size(4), unless DFS_NOATTR],
 *    "     (post)            thread address(4),
 *                              [fid, unless DFS_NOFID],
 *                              newsize(4),
 *                              [path length(2), path, unless DFS_NOPATH]
 * utimes                     [fid, unless DFS_NOFID],
 *                              atime(8), mtime(8),
 *                              [file type(2), unless DFS_NOATTR],
 *                              [path length(2), path, unless DFS_NOPATH]
 * execve                     [fid, unless DFS_NOFID],
 *                              [size(4), owner uid(2), unless DFS_NOATTR],
 *                              [path length(2), path, unless DFS_NOPATH]
 * mknod (pre)                thread address(4),
 *                              [directory fid, unless DFS_NOFID]
 *   "   (post)               thread address(4),
 *                              dev(4),
 *                              [fid, unless DFS_NOFID],
 *                              mode(2),
 *                              [path length(2), path, unless DFS_NOPATH]
 *
 * rename (pre -- only if target exists)
                              thread address(4),
 *                              [to fid, unless DFS_NOFID]
 *                              [size(4), # links (2), unless DFS_NOATTR],
 *   "    (post)              thread address(4), object types(4)
 *                              [from dir fid, unless type[0] unset (-1)]
 *                              [from fid, unless type[1] unset]
 *                              [to dir fid, unless type[2] unset]
 *                              [file type(2), unless DFS_NOATTR]
 *                              [from path length(2), unless DFS_NOPATH],
 *                              [to path length(2), unless DFS_NOPATH2],
 *                              [from path, unless DFS_NOPATH],
 *                              [to path, unless DFS_NOPATH2]
 * link                       object types(4)
 *                              [from fid, unless type[0] unset],
 *                              [from dir fid, unless type[1] unset],
 *                              [to dir fid, unless type[2] unset],
 *                              [file type(2), unless DFS_NOATTR],
 *                              [path length(2), unless DFS_NOPATH],
 *                              [target path length(2), unless DFS_NOPATH2],
 *                              [path, unless DFS_NOPATH],
 *                              [target path, unless DFS_NOPATH2]
 * symlink                    [dir fid, unless DFS_NOFID],
 *                              [fid, unless DFS_NOFID2],
 *                              [target path length(2), unless DFS_NOPATH2],
 *                              [path length(2), unless DFS_NOPATH],
 *                              [target path, unless DFS_NOPATH2]
 *                              [path, unless DFS_NOPATH],
 * mount                      rwflag(4),
 *                              [fid, unless DFS_NOFID],
 *                              [path length(2), path, unless DFS_NOPATH]
 * unlink, rmdir (pre)        thread address(4),
 *                              [object fid, unless DFS_NOFID]
 *                              [dir fid, unless DFS_NOFID2]
 *                              [size(4), file type(2), #links(2),
 *                                        unless DFS_NOATTR],
 *      "        (post)       thread address(4),
 *                              [path length(2), path, unless DFS_NOPATH]
 * unmount (pre)              thread address(4),
 *                              [fid, unless DFS_NOFID]
 *     "   (post)             thread address(4),
 *                              [path length(2), path, unless DFS_NOPATH]
 * close                      file descriptor(2),
 *                              # reads(2), # writes(2), # seeks(2),
 *                              # bytes read(4), # bytes written(4),
 *                              file table index(2), reference count(2),
 *                              flags(2), whence(2),
 *                              [fid, unless DFS_NOFID],
 *                              [size(4), file type(2), unless DFS_NOATTR]
 * fork                       child pid(2), uid(2) (pid is zero if error)
 * exit                       header only
 * settimeofday               header only (don't bother w/timezone)
 * setreuid                   real uid(2), effective uid(2)
 * seek                       file descriptor(2), # reads(2), # writes(2),
 *                              ftindex(2), # bytes read(4), # bytes written(4)
 * lookup                     [parent fid, unless DFS_NOFID],
 *                              [component fid, unless DFS_NOFID2],
 *                              [file type(2), unless DFS_NOATTR],
 *                              [path length(2), component, unless DFS_NOPATH]
 * root (of file system)      [component fid, unless DFS_NOFID],
 *                              [target fid, unless DFS_NOFID2]
 *                              [path length(2), component, unless DFS_NOPATH]
 *
 * system call dump array of ints. Total length = DFS_MAXSYSCALL*4 */


/*
 * CheckV1FilePreamble -- returns whether or not the preamble makes sense.
 */
int CheckV1FilePreamble(preamblePtr)
     dfs_file_preamble_v1_t *preamblePtr;
{
    int truth = 0;

    /*
     * should be a cmu host: 128.2.xxx.xxx, and boot time should
     * be in correct range
     */
    if (((preamblePtr->hostAddress >> 24) == 128) &&
	(((preamblePtr->hostAddress & 0x00ff0000) >> 16) == 2) &&
	(preamblePtr->bootTime < V1_END) &&
	(preamblePtr->bootTime > V1_START))
	truth = 1;

    return(truth);
}

/*
 * V1FilePreamble -- interprets a buffer of DECODE_AMOUNT bytes
 * and decides whether or not it's a valid version 1 file preamble.
 * Returns 1 if so, 0 if not.
 */
int V1FilePreamble(buf)
     char *buf;
{
    dfs_file_preamble_v1_t preamble;
    int offset = 0;

    DFS_LOG_UNPACK_FOUR(preamble.hostAddress, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamble.bootTime, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamble.versionTime, buf, offset);

    return(CheckV1FilePreamble(&preamble));
}

/*
 * PrintV1FilePreamble
 */
void PrintV1FilePreamble(tfPtr)
     trace_file_t *tfPtr;
{
    dfs_file_preamble_v1_t *preamblePtr = (dfs_file_preamble_v1_t *) tfPtr->preamblePtr;

    printf("Trace of host %s, booted at %s",
	   Trace_NodeIdToStr(preamblePtr->hostAddress),
	   ctime(&preamblePtr->bootTime));
}

/*
 * UnpackV1FilePreamble -- reads v1 preamble written by collection server from file fp.
 * Also allocates space for the chunk preamble and chunk.
 */
int UnpackV1FilePreamble(tfPtr,buf)
     trace_file_t *tfPtr;
     char* buf;
{
    int offset = 0;
    dfs_file_preamble_v1_t *preamblePtr;

    buf = (char *) malloc(sizeof(dfs_file_preamble_v1_t));

    if (fread(buf, 1, sizeof(dfs_file_preamble_v1_t), tfPtr->fp) !=
	sizeof(dfs_file_preamble_v1_t)) {
	printf("Couldn't read file preamble!\n");
	free(buf);
	return(TRACE_FILEREADERROR);
    }

    tfPtr->preamblePtr = (char *) malloc(sizeof(dfs_file_preamble_v1_t));
    tfPtr->chunkPreamblePtr = (char *) malloc(sizeof(dfs_chunk_preamble_v1_t));
    tfPtr->chunk = (char *) malloc(DFS_TRACE_CHUNK_SIZE_V1);

    preamblePtr = (dfs_file_preamble_v1_t *) tfPtr->preamblePtr;
    DFS_LOG_UNPACK_FOUR(preamblePtr->hostAddress, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamblePtr->bootTime, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamblePtr->versionTime, buf, offset);

    tfPtr->traceStats.totalBytes += sizeof(dfs_file_preamble_v1_t);
    free(buf);
    return(TRACE_SUCCESS);
}

/*
 * UnpackV1ChunkPreamble -- reads chunk preamble written by agent from file fp.
 */
int UnpackV1ChunkPreamble(tfPtr)
     trace_file_t *tfPtr;
{
    char *buf;
    int offset = 0;
    int oldlost;
    time_t oldatime;
    dfs_chunk_preamble_v1_t *preamblePtr;

    buf = (char *) malloc(sizeof(dfs_chunk_preamble_v1_t));
    if (fread(buf, 1, sizeof(dfs_chunk_preamble_v1_t), tfPtr->fp) !=
	sizeof(dfs_chunk_preamble_v1_t)) {
	free(buf);
	return(TRACE_FILEREADERROR);
    }
    preamblePtr = (dfs_chunk_preamble_v1_t *) tfPtr->chunkPreamblePtr;

    oldlost = preamblePtr->bytesLost;
    oldatime = preamblePtr->agentBirthTime;
    DFS_LOG_UNPACK_FOUR(preamblePtr->agentBirthTime, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamblePtr->chunkSeq, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamblePtr->serverTroubles, buf, offset);
    DFS_LOG_UNPACK_FOUR(preamblePtr->bytesLost, buf, offset);
    if (verbose) {
	if (preamblePtr->agentBirthTime == oldatime)
	    printf("[.%d]", preamblePtr->chunkSeq);
	else
	    printf("[%ld.%d]", preamblePtr->agentBirthTime,
		   preamblePtr->chunkSeq);
	fflush(stdout);
    }
    if ((preamblePtr->bytesLost - oldlost > 0) && verbose &&
	(preamblePtr->bytesLost))  /* if 0, already printed it */
	printf("<%d, %d>", preamblePtr->bytesLost, preamblePtr->serverTroubles);

    tfPtr->traceStats.totalBytes += sizeof(dfs_chunk_preamble_v1_t);
    free(buf);
    return(TRACE_SUCCESS);
}

/*
 * UnpackV1Chunk -- unpacks a chunk and changes/resets the
 * appropriate counters. Returns TRACE_FILEREADERROR if
 * the whole chunk didn't appear, otherwise returns
 * TRACE_SUCCESS.
 */
int UnpackV1Chunk(tfPtr)
     trace_file_t *tfPtr;
{
    if (debug)
	printf("Reading new chunk at location %ld\n",tfPtr->traceStats.totalBytes);


    if (fread(tfPtr->chunk, 1, DFS_TRACE_CHUNK_SIZE_V1, tfPtr->fp) !=
	DFS_TRACE_CHUNK_SIZE_V1)
	return(TRACE_FILEREADERROR);

    tfPtr->chunkOffset = 0;
    tfPtr->traceStats.totalBytes += DFS_TRACE_CHUNK_SIZE_V1;
    return(TRACE_SUCCESS);
}

/*
 * UnpackFirstV1RecordTime -- picks out the time of the first trace
 * record from the decode buffer, puts it in the statistics block.
 */
void UnpackFirstV1RecordTime(tfPtr, buf)
     trace_file_t *tfPtr;
     char *buf;
{
    int offset = 36;

    /* file header(12), chunk preamble(16), rec header(16) */
    /* time is in last two longwords of rec header */
    DFS_LOG_UNPACK_FOUR(tfPtr->traceStats.firstTraceRecordTime.tv_sec,
			buf, offset);
    DFS_LOG_UNPACK_FOUR(tfPtr->traceStats.firstTraceRecordTime.tv_usec,
			buf, offset);
}

/*
 * UnpackV1Record -- unpacks the record at the current chunk offset,
 * reading in a new chunk and preamble if necessary, and
 * returns a record of the appropriate size and type in recordPtrPtr,
 * or NULL if there was an error.
 */
void UnpackV1Record(tfPtr, recordPtrPtr)
     trace_file_t *tfPtr;
     dfs_header_t **recordPtrPtr;
{
    short length;
    short curOffset;
    int i;
    u_char vntype;

    *recordPtrPtr = NULL;

    /* first check for end of chunk */
    if (debug) printf("Chunk offset is %d\n", tfPtr->chunkOffset);

    if ((tfPtr->chunkOffset == -1) ||
	(tfPtr->chunkOffset >= DFS_TRACE_CHUNK_SIZE_V1) ||   /* a problem */
	(*((short *) &tfPtr->chunk[tfPtr->chunkOffset]) == 0)) {
	/* get a new chunk. preamble first */
	if (UnpackV1ChunkPreamble(tfPtr)) {
	    if (verbose)
		printf("Trace in file %s ends at %s%ld bytes read.\n",
		       tfPtr->fileName,
		       ctime(&tfPtr->traceStats.lastTraceRecordTime.tv_sec),
		       tfPtr->traceStats.totalBytes);
	    return;
	}

	/* now the chunk */
	if (UnpackV1Chunk(tfPtr)) {
	    if (verbose)
		printf("Couldn't read whole chunk.\n");
	    return;
	}
    }

    curOffset = (short) tfPtr->chunkOffset;
    DFS_LOG_UNPACK_TWO(length, tfPtr->chunk, tfPtr->chunkOffset);

    if (curOffset + length >= DFS_TRACE_CHUNK_SIZE_V1) {
	printf("Warning: Record length (%d) is bogus. Not unpacking record.\n",
	       length);
	tfPtr->chunkOffset += 2;  /* go to next word and try again */
	return;
    }

    switch ((int) tfPtr->chunk[tfPtr->chunkOffset]) { /* switch on opcode */
    case DFS_OPEN:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_post_open));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_open));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_open *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_open *) *recordPtrPtr)->flags,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_open *) *recordPtrPtr)->mode,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_open *) *recordPtrPtr)->fd,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_open *) *recordPtrPtr)->findex,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_open *)*recordPtrPtr)->fid,
			       vntype,
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOATTR)) {
		DFS_LOG_UNPACK_FOUR(((struct dfs_post_open *)*recordPtrPtr)->size,
				    tfPtr->chunk, tfPtr->chunkOffset);
		DFS_LOG_UNPACK_TWO(((struct dfs_post_open *) *recordPtrPtr)->uid,
				   tfPtr->chunk, tfPtr->chunkOffset);
		DFS_LOG_UNPACK_TWO(((struct dfs_post_open *)*recordPtrPtr)->fileType,
				   tfPtr->chunk, tfPtr->chunkOffset);
	    }
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_open *) *recordPtrPtr)->path,
				((struct dfs_post_open *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {  /* written for create */
	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_pre_open));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_open));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_open *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_open *) *recordPtrPtr)->dirFid,
			       (vntype & 0x0f),
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_open *)*recordPtrPtr)->oldSize,
				tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_CLOSE:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_close));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_close));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),
			      tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->fd,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->numReads,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->numWrites,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->numSeeks,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_close *) *recordPtrPtr)->bytesRead,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_close *) *recordPtrPtr)->bytesWritten,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->findex,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->refCount,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->flags,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->whence,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_close *) *recordPtrPtr)->fid,
			   (vntype & 0x0f),
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR)) {
	    DFS_LOG_UNPACK_FOUR(((struct dfs_close *) *recordPtrPtr)->size,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_close *) *recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_stat));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_stat));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_stat *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR))
	    DFS_LOG_UNPACK_TWO(((struct dfs_stat *)*recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_stat *) *recordPtrPtr)->path,
			    ((struct dfs_stat *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_chdir));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_chdir));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_chdir *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_chdir *) *recordPtrPtr)->path,
			    ((struct dfs_chdir *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_SEEK:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_seek));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_seek));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_seek *) *recordPtrPtr)->fd,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_seek *) *recordPtrPtr)->numReads,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_seek *) *recordPtrPtr)->numWrites,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_seek *) *recordPtrPtr)->findex,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_seek *) *recordPtrPtr)->bytesRead,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_seek *) *recordPtrPtr)->bytesWritten,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_EXECVE:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_execve));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_execve));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_execve *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR)) {
	    DFS_LOG_UNPACK_FOUR(((struct dfs_execve *) *recordPtrPtr)->size,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_execve *) *recordPtrPtr)->owner,
			       tfPtr->chunk, tfPtr->chunkOffset);
	}
	DFS_LOG_UNPACK_PATH(((struct dfs_execve *) *recordPtrPtr)->path,
			    ((struct dfs_execve *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_FORK:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_fork));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_fork));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_fork *) *recordPtrPtr)->childPid,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_fork *) *recordPtrPtr)->userId,
			   tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_EXIT:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_exit));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_exit));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	break;
    case DFS_ACCESS:
    case DFS_CHMOD:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_access));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_access));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_access *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_access *) *recordPtrPtr)->mode,
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR))
	    DFS_LOG_UNPACK_TWO(((struct dfs_access *)*recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_access *) *recordPtrPtr)->path,
			    ((struct dfs_access *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_CREAT:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_post_creat));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_creat));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,
				  tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_creat *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_creat *) *recordPtrPtr)->fd,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_creat *) *recordPtrPtr)->mode,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_creat *) *recordPtrPtr)->fid,
			       vntype,
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_creat *) *recordPtrPtr)->findex,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_creat *) *recordPtrPtr)->path,
				((struct dfs_post_creat *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_pre_creat));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_creat));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,
				  tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_creat *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_creat *) *recordPtrPtr)->dirFid,
			       (vntype & 0x0f),
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_creat *) *recordPtrPtr)->oldSize,
				tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_MKDIR:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_post_mkdir));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_mkdir));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,
				  tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_mkdir *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_mkdir *) *recordPtrPtr)->fid,
			       vntype,
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_mkdir *) *recordPtrPtr)->mode,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_mkdir *) *recordPtrPtr)->path,
				((struct dfs_post_mkdir *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_pre_mkdir));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_mkdir));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,
				  tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_mkdir *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_mkdir *) *recordPtrPtr)->dirFid,
			       (vntype & 0x0f),
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_CHOWN:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_chown));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_chown));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_chown *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_chown *) *recordPtrPtr)->owner,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_chown *) *recordPtrPtr)->group,
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR))
	    DFS_LOG_UNPACK_TWO(((struct dfs_chown *) *recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_chown *) *recordPtrPtr)->path,
			    ((struct dfs_chown *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_TRUNCATE:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_post_truncate));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_truncate));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_truncate *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_truncate *) *recordPtrPtr)->fid,
			       vntype,
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_truncate *) *recordPtrPtr)->newSize,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_truncate *) *recordPtrPtr)->path,
				((struct dfs_post_truncate *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_pre_truncate));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_truncate));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_truncate *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOATTR))
		DFS_LOG_UNPACK_FOUR(((struct dfs_pre_truncate *) *recordPtrPtr)->oldSize,
				    tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_UTIMES:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_utimes));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_utimes));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_utimes *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_utimes *) *recordPtrPtr)->atime.tv_sec,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_utimes *) *recordPtrPtr)->atime.tv_usec,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_utimes *) *recordPtrPtr)->mtime.tv_sec,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_utimes *) *recordPtrPtr)->mtime.tv_usec,
			    tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR))
	    DFS_LOG_UNPACK_TWO(((struct dfs_utimes *) *recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_utimes *) *recordPtrPtr)->path,
			    ((struct dfs_utimes *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_MKNOD:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_post_mknod));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_mknod));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,
				  tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_mknod *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_mknod *) *recordPtrPtr)->dev,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_mknod *) *recordPtrPtr)->fid,
			       vntype,
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_TWO(((struct dfs_post_mknod *) *recordPtrPtr)->mode,
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_mknod *) *recordPtrPtr)->path,
				((struct dfs_post_mknod *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_pre_mknod));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_mknod));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,
				  tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_mknod *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_mknod *) *recordPtrPtr)->dirFid,
			       (vntype & 0x0f),
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_RENAME:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    char vnode_types[3];

	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_post_rename));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_rename));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_rename *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    for (i = 0; i < 3; i++)
		vnode_types[i] = tfPtr->chunk[tfPtr->chunkOffset++];
	    tfPtr->chunkOffset++;  /* was packed in four bytes */
	    DFS_LOG_UNPACK_FID(((struct dfs_post_rename *)
				*recordPtrPtr)->fromDirFid,
			       vnode_types[0],
			       (vnode_types[0] == (char) 0xff),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_rename *)
				*recordPtrPtr)->fromFid,
			       vnode_types[1],
			       (vnode_types[1] == (char) 0xff),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_post_rename *)
				*recordPtrPtr)->toDirFid,
			       vnode_types[2],
			       (vnode_types[2] == (char) 0xff),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOATTR))
		DFS_LOG_UNPACK_TWO(((struct dfs_post_rename *)
				    *recordPtrPtr)->fileType,
				   tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOPATH))
		DFS_LOG_UNPACK_TWO(((struct dfs_post_rename *)
				    *recordPtrPtr)->fromPathLength,
				   tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOPATH2))
		DFS_LOG_UNPACK_TWO(((struct dfs_post_rename *)
				    *recordPtrPtr)->toPathLength,
				   tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOPATH))
		DFS_LOG_UNPACK_STRING(((struct dfs_post_rename *)
				       *recordPtrPtr)->fromPath,
				      ((struct dfs_post_rename *)*recordPtrPtr)->fromPathLength,
				      tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOPATH2))
		DFS_LOG_UNPACK_STRING(((struct dfs_post_rename *)
				       *recordPtrPtr)->toPath,
				      ((struct dfs_post_rename *)*recordPtrPtr)->toPathLength,
				      tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_pre_rename));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_rename));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_rename *)*recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_rename *)
				*recordPtrPtr)->toFid,
			       (vntype & 0x0f),
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOATTR)) {
		DFS_LOG_UNPACK_FOUR(((struct dfs_pre_rename *)*recordPtrPtr)->size,
				    tfPtr->chunk, tfPtr->chunkOffset);
		DFS_LOG_UNPACK_TWO(((struct dfs_pre_rename *)*recordPtrPtr)->numLinks,
				   tfPtr->chunk, tfPtr->chunkOffset);
	    }
	}
	break;
    case DFS_LINK: {
	char vnode_types[3];

	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_link));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_link));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	for (i = 0; i < 3; i++)
	    vnode_types[i] = tfPtr->chunk[tfPtr->chunkOffset++];
	tfPtr->chunkOffset++;  /* was packed in four bytes */
	DFS_LOG_UNPACK_FID(((struct dfs_link *) *recordPtrPtr)->fromFid,
			   vnode_types[0],
			   (vnode_types[0] == (char) 0xff),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_link *) *recordPtrPtr)->fromDirFid,
			   vnode_types[1],
			   (vnode_types[1] == (char) 0xff),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_link *) *recordPtrPtr)->toDirFid,
			   vnode_types[2],
			   (vnode_types[2] == (char) 0xff),
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR))
	    DFS_LOG_UNPACK_TWO(((struct dfs_link *) *recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH))
	    DFS_LOG_UNPACK_TWO(((struct dfs_link *)
				*recordPtrPtr)->fromPathLength,
			       tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH2))
	    DFS_LOG_UNPACK_TWO(((struct dfs_link *)
				*recordPtrPtr)->toPathLength,
			       tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH))
	    DFS_LOG_UNPACK_STRING(((struct dfs_link *)*recordPtrPtr)->fromPath,
				  ((struct dfs_link *) *recordPtrPtr)->fromPathLength,
				  tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH2))
	    DFS_LOG_UNPACK_STRING(((struct dfs_link *) *recordPtrPtr)->toPath,
				  ((struct dfs_link *) *recordPtrPtr)->toPathLength,
				  tfPtr->chunk, tfPtr->chunkOffset);
    }	break;
    case DFS_SYMLINK:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_symlink));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_symlink));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_symlink *) *recordPtrPtr)->dirFid,
			   (vntype & 0x0f),
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_symlink *) *recordPtrPtr)->fid,
			   (vntype >> 4),
			   ((*recordPtrPtr)->flags & DFS_NOFID2),
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH2))
	    DFS_LOG_UNPACK_TWO(((struct dfs_symlink *)
				*recordPtrPtr)->targetPathLength,
			       tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH))
	    DFS_LOG_UNPACK_TWO(((struct dfs_symlink *)
				*recordPtrPtr)->linkPathLength,
			       tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH2))
	    DFS_LOG_UNPACK_STRING(((struct dfs_symlink *)
				   *recordPtrPtr)->targetPath,
				  ((struct dfs_symlink *) *recordPtrPtr)->targetPathLength,
				  tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOPATH))
	    DFS_LOG_UNPACK_STRING(((struct dfs_symlink *)
				   *recordPtrPtr)->linkPath,
				  ((struct dfs_symlink *) *recordPtrPtr)->linkPathLength,
				  tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *)malloc(sizeof(struct dfs_post_rmdir));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_rmdir));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),
				  tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_rmdir *)
				 *recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_rmdir *) *recordPtrPtr)->path,
				((struct dfs_post_rmdir *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_pre_rmdir));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_rmdir));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),
				  tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_rmdir *)
				 *recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_rmdir *) *recordPtrPtr)->fid,
			       (vntype & 0x0f),
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_rmdir *) *recordPtrPtr)->dirFid,
			       (vntype >> 4),
			       ((*recordPtrPtr)->flags & DFS_NOFID2),
			       tfPtr->chunk, tfPtr->chunkOffset);
	    if (!((*recordPtrPtr)->flags & DFS_NOATTR)) {
		DFS_LOG_UNPACK_FOUR(((struct dfs_pre_rmdir *)*recordPtrPtr)->size,
				    tfPtr->chunk, tfPtr->chunkOffset);
		DFS_LOG_UNPACK_TWO(((struct dfs_pre_rmdir *)*recordPtrPtr)->fileType,
				   tfPtr->chunk, tfPtr->chunkOffset);
		DFS_LOG_UNPACK_TWO(((struct dfs_pre_rmdir *)*recordPtrPtr)->numLinks,
				   tfPtr->chunk, tfPtr->chunkOffset);
	    }
	}
	break;
    case DFS_UNMOUNT:
	if (tfPtr->chunk[tfPtr->chunkOffset+1] & DFS_POST) {
	    *recordPtrPtr = (dfs_header_t *)
		malloc(sizeof(struct dfs_post_unmount));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_post_unmount));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),
				  tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_post_unmount *)
				 *recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_PATH(((struct dfs_post_unmount *)*recordPtrPtr)->path,
				((struct dfs_post_unmount *) *recordPtrPtr)->pathLength,
				(*recordPtrPtr)->flags,
				tfPtr->chunk, tfPtr->chunkOffset);
	} else {
	    *recordPtrPtr = (dfs_header_t *)
		malloc(sizeof(struct dfs_pre_unmount));
	    bzero((char *)*recordPtrPtr, sizeof(struct dfs_pre_unmount));
	    DFS_LOG_UNPACK_HEADER((**recordPtrPtr),
				  tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FOUR(((struct dfs_pre_unmount *)
				 *recordPtrPtr)->threadAddr,
				tfPtr->chunk, tfPtr->chunkOffset);
	    DFS_LOG_UNPACK_FID(((struct dfs_pre_unmount *) *recordPtrPtr)->fid,
			       vntype,
			       ((*recordPtrPtr)->flags & DFS_NOFID),
			       tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_MOUNT:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_mount));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_mount));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FOUR(((struct dfs_mount *) *recordPtrPtr)->rwflag,
			    tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_mount *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_mount *) *recordPtrPtr)->path,
			    ((struct dfs_mount *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_SETREUID:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_setreuid));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_setreuid));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_setreuid *) *recordPtrPtr)->ruid,
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_TWO(((struct dfs_setreuid *) *recordPtrPtr)->euid,
			   tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_SETTIMEOFDAY:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_settimeofday));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_settimeofday));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,tfPtr->chunkOffset);
	break;
    case DFS_SYSCALLDUMP_V1:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_call));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_call));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,tfPtr->chunkOffset);
	/* translate to opcode in tracelib.h */
	(*recordPtrPtr)->opcode = DFS_SYSCALLDUMP;

	{
	    int i;
	    for (i = 1; i <= DFS_MAXSYSCALL_V1; i++)
		DFS_LOG_UNPACK_FOUR(((struct dfs_call *) *recordPtrPtr)->count[i],
				    tfPtr->chunk, tfPtr->chunkOffset);
	}
	break;
    case DFS_LOOKUP_V1:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_lookup));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_lookup));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,tfPtr->chunkOffset);
	/* translate to opcode in tracelib.h */
	(*recordPtrPtr)->opcode = DFS_LOOKUP;

	DFS_LOG_UNPACK_FID(((struct dfs_lookup *) *recordPtrPtr)->parentFid,
			   (vntype & 0x0f),
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_lookup *) *recordPtrPtr)->compFid,
			   (vntype >> 4),
			   ((*recordPtrPtr)->flags & DFS_NOFID2),
			   tfPtr->chunk, tfPtr->chunkOffset);
	if (!((*recordPtrPtr)->flags & DFS_NOATTR))
	    DFS_LOG_UNPACK_TWO(((struct dfs_lookup *) *recordPtrPtr)->fileType,
			       tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_lookup *) *recordPtrPtr)->path,
			    ((struct dfs_lookup *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_GETSYMLINK_V1:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_getsymlink));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_getsymlink));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr),tfPtr->chunk,tfPtr->chunkOffset);
	/* translate to opcode in tracelib.h */
	(*recordPtrPtr)->opcode = DFS_GETSYMLINK;

	DFS_LOG_UNPACK_FID(((struct dfs_getsymlink *) *recordPtrPtr)->fid,
			   vntype,
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_getsymlink *) *recordPtrPtr)->path,
			    ((struct dfs_getsymlink *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    case DFS_ROOT_V1:
	*recordPtrPtr = (dfs_header_t *) malloc(sizeof(struct dfs_root));
	bzero((char *)*recordPtrPtr, sizeof(struct dfs_root));
	DFS_LOG_UNPACK_HEADER((**recordPtrPtr), tfPtr->chunk,tfPtr->chunkOffset);
	/* translate to opcode in tracelib.h */
	(*recordPtrPtr)->opcode = DFS_ROOT;

	DFS_LOG_UNPACK_FID(((struct dfs_root *) *recordPtrPtr)->compFid,
			   (vntype & 0x0f),
			   ((*recordPtrPtr)->flags & DFS_NOFID),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_FID(((struct dfs_root *) *recordPtrPtr)->targetFid,
			   (vntype >> 4),
			   ((*recordPtrPtr)->flags & DFS_NOFID2),
			   tfPtr->chunk, tfPtr->chunkOffset);
	DFS_LOG_UNPACK_PATH(((struct dfs_root *) *recordPtrPtr)->path,
			    ((struct dfs_root *) *recordPtrPtr)->pathLength,
			    (*recordPtrPtr)->flags,
			    tfPtr->chunk, tfPtr->chunkOffset);
	break;
    default:
	printf("Bogus bytes! chunk at %ld, offset %d, opcode = 0x%x, flags = 0x%x\n",
	       tfPtr->traceStats.totalBytes - DFS_TRACE_CHUNK_SIZE_V1, tfPtr->chunkOffset,
	       tfPtr->chunk[tfPtr->chunkOffset],
	       tfPtr->chunk[tfPtr->chunkOffset+1]);
	tfPtr->chunkOffset += 2;  /* get to next word, try again */
	return;  /* recordPtr will be null */
    }

    /* more sanity checks: check length */
    if (tfPtr->chunkOffset - curOffset != length) {
	printf("Warning: Record length (%d) does not match amount read (%d)!\n",
	       length, tfPtr->chunkOffset - curOffset);
	tfPtr->chunkOffset = curOffset + length;
    }

    /* check timestamps */
    if (timercmp(&(*recordPtrPtr)->time, &tfPtr->traceStats.lastTraceRecordTime,<)) {
	printf("Warning: Decreasing timestamp. Chunk %ld, offset %d, opcode %s.\n\tLast timestamp was %s",
	       tfPtr->traceStats.totalBytes - DFS_TRACE_CHUNK_SIZE_V1, curOffset,
	       Trace_OpcodeToStr((*recordPtrPtr)->opcode),
	       ctime(&tfPtr->traceStats.lastTraceRecordTime.tv_sec));
	printf("\tCurrent time is %s", ctime(&(*recordPtrPtr)->time.tv_sec));
    }
    tfPtr->chunkOffset = (tfPtr->chunkOffset + 3) & ~0x3; /* word align */
    return;
}
