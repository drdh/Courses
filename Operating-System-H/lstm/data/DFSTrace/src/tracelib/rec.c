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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/rec.c,v 1.3 1998/10/23 03:37:57 tmkr Exp $";
*/
#endif _BLURB_


/*
 * Rec.c -- routines for manipulating dfs trace records.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>
#include "tracelib.h"

#define	STREQ(a, b) (strcasecmp(a, b) == 0)
#define CHECK_NULL_STRING(string) (string ?   string : "(nil)")

/*
 * Trace_GetFileType -- given a pointer to a trace record, returns the file
 * type (appropriately masked) if such a field exists in that record.
 * Otherwise, returns DFS_IFMT (which wouldn't make sense as a file type).
 */
short Trace_GetFileType(recPtr)
     dfs_header_t *recPtr;
{
    short fileType;

    switch (recPtr->opcode) {
    case DFS_CLOSE:
	fileType = ((struct dfs_close *)recPtr)->fileType;
	break;
    case DFS_OPEN:
	fileType = ((struct dfs_open *)recPtr)->fileType;
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	fileType = ((struct dfs_rmdir *)recPtr)->fileType;
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	fileType = ((struct dfs_stat *)recPtr)->fileType;
	break;
    case DFS_ACCESS:
    case DFS_CHMOD:
	fileType = ((struct dfs_access *)recPtr)->fileType;
	break;
    case DFS_CHOWN:
	fileType = ((struct dfs_chown *)recPtr)->fileType;
	break;
    case DFS_RENAME:
	fileType = ((struct dfs_rename *)recPtr)->fileType;
	break;
    case DFS_LINK:
	fileType = ((struct dfs_link *)recPtr)->fileType;
	break;
    case DFS_LOOKUP:
	fileType = ((struct dfs_lookup *)recPtr)->fileType;
	break;
    case DFS_UTIMES:
	fileType = ((struct dfs_utimes *)recPtr)->fileType;
	break;
    default:
	fileType = -1;
	break;
    }
    return(fileType & DFS_IFMT);
}

short Trace_GetFileIndex(recPtr)
     dfs_header_t *recPtr;
{
    short findex;

    switch (recPtr->opcode) {
    case DFS_CLOSE:
	findex = ((struct dfs_close *)recPtr)->findex;
	break;
    case DFS_OPEN:
	findex = ((struct dfs_open *)recPtr)->findex;
	break;
    case DFS_CREAT:
	findex = ((struct dfs_creat *)recPtr)->findex;
	break;
    case DFS_SEEK:
	findex = ((struct dfs_seek *)recPtr)->findex;
	break;
    case DFS_READ:
    case DFS_WRITE:
	findex = ((struct dfs_read *)recPtr)->findex;
	break;
    case DFS_FSYNC:
	findex = ((struct dfs_fsync *)recPtr)->findex;
	break;
    default:
	findex = -1;
	break;
    }
    return(findex);
}

/*
 * Trace_GetRefcount -- given a pointer to a trace record, returns the
 * reference count if such a field exists in that record.
 * Otherwise, returns -1.
 */
short Trace_GetRefCount(recPtr)
     dfs_header_t *recPtr;
{
    short refCount;

    switch (recPtr->opcode) {
    case DFS_CLOSE:
	refCount = ((struct dfs_close *)recPtr)->refCount;
	break;
    default:
	refCount = -1;
	break;
    }
    return(refCount);
}

/*
 * Trace_GetFid -- given a record ptr, returns as out parameters
 * the number of fids in the record, and a list of fid ptrs. If there
 * aren't any fids, returns 0  as the number of fids.
 */
void Trace_GetFid(recPtr, fidplist, nump)
     dfs_header_t *recPtr;
     generic_fid_t *fidplist[];
     int *nump;
{
    *nump = 0;
    switch(recPtr->opcode) {
    case DFS_OPEN:
	*nump = 2;
	fidplist[0] = &((struct dfs_open *)recPtr)->fid;
	fidplist[1] = &((struct dfs_open *)recPtr)->dirFid;
	break;
    case DFS_CLOSE:
	*nump =1;
	fidplist[0] = &((struct dfs_close *)recPtr)->fid;
	break;
    case DFS_CREAT:
	*nump = 2;
	fidplist[0] = &((struct dfs_creat *)recPtr)->fid;
	fidplist[1] = &((struct dfs_creat *)recPtr)->dirFid;
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	*nump = 1;
	fidplist[0] = &((struct dfs_stat *)recPtr)->fid;
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	*nump = 2;
	fidplist[0] = &((struct dfs_rmdir *)recPtr)->fid;
	fidplist[1] = &((struct dfs_rmdir *)recPtr)->dirFid;
	break;
    case DFS_UNMOUNT:
	*nump = 1;
	fidplist[0] = &((struct dfs_unmount *)recPtr)->fid;
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	*nump = 1;
	fidplist[0] = &((struct dfs_chdir *)recPtr)->fid;
	break;
    case DFS_EXECVE:
	*nump = 1;
	fidplist[0] = &((struct dfs_execve *)recPtr)->fid;
	break;
    case DFS_ACCESS:
    case DFS_CHMOD:
	*nump = 1;
	fidplist[0] = &((struct dfs_access *)recPtr)->fid;
	break;
    case DFS_MKDIR:
	*nump = 2;
	fidplist[0] = &((struct dfs_mkdir *)recPtr)->fid;
	fidplist[1] = &((struct dfs_mkdir *)recPtr)->dirFid;
	break;
    case DFS_CHOWN:
	*nump = 1;
	fidplist[0] = &((struct dfs_chown *)recPtr)->fid;
	break;
    case DFS_RENAME:
	*nump = 4;
	fidplist[0] = &((struct dfs_rename *)recPtr)->fromFid;
	fidplist[1] = &((struct dfs_rename *)recPtr)->fromDirFid;
	fidplist[2] = &((struct dfs_rename *)recPtr)->toFid;
	fidplist[3] = &((struct dfs_rename *)recPtr)->toDirFid;
	break;
    case DFS_LINK:
	*nump = 3;
	fidplist[0] = &((struct dfs_link *)recPtr)->fromFid;
	fidplist[1] = &((struct dfs_link *)recPtr)->fromDirFid;
	fidplist[2] = &((struct dfs_link *)recPtr)->toDirFid;
	break;
    case DFS_SYMLINK:
	*nump = 2;
	fidplist[0] = &((struct dfs_symlink *)recPtr)->fid;
	fidplist[1] = &((struct dfs_symlink *)recPtr)->dirFid;
	break;
    case DFS_TRUNCATE:
	*nump = 1;
	fidplist[0] = &((struct dfs_truncate *)recPtr)->fid;
	break;
    case DFS_UTIMES:
	*nump = 1;
	fidplist[0] = &((struct dfs_utimes *)recPtr)->fid;
	break;
    case DFS_MKNOD:
	*nump = 2;
	fidplist[0] = &((struct dfs_mknod *)recPtr)->fid;
	fidplist[1] = &((struct dfs_mknod *)recPtr)->dirFid;
	break;
    case DFS_MOUNT:
	*nump = 1;
	fidplist[0] = &((struct dfs_mount *)recPtr)->fid;
	break;
    case DFS_LOOKUP:
	*nump = 2;
	fidplist[0] = &((struct dfs_lookup *)recPtr)->compFid;
	fidplist[1] = &((struct dfs_lookup *)recPtr)->parentFid;
	break;
    case DFS_ROOT:
	*nump = 2;
	fidplist[0] = &((struct dfs_root *)recPtr)->compFid;
	fidplist[1] = &((struct dfs_root *)recPtr)->targetFid;
	break;
    case DFS_GETSYMLINK:
	*nump = 1;
	fidplist[0] = &((struct dfs_getsymlink *)recPtr)->fid;
	break;
    case DFS_SEEK:
	*nump = 1;
	fidplist[0] = &((struct dfs_seek *)recPtr)->fid;
	break;
    case DFS_READ:
    case DFS_WRITE:
	*nump = 1;
	fidplist[0] = &((struct dfs_read *)recPtr)->fid;
	break;
    case DFS_SETTIMEOFDAY:
    case DFS_FORK:
    case DFS_EXIT:
    case DFS_SETREUID:
    case DFS_SYSCALLDUMP:
    default:
	break;
    }
}

/*
 * Trace_GetPath -- returns as out parameters the number of path names
 * and a list of pathname ptrs, if any.
 */
void Trace_GetPath(recPtr, pathplist, nump)
     dfs_header_t *recPtr;
     char *pathplist[];
     int *nump;
{
    *nump = 0;
    switch (recPtr->opcode) {
    case DFS_OPEN:
	*nump = 1;
	pathplist[0] = ((struct dfs_open *)recPtr)->path;
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	*nump = 1;
	pathplist[0] = ((struct dfs_stat *)recPtr)->path;
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	*nump = 1;
	pathplist[0] = ((struct dfs_chdir *)recPtr)->path;
	break;
    case DFS_EXECVE:
	*nump = 1;
	pathplist[0] = ((struct dfs_execve *)recPtr)->path;
	break;
    case DFS_CHMOD:
    case DFS_ACCESS:
	*nump = 1;
	pathplist[0] = ((struct dfs_access *)recPtr)->path;
	break;
    case DFS_CREAT:
	*nump = 1;
	pathplist[0] = ((struct dfs_creat *)recPtr)->path;
	break;
    case DFS_MKDIR:
	*nump = 1;
	pathplist[0] = ((struct dfs_mkdir *)recPtr)->path;
	break;
    case DFS_CHOWN:
	*nump = 1;
	pathplist[0] = ((struct dfs_chown *)recPtr)->path;
	break;
    case DFS_TRUNCATE:
	*nump = 1;
	pathplist[0] = ((struct dfs_truncate *)recPtr)->path;
	break;
    case DFS_UTIMES:
	*nump = 1;
	pathplist[0] = ((struct dfs_utimes *)recPtr)->path;
	break;
    case DFS_MKNOD:
	*nump = 1;
	pathplist[0] = ((struct dfs_mknod *)recPtr)->path;
	break;
    case DFS_LOOKUP:
	*nump = 1;
	pathplist[0] = ((struct dfs_lookup *)recPtr)->path;
	break;
    case DFS_ROOT:
	*nump = 1;
	pathplist[0] = ((struct dfs_root *)recPtr)->path;
	break;
    case DFS_GETSYMLINK:
	*nump = 2;
	pathplist[0] = ((struct dfs_getsymlink *)recPtr)->compPath;
	pathplist[1] = ((struct dfs_getsymlink *)recPtr)->path;
	break;
    case DFS_RENAME:
	*nump = 2;
	pathplist[0] = ((struct dfs_rename *)recPtr)->fromPath;
	pathplist[1] = ((struct dfs_rename *)recPtr)->toPath;
	break;
    case DFS_LINK:
	*nump = 2;
	pathplist[0] = ((struct dfs_link *)recPtr)->fromPath;
	pathplist[1] = ((struct dfs_link *)recPtr)->toPath;
	break;
    case DFS_SYMLINK:
	*nump = 2;
	pathplist[0] = ((struct dfs_symlink *)recPtr)->targetPath;
	pathplist[1] = ((struct dfs_symlink *)recPtr)->linkPath;
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	*nump = 1;
	pathplist[0] = ((struct dfs_rmdir *)recPtr)->path;
	break;
    case DFS_UNMOUNT:
	*nump = 1;
	pathplist[0] = ((struct dfs_unmount *)recPtr)->path;
	break;
    case DFS_SEEK:
	*nump = 1;
	pathplist[0] = ((struct dfs_seek *)recPtr)->path;
	break;
    case DFS_READ:
    case DFS_WRITE:
	*nump = 1;
	pathplist[0] = ((struct dfs_read *)recPtr)->path;
	break;
    case DFS_CLOSE:
	*nump = 1;
	pathplist[0] = ((struct dfs_close *)recPtr)->path;
	break;
    case DFS_SETTIMEOFDAY:
    case DFS_FORK:
    case DFS_EXIT:
    case DFS_SETREUID:
    default:
	break;
    }
}

/*
 * Trace_FidsEqual -- returns 1 if they're equal, 0 if they're not,
 * or if something was bogus.
 */
int Trace_FidsEqual(fid1Ptr, fid2Ptr)
     generic_fid_t *fid1Ptr;
     generic_fid_t *fid2Ptr;
{
    int rc = 0;

    if ((fid1Ptr == NULL) || (fid2Ptr == NULL))
	return(0);

    if (fid1Ptr->tag != fid2Ptr->tag)
	return(0);

    switch (fid1Ptr->tag) {
    case DFS_ITYPE_AFS:
	if ((fid1Ptr->value.afs.Cell == fid2Ptr->value.afs.Cell) &&
	    (fid1Ptr->value.afs.Fid.Volume == fid2Ptr->value.afs.Fid.Volume) &&
	    (fid1Ptr->value.afs.Fid.Vnode == fid2Ptr->value.afs.Fid.Vnode) &&
	    (fid1Ptr->value.afs.Fid.Unique == fid2Ptr->value.afs.Fid.Unique))
	    rc = 1;
	break;
    case DFS_ITYPE_CFS:
	if ((fid1Ptr->value.cfs.Volume == fid2Ptr->value.cfs.Volume) &&
	    (fid1Ptr->value.cfs.Vnode == fid2Ptr->value.cfs.Vnode) &&
	    (fid1Ptr->value.cfs.Unique == fid2Ptr->value.cfs.Unique))
	    rc = 1;
	break;
    case DFS_ITYPE_UFS:
    case DFS_ITYPE_NFS:
	if ((fid1Ptr->value.local.device == fid2Ptr->value.local.device) &&
	    (fid1Ptr->value.local.number == fid2Ptr->value.local.number))
	    rc = 1;
	break;
    default:
	break;
    }
    return(rc);
}

/*
 * Trace_CopyRecord -- makes a safe copy of a trace record.
 */
void Trace_CopyRecord(sp, dpp)
     dfs_header_t *sp;
     dfs_header_t **dpp;
{
    *dpp = NULL;

    switch(sp->opcode) {
    case DFS_OPEN:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_open));
	bcopy(sp, *dpp, sizeof(struct dfs_open));
	if (((struct dfs_open *)sp)->path) {
	    ((struct dfs_open *)*dpp)->path =
		(char *) malloc(((struct dfs_open *)sp)->pathLength+1);
	    strncpy(((struct dfs_open *)*dpp)->path,
		    ((struct dfs_open *)sp)->path,
		    ((struct dfs_open *)*dpp)->pathLength+1);
	}
	break;
    case DFS_CREAT:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_creat));
	bcopy(sp, *dpp, sizeof(struct dfs_creat));
	if (((struct dfs_creat *)sp)->path) {
	    ((struct dfs_creat *)*dpp)->path =
		(char *) malloc(((struct dfs_creat *)sp)->pathLength+1);
	    strncpy(((struct dfs_creat *)*dpp)->path,
		    ((struct dfs_creat *)sp)->path,
		    ((struct dfs_creat *)*dpp)->pathLength+1);
	}
	break;
    case DFS_CLOSE:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_close));
	bcopy(sp, *dpp, sizeof(struct dfs_close));
	if (((struct dfs_close *)sp)->path) {
	    ((struct dfs_close *)*dpp)->path =
		(char *) malloc(strlen(((struct dfs_close *)sp)->path)+1);
	    (void) strcpy(((struct dfs_close *)*dpp)->path,
			  ((struct dfs_close *)sp)->path);
	}
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_stat));
	bcopy(sp, *dpp, sizeof(struct dfs_stat));
	if (((struct dfs_stat *)sp)->path) {
	    ((struct dfs_stat *)*dpp)->path =
		(char *) malloc(((struct dfs_stat *)sp)->pathLength+1);
	    strncpy(((struct dfs_stat *)*dpp)->path,
		    ((struct dfs_stat *)sp)->path,
		    ((struct dfs_stat *)*dpp)->pathLength+1);
	}
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_chdir));
	bcopy(sp, *dpp, sizeof(struct dfs_chdir));
	if (((struct dfs_chdir *)sp)->path) {
	    ((struct dfs_chdir *)*dpp)->path =
		(char *) malloc(((struct dfs_chdir *)sp)->pathLength+1);
	    strncpy(((struct dfs_chdir *)*dpp)->path,
		    ((struct dfs_chdir *)sp)->path,
		    ((struct dfs_chdir *)*dpp)->pathLength+1);
	}
	break;
    case DFS_EXECVE:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_execve));
	bcopy(sp, *dpp, sizeof(struct dfs_execve));
	if (((struct dfs_execve *)sp)->path) {
	    ((struct dfs_execve *)*dpp)->path =
		(char *) malloc(((struct dfs_execve *)sp)->pathLength+1);
	    strncpy(((struct dfs_execve *)*dpp)->path,
		    ((struct dfs_execve *)sp)->path,
		    ((struct dfs_execve *)*dpp)->pathLength+1);
	}
	break;
    case DFS_CHMOD:
    case DFS_ACCESS:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_access));
	bcopy(sp, *dpp, sizeof(struct dfs_access));
	if (((struct dfs_access *)sp)->path) {
	    ((struct dfs_access *)*dpp)->path =
		(char *) malloc(((struct dfs_access *)sp)->pathLength+1);
	    strncpy(((struct dfs_access *)*dpp)->path,
		    ((struct dfs_access *)sp)->path,
		    ((struct dfs_access *)*dpp)->pathLength+1);
	}
	break;
    case DFS_MKDIR:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_mkdir));
	bcopy(sp, *dpp, sizeof(struct dfs_mkdir));
	if (((struct dfs_mkdir *)sp)->path) {
	    ((struct dfs_mkdir *)*dpp)->path =
		(char *) malloc(((struct dfs_mkdir *)sp)->pathLength+1);
	    strncpy(((struct dfs_mkdir *)*dpp)->path,
		    ((struct dfs_mkdir *)sp)->path,
		    ((struct dfs_mkdir *)*dpp)->pathLength+1);
	}
	break;
    case DFS_CHOWN:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_chown));
	bcopy(sp, *dpp, sizeof(struct dfs_chown));
	if (((struct dfs_chown *)sp)->path) {
	    ((struct dfs_chown *)*dpp)->path =
		(char *) malloc(((struct dfs_chown *)sp)->pathLength+1);
	    strncpy(((struct dfs_chown *)*dpp)->path,
		    ((struct dfs_chown *)sp)->path,
		    ((struct dfs_chown *)*dpp)->pathLength+1);
	}
	break;
    case DFS_RENAME:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_rename));
	bcopy(sp, *dpp, sizeof(struct dfs_rename));
	if (((struct dfs_rename *)sp)->toPath) {
	    ((struct dfs_rename *)*dpp)->toPath =
		(char *) malloc(((struct dfs_rename *)sp)->toPathLength+1);
	    strncpy(((struct dfs_rename *)*dpp)->toPath,
		    ((struct dfs_rename *)sp)->toPath,
		    ((struct dfs_rename *)*dpp)->toPathLength+1);
	}
	if (((struct dfs_rename *)sp)->fromPath) {
	    ((struct dfs_rename *)*dpp)->fromPath =
		(char *) malloc(((struct dfs_rename *)sp)->fromPathLength+1);
	    strncpy(((struct dfs_rename *)*dpp)->fromPath,
		    ((struct dfs_rename *)sp)->fromPath,
		    ((struct dfs_rename *)*dpp)->fromPathLength+1);
	}
	break;
    case DFS_LINK:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_link));
	bcopy(sp, *dpp, sizeof(struct dfs_link));
	if (((struct dfs_link *)sp)->toPath) {
	    ((struct dfs_link *)*dpp)->toPath =
		(char *) malloc(((struct dfs_link *)sp)->toPathLength+1);
	    strncpy(((struct dfs_link *)*dpp)->toPath,
		    ((struct dfs_link *)sp)->toPath,
		    ((struct dfs_link *)*dpp)->toPathLength+1);
	}
	if (((struct dfs_link *)sp)->fromPath) {
	    ((struct dfs_link *)*dpp)->fromPath =
		(char *) malloc(((struct dfs_link *)sp)->fromPathLength+1);
	    strncpy(((struct dfs_link *)*dpp)->fromPath,
		    ((struct dfs_link *)sp)->fromPath,
		    ((struct dfs_link *)*dpp)->fromPathLength+1);
	}
	break;
    case DFS_SYMLINK:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_symlink));
	bcopy(sp, *dpp, sizeof(struct dfs_symlink));
	if (((struct dfs_symlink *)sp)->linkPath) {
	    ((struct dfs_symlink *)*dpp)->linkPath =
		(char *) malloc(((struct dfs_symlink *)sp)->linkPathLength+1);
	    strncpy(((struct dfs_symlink *)*dpp)->linkPath,
		    ((struct dfs_symlink *)sp)->linkPath,
		    ((struct dfs_symlink *)*dpp)->linkPathLength+1);
	}
	if (((struct dfs_symlink *)sp)->targetPath) {
	    ((struct dfs_symlink *)*dpp)->targetPath =
		(char *) malloc(((struct dfs_symlink *)
				 sp)->targetPathLength+1);
	    strncpy(((struct dfs_symlink *)*dpp)->targetPath,
		    ((struct dfs_symlink *)sp)->targetPath,
		    ((struct dfs_symlink *)*dpp)->targetPathLength+1);
	}
	break;
    case DFS_TRUNCATE:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_truncate));
	bcopy(sp, *dpp, sizeof(struct dfs_truncate));
	if (((struct dfs_truncate *)sp)->path) {
	    ((struct dfs_truncate *)*dpp)->path =
		(char *) malloc(((struct dfs_truncate *)sp)->pathLength+1);
	    strncpy(((struct dfs_truncate *)*dpp)->path,
		    ((struct dfs_truncate *)sp)->path,
		    ((struct dfs_truncate *)*dpp)->pathLength+1);
	}
	break;
    case DFS_UTIMES:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_utimes));
	bcopy(sp, *dpp, sizeof(struct dfs_utimes));
	if (((struct dfs_utimes *)sp)->path) {
	    ((struct dfs_utimes *)*dpp)->path =
		(char *) malloc(((struct dfs_utimes *)sp)->pathLength+1);
	    strncpy(((struct dfs_utimes *)*dpp)->path,
		    ((struct dfs_utimes *)sp)->path,
		    ((struct dfs_utimes *)*dpp)->pathLength+1);
	}
	break;
    case DFS_MKNOD:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_mknod));
	bcopy(sp, *dpp, sizeof(struct dfs_mknod));
	if (((struct dfs_mknod *)sp)->path) {
	    ((struct dfs_mknod *)*dpp)->path =
		(char *) malloc(((struct dfs_mknod *)sp)->pathLength+1);
	    strncpy(((struct dfs_mknod *)*dpp)->path,
		    ((struct dfs_mknod *)sp)->path,
		    ((struct dfs_mknod *)*dpp)->pathLength+1);
	}
	break;
    case DFS_MOUNT:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_mount));
	bcopy(sp, *dpp, sizeof(struct dfs_mount));
	if (((struct dfs_mount *)sp)->path) {
	    ((struct dfs_mount *)*dpp)->path =
		(char *) malloc(((struct dfs_mount *)sp)->pathLength+1);
	    strncpy(((struct dfs_mount *)*dpp)->path,
		    ((struct dfs_mount *)sp)->path,
		    ((struct dfs_mount *)*dpp)->pathLength+1);
	}
	break;
    case DFS_SEEK:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_seek));
	bcopy(sp, *dpp, sizeof(struct dfs_seek));
	if (((struct dfs_seek *)sp)->path) {
	    ((struct dfs_seek *)*dpp)->path =
		(char *) malloc(strlen(((struct dfs_seek *)sp)->path)+1);
	    (void) strcpy(((struct dfs_seek *)*dpp)->path,
			  ((struct dfs_seek *)sp)->path);
	}
	break;
    case DFS_FORK:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_fork));
	bcopy(sp, *dpp, sizeof(struct dfs_fork));
	break;
    case DFS_EXIT:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_exit));
	bcopy(sp, *dpp, sizeof(struct dfs_exit));
	break;
    case DFS_SETTIMEOFDAY:
	*dpp = (dfs_header_t *)malloc(sizeof(struct dfs_settimeofday));
	bcopy(sp, *dpp, sizeof(struct dfs_settimeofday));
	break;
    case DFS_SETREUID:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_setreuid));
	bcopy(sp, *dpp, sizeof(struct dfs_setreuid));
	break;
    case DFS_LOOKUP:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_lookup));
	bcopy(sp, *dpp, sizeof(struct dfs_lookup));
	if (((struct dfs_lookup *)sp)->path) {
	    ((struct dfs_lookup *)*dpp)->path =
		(char *) malloc(((struct dfs_lookup *)sp)->pathLength+1);
	    strncpy(((struct dfs_lookup *)*dpp)->path,
		    ((struct dfs_lookup *)sp)->path,
		    ((struct dfs_lookup *)*dpp)->pathLength+1);
	}
	break;
    case DFS_ROOT:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_root));
	bcopy(sp, *dpp, sizeof(struct dfs_root));
	if (((struct dfs_root *)sp)->path) {
	    ((struct dfs_root *)*dpp)->path =
		(char *) malloc(((struct dfs_root *)sp)->pathLength+1);
	    strncpy(((struct dfs_root *)*dpp)->path,
		    ((struct dfs_root *)sp)->path,
		    ((struct dfs_root *)*dpp)->pathLength+1);
	}
	break;
    case DFS_GETSYMLINK:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_getsymlink));
	bcopy(sp, *dpp, sizeof(struct dfs_getsymlink));
	if (((struct dfs_getsymlink *)sp)->path) {
	    ((struct dfs_getsymlink *)*dpp)->path =
		(char *) malloc(((struct dfs_getsymlink *)sp)->pathLength+1);
	    strncpy(((struct dfs_getsymlink *)*dpp)->path,
		    ((struct dfs_getsymlink *)sp)->path,
		    ((struct dfs_getsymlink *)*dpp)->pathLength+1);
	}
	if (((struct dfs_getsymlink *)sp)->compPath) {
	    ((struct dfs_getsymlink *)*dpp)->compPath =
		(char *) malloc(((struct dfs_getsymlink *)
				 sp)->compPathLength+1);
	    strncpy(((struct dfs_getsymlink *)*dpp)->compPath,
		    ((struct dfs_getsymlink *)sp)->compPath,
		    ((struct dfs_getsymlink *)*dpp)->compPathLength+1);
	}
	break;
    case DFS_READ:
    case DFS_WRITE:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_read));
	bcopy(sp, *dpp, sizeof(struct dfs_read));
	if (((struct dfs_read *)sp)->path) {
	    ((struct dfs_read *)*dpp)->path =
		(char *) malloc(strlen(((struct dfs_read *)sp)->path)+1);
	    (void) strcpy(((struct dfs_read *)*dpp)->path,
			  ((struct dfs_read *)sp)->path);
	}
	break;
    case DFS_NOTE:
	*dpp = (dfs_header_t *) malloc(sizeof(struct dfs_note));
	bcopy(sp, *dpp, sizeof(struct dfs_note));
	if (((struct dfs_note *)sp)->note) {
	    ((struct dfs_note *)*dpp)->note =
		(char *) malloc(((struct dfs_note *)sp)->length+1);
	    strncpy(((struct dfs_note *)*dpp)->note,
		    ((struct dfs_note *)sp)->note,
		    ((struct dfs_note *)*dpp)->length+1);
	}
	break;
    default:
	break;
    }
}

char *Trace_NodeIdToStr(addr)
     int addr;
{
    u_int n = (u_int) htonl((u_long) addr);
    static char buf[100];

    sprintf(buf, "%u.%u.%u.%u",
	    ((u_char *)&n)[0], ((u_char *)&n)[1],
	    ((u_char *)&n)[2], ((u_char *)&n)[3]);
    return(buf);
}

char *Trace_OpcodeToStr(opcode)
     u_char opcode;
{
    static char buf[100];

    switch ((int) opcode) {
    case DFS_OPEN:
	(void) strcpy(buf, "OPEN");
	break;
    case DFS_CLOSE:
	(void) strcpy(buf, "CLOSE");
	break;
    case DFS_STAT:
	(void) strcpy(buf, "STAT");
	break;
    case DFS_LSTAT:
	(void) strcpy(buf, "LSTAT");
	break;
    case DFS_SEEK:
	(void) strcpy(buf, "SEEK");
	break;
    case DFS_EXECVE:
	(void) strcpy(buf, "EXECVE");
	break;
    case DFS_EXIT:
	(void) strcpy(buf, "EXIT");
	break;
    case DFS_CHDIR:
	(void) strcpy(buf, "CHDIR");
	break;
    case DFS_FORK:
	(void) strcpy(buf, "FORK");
	break;
    case DFS_UNLINK:
	(void) strcpy(buf, "UNLINK");
	break;
    case DFS_ACCESS:
	(void) strcpy(buf, "ACCESS");
	break;
    case DFS_READLINK:
	(void) strcpy(buf, "READLINK");
	break;
    case DFS_CREAT:
	(void) strcpy(buf, "CREAT");
	break;
    case DFS_CHMOD:
	(void) strcpy(buf, "CHMOD");
	break;
    case DFS_SETREUID:
	(void) strcpy(buf, "SETREUID");
	break;
    case DFS_RENAME:
	(void) strcpy(buf, "RENAME");
	break;
    case DFS_RMDIR:
	(void) strcpy(buf, "RMDIR");
	break;
    case DFS_LINK:
	(void) strcpy(buf, "LINK");
	break;
    case DFS_CHOWN:
	(void) strcpy(buf, "CHOWN");
	break;
    case DFS_MKDIR:
	(void) strcpy(buf, "MKDIR");
	break;
    case DFS_SYMLINK:
	(void) strcpy(buf, "SYMLINK");
	break;
    case DFS_SETTIMEOFDAY:
	(void) strcpy(buf, "SETTIMEOFDAY");
	break;
    case DFS_MOUNT:
	(void) strcpy(buf, "MOUNT");
	break;
    case DFS_UNMOUNT:
	(void) strcpy(buf, "UNMOUNT");
	break;
    case DFS_TRUNCATE:
	(void) strcpy(buf, "TRUNCATE");
	break;
    case DFS_CHROOT:
	(void) strcpy(buf, "CHROOT");
	break;
    case DFS_MKNOD:
	(void) strcpy(buf, "MKNOD");
	break;
    case DFS_UTIMES:
	(void) strcpy(buf, "UTIMES");
	break;
    case DFS_LOOKUP:
	(void) strcpy(buf, "LOOKUP");
	break;
    case DFS_GETSYMLINK:
	(void) strcpy(buf, "GETSYMLINK");
	break;
    case DFS_ROOT:
	(void) strcpy(buf, "ROOT");
	break;
    case DFS_SYSCALLDUMP:
	(void) strcpy(buf, "Syscall Dump");
	break;
    case DFS_READ:
	(void) strcpy(buf, "READ");
	break;
    case DFS_WRITE:
	(void) strcpy(buf, "WRITE");
	break;
    case DFS_NOTE:
	(void) strcpy(buf, "NOTE");
	break;
    case DFS_SYNC:
	(void) strcpy(buf, "SYNC");
	break;
    case DFS_FSYNC:
	(void) strcpy(buf, "FSYNC");
	break;
    default:
	(void) strcpy(buf, "Unknown!");
	break;
    }
    return(buf);
}

char *Trace_OpenFlagsToStr(flags)
     u_short flags;
{
    static char buf[100];

    buf[0]=0;
    if (flags & DFS_FREAD) strcat(buf, "read ");
    if (flags & DFS_FWRITE) strcat(buf, "write ");
    if (flags & DFS_FAPPEND) strcat(buf, "append ");
    if (flags & DFS_FCREAT) strcat(buf, "create ");
    if (flags & DFS_FTRUNC) strcat(buf, "truncate ");
    if (flags & DFS_FEXCL) strcat(buf, "excl create ");
    if (flags & DFS_FNDELAY) strcat(buf, "non-blocking ");
    return(buf);
}

char *Trace_RecTimeToStr(recPtr)
     dfs_header_t *recPtr;
{
    static char timeStr[20];

    (void) strncpy(timeStr, ctime(&recPtr->time.tv_sec)+4, 20);
    return(timeStr);
}

/*
 * StrToInodeType -- converts a token string to an inode type.
 * Returns -1 if the type is bogus.
 */
char StrToInodeType(str)
     char *str;
{
    if (STREQ(str, "local") || STREQ(str, "ufs")) return(DFS_ITYPE_UFS);
    if (STREQ(str, "afs")) return(DFS_ITYPE_AFS);
    if (STREQ(str, "cfs")) return(DFS_ITYPE_CFS);
    if (STREQ(str, "bdev") || STREQ(str, "block")) return(DFS_ITYPE_BDEV);
    if (STREQ(str, "spec") || STREQ(str, "special")) return(DFS_ITYPE_SPEC);
    if (STREQ(str, "nfs")) return(DFS_ITYPE_NFS);
    return(-1);
}

/*
 * StrToOpcode -- converts a token string to an opcode. Returns 0
 * if the opcode is bogus.
 */
u_char StrToOpcode(str)
     char *str;
{
    if (STREQ(str, "read")) return(DFS_READ);
    if (STREQ(str, "write")) return(DFS_WRITE);
    if (STREQ(str, "lookup")) return(DFS_LOOKUP);
    if (STREQ(str, "getsymlink")) return(DFS_GETSYMLINK);
    if (STREQ(str, "root")) return(DFS_ROOT);
    if (STREQ(str, "open")) return(DFS_OPEN);
    if (STREQ(str, "close")) return(DFS_CLOSE);
    if (STREQ(str, "stat")) return(DFS_STAT);
    if (STREQ(str, "lstat")) return(DFS_LSTAT);
    if (STREQ(str, "seek")) return(DFS_SEEK);
    if (STREQ(str, "execve") || STREQ(str, "exec")) return(DFS_EXECVE);
    if (STREQ(str, "exit")) return(DFS_EXIT);
    if (STREQ(str, "fork")) return(DFS_FORK);
    if (STREQ(str, "chdir")) return(DFS_CHDIR);
    if (STREQ(str, "unlink")) return(DFS_UNLINK);
    if (STREQ(str, "access")) return(DFS_ACCESS);
    if (STREQ(str, "readlink")) return(DFS_READLINK);
    if (STREQ(str, "creat")) return(DFS_CREAT);
    if (STREQ(str, "chmod")) return(DFS_CHMOD);
    if (STREQ(str, "setreuid")) return(DFS_SETREUID);
    if (STREQ(str, "rename")) return(DFS_RENAME);
    if (STREQ(str, "rmdir")) return(DFS_RMDIR);
    if (STREQ(str, "link")) return(DFS_LINK);
    if (STREQ(str, "chown")) return(DFS_CHOWN);
    if (STREQ(str, "mkdir")) return(DFS_MKDIR);
    if (STREQ(str, "symlink")) return(DFS_SYMLINK);
    if (STREQ(str, "settimeofday")) return(DFS_SETTIMEOFDAY);
    if (STREQ(str, "mount")) return(DFS_MOUNT);
    if (STREQ(str, "unmount")) return(DFS_UNMOUNT);
    if (STREQ(str, "truncate")) return(DFS_TRUNCATE);
    if (STREQ(str, "utimes")) return(DFS_UTIMES);
    if (STREQ(str, "chroot")) return(DFS_CHROOT);
    if (STREQ(str, "mknod")) return(DFS_MKNOD);
    if (STREQ(str, "syscalldump")) return(DFS_SYSCALLDUMP);
    if (STREQ(str, "note")) return(DFS_NOTE);
    return(DFS_UNUSED);
}

/*
 * StrToFileType -- converts a token string to a file type. Returns
 * 0 if the file type is bogus.
 */
u_short StrToFileType(str)
     char *str;
{
    if (STREQ(str, "char") || STREQ(str, "character")) return(DFS_IFCHR);
    if (STREQ(str, "dir") || STREQ(str, "directory")) return(DFS_IFDIR);
    if (STREQ(str, "blk") || STREQ(str, "block")) return(DFS_IFBLK);
    if (STREQ(str, "reg") || STREQ(str, "regular")) return(DFS_IFREG);
    if (STREQ(str, "lnk") || STREQ(str, "link")) return(DFS_IFLNK);
    if (STREQ(str, "sock") || STREQ(str, "socket")) return(DFS_IFSOCK);
    return(0);
}

/*
 * Trace_FileTypeToStr
 */
char *Trace_FileTypeToStr(type)
     u_short type;
{
    static char buf[100];

    switch (type) {
    case DFS_IFREG:
	(void) strcpy(buf, "regular");
	break;
    case DFS_IFDIR:
	(void) strcpy(buf, "directory");
	break;
    case DFS_IFBLK:
	(void) strcpy(buf, "block special");
	break;
    case DFS_IFLNK:
	(void) strcpy(buf, "link");
	break;
    case DFS_IFCHR:
	(void) strcpy(buf, "character special");
	break;
    case DFS_IFSOCK:
	(void) strcpy(buf, "socket");
	break;
    default:
	(void) strcpy(buf, "INVALID");
	break;
    }
    return(buf);
}

char *Trace_InodeTypeToStr(type)
     int type;
{
    static char buf[100];

    switch (type) {
    case DFS_ITYPE_UFS:
	(void) strcpy(buf, "Unix Inode");
	break;
    case DFS_ITYPE_NFS:
	(void) strcpy(buf, "NFS Inode");
	break;
    case DFS_ITYPE_AFS:
	(void) strcpy(buf, "AFS Inode");
	break;
    case DFS_ITYPE_BDEV:
	(void) strcpy(buf, "Block Device");
	break;
    case DFS_ITYPE_SPEC:
	(void) strcpy(buf, "Special Inode");
	break;
    case DFS_ITYPE_CFS:
	(void) strcpy(buf, "Coda Inode");
	break;
    }
    return(buf);
}

char *Trace_FlagsToStr(flags)
     u_char flags;
{
    static char buf[100];

    buf[0]=0;
    if (flags & DFS_POST)
	(void) strcpy(buf, "Post record");
    if (flags & DFS_NOFID)
	(void) strcat(buf, " No fid");
    if (flags & DFS_NOPATH)
	(void) strcat(buf, " No pathname");
    if (flags & DFS_NOATTR)
	(void) strcat(buf, " No attributes");
    if (flags & DFS_TRUNCATED)
	(void) strcat(buf, " Pathname truncated");
    if (flags & DFS_NOFID2)
	(void) strcat(buf, " No fid2");
    if (flags & DFS_NOPATH2)
	(void) strcat(buf, " No pathname2");
    if (flags & DFS_TRUNCATED2)
	(void) strcat(buf, " Pathname2 truncated");
    return(buf);
}

char *Trace_FidPtrToStr(fidPtr)
     generic_fid_t *fidPtr;
{
    static char buf[100];

    switch (fidPtr->tag) {
    case DFS_ITYPE_UFS:
	sprintf(buf,"Unix Vnode (0x%lx,0x%lx)",
		fidPtr->value.local.device,
		fidPtr->value.local.number);
	break;
    case DFS_ITYPE_AFS:
	sprintf(buf, "AFS Vnode 0x%lx.0x%lx.0x%lx.0x%lx",
		fidPtr->value.afs.Cell,
		fidPtr->value.afs.Fid.Volume,
		fidPtr->value.afs.Fid.Vnode,
		fidPtr->value.afs.Fid.Unique);
	break;
    case DFS_ITYPE_CFS:
	sprintf(buf, "CFS Vnode 0x%lx.0x%lx.0x%lx",
		fidPtr->value.cfs.Volume,
		fidPtr->value.cfs.Vnode,
		fidPtr->value.cfs.Unique);
	break;
    case DFS_ITYPE_NFS:
	sprintf(buf, "NFS Vnode (0x%lx,0x%lx)",
		fidPtr->value.local.device,
		fidPtr->value.local.number);
	break;
    case DFS_ITYPE_BDEV:
	(void) strcpy(buf, "Block Device");
	break;
    case DFS_ITYPE_SPEC:
	(void) strcpy(buf, "Special Inode");
	break;
    default:
	(void) strcpy(buf, "INVALID");
	break;
    }
    return(buf);
}

char *ShortTime(timePtr)
     time_t *timePtr;
{
    static char timeStr[15];

    (void) strncpy(timeStr, ctime(timePtr)+4, 15);
    return(timeStr);
}

void Trace_PrintRecord(recPtr)
     dfs_header_t *recPtr;
{
    (void) printf("%s.%06ld %d %s ", ShortTime(&recPtr->time.tv_sec),
		  recPtr->time.tv_usec, recPtr->pid,
		  Trace_OpcodeToStr(recPtr->opcode));

    switch ((int) recPtr->opcode) {
    case DFS_READ:
    case DFS_WRITE:
	(void) printf("%s (%d), %d bytes",
		      CHECK_NULL_STRING (((struct dfs_read *)recPtr)->path),
		      ((struct dfs_read *)recPtr)->findex,
		      ((struct dfs_read *)recPtr)->amount);
	break;
    case DFS_CLOSE:
	(void) printf("%s (%d) %d ref, %d rd (%ld), %d wr (%ld), %d sk",
		      CHECK_NULL_STRING (((struct dfs_close *)recPtr)->path),
		      ((struct dfs_close *)recPtr)->findex,
		      ((struct dfs_close *)recPtr)->refCount,
		      ((struct dfs_close *)recPtr)->numReads,
		      ((struct dfs_close *)recPtr)->bytesRead,
		      ((struct dfs_close *)recPtr)->numWrites,
		      ((struct dfs_close *)recPtr)->bytesWritten,
		      ((struct dfs_close *)recPtr)->numSeeks);
	break;
    case DFS_SEEK:
	(void) printf("%s (%d) from %d to %d",
		      CHECK_NULL_STRING (((struct dfs_seek *)recPtr)->path),
		      ((struct dfs_seek *)recPtr)->findex,
		      ((struct dfs_seek *)recPtr)->oldOffset,
		      ((struct dfs_seek *)recPtr)->offset);
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	(void) printf("%s links %d",
		      CHECK_NULL_STRING (((struct dfs_rmdir *)recPtr)->path),
		      ((struct dfs_rmdir *)recPtr)->numLinks);
	break;
    case DFS_UNMOUNT:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_unmount *)recPtr)->path));
	break;
    case DFS_OPEN:
	(void) printf("%s (%d) %s",
		      CHECK_NULL_STRING (((struct dfs_open *)recPtr)->path),
		      ((struct dfs_open *)recPtr)->findex,
		      Trace_OpenFlagsToStr(((struct dfs_open *)recPtr)->flags));
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_stat *)recPtr)->path));
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_chdir *)recPtr)->path));
	break;
    case DFS_EXECVE:
	(void) printf("%s",
		      (recPtr->flags & DFS_NOPATH)?"":((struct dfs_execve *)recPtr)->path);
	break;
    case DFS_ACCESS:
    case DFS_CHMOD:
	(void) printf("%s %o",
		      CHECK_NULL_STRING (((struct dfs_access *)recPtr)->path),
		      ((struct dfs_access *)recPtr)->mode);
	break;
    case DFS_MKDIR:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_mkdir *)recPtr)->path));
	break;
    case DFS_CREAT:
	(void) printf("%s (%d)",
		      CHECK_NULL_STRING (((struct dfs_creat *)recPtr)->path),
		      ((struct dfs_creat *)recPtr)->findex);
	break;
    case DFS_RENAME:
	(void) printf("%s %s",
		      CHECK_NULL_STRING (((struct dfs_rename *)recPtr)->fromPath),
		      CHECK_NULL_STRING (((struct dfs_rename *)recPtr)->toPath));
	break;
    case DFS_LINK:
	(void) printf("%s %s",
		      CHECK_NULL_STRING (((struct dfs_link *)recPtr)->fromPath),
		      CHECK_NULL_STRING (((struct dfs_link *)recPtr)->toPath));
	break;
    case DFS_SYMLINK:
	(void) printf("%s %s",
		      CHECK_NULL_STRING (((struct dfs_symlink *)recPtr)->targetPath),
		      CHECK_NULL_STRING (((struct dfs_symlink *)recPtr)->linkPath));
	break;
    case DFS_CHOWN:
	(void) printf("%s %d.%d",
		      CHECK_NULL_STRING (((struct dfs_chown *)recPtr)->path),
		      ((struct dfs_chown *)recPtr)->group,
		      ((struct dfs_chown *)recPtr)->owner);
	break;
    case DFS_TRUNCATE:
	(void) printf("%s %ld",
		      CHECK_NULL_STRING (((struct dfs_truncate *)recPtr)->path),
		      ((struct dfs_truncate *)recPtr)->newSize);
	break;
    case DFS_UTIMES:
	(void) printf("%s %s",
		      CHECK_NULL_STRING (((struct dfs_utimes *)recPtr)->path),
		      ShortTime(&((struct dfs_utimes *)recPtr)->atime.tv_sec));
	(void) printf(" %s",
		      ShortTime(&((struct dfs_utimes *)recPtr)->mtime.tv_sec));
	break;
    case DFS_MKNOD:
	(void) printf("%s %d",
		      CHECK_NULL_STRING (((struct dfs_mknod *)recPtr)->path),
		      ((struct dfs_mknod *)recPtr)->dev);
	break;
    case DFS_MOUNT:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_mount *)recPtr)->path));
	break;
    case DFS_FORK:
	(void) printf("user %d child pid %d",
		      ((struct dfs_fork *)recPtr)->userId,
		      ((struct dfs_fork *)recPtr)->childPid);
	break;
    case DFS_SETREUID:
	(void) printf("real %d, effective %d",
		      ((struct dfs_setreuid *)recPtr)->ruid,
		      ((struct dfs_setreuid *)recPtr)->euid);
	break;
    case DFS_LOOKUP:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_lookup *)recPtr)->path));
	break;
    case DFS_ROOT:
	(void) printf("%s",
		      CHECK_NULL_STRING (((struct dfs_root *)recPtr)->path));
	break;
    case DFS_GETSYMLINK:
	(void) printf("%s -> %s",
		      CHECK_NULL_STRING (((struct dfs_getsymlink *)recPtr)->compPath),
		      CHECK_NULL_STRING (((struct dfs_getsymlink *)recPtr)->path));
	break;
    case DFS_NOTE:
	(void) printf("%s", ((struct dfs_note *)recPtr)->note);
	break;
    default:
	break;
    }
    if (recPtr->error)
	(void) printf(" err %d\n", recPtr->error);
    else
	(void) printf("\n");
}

void Trace_DumpRecord(recPtr)
     dfs_header_t *recPtr;
{
    printf("%s, %s, error %d, pid = %d at %s",
	   Trace_OpcodeToStr(recPtr->opcode), Trace_FlagsToStr(recPtr->flags),
	   recPtr->error, recPtr->pid, ctime(&recPtr->time.tv_sec));

    switch ((int) recPtr->opcode) {
    case DFS_READ:
    case DFS_WRITE:
	(void) printf("\tdesc %d (file table index %d), %d bytes\n\tfid = %s\n\t%s\n",
		      ((struct dfs_read *)recPtr)->fd,
		      ((struct dfs_read *)recPtr)->findex,
		      ((struct dfs_read *)recPtr)->amount,
		      Trace_FidPtrToStr(&((struct dfs_read *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_read *)recPtr)->path));
	break;
    case DFS_CLOSE:
	(void) printf(
		      "\tdesc %d (file table index %d) closed as part of %s\n\t%d reads (%ld bytes), %d writes (%ld bytes), %d seeks, old size = %ld, size = %ld\n\tfile type (octal) = %o, open count %d, flags (octal) = %o\n\tfid = %s\n\t%s\n",
		      ((struct dfs_close *)recPtr)->fd,
		      ((struct dfs_close *)recPtr)->findex,
		      (((struct dfs_close *)recPtr)->whence == 0)?"DUP":
		      Trace_OpcodeToStr((u_char)
					((struct dfs_close *)recPtr)->whence),
		      ((struct dfs_close *)recPtr)->numReads,
		      ((struct dfs_close *)recPtr)->bytesRead,
		      ((struct dfs_close *)recPtr)->numWrites,
		      ((struct dfs_close *)recPtr)->bytesWritten,
		      ((struct dfs_close *)recPtr)->numSeeks,
		      ((struct dfs_close *)recPtr)->oldSize,
		      (recPtr->flags & DFS_NOATTR)?(-1):
		      (((struct dfs_close *)recPtr)->size),
		      ((struct dfs_close *)recPtr)->fileType,
		      ((struct dfs_close *)recPtr)->refCount,
		      ((struct dfs_close *)recPtr)->flags,
		      Trace_FidPtrToStr(&((struct dfs_close *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_close *)recPtr)->path));
	break;
    case DFS_SEEK:
	(void) printf("\tdesc %d (findex %d): from %d to %d. %d reads (%d bytes), %d writes (%d bytes)\n\tfid = %s\n\t%s\n",
		      ((struct dfs_seek *)recPtr)->fd,
		      ((struct dfs_seek *)recPtr)->findex,
		      ((struct dfs_seek *)recPtr)->oldOffset,
		      ((struct dfs_seek *)recPtr)->offset,
		      ((struct dfs_seek *)recPtr)->numReads,
		      ((struct dfs_seek *)recPtr)->bytesRead,
		      ((struct dfs_seek *)recPtr)->numWrites,
		      ((struct dfs_seek *)recPtr)->bytesWritten,
		      Trace_FidPtrToStr(&((struct dfs_seek *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_seek *)recPtr)->path));
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	(void) printf("\tfile type (octal) = %o size = %ld links = %d\n\tfid = %s\n",
		      (unsigned) ((struct dfs_rmdir *)recPtr)->fileType,
		      ((struct dfs_rmdir *)recPtr)->size,
		      ((struct dfs_rmdir *)recPtr)->numLinks,
		      Trace_FidPtrToStr(&((struct dfs_rmdir *)recPtr)->fid));
	(void) printf("\tdir fid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_rmdir *) recPtr)->dirFid),
		      CHECK_NULL_STRING (((struct dfs_rmdir *)recPtr)->path),
		      ((struct dfs_rmdir *)recPtr)->pathLength);
	break;
    case DFS_UNMOUNT:
	(void) printf("\tfid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_unmount *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_unmount *)recPtr)->path),
		      ((struct dfs_unmount *)recPtr)->pathLength);
	break;
    case DFS_OPEN:
	(void) printf(
		      "\tfile type (octal) = %o\n\tdesc %d (file table index %d) flags=0x%x, mode=0x%x uid=%d size=%ld\n\tfid = %s\n",
		      (unsigned) ((struct dfs_open *)recPtr)->fileType,
		      ((struct dfs_open *)recPtr)->fd,
		      ((struct dfs_open *)recPtr)->findex,
		      ((struct dfs_open *)recPtr)->flags,
		      ((struct dfs_open *)recPtr)->mode,
		      (recPtr->flags & DFS_NOATTR)?(-1):
		      (((struct dfs_open *)recPtr)->uid),
		      (recPtr->flags & DFS_NOATTR)?(-1):
		      (((struct dfs_open *)recPtr)->size),
		      Trace_FidPtrToStr(&((struct dfs_open *) recPtr)->fid));
	(void) printf("\told size = %ld\n\tdir fid = %s\n\t%s (%d)\n",
		      ((struct dfs_open *)recPtr)->oldSize,
		      Trace_FidPtrToStr(&((struct dfs_open *) recPtr)->dirFid),
		      CHECK_NULL_STRING (((struct dfs_open *)recPtr)->path),
		      ((struct dfs_open *)recPtr)->pathLength);
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	(void) printf("\tfile type (octal) = %o\n\tfid = %s\n\t%s (%d)\n",
		      (unsigned) ((struct dfs_stat *)recPtr)->fileType,
		      Trace_FidPtrToStr(&((struct dfs_stat *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_stat *)recPtr)->path),
		      ((struct dfs_stat *)recPtr)->pathLength);
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	(void) printf("\tfid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_chdir *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_chdir *)recPtr)->path),
		      ((struct dfs_chdir *)recPtr)->pathLength);
	break;
    case DFS_EXECVE:
	(void) printf("\tfid = %s size = %ld owner = %d\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_execve *)recPtr)->fid),
		      (recPtr->flags & DFS_NOATTR)?(-1):((struct dfs_execve *)recPtr)->size,
		      (recPtr->flags & DFS_NOATTR)?(-1):((struct dfs_execve *)recPtr)->owner,
		      (recPtr->flags & DFS_NOPATH)?"":((struct dfs_execve *)recPtr)->path,
		      ((struct dfs_execve *)recPtr)->pathLength);
	break;
    case DFS_ACCESS:
    case DFS_CHMOD:
	(void) printf("\tfile type (octal) = %o mode (octal) = %o\n\tfid = %s\n\t%s (%d)\n",
		      (unsigned) ((struct dfs_access *)recPtr)->fileType,
		      ((struct dfs_access *)recPtr)->mode,
		      Trace_FidPtrToStr(&((struct dfs_access *)recPtr)->fid),
		      CHECK_NULL_STRING (((struct dfs_access *)recPtr)->path),
		      ((struct dfs_access *)recPtr)->pathLength);
	break;
    case DFS_MKDIR:
	(void) printf("\tmode=0x%x\n\tfid = %s\n",
		      ((struct dfs_mkdir *)recPtr)->mode,
		      Trace_FidPtrToStr(&((struct dfs_mkdir *) recPtr)->fid));
	(void) printf("\tdir fid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_mkdir *) recPtr)->dirFid),
		      ((struct dfs_mkdir *)recPtr)->path,
		      ((struct dfs_mkdir *)recPtr)->pathLength);
	break;
    case DFS_CREAT:
	(void) printf("\tdesc %d (file table index %d) mode=0x%x old size = %ld\n\tfid = %s\n",
		      ((struct dfs_creat *)recPtr)->fd,
		      ((struct dfs_creat *)recPtr)->findex,
		      ((struct dfs_creat *)recPtr)->mode,
		      ((struct dfs_creat *)recPtr)->oldSize,
		      Trace_FidPtrToStr(&((struct dfs_creat *) recPtr)->fid));
	(void) printf("\tdir fid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_creat *) recPtr)->dirFid),
		      ((struct dfs_creat *)recPtr)->path,
		      ((struct dfs_creat *)recPtr)->pathLength);
	break;
    case DFS_RENAME:
	(void) printf("\tfrom %s (len %d) (%s)\n",
		      ((struct dfs_rename *)recPtr)->fromPath,
		      ((struct dfs_rename *)recPtr)->fromPathLength,
		      Trace_FidPtrToStr(&((struct dfs_rename *) recPtr)->fromFid));
	(void) printf("\t\tparent dir fid %s\n",
		      Trace_FidPtrToStr(&((struct dfs_rename *)recPtr)->fromDirFid));
	(void) printf("\tto %s (len %d) (%s)\n",
		      ((struct dfs_rename *)recPtr)->toPath,
		      ((struct dfs_rename *)recPtr)->toPathLength,
		      Trace_FidPtrToStr(&((struct dfs_rename *) recPtr)->toFid));
	(void) printf("\t\tparent dir fid %s\n\tfile type (octal) = %o size = %ld links = %d\n",
		      Trace_FidPtrToStr(&((struct dfs_rename *) recPtr)->toDirFid),
		      (unsigned) ((struct dfs_rename *)recPtr)->fileType,
		      ((struct dfs_rename *)recPtr)->size,
		      ((struct dfs_rename *)recPtr)->numLinks);
	break;
    case DFS_LINK: /* doesn't work if the fidptrtostr calls are in same printf */
	(void) printf("\tfrom %s (len %d) (%s)\n",
		      ((struct dfs_link *)recPtr)->fromPath,
		      ((struct dfs_link *)recPtr)->fromPathLength,
		      Trace_FidPtrToStr(&((struct dfs_link *)recPtr)->fromFid));
	(void) printf("\tfile type (octal) is %o\n\tparent dir fid %s\n",
		      (unsigned) ((struct dfs_link *)recPtr)->fileType,
		      Trace_FidPtrToStr(&((struct dfs_link *)recPtr)->fromDirFid));
	(void) printf("\tto dir %s (len %d) (%s)\n",
		      ((struct dfs_link *)recPtr)->toPath,
		      ((struct dfs_link *)recPtr)->toPathLength,
		      Trace_FidPtrToStr(&((struct dfs_link *)recPtr)->toDirFid));
	break;
    case DFS_SYMLINK:
	(void) printf("\ttarget %s (len %d)\n\tlink %s (len %d) (%s)\n",
		      ((struct dfs_symlink *)recPtr)->targetPath,
		      ((struct dfs_symlink *)recPtr)->targetPathLength,
		      ((struct dfs_symlink *)recPtr)->linkPath,
		      ((struct dfs_symlink *)recPtr)->linkPathLength,
		      Trace_FidPtrToStr(&((struct dfs_symlink *)recPtr)->fid));
	(void) printf("\tdir fid %s\n",
		      Trace_FidPtrToStr(&((struct dfs_symlink *)recPtr)->dirFid));
	break;
    case DFS_CHOWN:
	(void) printf("\tfile type (octal) = %o group = %d owner = %d\n\tfid = %s\n\t%s (%d)\n",
		      (unsigned) ((struct dfs_chown *)recPtr)->fileType,
		      ((struct dfs_chown *)recPtr)->group,
		      ((struct dfs_chown *)recPtr)->owner,
		      Trace_FidPtrToStr(&((struct dfs_chown *)recPtr)->fid),
		      ((struct dfs_chown *)recPtr)->path,
		      ((struct dfs_chown *)recPtr)->pathLength);
	break;
    case DFS_TRUNCATE:
	(void) printf("\told length = %ld new length = %ld\n\tfid = %s\n\t%s (%d)\n",
		      ((struct dfs_truncate *)recPtr)->oldSize,
		      ((struct dfs_truncate *)recPtr)->newSize,
		      Trace_FidPtrToStr(&((struct dfs_truncate *) recPtr)->fid),
		      ((struct dfs_truncate *)recPtr)->path,
		      ((struct dfs_truncate *)recPtr)->pathLength);
	break;
    case DFS_UTIMES:
	(void) printf("\tfile type (octal) = %o\n\tfid = %s\n\tatime = %s",
		      (unsigned) ((struct dfs_utimes *)recPtr)->fileType,
		      Trace_FidPtrToStr(&((struct dfs_utimes *)recPtr)->fid),
		      ctime(&((struct dfs_utimes *)recPtr)->atime.tv_sec));
	(void) printf("\tmtime = %s\t%s (%d)\n",
		      ctime(&((struct dfs_utimes *)recPtr)->mtime.tv_sec),
		      ((struct dfs_utimes *)recPtr)->path,
		      ((struct dfs_utimes *)recPtr)->pathLength);
	break;
    case DFS_MKNOD:
	(void) printf("\tdev = %d mode = 0x%x\n\tfid = %s\n",
		      ((struct dfs_mknod *)recPtr)->dev,
		      ((struct dfs_mknod *)recPtr)->mode,
		      Trace_FidPtrToStr(&((struct dfs_mknod *) recPtr)->fid));
	(void) printf("\tdir fid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_mknod *) recPtr)->dirFid),
		      ((struct dfs_mknod *)recPtr)->path,
		      ((struct dfs_mknod *)recPtr)->pathLength);
	break;
    case DFS_MOUNT:
	(void) printf("\tfid = %s rwflag = %d\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_mount *)recPtr)->fid),
		      ((struct dfs_mount *)recPtr)->rwflag,
		      ((struct dfs_mount *)recPtr)->path,
		      ((struct dfs_mount *)recPtr)->pathLength);
	break;
    case DFS_FORK:
	(void) printf("\tuser %d forked child pid %d\n",
		      ((struct dfs_fork *)recPtr)->userId,
		      ((struct dfs_fork *)recPtr)->childPid);
	break;
    case DFS_SETREUID:
	(void) printf("\treal uid = %d, effective = %d\n",
		      ((struct dfs_setreuid *)recPtr)->ruid,
		      ((struct dfs_setreuid *)recPtr)->euid);
	break;
    case DFS_LOOKUP:
	(void) printf("\tfile type (octal) = %o\n\tparent fid = %s\n",
		      ((unsigned) ((struct dfs_lookup *)recPtr)->fileType),
		      Trace_FidPtrToStr(&((struct dfs_lookup *)recPtr)->parentFid));
	(void) printf("\tcomponent fid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_lookup *)recPtr)->compFid),
		      ((struct dfs_lookup *)recPtr)->path,
		      ((struct dfs_lookup *)recPtr)->pathLength);
	break;
    case DFS_ROOT:
	(void) printf("\tcomponent fid = %s\n",
		      Trace_FidPtrToStr(&((struct dfs_root *)recPtr)->compFid));
	(void) printf("\ttarget fid = %s\n\t%s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_root *)recPtr)->targetFid),
		      ((struct dfs_root *)recPtr)->path,
		      ((struct dfs_root *)recPtr)->pathLength);
	break;
    case DFS_GETSYMLINK:
	(void) printf("\tfid = %s\n\t%s (%d) -> %s (%d)\n",
		      Trace_FidPtrToStr(&((struct dfs_getsymlink *)recPtr)->fid),
		      ((struct dfs_getsymlink *)recPtr)->compPath,
		      ((struct dfs_getsymlink *)recPtr)->compPathLength,
		      ((struct dfs_getsymlink *)recPtr)->path,
		      ((struct dfs_getsymlink *)recPtr)->pathLength);
	break;
    case DFS_NOTE:
	(void) printf("%s\n", ((struct dfs_note *)recPtr)->note);
	break;
    case DFS_SYSCALLDUMP: {
	u_char i;

	(void) printf("\tsystem call\ttimes called\n");
	for (i = 1; i <= DFS_MAXSYSCALL; i++)
	    printf("\t%s\t%d\n", Trace_OpcodeToStr(i),
		   ((struct dfs_call *)recPtr)->count[i]);
	break;
    }
    default:
	break;
    }
}
