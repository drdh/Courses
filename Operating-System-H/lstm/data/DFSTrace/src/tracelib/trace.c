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



static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/trace.c,v 1.5 1998/10/23 03:37:57 tmkr Exp $";

*/
#endif _BLURB_


/*
 *  Trace.c -- routines for reading trace records.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/errno.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>

#include "filter.h"
#include "ftable.h"
#include "pid.h"
#include "split.h"
#include "tracelib.h"
#include "trace.h"
#include "unpack.h"

char verbose = 0;
char debug = 0;

trace_file_t *TraceFileTable[TRACE_HASH_SIZE];

trace_file_t *TraceFileLookup(key)
FILE *key;
{
    trace_file_t *current =
	TraceFileTable[((unsigned int) key) % TRACE_HASH_SIZE];

    while (current != (trace_file_t *) 0) {
	if (key == current->fp)
	    return(current);
	current = current->traceFileLink;
    }

    return((trace_file_t *) 0);
}

/**
 * Determines which bucket this record hashes  to.
 *set the link for the new record to point to  the record
 *currently in that bucket, and puts the new record's
 *address in the bucket.
 */
void TraceFileInsert(pRecord)
trace_file_t *pRecord;
{
    trace_file_t **bucketPtr =
	&TraceFileTable[((unsigned int) pRecord->fp) % TRACE_HASH_SIZE];
    pRecord->traceFileLink = *bucketPtr;
    *bucketPtr = pRecord;
}

trace_file_t *TraceFileDelete(key)
FILE *key;
{
    trace_file_t **linkPtr =
	&TraceFileTable[((unsigned int) key) % TRACE_HASH_SIZE];

    while (*linkPtr != (trace_file_t *) 0) {
	if (key == (*linkPtr)->fp) {
	    trace_file_t *current = *linkPtr;
	    (*linkPtr) = current->traceFileLink;
	    return(current);
	}
	linkPtr = &(*linkPtr)->traceFileLink;
    }

    return((trace_file_t *) 0);
}


/*
 * Trace_Stats -- gets info stored in hash table entry.
 */
int Trace_Stats(fp, statPtr)
     FILE *fp;
     trace_stat_t *statPtr;
{
    trace_file_t *tfPtr;

    if ((tfPtr = TraceFileLookup(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    *statPtr = tfPtr->traceStats;
    return(TRACE_SUCCESS);
}

int Trace_GetVersion(fp, vp)
     FILE *fp;
     char *vp;
{
    trace_file_t *tfPtr;

    if ((tfPtr = TraceFileLookup(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    *vp = tfPtr->version;
    return(TRACE_SUCCESS);
}

void FreeTraceFile(tfPtr)
     trace_file_t *tfPtr;
{
    DeleteAllUsers(tfPtr);
    free(tfPtr->fileName);
    free(tfPtr->filterPtr);
    free(tfPtr->chunk);
    free(tfPtr->chunkPreamblePtr);
    free(tfPtr->preamblePtr);
    free(tfPtr->command);
    free(tfPtr);
}

/**
 * opens and decodes a new trace file
 *modified October 98 to check if file is compressed and if
 *so use popen with gnu zip
 */
trace_file_t *NewTraceFile(name)
     char *name;
{
    trace_file_t *traceFilePtr;
    int rc,length;
    char *command;

    traceFilePtr = (trace_file_t *) malloc(sizeof(trace_file_t));
    bzero((char *) traceFilePtr, sizeof(trace_file_t));

    traceFilePtr->chunkOffset = -1;  /* no chunks yet */
    traceFilePtr->fileName = (char *) malloc(strlen(name)+1);
    InitFilter(&traceFilePtr->filterPtr);
    (void) strcpy(traceFilePtr->fileName, name);

    /* Check if the file is compressed if so use a popen */
    length =strlen(name);
    if ( length >3  && (strncmp(traceFilePtr->fileName+length-3,
				".gz", 3)==0)) {
	command =(char *) malloc (length + 40);
	sprintf(command, " gzip -dc %s 2> /dev/null",name);
	if ((traceFilePtr->fp =popen(command, "r")) == NULL) {
	    FreeTraceFile(traceFilePtr);
	    return(NULL);
	}
	traceFilePtr-> command = command;
	traceFilePtr-> compressed = 1;
    }
    /* Otherwise open the file with fopen */
    else {
	if ((traceFilePtr->fp =fopen(traceFilePtr->fileName, "r")) == NULL) {
	    FreeTraceFile(traceFilePtr);
	    return(NULL);
	}
	traceFilePtr-> compressed = 0;
	traceFilePtr-> command = NULL;
    }

    rc = DecodeTrace(traceFilePtr);

    switch (rc) {
    case TRACE_BADVERSION:
	printf("Bad trace version, file %s\n", traceFilePtr->fileName);
	FreeTraceFile(traceFilePtr);
	traceFilePtr = NULL;
	break;
    case TRACE_FILEREADERROR:
	printf("Error decoding trace %s\n", traceFilePtr->fileName);
	FreeTraceFile(traceFilePtr);
	traceFilePtr = NULL;
	break;
    default:
	if (verbose)
	    PrintPreamble(traceFilePtr);
    }
    return(traceFilePtr);
}

char *VersionToStr(version)
     u_short version;
{
    static char buf[25];
    sprintf(buf, "%d.%d", version >> 8, version & 0x0f);
    return(buf);
}

/*
 * Trace_Open -- opens a trace file, reads the preamble, and
 * returns a FILE *.
 */
FILE *Trace_Open(name)
     char *name;
{
    trace_file_t *traceFilePtr;

    if ((traceFilePtr = NewTraceFile(name)) == NULL)
	return(NULL);

    TraceFileInsert(traceFilePtr);
    return(traceFilePtr->fp);
}

/*
 * Trace_SetFilter -- shell for ParseFilterFile, which reads
 * in a filter file and sets the filter accordingly.
 */
int Trace_SetFilter(fp, fileName)
     FILE *fp;
     char *fileName;
{
    FILE *ffp;
    trace_file_t *tfPtr;

    if ((tfPtr = TraceFileLookup(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    if ((ffp = fopen(fileName, "r")) == NULL)
	return(TRACE_FILENOTFOUND);

    ParseFilterFile(ffp, tfPtr->filterPtr);
    fclose(ffp);
    return(TRACE_SUCCESS);
}

/*
 * Trace_GetUser -- shell for GetPidUser, which
 * searches the user table for the pid.
 */
int Trace_GetUser(fp, pid, uidp)
     FILE *fp;
     short pid;
     uid_t *uidp;
{
    trace_file_t *tfPtr;

    if ((tfPtr = TraceFileLookup(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    *uidp = GetPidUser(tfPtr, pid);
    return(TRACE_SUCCESS);
}

/*
 * Trace_PrintPreamble -- shell for PrintPreamble.
 */
int Trace_PrintPreamble(fp)
     FILE *fp;
{
    trace_file_t *tfPtr;

    if ((tfPtr = TraceFileLookup(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    PrintPreamble(tfPtr);
    return(TRACE_SUCCESS);
}

/*
 * Trace_FreeRecord -- frees the storage allocated at recPtr. Needed
 * because some records have storage for pathnames and other associated
 * data structures that must also be freed.
 */
int Trace_FreeRecord(fp, recPtr)
     FILE *fp;
     dfs_header_t *recPtr;
{
    trace_file_t *tfPtr;

    /* find trace file info */
    if ((tfPtr = TraceFileLookup(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    switch ((int) recPtr->opcode) {
    case DFS_OPEN:
	if (((struct dfs_open *)recPtr)->pathLength)
	    free(((struct dfs_open *)recPtr)->path);
	break;
    case DFS_STAT:
    case DFS_LSTAT:
	if (((struct dfs_stat *)recPtr)->pathLength)
	    free(((struct dfs_stat *)recPtr)->path);
	break;
    case DFS_CHDIR:
    case DFS_CHROOT:
    case DFS_READLINK:
	if (((struct dfs_chdir *)recPtr)->pathLength)
	    free(((struct dfs_chdir *)recPtr)->path);
	break;
    case DFS_EXECVE:
	if (((struct dfs_execve *)recPtr)->pathLength)
	    free(((struct dfs_execve *)recPtr)->path);
	break;
    case DFS_ACCESS:
    case DFS_CHMOD:
	if (((struct dfs_access *)recPtr)->pathLength)
	    free(((struct dfs_access *)recPtr)->path);
	break;
    case DFS_CREAT:
	if (((struct dfs_creat *)recPtr)->pathLength)
	    free(((struct dfs_creat *)recPtr)->path);
	break;
    case DFS_MKDIR:
	if (((struct dfs_mkdir *)recPtr)->pathLength)
	    free(((struct dfs_mkdir *)recPtr)->path);
	break;
    case DFS_CHOWN:
	if (((struct dfs_chown *)recPtr)->pathLength)
	    free(((struct dfs_chown *)recPtr)->path);
	break;
    case DFS_RENAME:
	if (((struct dfs_rename *)recPtr)->fromPathLength)
	    free(((struct dfs_rename *)recPtr)->fromPath);
	if (((struct dfs_rename *)recPtr)->toPathLength)
	    free(((struct dfs_rename *)recPtr)->toPath);
	break;
    case DFS_LINK:
	if (((struct dfs_link *)recPtr)->fromPathLength)
	    free(((struct dfs_link *)recPtr)->fromPath);
	if (((struct dfs_link *)recPtr)->toPathLength)
	    free(((struct dfs_link *)recPtr)->toPath);
	break;
    case DFS_SYMLINK:
	if (((struct dfs_symlink *)recPtr)->targetPathLength)
	    free(((struct dfs_symlink *)recPtr)->targetPath);
	if (((struct dfs_symlink *)recPtr)->linkPathLength)
	    free(((struct dfs_symlink *)recPtr)->linkPath);
	break;
    case DFS_RMDIR:
    case DFS_UNLINK:
	if (((struct dfs_rmdir *)recPtr)->pathLength)
	    free(((struct dfs_rmdir *)recPtr)->path);
	break;
    case DFS_UNMOUNT:
	if (((struct dfs_unmount *)recPtr)->pathLength)
	    free(((struct dfs_unmount *)recPtr)->path);
	break;
    case DFS_TRUNCATE:
	if (((struct dfs_truncate *)recPtr)->pathLength)
	    free(((struct dfs_truncate *)recPtr)->path);
	break;
    case DFS_UTIMES:
	if (((struct dfs_utimes *)recPtr)->pathLength)
	    free(((struct dfs_utimes *)recPtr)->path);
	break;
    case DFS_MKNOD:
	if (((struct dfs_mknod *)recPtr)->pathLength)
	    free(((struct dfs_mknod *)recPtr)->path);
	break;
    case DFS_MOUNT:
	if (((struct dfs_mount *)recPtr)->pathLength)
	    free(((struct dfs_mount *)recPtr)->path);
	break;
    case DFS_LOOKUP:
	if (((struct dfs_lookup *)recPtr)->pathLength)
	    free(((struct dfs_lookup *)recPtr)->path);
	break;
    case DFS_ROOT:
	if (((struct dfs_root *)recPtr)->pathLength)
	    free(((struct dfs_root *)recPtr)->path);
	break;
    case DFS_GETSYMLINK:
	if (((struct dfs_getsymlink *)recPtr)->pathLength)
	    free(((struct dfs_getsymlink *)recPtr)->path);
	if (((struct dfs_getsymlink *)recPtr)->compPathLength)
	    free(((struct dfs_getsymlink *)recPtr)->compPath);
	break;
    case DFS_CLOSE:
	if (((struct dfs_close *)recPtr)->path)
	    free(((struct dfs_close *)recPtr)->path);
	break;
    case DFS_SEEK:
	if (((struct dfs_seek *)recPtr)->path)
	    free(((struct dfs_seek *)recPtr)->path);
	break;
    case DFS_READ:
    case DFS_WRITE:
	if (((struct dfs_read *)recPtr)->path)
	    free(((struct dfs_read *)recPtr)->path);
	break;
    case DFS_EXIT:
	/* free pid structure */
	if (tfPtr->version != TRACE_VERSION_UCB1)
	    DoneWithPid(tfPtr, recPtr);
	break;
    case DFS_FORK:
    case DFS_SETREUID:
    case DFS_SETTIMEOFDAY:
    case DFS_SYSCALLDUMP:
    default:
	break;
    }
    free(recPtr);
    return(TRACE_SUCCESS);
}

/*
 * Trace_GetRecord -- reads trace data from file fp, and returns a pointer to
 * a trace record.  Storage for the record is allocated in this module,
 * but must be freed by calling Trace_FreeRecord.
 */
dfs_header_t *Trace_GetRecord(fp)
     FILE *fp;
{
    trace_file_t *tfPtr;
    dfs_header_t *recordPtr = NULL;
    dfs_header_t *prePtr = NULL;
    dfs_header_t *postPtr = NULL;
    char incomplete;

    /* find trace file info */
    if ((tfPtr = TraceFileLookup(fp)) == NULL) {
	printf("Couldn't find trace file!\n");
	return(NULL);
    }

    do {
	tfPtr->unpackRecordProc(tfPtr, &recordPtr);
	if (recordPtr == NULL)
	    return(NULL);

	tfPtr->traceStats.totalRecords++;
	if (!(timerisset(&tfPtr->traceStats.firstTraceRecordTime))) {
	    tfPtr->traceStats.firstTraceRecordTime = recordPtr->time;
	    if (verbose)
		printf("Trace begins at %s",
		       ctime(&recordPtr->time.tv_sec));
	}
	tfPtr->traceStats.lastTraceRecordTime = recordPtr->time;

	incomplete = 0;

	/* merge split records */
	if ((tfPtr->version != TRACE_VERSION_UCB1) &&
	    (record_is_split(recordPtr->opcode))) {
	    if (recordPtr->flags & DFS_POST) {
		/* look for pre record and reassemble */
		postPtr = recordPtr;
		GetSplitRecord(postPtr, &prePtr);
		recordPtr = MergeSplitRecords(prePtr, postPtr);
	    } else {
		/* stash pre record and keep going */
		StashSplitRecord(recordPtr);
		incomplete = 1;
	    }
	}

	if (!incomplete) {
	    /* check time against filter end time to see if we should continue */
	    if (TimeIsUp(tfPtr->filterPtr, recordPtr)) {
		(void) Trace_FreeRecord(fp, recordPtr);
		return(NULL);
	    }
	    tfPtr->traceStats.recordsRead++;
	    /* Update   the table of currently open files*/
	    if (recordPtr->error == 0)
		DoFileTable(tfPtr, recordPtr);
	    if (tfPtr->version != TRACE_VERSION_UCB1)
		AddPid(tfPtr, recordPtr);
	}
    } while (incomplete || FilteredOut(tfPtr, recordPtr));

    tfPtr->traceStats.recordsUsed++;
    return(recordPtr);
}

int Trace_Close(fp)
     FILE *fp;
{
    trace_file_t *tfPtr;

    if ((tfPtr = TraceFileDelete(fp)) == NULL)
	return(TRACE_FILERECORDNOTFOUND);

    if (verbose)
	printf("Closing trace file %s\n", tfPtr->fileName);

    FreeTraceFile(tfPtr);
    pclose(fp);
    return(TRACE_SUCCESS);
}
