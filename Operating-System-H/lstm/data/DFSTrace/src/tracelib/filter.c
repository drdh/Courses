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


static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/filter.c,v 1.2 1998/10/06 19:40:30 tmkr Exp $";
 */
#endif _BLURB_

/*
 * filter.c  -- filter implementation
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#ifdef __SunOS__
#include <re_comp.h>
#else
#include <regex.h>
#endif

#include "filter.h"
#include "ftable.h"
#include "trace.h"
#include "pid.h"
#include "rec.h"
#include "tracelib.h"

/* converts ASCII string to time*/
extern time_t atot();

int ParseEnd();
int ParseError();
int ParseInodeType();
int ParseMatchFds();
int ParseOpcode();
int ParsePath();
int ParsePid();
int ParseRef();
int ParseStart();
int ParseType();
int ParseUser();

int MatchFds();
int IsPidAncestor();
void MarkFds();

/*
 * GetToken --
 * Copies next token out of first argument and into second.  Token is null-terminated.
 * Function returns first argument or NULL if no token was found.
 * Third argument is set to point at first char after this token (or NULL).
 */
char *GetToken(buf, token, nextpp)
     char *buf;
     char *token;
     char **nextpp;
{
    char *bp, *tp;
    int intoken = 0;
    char c;

    *nextpp = NULL;
    if (buf == NULL) return(NULL);

    bp = buf;
    tp = token;

    while ((c = *bp++) && c != '\n') {
	if (isspace(c)) {
	    if (!intoken) continue;
	    break;
	}
	if (!intoken) intoken = 1;
	*tp++ = c;
    }

    if (!intoken) return(NULL);
    *tp = '\0';
    *nextpp = bp;
    return(buf);
}

u_long opcodeMasks[DFS_MAXOPCODE][MAXWIDTH] = {
    { 0x0, 0x1 }, { 0x0, 0x2 }, { 0x0, 0x4 }, { 0x0, 0x8 },
    { 0x0, 0x10 }, { 0x0, 0x20 }, { 0x0, 0x40 }, { 0x0, 0x80 },
    { 0x0, 0x100 }, { 0x0, 0x200 }, { 0x0, 0x400 }, { 0x0, 0x800 },
    { 0x0, 0x1000 }, { 0x0, 0x2000 }, { 0x0, 0x4000 }, { 0x0, 0x8000 },
    { 0x0, 0x10000 }, { 0x0, 0x20000 }, { 0x0, 0x40000 }, { 0x0, 0x80000 },
    { 0x0, 0x100000 }, { 0x0, 0x200000 }, { 0x0, 0x400000 }, { 0x0, 0x800000 },
    { 0x0, 0x1000000 }, { 0x0, 0x2000000 }, { 0x0, 0x4000000 }, { 0x0, 0x8000000 },
    { 0x0, 0x10000000 }, { 0x0, 0x20000000 }, { 0x0, 0x40000000 }, { 0x0, 0x80000000 },
    { 0x1, 0x0 }, { 0x2, 0x0 }, { 0x4, 0x0 }, { 0x8, 0x0 },
    { 0x10, 0x0 }
};

/* InitFilter -- sets filter defaults */
void InitFilter(filterPtrPtr)
     filter_t **filterPtrPtr;
{
    int i;

    *filterPtrPtr = (filter_t *) malloc(sizeof(filter_t));
    bzero(*filterPtrPtr, sizeof(filter_t));

    for (i = 0; i < MAXWIDTH; i++)
	(*filterPtrPtr)->opcodes[i] = -1;   /* get all opcodes */
    (*filterPtrPtr)->inodeTypes = -1;  /* all object types */
    (*filterPtrPtr)->fileTypes = -1;  /* all file types */
    (*filterPtrPtr)->refCount = -1;  /* ignore refcounts */
    (*filterPtrPtr)->pidFlag = 0;
    (*filterPtrPtr)->userFlag = 0;
    (*filterPtrPtr)->pidList = NULL;  /* no specific pids */
    (*filterPtrPtr)->userList = NULL;  /* no specific users */
    (*filterPtrPtr)->errorList = NULL;  /* no specific errors */
    (*filterPtrPtr)->pathList = NULL;  /* no paths */
    (*filterPtrPtr)->matchfds = 0;      /* don't match by default */
    timerclear(&(*filterPtrPtr)->startTime);
    timerclear(&(*filterPtrPtr)->endTime);
    return;
}

struct parsefilter {
    char cmd[MAXCMDLEN];
    int  (*parsefun)();
};

struct parsefilter commands[] = {
    {"end", ParseEnd},
    {"error", ParseError},
    {"matchfds", ParseMatchFds},
    {"inodetype", ParseInodeType},
    {"opcode", ParseOpcode},
    {"path", ParsePath},
    {"pid", ParsePid},
    {"refcount", ParseRef},
    {"start", ParseStart},
    {"type", ParseType},
    {"user", ParseUser},
    { "END", NULL}
    /*{(char *) NULL, (int (*)()) NULL }*/
};

/*
 * ParseFilterFile -- reads in a filter file and dispatches commands
 * to appropriate parsing routines, which set the filter structure.
 * Parsing routines return non-zero if there was a problem.
 */
void ParseFilterFile(ffp, filterPtr)
     FILE *ffp;
     filter_t *filterPtr;
{
    char line[MAXCMDLEN];
    char token[MAXCMDLEN];
    char *cp;
    int i;

    while (fgets(line, MAXCMDLEN, ffp) != NULL) {
	cp = line;
	if (GetToken(cp, token, &cp) == NULL) continue;

	i = 0;
	while ((commands[i].parsefun != NULL) &&
	       (!(STREQ(commands[i].cmd, token))))
	    i++;

	if ((commands[i].parsefun== NULL) ||
	    (commands[i].parsefun(filterPtr, cp)))
	    printf("ParseFilterFile: error parsing command %s", token);
    }
    return;
}

int ParseEnd(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int rc = 0;

    if (cp)
	filterPtr->endTime.tv_sec = (long) atot(cp);
    else
	rc = 1;

    return(rc);
}

int ParseError(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    char token[MAXCMDLEN];
    int i = 0, rc = 0;

    filterPtr->errorList = (int **) malloc(MAXERRARRAY*sizeof(int *));
    bzero(filterPtr->errorList, MAXERRARRAY*sizeof(int *));
    while ((GetToken(cp, token, &cp) != NULL) &&
	   (i < MAXERRARRAY)) {
	filterPtr->errorList[i] = (int *) malloc(sizeof(int));
	*(filterPtr->errorList[i++]) = atoi(token);
    }
    if (i >= MAXERRARRAY)
	rc = 1;

    return(rc);
}

int ParseInodeType(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    char it;
    char token[MAXCMDLEN];
    int rc = 0;

    filterPtr->inodeTypes = 0;
    while (GetToken(cp, token, &cp) != NULL) {
	if ((it = StrToInodeType(token)) != -1)
	    filterPtr->inodeTypes |= (1 << it);
	else
	    rc = 1;
    }
    return(rc);
}

int ParseMatchFds(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    filterPtr->matchfds = 1;
    return(0);
}

int ParseOpcode(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int i = 0, rc = 0;
    u_char op;
    char token[MAXCMDLEN];
    char *ocp;

    ocp = cp;
    GetToken(cp, token, &cp);
    if (STREQ(token, "exclude")) {
	for (i = 0; i < MAXWIDTH; i++)
	    filterPtr->opcodes[i] = -1;
    } else {
	for (i = 0; i < MAXWIDTH; i++)
	    filterPtr->opcodes[i] = 0;
	if (!STREQ(token, "include"))
	    cp = ocp;
    }
    while (GetToken(cp, token, &cp) != NULL) {
	op = StrToOpcode(token);
	if (op) {
	    for (i = 0; i < MAXWIDTH; i++)
		filterPtr->opcodes[i] ^=
		    opcodeMasks[op-1][i];
	} else
	    rc = 1;
    }
    return(rc);
}

int ParsePath(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int i = 0, rc = 0;
    char token[MAXCMDLEN];
    char *ocp;
    regex_t r;

    filterPtr->pathList = (char **) malloc(sizeof(char *)*MAXPATHARRAY);
    bzero(filterPtr->pathList, MAXPATHARRAY*sizeof(char *));
    ocp = cp;
    GetToken(cp, token, &cp);
    if (STREQ(token, "exclude"))
	filterPtr->pathFlag |= FILTER_EXCLUDE;
    else if (!STREQ(token, "include"))
	cp = ocp;
    while ((GetToken(cp, token, &cp) != NULL) &&
	   (i < MAXPATHARRAY)) {

	/* compare the regular expansion for correctness */
	if (regcomp(&r, token, 0)) {
	    printf("Can't process regexp %s, ignoring\n", token);
	    rc = 1;
	} else {
	    filterPtr->pathList[i] =
		(char *) malloc(strlen(token)+1);
	    strcpy(filterPtr->pathList[i], token);
	}
	i++;
    }

    if (i >= MAXPATHARRAY)
	rc = 1;

    return(rc);
}

int ParsePid(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int i = 0, rc = 0;
    char token[MAXCMDLEN];
    char *ocp;

    filterPtr->pidList = (pfil_t **) malloc(sizeof(pfil_t *)*MAXPIDARRAY);
    bzero(filterPtr->pidList, MAXPIDARRAY*sizeof(pfil_t *));
    ocp = cp;
    GetToken(cp, token, &cp);
    if (STREQ(token, "exclude"))
	filterPtr->pidFlag |= FILTER_EXCLUDE;
    else if (!STREQ(token, "include"))
	cp = ocp;
    while ((GetToken(cp, token, &cp) != NULL) &&
	   (i < MAXPIDARRAY)) {
	filterPtr->pidList[i] = (pfil_t *) malloc(sizeof(pfil_t));
	if (token[strlen(token)-1] == '+')
	    filterPtr->pidList[i]->kid = 1;
	else
	    filterPtr->pidList[i]->kid = 0;
	filterPtr->pidList[i++]->pid = atoi(token);
    }

    if (i >= MAXPIDARRAY)
	rc = 1;

    return(rc);
}

int ParseRef(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int rc = 0;
    char token[MAXCMDLEN];

    if (GetToken(cp, token, &cp) != NULL)
	filterPtr->refCount = (short) atoi(token);

    return(rc);
}

int ParseStart(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int rc = 0;

    if (cp)
	filterPtr->startTime.tv_sec = (long) atot(cp);
    else
	rc = 1;

    return(rc);
}

int ParseType(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int rc = 0;
    char token[MAXCMDLEN];

    filterPtr->fileTypes = 0;  /* clear */
    while (GetToken(cp, token, &cp) != NULL) {
	u_short fileType;

	fileType = StrToFileType(token);
	if (fileType)
	    filterPtr->fileTypes |=
		(1 << (fileType >> 12));
    }
    return(rc);
}

int ParseUser(filterPtr, cp)
     filter_t *filterPtr;
     char *cp;
{
    int rc = 0, i = 0;
    char token[MAXCMDLEN];
    char *ocp;

    filterPtr->userList = (uid_t **) malloc(sizeof(uid_t *)*MAXUSERARRAY);
    bzero(filterPtr->userList, MAXUSERARRAY*sizeof(uid_t *));
    ocp = cp;
    GetToken(cp, token, &cp);
    if (STREQ(token, "exclude"))
	filterPtr->userFlag |= FILTER_EXCLUDE;
    else if (!STREQ(token, "include"))
	cp = ocp;
    while ((GetToken(cp, token, &cp) != NULL) &&
	   (i < MAXUSERARRAY)) {
	filterPtr->userList[i] = (uid_t *) malloc(sizeof(uid_t));
	*filterPtr->userList[i++] = atoi(token);
    }
    if (i >= MAXUSERARRAY)
	rc = 1;

    return(rc);
}

/*
 * FilteredOut -- internal routine to screen records within a chunk.
 * If a record is filtered out, it is freed at this point.
 * The filter is set using Trace_SetFilter.
 */
int FilteredOut(tfPtr, recordPtr)
     trace_file_t *tfPtr;
     dfs_header_t *recordPtr;
{
    u_char youlose = 0;
    u_short ft;
    int count, i, num;
    int sum = 0;
    generic_fid_t *fidplist[DFS_MAXFIDS];
    filter_t *filterPtr;

    filterPtr = tfPtr->filterPtr;

    /* check matching (doesn't make sense to check for failed calls) */
    if ((filterPtr->matchfds) && (recordPtr->error == 0) &&
	MatchFds(tfPtr, recordPtr)) {
	goto lose;
    }

    /* check times */
    if (timerisset(&filterPtr->startTime) &&
	timercmp(&recordPtr->time, &filterPtr->startTime, <)) {
	goto lose;
    }
    if (timerisset(&filterPtr->endTime) &&
	timercmp(&recordPtr->time, &filterPtr->endTime, >)) {
	goto lose;
    }

    /* check opcode */
    for (i = 0; i < MAXWIDTH; i++)
	sum |= filterPtr->opcodes[i] & opcodeMasks[recordPtr->opcode - 1][i];
    if (sum == 0) {
	goto lose;
    }

    /* check error field */
    if (filterPtr->errorList != NULL) {
	int i = 0;
	char foundit = 0;

	while (filterPtr->errorList[i] != NULL)
	    if (*filterPtr->errorList[i++] == recordPtr->error)
		foundit = 1;
	if (!foundit) {
	    goto lose;
	}
    }

    /* check pid */
    if (filterPtr->pidList != NULL) {
	int i = 0;

	if (filterPtr->pidFlag & FILTER_EXCLUDE) {
	    while (filterPtr->pidList[i] != NULL) {
		if ((filterPtr->pidList[i]->pid == recordPtr->pid) ||
		    (filterPtr->pidList[i]->kid &&
		     IsPidAncestor(tfPtr, filterPtr->pidList[i]->pid,
				   recordPtr->pid)))
		    goto lose;
		i++;
	    }
	} else { /* include */
	    char foundit = 0;

	    while (filterPtr->pidList[i] != NULL) {
		if ((filterPtr->pidList[i]->pid == recordPtr->pid) ||
		    (filterPtr->pidList[i]->kid &&
		     IsPidAncestor(tfPtr, filterPtr->pidList[i]->pid,
				   recordPtr->pid)))
		    foundit = 1;
		i++;
	    }
	    if (!foundit) {
		goto lose;
	    }
	}
    }

    /* check user */
    if (filterPtr->userList != NULL) {
	int i = 0;

	if (filterPtr->userFlag & FILTER_EXCLUDE) {
	    while (filterPtr->userList[i] != NULL) {
		if (*filterPtr->userList[i] ==
		    GetPidUser(tfPtr, recordPtr->pid))
		    goto lose;
		i++;
	    }
	} else { /* include */
	    char foundit = 0;

	    while (filterPtr->userList[i] != NULL) {
		if (*filterPtr->userList[i] ==
		    GetPidUser(tfPtr, recordPtr->pid))
		    foundit = 1;
		i++;
	    }
	    if (!foundit)
		goto lose;
	}
    }

    /* check path */
    if (filterPtr->pathList != NULL) {
	int i = 0;
	int nump = 0;
	int j;
	char *pathplist[DFS_MAXPATHS];

	Trace_GetPath(recordPtr, pathplist, &nump);

	if (filterPtr->pathFlag & FILTER_EXCLUDE) {
	    while (filterPtr->pathList[i] != NULL) {

		regex_t r;
		(void) regcomp(&r, filterPtr->pathList[i],
			       REG_ICASE|REG_NOSUB);

		for (j = 0; j < nump; j++)
		    if (pathplist[j] &&
			regexec(&r, pathplist[j], 0, 0, 0) == 0)
			    goto lose;
		i++;
	    }
	} else { /* include */
	    char foundit = 0;

	    while (filterPtr->pathList[i] != NULL) {

		regex_t r;
		(void) regcomp(&r, filterPtr->pathList[i],
			       REG_ICASE|REG_NOSUB);

		for (j = 0; j < nump; j++)

		    if (pathplist[j] && regexec(&r, pathplist[j], 0, 0, 0) == 0)

			    foundit = 1;
		i++;
	    }
	    if (!foundit)
		goto lose;
	}
    }

    /* check file type */
    ft = Trace_GetFileType(recordPtr);
    if ((ft != DFS_IFMT) && (!(filterPtr->fileTypes & (1 << (ft>>12) )))) {
	/* HACK ALERT
	 * If this is a close, don't filter on file type. Reason: if
	 * a creat was passed through, the close needs to pass also.
	 * if the open/creat was filtered out, we wouldn't be here
	 * with a close record anyway.
	 */
	if (recordPtr->opcode != DFS_CLOSE)
	    goto lose;
    }

    /* check refcount */
    if (filterPtr->refCount != -1) {
	count = Trace_GetRefCount(recordPtr);
	if ((count != -1) && (count != filterPtr->refCount)) {
	    goto lose;
	}
    }

    /* check inode type */
    Trace_GetFid(recordPtr, fidplist, &num);
    for (i = 0; i < num; i++)
	if ((fidplist[i]->tag != -1) &&
	    (!(filterPtr->inodeTypes & (1 << (fidplist[i]->tag)))))
	    goto lose;

    goto out;
 lose:
    youlose = 1;
    if ((filterPtr->matchfds) && (recordPtr->error == 0))
	MarkFds(tfPtr, recordPtr);
    (void) Trace_FreeRecord(tfPtr->fp, recordPtr);
 out:
    return(youlose);
}

/*
 * TimeIsUp -- abbreviated version of FilteredOut that checks to see
 * if we've passed the end time on a filter. In this case, we would
 * want to stop early. Returns 0 if we're still before the end time,
 * and 1 if our time is up.
 */
int TimeIsUp(filterPtr, recordPtr)
     filter_t *filterPtr;
     dfs_header_t *recordPtr;
{
    if (timerisset(&filterPtr->endTime) &&
	timercmp(&recordPtr->time, &filterPtr->endTime, >))
	return(1);
    return(0);
}

/*
 * MatchFds -- routine to check file descriptor based operations with
 * their previous opens or creats. If a match is found, returns 0.
 * If the previous open or creat was filtered out, this routine returns 1.
 * The file table entry is filled out in DoFileTable.
 */
int MatchFds(tfPtr, recordPtr)
     trace_file_t *tfPtr;
     dfs_header_t *recordPtr;
{
    int rc = 0;
    short findex;

    findex = Trace_GetFileIndex(recordPtr);

    switch (recordPtr->opcode) {
    case DFS_SEEK:
    case DFS_READ:
    case DFS_WRITE:
	if ((!tfPtr->kernFileTable[findex].allocated) ||
	    (tfPtr->kernFileTable[findex].filtered))
	    rc = 1;
	break;
    case DFS_CLOSE:
	if (tfPtr->kernFileTable[findex].allocated) {
	    /*
	     * got it. now decide if this record should be passed
	     * on to the user. If the open/creat was filtered out, we
	     * should also filter the close.
	     */
	    if (tfPtr->kernFileTable[findex].filtered)
		rc = 1;
	    /*
	     * the close must not be filtered out here based on
	     * reference count. If it is the last one, reset
	     * the entry.
	     */
	    if (((struct dfs_close *)recordPtr)->refCount == 1)
		FileTableDelete(tfPtr, recordPtr);
	} else {
	    rc = 1; /* couldn't find open for this close */
	}
	break;
    default:
	break;
    }
    return(rc);
}

/*
 * MarkFds -- routine to mark opens and creats in the file table as
 * filtered out, so subsequent matches on those filtered records will fail.
 */
void MarkFds(tfPtr, recordPtr)
     trace_file_t *tfPtr;
     dfs_header_t *recordPtr;
{
    short findex;

    findex = Trace_GetFileIndex(recordPtr);
    if ((recordPtr->opcode == DFS_OPEN ||
	 (recordPtr->opcode == DFS_CREAT)) &&
	(findex >= 0))
	tfPtr->kernFileTable[findex].filtered = 1;
}
