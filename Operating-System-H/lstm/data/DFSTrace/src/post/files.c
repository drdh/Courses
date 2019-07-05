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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/files.c,v 1.3 1998/10/13 21:14:31 tmkr Exp $";
*/
#endif _BLURB_


/*
 *  files.c -- lists file names by frequency of access.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <sys/param.h>
#include <time.h>
#include <string.h>
#include "tracelib.h"


typedef struct fref {
	char         *name;
	u_long        refs;
	generic_fid_t fid;
	struct fref  *frefLink;
} fref_t;


/* for hashing by paths */
#define FREF_HASH_KEY(x)  (x[0]+x[strlen(x)-1]+strlen(x))

#define FREF_HASH_SIZE 101
fref_t *FRefTable[FREF_HASH_SIZE];

#define TOPN 100   /* number of files to report from list */

int slot = 0;
fref_t **base;
int absolute = 0;  /* absolute pathnames only */
int inodes = 0;    /* default is don't print vnodes */
int top = TOPN;    /* report top 100 by default */


fref_t *FRefLookup(key)
char *key;
{
    fref_t *current =
	FRefTable[((unsigned int) FREF_HASH_KEY(key)) % FREF_HASH_SIZE];
    while (current != (fref_t *) 0) {
	if ((strcmp(key, (current)->name) == 0))
	    return(current);
	current = current->frefLink;
    }
    return((fref_t *) 0);
}

fref_t *FRefDelete(key)
char *key;
{
    fref_t **linkPtr =
	&FRefTable[((unsigned int) FREF_HASH_KEY(key)) % FREF_HASH_SIZE];
    while (*linkPtr != (fref_t *) 0) {
	if ((strcmp(key,(*linkPtr)->name) == 0)) {
	    fref_t *current = *linkPtr;
	    (*linkPtr) = current->frefLink;
	    return(current);
	}
	linkPtr = &(*linkPtr)->frefLink;
    }
    return((fref_t *) 0);
}

void FRefInsert(pRecord)
fref_t *pRecord;
{
    fref_t **bucketPtr =
	&FRefTable[((unsigned int) FREF_HASH_KEY(pRecord->name)) %
		   FREF_HASH_SIZE];
    pRecord->frefLink = *bucketPtr;
    *bucketPtr = pRecord;
}

void FRefForall(f)
void (*f)();
{
    unsigned int i;
    fref_t *current,*next;
    for (i=0; i< FREF_HASH_SIZE; i++) {
	current = FRefTable[i];
	while (current != (fref_t *) 0) {
	    next = current->frefLink;
	    f(current);
	    current = next;
	}
    }
}

int GoodRef(recPtr)
dfs_header_t *recPtr;
{
	char *pathplist[DFS_MAXPATHS];
	int num;

	Trace_GetPath(recPtr, pathplist, &num);
	if ((recPtr->opcode == DFS_OPEN ||
	     recPtr->opcode == DFS_CREAT ||
	     recPtr->opcode == DFS_MKDIR ||
	     recPtr->opcode == DFS_RMDIR ||
	     recPtr->opcode == DFS_UNLINK ||
	     recPtr->opcode == DFS_STAT ||
	     recPtr->opcode == DFS_CHDIR ||
	     recPtr->opcode == DFS_LSTAT ||
	     recPtr->opcode == DFS_EXECVE ||
	     recPtr->opcode == DFS_ACCESS ||
	     recPtr->opcode == DFS_CHMOD ||
	     recPtr->opcode == DFS_CHOWN ||
	     recPtr->opcode == DFS_TRUNCATE ||
	     recPtr->opcode == DFS_RENAME ||
	     recPtr->opcode == DFS_LINK ||
	     recPtr->opcode == DFS_SYMLINK ||
	     recPtr->opcode == DFS_UTIMES ||
	     recPtr->opcode == DFS_MKNOD ||
	     recPtr->opcode == DFS_READLINK ||
	     recPtr->opcode == DFS_CHROOT) &&
	    (recPtr->error == 0) &&
	    pathplist[0])
		return(1);
	return(0);
}

fref_t *MakeFRef(recPtr)
dfs_header_t *recPtr;
{
	generic_fid_t *fidplist[DFS_MAXFIDS];
	char *pathplist[DFS_MAXPATHS];
	int i;
	fref_t *frefPtr;

	Trace_GetFid(recPtr, fidplist, &i);
	Trace_GetPath(recPtr, pathplist, &i);
	if (!GoodRef(recPtr) ||
	    (fidplist[0] == NULL) ||
	    (fidplist[0]->tag == -1) ||
	    (pathplist[0] == NULL))
		return(NULL);

	frefPtr = (fref_t *) malloc(sizeof(fref_t));
	bzero(frefPtr, sizeof(fref_t));
	frefPtr->refs = 1;
	frefPtr->fid = *fidplist[0];
	frefPtr->name = (char *) malloc(strlen(pathplist[0])+1);
	strcpy(frefPtr->name, pathplist[0]);
	return(frefPtr);
}

void CopyToBase(frefPtr)
fref_t *frefPtr;
{
	base[slot++] = frefPtr;
}

int FRefCompare(f1, f2)
fref_t **f1, **f2;
{
	return((*f2)->refs - (*f1)->refs);
}

void PrintUsage()
{
	printf("Usage: files [-a] [-d] [-i] [-v] [-n numfiles] [-f filter] file\n");
	exit(1);
}

int main(argc, argv)
int argc;
char *argv[];
{
	extern int optind;
	extern char *optarg;
	FILE *inFile;
	dfs_header_t *recPtr;
	fref_t *frefPtr, *nrefPtr;
	int numFRefs, i, limit;
	char filterName[MAXPATHLEN];

	filterName [0] = 0;
	numFRefs =0;

	/* get filename */
	if (argc < 2)
		PrintUsage();

	/* Obtain invocation options */
	while ((i = getopt(argc, argv, "avdin:f:")) != EOF)
		switch (i) {
		case 'a':
			absolute = 1;
			break;
		case 'v':
			verbose = 1;
			break;
		case 'd':
			debug = 1;
			break;
		case 'i':
			inodes = 1;
			break;
		case 'n':
			top = atoi(optarg);
			break;
		case 'f':
			(void) strcpy(filterName, optarg);
			break;
		default:
			PrintUsage();
		}


	if (((inFile = Trace_Open(argv[optind])) == NULL) ||
	    (filterName[0] && Trace_SetFilter(inFile, filterName))) {
		printf("files: can't process file %s.\n", argv[optind]);
		exit(1);
	}

	while ((recPtr = Trace_GetRecord(inFile))!= NULL) {
		if ((nrefPtr = MakeFRef(recPtr))!=  NULL ) {
			if ((frefPtr = FRefLookup(nrefPtr->name))!=  NULL)
				frefPtr->refs++;
			else {
				numFRefs++;
				FRefInsert(nrefPtr);
			}
		}
		(void) Trace_FreeRecord(inFile, recPtr);
	}
	(void) Trace_Close(inFile);

	/* sort the results */
	base = (fref_t **) malloc(numFRefs * sizeof(fref_t *));
	FRefForall(CopyToBase);

	qsort((char *) base, numFRefs, sizeof(fref_t *), FRefCompare);
	limit = (top < numFRefs) ? top : numFRefs;
	if (top==-1) {
	  limit = numFRefs;
	}
	if (inodes) {
		printf("refs\tname\tfid\n");
		for (i = 0; i < limit; i++)
			printf("%ld\t%s\t%s\n", base[i]->refs, base[i]->name,
			       Trace_FidPtrToStr(&base[i]->fid));
	} else {
		printf("refs\tname\n");
		for (i = 0; i < limit; i++)
			printf("%ld\t%s\n", base[i]->refs, base[i]->name);
	}
	return 0;
}
