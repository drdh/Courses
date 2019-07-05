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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/untrace.c,v 1.4 1998/11/06 03:38:35 tmkr Exp $";
*/
#endif _BLURB_

/*
 *
 *    Untrace: extract skeleton and replay command files from a
 * 	trace file
 *		original version by jjk
 *		modifications by lily, luqi
 *
 *    The program makes two passes over the input tracefile.
 *    The first simply builds data structures which allow the
 *    set of skeleton-building commands to be emitted at completion.
 *    The second pass emits the replay commands as it processes each
 *    record.
 *
 *    The procedure is difficult (much more difficult than I originally
 *    thought!) because it must cope with incomplete data, i.e., the trace
 *    file does not (in general) contain all of the information upon which
 *    trace interpretation depends.  Thus, certain "inferences" of
 *    "extra-trace" activity must be made.  The potentially missing data
 *    includes:
 *       - operations which are filtered out of the trace file (due to
 *	   pid/uid selection, open/close matching, etc)
 *       - operations which occurred on a different client of the distributed
 *	   file system (e.g., a traced object was created, removed, or
 *	   renamed on another workstation)
 *       - operations affecting the file system structure which are not traced;
 *         e.g., initial resolution of an AFS/CFS volume mount point,
 *	   reevaulation of an AFS/CFS mount point, "metamorphosis" of a CFS
 *	   object (i.e., to/from dangling link)
 *
 *    NOTE: for proper matching of open and close calls, this program
 *	must be used with a filter file (-f) containing the following
 *	commands:
 * 			error 0
 * 			type regular link directory
 * 			matchfds
 * 			refcount 1
 *
 *    ToDo:
 *       - handle AFS/CFS volume releases correctly
 *	 - remove filter file requirement
 *
 */

#ifdef __cplusplus
extern "C" {
#endif __cplusplus

#include <sys/types.h>
#include <fcntl.h>
#include <strings.h>
#include <dirent.h>
#include <sys/param.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#ifdef CMUCS
#include <libc.h>
#else
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#endif

#include "tracelib.h"

#ifdef __cplusplus
}
#endif __cplusplus


#include "dlist.h"
#include "dhash.h"


#define CHECK_BAD_TYPE(type) ((type & DFS_IFMT)==DFS_IFSOCK || \
                              (type & DFS_IFMT) == DFS_IFBLK || \
                              (type & DFS_IFMT) ==  DFS_IFCHR)
/*  *****  Forward Declarations  *****  */

struct direntry;
struct child_iterator;
struct parent_iterator;
struct fsobj;
struct fsobj_iterator;
extern dhashtab FSDB;
extern int FidHash(generic_fid *);

/*  *****  File System Objects  *****  */

const int nFidBuckets =	1024;	    /* buckets in Fid hash table */
const int nDirBuckets =	16;	    /* buckets in Directory hash tables */

enum MountStatus { Root, MountPoint, Normal };

struct fsobj : dlink {
    generic_fid fid;

    int	type;			/* {DFS_IFREG, DFS_IFDIR, DFS_IFLNK} */
    dlist parents;		/* parent direntries */
    union {
	struct {		/* DFS_IFREG */
	    char *path;		/* used as source of hard link */
	} file;

	struct {		/* DFS_IFDIR */
	    dhashtab *children;	/* child direntries */

	    MountStatus	mstat;	/* {Root, MountPoint, Normal} */
	    union {
		fsobj *mtpt;	/* Root */
		fsobj *root;	/* MountPoint */
				/* Normal */
	    } m;
	} dir;

	struct {		/* DFS_IFLNK */
	    char *contents;	/* not really needed! */
	} symlink;
    } u;

    unsigned initial : 1;	/* did object exist prior to start of trace? */
    unsigned deleted : 1;	/* has object been deleted? */
    unsigned mutated : 1;	/* was object (or some descendent) mutated in course of trace? */

    fsobj(generic_fid *, int, int);
    ~fsobj();

    fsobj *GetChild(char *, generic_fid *, int);
    fsobj *CreateChild(char *, generic_fid *, int);
    void DeleteChild(fsobj *, char *, int =0);
    direntry *CreateChildEntry(fsobj *, char *);

    direntry *FindChildEntry(char *, int * =0);
    direntry *MakeChildEntry(fsobj *, char *, int);
    void DestroyChildEntry(direntry *, int =0);
    direntry *FindParentEntry(char *, generic_fid *, int * =0);
    direntry *MakeParentEntry(fsobj *, char *, int);
    void DestroyParentEntry(direntry *);

    int nParents()
	{ return(parents.count()); }
    int nUndeletedParents();
    int nDeletedParents();
    fsobj *GetParent();

    void getpath(char *, int =1);/* return (a) current, non-deleted path to object */
                                 /* second arg is "ignore deleted objects" */
    void propagate();		/* propagates mutated state upwards (if TRUE) */
    void skeletize(char *);	/* emit skeleton-forming command, and visit descendents */
    void dump(int);		/* for debugging */
};

struct fsobj_iterator : dhashtab_iterator {
    fsobj_iterator(generic_fid *fid =(generic_fid *)-1) : dhashtab_iterator(FSDB, fid)
	{ ; }
    ~fsobj_iterator()
	{ ; }
    fsobj *operator()()
	{ return((fsobj *)dhashtab_iterator::operator()()); }
};

dhashtab FSDB(nFidBuckets, (int (*)(void *))&FidHash, 0);
generic_fid RootFid;		/* ought to be a constant, but union types can't be initialized! */
fsobj *root;

int FidHash(generic_fid *);
fsobj *FindFsobj(generic_fid *, int * =0);
fsobj *MakeFsobj(generic_fid *, int, int);
void DestroyFsobj(fsobj *);
fsobj *GetFsobj(generic_fid *, int);
void RenameChildEntry(fsobj *, char *, fsobj *, char *);
fsobj *InferCreate(generic_fid *, int);
fsobj *InferCreate(fsobj *, char *, generic_fid *, int);
void InferDelete(fsobj *);
void InferDelete(fsobj *, fsobj *, char *);
void InferLink(fsobj *, fsobj *, char *);
void InferRename(fsobj *, fsobj *, char *);


/*  *****  Directory Entries  *****  */

/* A direntry may represent either a standard "ParentDirectory --> ChildObject" link, */
/* or it may represent a "ChildObject --> ParentDirectory" backwards link. */
/* In the former case, the direntry is linked into the parent's u.dir.children hash table, */
/* and in the latter it is linked into the child's parents list. */
/* Note that all three types of objects have a parents list, and that in a "parent direntry" the */
/* comp field is NOT "..", but is the same as the comp field in the corresponding "child direntry." */

struct direntry : dlink {
    char *comp;			/* component string */
    generic_fid	fid;		/* associated fid */
    unsigned initial : 1;	/* did entry exist prior to start of trace? */
    unsigned deleted : 1;	/* has entry been deleted? */
				/* Don't bother with mutated flag. */
				/* We simply assume mutated state for */
				/* ALL entries referencing a mutated object! */

    direntry(char *, generic_fid *, int);
    ~direntry();

    void dump(int);		/* for debugging */
};

struct child_iterator : dhashtab_iterator {
    child_iterator(fsobj *f, char *comp =(char *)-1) : dhashtab_iterator(*f->u.dir.children, comp)
	{ ; }
    ~child_iterator()
	{ ; }
    direntry *operator()()
	{ return((direntry *)dhashtab_iterator::operator()()); }
};

struct parent_iterator : dlist_iterator {
    parent_iterator(fsobj *f) : dlist_iterator(f->parents)
	{ ; }
    ~parent_iterator()
	{ ; }
    direntry *operator()()
	{ return((direntry *)dlist_iterator::operator()()); }
};

int DirHash(char *);


/*  *****  Miscellaneous  *****  */

# ifndef NDEBUG
# define _assert(ex)\
{\
    if (!(ex)) {\
	Die(0, 0, 0, "Assertion failed: file \"%s\", line %d\n", __FILE__, __LINE__);\
    }\
}
# define assert(ex)	_assert(ex)
# else
# define _assert(ex)
# define assert(ex)
# endif

/* macro to subtract one timeval from another */
#define SUBTIME(fromp, subp)\
do {\
    if ((subp)->tv_usec > (fromp)->tv_usec) \
      { (fromp)->tv_sec--; (fromp)->tv_usec += 1000000; }\
    (fromp)->tv_sec -= (subp)->tv_sec;\
    (fromp)->tv_usec -= (subp)->tv_usec;\
} while(0);

#define	STREQ(a, b) (strcmp((a), (b)) == 0)

#define	ISDOT(comp) (STREQ((comp), "."))

#define	ISDOTDOT(comp) (STREQ((comp), ".."))

const generic_fid UnsetFid = { (char)-1 };

#define	UFS_FID_EQ(fid1, fid2)\
(\
    ((fid1).value.local.device == (fid2).value.local.device) &&\
    ((fid1).value.local.number == (fid2).value.local.number)\
)
#define	AFS_FID_EQ(fid1, fid2)\
(\
    ((fid1).value.afs.Cell == (fid2).value.afs.Cell) &&\
    ((fid1).value.afs.Fid.Volume == (fid2).value.afs.Fid.Volume) &&\
    ((fid1).value.afs.Fid.Vnode == (fid2).value.afs.Fid.Vnode) &&\
    ((fid1).value.afs.Fid.Unique == (fid2).value.afs.Fid.Unique)\
)
#define	CFS_FID_EQ(fid1, fid2)\
(\
    ((fid1).value.cfs.Volume == (fid2).value.cfs.Volume) &&\
    ((fid1).value.cfs.Vnode == (fid2).value.cfs.Vnode) &&\
    ((fid1).value.cfs.Unique == (fid2).value.cfs.Unique)\
)
#define	FID_EQ(fid1, fid2)\
(\
    ((fid1).tag == (fid2).tag) &&\
    (((fid1).tag == DFS_ITYPE_UFS && UFS_FID_EQ(fid1, fid2)) ||\
     ((fid1).tag == DFS_ITYPE_AFS && AFS_FID_EQ(fid1, fid2)) ||\
     ((fid1).tag == DFS_ITYPE_CFS && CFS_FID_EQ(fid1, fid2)))\
)

#define	FID_UNSET(fid)\
    ((fid).tag == (char)-1)

char *TraceFile;
char *FilterFile;
char *SkeletonFile = "skeleton.cmd";
char *ReplayFile = "replay.cmd";
int MutationsOnly;
int CrossParent;
int Verbose;
int Harsh;
int Experimental;
struct timeval Delta = {0, 0};
struct timeval LastTime = {0, 0};
FILE *tracefp;
dfs_header_t *recPtr;
FILE *skeletonfp;
FILE *replayfp;
int Pass;				    /* 1 --> DeriveSkeleton, 2 --> DeriveReplay */
generic_fid BreakFid = { (char)-1 };
struct {
    int RecordsHandled;
    int CreatesInferred;
    int DeletesInferred;
    int LinksInferred;
    int RenamesInferred;
} Stats[2];

void ParseArgs(int, char **);
void DeriveSkeleton();
void DeriveReplay();
void OpenTraceFile();
void CloseTraceFile();
void HandleRecord();
void RollBackToInitial();
void PurgeNonMutated();
void GetPath(char *, generic_fid *, char * =0, int =1);
char *LastComp(char *);
void MakeFakeName(char *, generic_fid *);
void DumpTree();
void DumpRoots();
void DumpStats();
void dprint(int, int, int, char * ...);
void Die(int, int, int, char * ...);


int main(int argc, char *argv[]) {
    ParseArgs(argc, argv);

    /* Open output files. */
    if ((skeletonfp = fopen(SkeletonFile, "w+")) == NULL) {
	fprintf(stderr, "fopen(%s, \"w+\") failed\n", SkeletonFile);
	exit(-1);
    }
    if ((replayfp = fopen(ReplayFile, "w+")) == NULL) {
	fprintf(stderr, "fopen(%s, \"w+\") failed\n", ReplayFile);
	exit(-1);
    }

    /* Create Root object. */
    RootFid.tag = DFS_ITYPE_UFS;
    RootFid.value.local.device = -1;
    RootFid.value.local.number = -1;
    root = MakeFsobj(&RootFid, DFS_IFDIR, 1);

    /* Pass 1. */
    DeriveSkeleton();
    fflush(skeletonfp);

    /* Pass 2. */
    DeriveReplay();
    fflush(replayfp);

    DumpStats();
}


void ParseArgs(int argc, char **argv) {
    int i;
    extern int optind;
    extern char *optarg;

    if (argc < 2) goto usage;

    while ((i = getopt(argc, argv, "d:ef:hms:r:vx")) != EOF)
	    switch (i) {
	    case 'd':
		Delta.tv_sec = atoi(optarg);
		break;
	    case 'e':
		Experimental = 1;
		break;
	    case 'f':
		FilterFile = optarg;
		break;
	    case 'h':
		Harsh = 1;
		break;
	    case 'm':
		MutationsOnly = 1;
		break;
	    case 'r':
		ReplayFile = optarg;
		break;
	    case 's':
		SkeletonFile = optarg;
		break;
	    case 'v':
		Verbose = 1;
		break;
	    case 'x':
		CrossParent = 1;
		break;
     	    default:
		goto usage;
	    }

    if (argc <= optind) goto usage;

    TraceFile = argv[optind];
    return;

usage:
    fprintf(stderr, "usage: untrace [-f filterfile] [-s skeletonfile] [-r replayfile] [-d delay] [-m] [-x] [-v] tracefile\n");
    exit(-1);
}


/* Cycle through all trace records, building an internal representation of the file system graph. */
/* When all records have been internalized, "roll-back" the state to what it must have been initially. */
/* Lastly, walk the graph, emitting appropriate creation commands into the skeleton file. */
void DeriveSkeleton() {
    Pass = 1;
    dprint(0, 0, 0, "DeriveSkeleton...");

    OpenTraceFile();
    for (;;) {
	recPtr = Trace_GetRecord(tracefp);
	if (recPtr == NULL) break;

	if (recPtr->error == 0) HandleRecord();

	(void) Trace_FreeRecord(tracefp, recPtr);
	recPtr = 0;
    }
    CloseTraceFile();

/*    if (Verbose) DumpTree();*/
    RollBackToInitial();
    if (MutationsOnly && Experimental) {
	PurgeNonMutated();
/*	if (Verbose) DumpTree();*/
    }
/*    if (Verbose) DumpTree();*/

    char path[MAXNAMLEN];
    strcpy(path, "root");
    root->skeletize(path);
}


/* Cycle through all trace records, updating the interna1 representation and */
/* emitting corresponding commands into the mutation file. */
void DeriveReplay() {
    Pass = 2;
    dprint(0, 0, 0, "DeriveReplay...");

    OpenTraceFile();
    for (;;) {
	recPtr = Trace_GetRecord(tracefp);
	if (recPtr == NULL) break;

	if (recPtr->error == 0) HandleRecord();

	(void) Trace_FreeRecord(tracefp, recPtr);
	recPtr = 0;
    }
    CloseTraceFile();
}


void OpenTraceFile() {
    if (tracefp != NULL)
	Die(0, 0, 0, "OpenTraceFile: open fp (%x)", tracefp);

    if ((tracefp = Trace_Open(TraceFile)) == NULL)
	Die(0, 0, 0, "OpenTraceFile: OpenFile(%s) failed", TraceFile);
    if (FilterFile && Trace_SetFilter(tracefp, FilterFile) != 0)
	Die(0, 0, 0, "OpenTraceFile: SetFilter(%s) failed", FilterFile);
}


void CloseTraceFile() {
    if (tracefp == NULL)
	Die(0, 0, 0, "CloseTraceFile: fp is null");

    (void) Trace_Close(tracefp);
    tracefp = NULL;
}


/* The fundamental routine.  Note that behavior is different depending on Pass (1 or 2). */
void HandleRecord() {
    /* The handling of each record has three phases (some of which may be null): */
    /*    1. Getting the object(s) that are involved in the operation */
    /*    2. Reflecting the operation in the internal data structures */
    /*    3. Emitting the replay command (during Pass 2 only) */
    /* Note that phases 1 and 2 may cause operations to be inferred. */

    Stats[Pass - 1].RecordsHandled++;
    if (Verbose && (Stats[Pass - 1].RecordsHandled % 100) == 0)
	dprint(0, 0, 0, "%d", Stats[Pass - 1].RecordsHandled);

    if (Pass == 2 && timerisset(&Delta)) {
	struct timeval diff;
	timerclear(&diff);

	if (timerisset(&LastTime)) {
	    if (timercmp(&recPtr->time, &LastTime, >)) {
		diff = recPtr->time;
		SUBTIME(&diff, &LastTime);
	    }

	    if (timercmp(&diff, &Delta, >))
		fprintf(replayfp, "sleep %ld.%06ld\n", diff.tv_sec, diff.tv_usec);
	}

	LastTime = recPtr->time;
    }


    char path1[MAXPATHLEN];
    path1[0] = '\0';
    char path2[MAXPATHLEN];
    path1[1] = '\0';
    switch(recPtr->opcode) {
	case DFS_OPEN: {
	    dfs_open *r = (dfs_open *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    int flags;
	    if (r->flags & DFS_FWRITE) {
		flags = O_RDWR;
		if (r->oldSize == -1) {
		    flags |= (O_CREAT | O_EXCL);
		    fsobj *pf = GetFsobj(&r->dirFid, DFS_IFDIR);

		     pf->CreateChild(LastComp(r->path),
						&r->fid, DFS_IFREG);
		}
		else {
		    if (r->size == 0) flags |= O_TRUNC;
		    fsobj *cf = GetFsobj(&r->fid, DFS_IFREG);

		    cf->mutated = 1;
		}
	    }
	    else {
		flags = O_RDONLY;
		GetFsobj(&r->fid, r->fileType & DFS_IFMT);
	    }

	    if (Pass == 2 && ((r->flags & DFS_FWRITE) || !MutationsOnly)) {
		GetPath(path1, &r->fid);
		fprintf(replayfp, "open %s %d %d\n",
			path1, flags, r->findex);
	    }
	    }
	    break;

	case DFS_CLOSE: {
	    dfs_close *r = (dfs_close *)recPtr;
	    if ((r->fileType & DFS_IFMT) ==DFS_IFSOCK) return;

	    if (Pass == 2 && ((r->flags & DFS_FWRITE) || !MutationsOnly)) {
		fprintf(replayfp, "close %d %ld\n",
			r->findex, (r->flags & DFS_FWRITE) ? r->size :(unsigned  long) -1);
	    }
	    }
	    break;

	case DFS_CHMOD: {
	    dfs_access *r = (dfs_access *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    fsobj *f = GetFsobj(&r->fid, r->fileType & DFS_IFMT);

	    f->mutated = 1;

	    if (Pass == 2) {
		GetPath(path1, &r->fid);
		fprintf(replayfp, "chmod %s\n", path1);
	    }
	    }
	    break;

	case DFS_CHOWN: {
	    dfs_chown *r = (dfs_chown *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    fsobj *f = GetFsobj(&r->fid, r->fileType & DFS_IFMT);

	    f->mutated = 1;

	    if (Pass == 2) {
		GetPath(path1, &r->fid);
		fprintf(replayfp, "chown %s\n", path1);
	    }
	    }
	    break;

	case DFS_TRUNCATE: {
	    dfs_truncate *r = (dfs_truncate *)recPtr;

	    fsobj *f = GetFsobj(&r->fid, DFS_IFREG);

	    f->mutated = 1;

	    if (Pass == 2) {
		GetPath(path1, &r->fid);
		fprintf(replayfp, "truncate %s %ld\n", path1, r->newSize);
	    }
	    }
	    break;

	case DFS_UTIMES: {
	    dfs_utimes *r = (dfs_utimes *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    fsobj *f = GetFsobj(&r->fid, r->fileType & DFS_IFMT);

	    f->mutated = 1;

	    if (Pass == 2) {
		GetPath(path1, &r->fid);
		fprintf(replayfp, "utimes %s\n", path1);
	    }
	    }
	    break;

	case DFS_CREAT: {
	    dfs_creat *r = (dfs_creat *)recPtr;

	    int flags = O_RDWR;
	    if (r->oldSize == -1) {
		flags |= (O_CREAT | O_EXCL);
		fsobj *pf = GetFsobj(&r->dirFid, DFS_IFDIR);

		pf->CreateChild(LastComp(r->path),
					    &r->fid, DFS_IFREG);
	    }
	    else {
		flags |= O_TRUNC;
		fsobj *cf = GetFsobj(&r->fid, DFS_IFREG);

		cf->mutated = 1;
	    }

	    if (Pass == 2) {
		GetPath(path1, &r->fid);
		fprintf(replayfp, "open %s %d %d\n",
			path1, flags, r->findex);
	    }
	    }
	    break;

	case DFS_UNLINK: {
	    dfs_rmdir *r = (dfs_rmdir *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    char *comp = LastComp(r->path);
	    int type = r->fileType & DFS_IFMT;
	    fsobj *pf = GetFsobj(&r->dirFid, DFS_IFDIR);
	    fsobj *cf = pf->GetChild(comp, &r->fid, type);

	    pf->DeleteChild(cf, comp, r->numLinks == 1);

	    if (Pass == 2) {
		GetPath(path1, &r->dirFid, comp);
		fprintf(replayfp, "unlink %s\n", path1);
	    }
	    }
	    break;

	case DFS_LINK: {
	    dfs_link *r = (dfs_link *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    int SameParent = FID_EQ(r->fromDirFid, r->toDirFid);
	    char *scomp = LastComp(r->fromPath);
	    char *tcomp = LastComp(r->toPath);
	    int type = r->fileType & DFS_IFMT;
	    fsobj *spf = GetFsobj(&r->fromDirFid, DFS_IFDIR);
	    fsobj *sf = spf->GetChild(scomp, &r->fromFid, type);
	    fsobj *tpf = SameParent
	      ? spf
	      : GetFsobj(&r->toDirFid, DFS_IFDIR);

	    /* Sanity checks. */
	    if (sf->type != DFS_IFREG) {
		/* This would screw things up mightily! */
		fprintf(stderr, "link of non-file (illegal)\n");
		assert(0);
	    }
	    if (!SameParent) {
		/*
		 * This will fail if replayed in AFS or CFS!
		 * So, ignore the operation, unless instructed to do otherwise.
		 */
		dprint(0, 0, 0, "cross parent link");
		if (!CrossParent)
		    break;
	    }
	    spf->mutated = 1;
	    tpf->CreateChildEntry(sf, tcomp);

	    if (Pass == 2) {
		GetPath(path1, &r->fromDirFid, scomp);
		GetPath(path2, &r->toDirFid, tcomp);
		fprintf(replayfp, "link %s %s\n", path1, path2);
	    }
	    }
	    break;

	    /* N.B.  We assume rename target doesn't exist (so a delete is always inferred if it does). */
	case DFS_RENAME: {
	    dfs_rename *r = (dfs_rename *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    int SameParent = FID_EQ(r->fromDirFid, r->toDirFid);
	    char *scomp = LastComp(r->fromPath);
	    char *tcomp = LastComp(r->toPath);
	    int type = r->fileType & DFS_IFMT;
	    fsobj *spf = GetFsobj(&r->fromDirFid, DFS_IFDIR);
	    fsobj *sf = spf->GetChild(scomp, &r->fromFid, type);
	    fsobj *tpf = GetFsobj(&r->toDirFid, DFS_IFDIR);

	    /* Sanity check. */
	    if (!SameParent && type == DFS_IFREG && sf->nUndeletedParents() > 1) {
		/* This will fail if replayed in AFS or CFS! */
		/* So, ignore the operation, unless instructed to do otherwise. */
		dprint(0, 0, 0, "cross parent link (via rename)");
		if (!CrossParent)
		    break;
	    }
	    RenameChildEntry(spf, scomp, tpf, tcomp);

	    if (Pass == 2) {
		GetPath(path1, &r->fromDirFid, scomp);
		GetPath(path2, &r->toDirFid, tcomp);
		fprintf(replayfp, "rename %s %s\n", path1, path2);
	    }
	    }
	    break;

	case DFS_MKDIR: {
	    dfs_mkdir *r = (dfs_mkdir *)recPtr;

	    char *comp = LastComp(r->path);
	    fsobj *pf = GetFsobj(&r->dirFid, DFS_IFDIR);

	    pf->CreateChild(comp, &r->fid, DFS_IFDIR);

	    if (Pass == 2) {
		GetPath(path1, &r->dirFid, comp);
		fprintf(replayfp, "mkdir %s\n", path1);
	    }
	    }
	    break;

	case DFS_RMDIR: {
	    dfs_rmdir *r = (dfs_rmdir *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    char *comp = LastComp(r->path);
	    fsobj *pf = GetFsobj(&r->dirFid, DFS_IFDIR);
	    fsobj *cf = pf->GetChild(comp, &r->fid, DFS_IFDIR);

	    pf->DeleteChild(cf, comp, r->numLinks == 1);

	    if (Pass == 2) {
		GetPath(path1, &r->dirFid, comp);
		fprintf(replayfp, "rmdir %s\n", path1);
	    }
	    }
	    break;

	case DFS_SYMLINK: {
	    dfs_symlink *r = (dfs_symlink *)recPtr;

	    char *tcomp = LastComp(r->linkPath);
	    fsobj *pf = GetFsobj(&r->dirFid, DFS_IFDIR);

	    /* fsobj *cf = pf->CreateChild(tcomp, &r->fid, DFS_IFLNK); */
	    /*
	     * symlink records are a bit wierd. A successful lookup will
	     * precede it, deposited during the creation of the link.
	     * So by now this link should already exist in the directory;
	     * it will have been inferred from the lookup record.  However,
	     * we must undo some of the inferences that may have been made.
	     */
	    fsobj *cf = pf->GetChild(tcomp, &r->fid, DFS_IFLNK);
	    assert(cf != 0);
	    cf->initial = 0;
	    direntry *cde = pf->FindChildEntry(tcomp);
	    assert(cde != 0);
	    cde->initial = 0;
	    direntry *pde = cf->FindParentEntry(tcomp, &pf->fid);
	    assert(pde != 0);
	    pde->initial = 0;
	    {
		/* This really isn't necessary! */
		cf->u.symlink.contents = new char[strlen(r->targetPath) + 1];
		strcpy(cf->u.symlink.contents, r->targetPath);
	    }

	    if (Pass == 2) {
		GetPath(path2, &r->dirFid, tcomp);
		fprintf(replayfp, "symlink %s %s\n", r->targetPath, path2);
	    }
	    }
	    break;

	case DFS_STAT:
	case DFS_LSTAT: {
	    dfs_stat *r = (dfs_stat *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    if (!MutationsOnly) {
		GetFsobj(&r->fid, r->fileType & DFS_IFMT);

		if (Pass == 2) {
		    GetPath(path1, &r->fid);
		    fprintf(replayfp, "%s %s\n",
			    recPtr->opcode == DFS_STAT ? "stat" : "lstat", path1);
		}
	    }
	    }
	    break;

	case DFS_ACCESS: {
	    dfs_access *r = (dfs_access *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    if (!MutationsOnly) {
		GetFsobj(&r->fid, r->fileType & DFS_IFMT);

		if (Pass == 2) {
		    GetPath(path1, &r->fid);
		    fprintf(replayfp, "access %s\n", path1);
		}
	    }
	    }
	    break;

	case DFS_READLINK: {
	    dfs_chdir *r = (dfs_chdir *)recPtr;

	    if (!MutationsOnly) {
		 GetFsobj(&r->fid, DFS_IFLNK);

		if (Pass == 2) {
		    GetPath(path1, &r->fid);
		    fprintf(replayfp, "readlink %s\n", path1);
		}
	    }
	    }
	    break;

	case DFS_LOOKUP: {
	    dfs_lookup *r = (dfs_lookup *)recPtr;
	    if (CHECK_BAD_TYPE(r->fileType)) return;

	    if (!ISDOTDOT(r->path) && !ISDOT(r->path)) {
		fsobj *pf = GetFsobj(&r->parentFid, DFS_IFDIR);
		pf->GetChild(r->path, &r->compFid,
					 r->fileType & DFS_IFMT);
	    }
	    }
	    break;

	case DFS_GETSYMLINK: {
	    dfs_getsymlink *r = (dfs_getsymlink *)recPtr;

	    fsobj *f = GetFsobj(&r->fid, DFS_IFLNK);
	    {
		/* This really isn't necessary! */
		if (f->u.symlink.contents == 0) {
			if (r->path) {
				f->u.symlink.contents = new char[strlen(r->path) + 1];
				strcpy(f->u.symlink.contents, r->path);
			}
		}
		else {
		    if (!STREQ(f->u.symlink.contents, r->path)) {
			dprint(0, 0, 0, "link contents changing %s --> %s",
			       f->u.symlink.contents, r->path);
			delete f->u.symlink.contents;
			if (r->path) {
				f->u.symlink.contents = new char[strlen(r->path) + 1];
				strcpy(f->u.symlink.contents, r->path);
			}
		    }
		}
	    }
	    }
	    break;

	    /* N.B.  We assume that mount state is static, and initialized prior to start of trace! */
	case DFS_ROOT: {
	    dfs_root *r = (dfs_root *)recPtr;

	    fsobj *pf = GetFsobj(&r->compFid, DFS_IFDIR);

	    /* Don't do regular GetFsobj for VFS Root objects! */
	    int AnyDeleted = 0;
	    fsobj *cf = FindFsobj(&r->targetFid, &AnyDeleted);
	    assert(AnyDeleted == 0);
	    if (cf == 0) {
		cf = MakeFsobj(&r->targetFid, DFS_IFDIR, 1);
	    }
	    else {
		assert(cf->type == DFS_IFDIR);
		assert(cf->initial);
		assert(cf->nParents() == 0 || ((cf->nParents() == 1) && (cf->GetParent() == root)));
		if (cf->nParents() == 1) {
		    /* de-link the child-parent relationship between "root" and "cf" */
		    /* remove "cf" from "root" children list */
		    /* the main difficulty here is that we do not have the "comp" for "cf" */
		    direntry *cde=NULL, *pde=NULL;
		    {
			dhashtab_iterator next(*(root->u.dir.children));
			dlink *d;
			while ((d = next())!=0) {
			    cde = (direntry *)d;
			    if (!bcmp((char *)&(cde->fid), (char *)&(cf->fid), (int)sizeof(generic_fid)))
			       /* found the entry */
			      break;
			}
			assert(cde != 0);
			assert(root->u.dir.children->remove(cde->comp, cde) == cde);
		    }

		    {
			/* remove "root" from "cf"'s parents list */
			parent_iterator next(cf);
			while ((pde = next())!=0) {
			    if (!strcmp(pde->comp, cde->comp))
			      break;
			}
			assert(pde != 0);
			assert(cf->parents.remove(pde) == pde);
			int OldPass = Pass;
			Pass = 2;
			delete pde;
			delete cde;
			Pass = OldPass;
		    }
		}
	    }

	    switch (pf->u.dir.mstat) {
		case Root:
		    assert(0);

		case MountPoint:
		    assert(pf->u.dir.m.root == cf);
		    break;

		case Normal:
		    pf->u.dir.mstat = MountPoint;
		    pf->u.dir.m.root = cf;
		    break;

		default:
		    assert(0);
	    }
	    switch (cf->u.dir.mstat) {
		case Root:
		    assert(cf->u.dir.m.mtpt == pf);
		    break;

		case MountPoint:
		    assert(0);

		case Normal:
		    cf->u.dir.mstat = Root;
		    cf->u.dir.m.mtpt = pf;
		    break;

		default:
		    assert(0);
	    }
	    }
	    break;

	   	case DFS_MOUNT:
	case DFS_UNMOUNT:
	    /* We may have to cope with these eventually! */
	  /* assert(0); */

	    /*
	case DFS_MKNOD:
	case DFS_CHROOT:
	case DFS_CHDIR:
	case DFS_FORK:
	case DFS_SEEK:
	case DFS_EXECVE:
	case DFS_EXIT:
	case DFS_SETREUID:
	case DFS_SETTIMEOFDAY:
	case DFS_SYSCALLDUMP:
	    break;

	case DFS_UNUSED:
	default:
	    assert(0);
*/
	default:
	    break;
    }
}


/* Discard non-initial direntries and objects. */
/* Also, undelete initial-and-deleted direntries and objects. */
/* This could be done via root-based tree-walk! */
void RollBackToInitial() {
    dprint(0, 0, 0, "RollBackToInitial...");

    if (Verbose)
	DumpRoots();

    /* Should we do some sanity-checking of invariants here? */
    /* 1. (pde <--> cde) ^ (pde->initial <--> cde->initial) ^ (pde->deleted <--> cde->deleted) */
    /* 2. !f->initial --> there does not exist an initial pde */
    /* 3. f->type = {DIR,LNK} --> there exists at most one initial pde */
    /* 4. cf->initial --> pf->initial */
    /* 5. f->nUndeletedParents == 0 <--> f->IsRoot */
    {
    }

    fsobj_iterator next;
    fsobj *f=NULL;
    int readahead = 0;
    while (readahead || (f = next())) {
	readahead = 0;

	/* Destroy non-initial direntries, and undelete initial ones. */
	{
	    parent_iterator pnext(f);
	    direntry *p=NULL;
	    int preadahead = 0;
	    while (preadahead || (p = pnext())) {
		preadahead = 0;

		if (p->initial) {
		    assert(f->initial);
		    p->deleted = 0;
		}
		else {
		    assert(!p->deleted);

		    direntry *tp = pnext();
		    preadahead = (tp != 0);
		    f->DestroyParentEntry(p);
		    p = tp;
		}
	    }
	}
	if (f->type == DFS_IFDIR) {
	    child_iterator cnext(f);
	    direntry *c=NULL;
	    int creadahead = 0;
	    while (creadahead || (c = cnext())) {
		creadahead = 0;

		if (c->initial) {
		    assert(f->initial);
		    c->deleted = 0;
		}
		else {
		    assert(!c->deleted);

		    direntry *tc = cnext();
		    creadahead = (tc != 0);
		    f->DestroyChildEntry(c);
		    c = tc;
		}
	    }
	}

	/* Destroy non-initial objects, and undelete initial ones. */
	if (f->initial) {
	    f->deleted = 0;
	}
	else {
	    assert(!f->deleted);

	    fsobj *tf = next();
	    readahead = (tf != 0);
	    DestroyFsobj(f);
	    f = tf;
	}
    }
}


/* Identify and discard that part of the skeleton not required to support "mutation-only" replay. */
void PurgeNonMutated() {
    dprint(0, 0, 0, "PurgeNonMutated...");

    /* 1. Propagate mutated flag from all nodes up to root. */
    {
	fsobj_iterator next;
	fsobj *f;
	while ((f = next())!=NULL) {
	    assert(f->initial);
	    assert(!f->deleted);

	    if (f->mutated)
		f->propagate();
	}
    }

    /* 2. Delete non-mutated objects and direntries. */
    /* N.B.  Alternatively, we could just have fsobj::skeletize() ignore non-mutated objects! */
    {
	/* Should we check some invariants here? */
	{
	}

	fsobj_iterator next;
	fsobj *f=NULL;
	int readahead = 0;
	while (readahead || (f = next())) {
	    assert(f->initial);
	    assert(!f->deleted);
	    readahead = 0;

	    /* Destroy direntries that reference non-mutated objects. */
	    {
		parent_iterator pnext(f);
		direntry *p=NULL;
		int preadahead = 0;
		while (preadahead || (p = pnext())) {
		    assert(p->initial);
		    assert(!p->deleted);
		    preadahead = 0;

		    fsobj *pf = FindFsobj(&p->fid);
		    assert(pf != 0);
		    if (!pf->mutated) {
			p->initial = 0;
			direntry *c = pf->FindChildEntry(p->comp);
			assert(c != 0);
			assert(c->initial);
			assert(!c->deleted);
			c->initial = 0;

			direntry *tp = pnext();
			preadahead = (tp != 0);
			f->DestroyParentEntry(p);
			p = tp;
		    }
		}
	    }
	    if (f->type == DFS_IFDIR) {
		child_iterator cnext(f);
		direntry *c = NULL;
		int creadahead = 0;
		while (creadahead || (c = cnext())) {
		    assert(c->initial);
		    assert(!c->deleted);
		    creadahead = 0;

		    fsobj *cf = FindFsobj(&c->fid);
		    assert(cf != 0);
		    if (!cf->mutated) {
			c->initial = 0;
			direntry *p = cf->FindParentEntry(c->comp, &f->fid);
			assert(p != 0);
			assert(p->initial);
			assert(!p->deleted);
			p->initial = 0;

			direntry *tc = cnext();
			creadahead = (tc != 0);
			f->DestroyChildEntry(c);
			c = tc;
		    }
		}
	    }

	    /* Destroy non-mutated objects. */
	    if (f->mutated) {
		;
	    }
	    else {
		f->initial = 0;

		fsobj *tf = next();
		readahead = (tf != 0);
		DestroyFsobj(f);
		f = tf;
	    }
	}
    }
}


fsobj *FindFsobj(generic_fid *fid, int *AnyDeleted) {
    if (AnyDeleted) *AnyDeleted = 0;

    fsobj_iterator next(fid);
    fsobj *f;
    fsobj *outf = 0;
    while ((f = (fsobj *)next())!=NULL)
	if (FID_EQ(f->fid, *fid)) {
	    if (f->deleted) {
		if (AnyDeleted) *AnyDeleted = 1;
		continue;
	    }

	    assert(outf == 0);
	    outf = f;
	}

    return(outf);
}


fsobj *MakeFsobj(generic_fid *fid, int type, int initial) {
    fsobj *f = FindFsobj(fid);
    assert(f == 0);

    f = new fsobj(fid, type, initial);
    FSDB.insert(fid, f);
    return(f);
}


void DestroyFsobj(fsobj *f) {
    /* Must destroy all direntries! */
    {
	parent_iterator pnext(f);
	direntry *p=NULL;
	int preadahead = 0;
	while (preadahead || (p = pnext())) {
	    preadahead = 0;

	    if (!p->deleted) {
		direntry *tp = pnext();
		preadahead = (tp != 0);
		f->DestroyParentEntry(p);
		p = tp;
	    }
	}
    }
    if (f->type == DFS_IFDIR) {
	child_iterator cnext(f);
	direntry *c=NULL;
	int creadahead = 0;
	while (creadahead || (c = cnext())) {
	    creadahead = 0;

	    if (!c->deleted) {
		direntry *tc = cnext();
		creadahead = (tc != 0);
		f->DestroyChildEntry(c);
		c = tc;
	    }
	}
    }

    if (Pass == 1 && f->initial) {
	f->deleted = 1;
    }
    else {
	assert(FSDB.remove(&f->fid, f) == f);
	delete f;
    }
}


fsobj *GetFsobj(generic_fid *fid, int type) {
    assert(!FID_UNSET(*fid));

    fsobj *f = FindFsobj(fid);
    if (f == 0) {
	f = InferCreate(fid, type);
    }
    else {
	if (f->type == type) {
	    /* This would appear to be the object we want (subsequent operations may show otherwise). */
	    ;
	}
	else {
	    /* We assume object was removed and re-created "extra-trace." */
	    InferDelete(f);
	    f = InferCreate(fid, type);
	}
    }

    return(f);
}


void RenameChildEntry(fsobj *spf, char *scomp, fsobj *tpf, char *tcomp) {
    direntry *sde = spf->FindChildEntry(scomp);
    assert(sde != 0);
    fsobj *sf = FindFsobj(&sde->fid);
    assert(sf != 0);
    spf->DestroyChildEntry(sde);
    spf->mutated = 1;
    sf->mutated = 1;

    tpf->CreateChildEntry(sf, tcomp);
    tpf->mutated = 1;
}


/* Inferred create in root directory. */
fsobj *InferCreate(generic_fid *cfid, int ctype) {
    int AnyDeleted = 0;
    fsobj *cf = FindFsobj(cfid, &AnyDeleted);
    assert(cf == 0);

    char fakename[MAXNAMLEN];
    MakeFakeName(fakename, cfid);
    if (AnyDeleted) {
	/* Must really infer a create. */
	cf = InferCreate(root, fakename, cfid, ctype);
    }
    else {
	/* Quietly create the object and make an initial entry in the root. */
	/* The hope is that the object will be quietly moved to its real parent before end of Pass 1. */
	assert(Pass == 1);
	dprint(0, 0, 0, "RootCreate: %s", fakename);
	cf = MakeFsobj(cfid, ctype, 1);
	(void)root->MakeChildEntry(cf, fakename, 1);
    }

    return(cf);
}


fsobj *InferCreate(fsobj *pf, char *comp, generic_fid *cfid, int ctype) {
    assert(pf->type == DFS_IFDIR);

    direntry *cde = pf->FindChildEntry(comp);
    if (cde != 0) {
	fsobj *tf = FindFsobj(&cde->fid);
	assert(tf != 0);
	InferDelete(pf, tf, comp);
    }

    fsobj *cf = MakeFsobj(cfid, ctype, 0);
    (void)pf->MakeChildEntry(cf, comp, 0);
    pf->mutated = 1;
    cf->mutated = 1;

    char path[MAXPATHLEN];
    GetPath(path, &pf->fid, comp);
    dprint(0, 0, 0, "InferCreate: %s", path);
    Stats[Pass - 1].CreatesInferred++;

    if (Pass == 2) {
	switch(ctype) {
	    case DFS_IFREG:
		fprintf(replayfp, "open %s %d -1\n",
			path, (O_RDWR | O_CREAT | O_EXCL));
		break;

	    case DFS_IFDIR:
		fprintf(replayfp, "mkdir %s\n", path);
		break;

	    case DFS_IFLNK:
		fprintf(replayfp, "symlink %s %s\n",
			"unknown_contents", path);
		break;

	    default:
		assert(0);
	}
    }

    return(cf);
}


/* Infer deletion of all links to the object (and therefore of the object itself). */
void InferDelete(fsobj *cf) {
    int n = cf->nUndeletedParents();
    if (n == 0) {
	/*
	 * This is an orphaned fso.  An unlink or rmdir was processed
	 * with a link count > 1, but untrace does not have state on
	 * any other links.  This could occur as a result of ignoring
	 * cross parent link records, or because of an initial link
	 * that has not manifested in any way except in the unlink or
	 * rmdir record.
	 *
	 * Since as far as the untrace is concerned the object is gone,
	 * destroy the fso without issuing any commands for Pass 2.
	 */
	char fakename[MAXNAMLEN];
	MakeFakeName(fakename, &cf->fid);
	dprint(0, 0, 0, "Destroying orphaned fso %s", fakename);
	DestroyFsobj(cf);
    }

    for (int i = 0; i < n; i++) {
	parent_iterator pnext(cf);
	direntry *pde;
	while ((pde = pnext())!=NULL)
	    if (!pde->deleted) {
		/* Must restart iteration after every delete since deletion invalidates iterator! */
		fsobj *pf = FindFsobj(&pde->fid);
		assert(pf != 0);
		char comp[MAXNAMLEN];
		strcpy(comp, pde->comp);
		InferDelete(pf, cf, comp);
		break;
	    }
    }
}


/* Infer deletion of a particular link to the object. */
/* If that link happens to be the last, destroy the object as well. */
/* N.B.  We must recursively delete directory objects! */
void InferDelete(fsobj *pf, fsobj *cf, char *comp) {
    assert(pf->type == DFS_IFDIR);

    int ctype = cf->type;

    pf->mutated = 1;
    cf->mutated = 1;

    /* Recursively remove directories. */
    if (ctype == DFS_IFDIR) {
	assert(cf->nUndeletedParents() == 1);
	child_iterator cnext(cf);
	direntry *c=NULL;
	int creadahead = 0;
	while (creadahead || (c = cnext())) {
	    creadahead = 0;

	    if (!c->deleted) {
		direntry *tc = cnext();
		creadahead = (tc != 0);

		fsobj *tf = FindFsobj(&c->fid);
		if (tf == 0) {
		    Trace_DumpRecord(recPtr);
		    exit(-1);
		}
		InferDelete(tf);

		c = tc;
	    }
	}
    }

    direntry *cde = pf->FindChildEntry(comp);
    assert(cde != 0);
    pf->DestroyChildEntry(cde);

    if (cf->nUndeletedParents() == 0)
	DestroyFsobj(cf);

    char path[MAXPATHLEN];
    GetPath(path, &pf->fid, comp);
    dprint(0, 0, 0, "InferDelete: %s", path);
    Stats[Pass - 1].DeletesInferred++;

    if (Pass == 2) {
	switch(ctype) {
	    case DFS_IFREG:
		fprintf(replayfp, "unlink %s\n", path);
		break;

	    case DFS_IFDIR:
		fprintf(replayfp, "rmdir %s\n", path);
		break;

	    case DFS_IFLNK:
		fprintf(replayfp, "unlink %s\n", path);
		break;

	    default:
		assert(0);
	}
    }
}


void InferLink(fsobj *sf, fsobj *tpf, char *tcomp) {
    assert(tpf->type == DFS_IFDIR);

    sf->mutated = 1;
    tpf->mutated = 1;

    char path1[MAXPATHLEN];
    GetPath(path1, &sf->fid);

    (void)tpf->CreateChildEntry(sf, tcomp);

    char path2[MAXPATHLEN];
    GetPath(path2, &tpf->fid, tcomp);
    dprint(0, 0, 0, "InferLink: %s %s", path1, path2);
    Stats[Pass - 1].LinksInferred++;

    if (Pass == 2) {
	fprintf(replayfp, "link %s %s\n", path1, path2);
    }
}


void InferRename(fsobj *sf, fsobj *tpf, char *tcomp) {
    assert(tpf->type == DFS_IFDIR);

    assert(sf->nUndeletedParents() == 1);
    parent_iterator pnext(sf);
    direntry *pde = 0;
    while ((pde = pnext())!=  NULL )
	if (!pde->deleted)
	    break;
    assert(pde != 0);
    fsobj *spf = FindFsobj(&pde->fid);
    assert(spf != 0);
    char *scomp = pde->comp;

    spf->mutated = 1;
    sf->mutated = 1;
    tpf->mutated = 1;

    RenameChildEntry(spf, scomp, tpf, tcomp);

    char path1[MAXPATHLEN];
    GetPath(path1, &spf->fid, scomp);
    char path2[MAXPATHLEN];
    GetPath(path2, &tpf->fid, tcomp);
    dprint(0, 0, 0, "InferRename: %s %s", path1, path2);
    Stats[Pass - 1].RenamesInferred++;

    if (Pass == 2) {
	fprintf(replayfp, "rename %s %s\n", path1, path2);
    }
}


fsobj::fsobj(generic_fid *Fid, int Type, int Initial) {
    fid = *Fid;
    type = Type;
    switch(type) {
	case DFS_IFREG:
	    u.file.path = 0;
	    break;

	case DFS_IFDIR:
	    u.dir.children = new dhashtab(nDirBuckets,
					  (int (*)(void *))&DirHash, 0);
	    u.dir.mstat = Normal;
	    bzero((char *)&u.dir.m, (int) sizeof(u.dir.m));
	    break;

	case DFS_IFLNK:
	    u.symlink.contents = 0;
	    break;

	default:
	    assert(0);
    }
    initial = Initial;
    deleted = 0;
    mutated = 0;
}


fsobj::~fsobj() {
    assert(Pass == 2 || !initial);

    assert(parents.count() == 0);
    switch(type) {
	case DFS_IFREG:
	    {
	    if (u.file.path != 0)
		delete u.file.path;
	    }
	    break;

	case DFS_IFDIR:
	    {
	    assert(u.dir.children->count() == 0);
	    delete u.dir.children;

	    /* Shouldn't we do something with MountPoints and Roots here? */
	    }
	    break;

	case DFS_IFLNK:
	    {
	    if (u.symlink.contents != 0)
		delete u.symlink.contents;
	    }
	    break;

	default:
	    assert(0);
    }
}


fsobj *fsobj::GetChild(char *comp, generic_fid *cfid, int ctype) {
    assert(type == DFS_IFDIR);

    int AnyObjectDeleted = 0;
    fsobj *cf = FindFsobj(cfid, &AnyObjectDeleted);
    if (cf == 0) {
	if (AnyObjectDeleted) {
	    /* Object with this fid once existed --> infer re-creation. */
	    cf = InferCreate(this, comp, cfid, ctype);
	}
	else {
	    /* Initial depends whether parent is initial and whether comp has ever existed in parent! */
	    int AnyDirentryDeleted = 0;
	    direntry *cde = FindChildEntry(comp, &AnyDirentryDeleted);
	    if (!initial || cde != 0 || AnyDirentryDeleted) {
		cf = InferCreate(this, comp, cfid, ctype);
	    }
	    else {
		cf = MakeFsobj(cfid, ctype, 1);
		cde = MakeChildEntry(cf, comp, 1);
	    }
	}
    }
    else {
	if (cf->type == ctype) {
	    int AnyDirentryDeleted = 0;
	    direntry *cde = FindChildEntry(comp, &AnyDirentryDeleted);
	    if (cde != 0 && FID_EQ(*cfid, cde->fid)) {
		/* This would appear to be the object we want (subsequent operations may show otherwise). */
		;
	    }
	    else {
		if (cde != 0) {
		    fsobj *tf = FindFsobj(&cde->fid);
		    assert(tf != 0);
		    InferDelete(tf);
		}

		/* Special case "delayed deduction" of initial link. */
		int SpecialCased = 0;
		if (Pass == 1 && initial && cde == 0 && !AnyDirentryDeleted &&
		    cf->initial) {
		    fsobj *pf = cf->GetParent();	/* could be NULL! */

		    if (pf == root) {
			/* Remove "fake" entry from root, and make initial entry in this object. */
			char fakename[MAXNAMLEN];
			MakeFakeName(fakename, cfid);
			cde = root->FindChildEntry(fakename);
			assert(cde != 0);
			{
			    /* Flip initial bits to ensure that destroyed links are really purged */
			    /* and not just marked deleted! */
			    cde->initial = 0;
			    direntry *pde = cf->FindParentEntry(fakename, &root->fid);
			    assert(pde != 0);
			    pde->initial = 0;
			}
			root->DestroyChildEntry(cde);
			MakeChildEntry(cf, comp, 1);
			SpecialCased = 1;

			char path2[MAXPATHLEN];
			GetPath(path2, &fid, comp);
			dprint(0, 0, 0, "SpecialRename: %s %s", fakename, path2);
		    }
		    else if (cf->type == DFS_IFREG &&
			     (CrossParent || pf == this || pf == NULL)) {
			char path1[MAXPATHLEN];
			GetPath(path1, &cf->fid, 0, 0);

			/* Make initial link in this object. */
			MakeChildEntry(cf, comp, 1);
			SpecialCased = 1;

			char path2[MAXPATHLEN];
			GetPath(path2, &fid, comp);
			dprint(0, 0, 0, "SpecialLink: %s %s", path1, path2);
		    }
		}

		/*
		 *   We know that one of the following must be true:
		 *      1. There was an extra-trace remove and re-create of this child
		 *      2. There was an extra-trace rename of this child
		 *      3. There are multiple links to this child (only legal for files).
		 *	   Note that, as a result of ignoring cross parent links,
		 *	   the fso's parent-child linkage may have been removed by
		 *	   by an unlink if the record link count was > 1.
		 *	   In this case, destroy the orphaned fso and recreate it.
		 *
		 *   Our strategy for inferring the unknown operation(s) depends on the
		 *   type of object:
		 *      IFREG:  if existing parent is same as new parent assume (3),
		 *		otherwise assume (1)
		 *      IFDIR:  assume (2)
		 *      IFLNK:  assume (1)
		 */
		if (!SpecialCased)
		    switch(cf->type) {
			case DFS_IFREG:
			    if (CrossParent ||
				(cf->nUndeletedParents() == 1 && cf->GetParent() == this))
				InferLink(cf, this, comp);
			    else {
				InferDelete(cf);
				cf = InferCreate(this, comp, cfid, ctype);
			    }
			    break;

			case DFS_IFDIR:
			    InferRename(cf, this, comp);
			    break;

			case DFS_IFLNK:
			    InferDelete(cf);
			    cf = InferCreate(this, comp, cfid, ctype);
			    break;

			default:
			    assert(0);
		    }
	    }
	}
	else {
	    /* Type discrepency --> child was removed and re-created "extra-trace." */
	    InferDelete(cf);
	    cf = InferCreate(this, comp, cfid, ctype);
	}
    }

    return(cf);
}


fsobj *fsobj::CreateChild(char *comp, generic_fid *cfid, int ctype) {
    assert(type == DFS_IFDIR);

    fsobj *cf = FindFsobj(cfid);
    if (cf != 0)
	InferDelete(cf);

    cf = MakeFsobj(cfid, ctype, 0);
    cf->mutated = 1;
    (void)CreateChildEntry(cf, comp);
    mutated = 1;

    return(cf);
}


/* N.B.  DeleteOthers is relevant only if cf->type == DFS_IFREG. */
/* If DeleteOthers is TRUE and there are other entries besides comp, then deletes are inferred for them. */
void fsobj::DeleteChild(fsobj *cf, char *comp, int DeleteOthers) {
    assert(type == DFS_IFDIR);

    cf->mutated = 1;
    mutated = 1;

    /* Recursively remove directories. */
    if (cf->type == DFS_IFDIR) {
	assert(cf->nUndeletedParents() == 1);
	child_iterator cnext(cf);
	direntry *c = NULL;
	int creadahead = 0;
	while (creadahead || (c = cnext())) {
	    creadahead = 0;

	    if (!c->deleted) {
		direntry *tc = cnext();
		creadahead = (tc != 0);

		fsobj *tf = FindFsobj(&c->fid);
		assert(tf != 0);
		InferDelete(tf);

		c = tc;
	    }
	}
    }

    direntry *cde = FindChildEntry(comp);
    assert(cde != 0);
    DestroyChildEntry(cde);

    if (cf->nUndeletedParents() == 0) {
	/*
	 * untrace believes this is (was) the last parent.
	 * However, the trace may indicate that there were
	 * multiple links to this object, unbeknownst to untrace
	 * thus far (DeleteOthers == 0).
	 */
	if (DeleteOthers)
	    DestroyFsobj(cf);
    }
    else {
	assert(cf->type == DFS_IFREG);

	if (DeleteOthers)
	    InferDelete(cf);
    }
}


direntry *fsobj::CreateChildEntry(fsobj *cf, char *comp) {
    assert(type == DFS_IFDIR);

    mutated = 1;
    cf->mutated = 1;

    /* Infer deletion of existing entry. */
    direntry *cde = FindChildEntry(comp);
    if (cde != 0) {
	fsobj *tf = FindFsobj(&cde->fid);
	assert(tf != 0);
	InferDelete(this, tf, comp);
    }

    cde = MakeChildEntry(cf, comp, 0);

    return(cde);
}


direntry *fsobj::FindChildEntry(char *comp, int *AnyDeleted) {
    assert(type == DFS_IFDIR);

    if (AnyDeleted) *AnyDeleted = 0;

    child_iterator next(this, comp);
    direntry *d;
    direntry *outd = 0;
    while ( (d = next())!=NULL)
	if (STREQ(comp, d->comp)) {
	    if (d->deleted) {
		if (AnyDeleted) *AnyDeleted = 1;
		continue;
	    }

	    assert(outd == 0);
	    outd = d;
	}

    return(outd);
}


direntry *fsobj::MakeChildEntry(fsobj *cf, char *comp, int Initial) {
    assert(type == DFS_IFDIR);

    direntry *cde = FindChildEntry(comp);
    assert(cde == 0);
    cde = new direntry(comp, &cf->fid, Initial);
    u.dir.children->insert(comp, cde);
    cf->MakeParentEntry(this, comp, Initial);

    return(cde);
}


void fsobj::DestroyChildEntry(direntry *d, int ParentEntryDeleted) {
    assert(type == DFS_IFDIR);
    assert(!ISDOT(d->comp));

    if (ParentEntryDeleted) {
	if (Pass == 1 && d->initial) {
	    d->deleted = 1;
	}
	else {
	    assert(u.dir.children->remove(d->comp, d) == d);
	    delete d;
	}
    }
    else {
	fsobj *cf = FindFsobj(&d->fid);
	assert(cf != 0);
	direntry *pde = cf->FindParentEntry(d->comp, &fid);
	assert(pde != 0);
	cf->DestroyParentEntry(pde);
    }
}


direntry *fsobj::FindParentEntry(char *comp, generic_fid *fid, int *AnyDeleted) {
    if (AnyDeleted) *AnyDeleted = 0;

    parent_iterator next(this);
    direntry *d;
    direntry *outd = 0;
    while ( (d = next())!= NULL)
	if (STREQ(comp, d->comp) && FID_EQ(*fid, d->fid)) {
	    if (d->deleted) {
		if (AnyDeleted) *AnyDeleted = 1;
		continue;
	    }

	    assert(outd == 0);
	    outd = d;
	}

    return(outd);
}


direntry *fsobj::MakeParentEntry(fsobj *pf, char *comp, int Initial) {
    direntry *pde = FindParentEntry(comp, &pf->fid);
    assert(pde == 0);
    pde = new direntry(comp, &pf->fid, Initial);
    parents.insert(pde);

    return(pde);
}


void fsobj::DestroyParentEntry(direntry *d) {
    fsobj *pf = FindFsobj(&d->fid);
    assert(pf != 0);
    direntry *cde = pf->FindChildEntry(d->comp);
    assert(cde != 0);
    pf->DestroyChildEntry(cde, 1);

    if (Pass == 1 && d->initial) {
	d->deleted = 1;
    }
    else {
	assert(parents.remove(d) == d);
	delete d;
    }
}


int fsobj::nUndeletedParents() {
    int count = 0;
    parent_iterator pnext(this);
    direntry *pde;
    while ((pde = pnext())!=NULL)
	if (!pde->deleted)
	    count++;

    if (count > 1)
	assert(type == DFS_IFREG);
    return(count);
}


int fsobj::nDeletedParents() {
    int count = 0;
    parent_iterator pnext(this);
    direntry *pde;
    while ((pde = pnext())!=NULL)
	if (pde->deleted)
	    count++;

    return(count);
}


/* N.B.  Returns arbitrarily selected parent when there are multiple candidates (i.e., for file with > 1 links). */
fsobj *fsobj::GetParent() {
    assert(!deleted);

    if (type == DFS_IFDIR && u.dir.mstat == Root) {
	return(u.dir.m.mtpt->GetParent());
    }

    /* Select first non-deleted path. */
    parent_iterator pnext(this);
    direntry *pde;
    while ((pde = pnext())!=NULL)
	if (!pde->deleted) {
	    fsobj *pf = FindFsobj(&pde->fid);
	    assert(pf != 0);
	    return(pf);
	}

    /* Not Found. */
    return(0);
}


/*
 * N.B.  Returns arbitrarily selected path when there are multiple candidates
 * (i.e., for file with > 1 links).
 * If ignoreDeleted == 0, can look at deleted entries.
 */
void fsobj::getpath(char *buf, int ignoreDeleted) {
    if (ignoreDeleted)
	assert(!deleted);

    if (type == DFS_IFDIR && u.dir.mstat == Root) {
	u.dir.m.mtpt->getpath(buf, ignoreDeleted);
	return;
    }

    /* Select first non-deleted path. */
    parent_iterator pnext(this);
    direntry *pde;
    while ((pde = pnext())!= NULL)
	if (pde->deleted && ignoreDeleted)
	    continue;
        else {
	    fsobj *pf = FindFsobj(&pde->fid);
	    assert(pf != 0);
	    pf->getpath(buf, ignoreDeleted);
	    strcat(buf, "/");
	    strcat(buf, pde->comp);
	    return;
	}

    /* Not Found. */
    MakeFakeName(buf, &fid);
}


/* Valid only when deletions have been purged. */
void fsobj::propagate() {
    assert(!deleted);

    if (!mutated) return;

    if (type == DFS_IFDIR && u.dir.mstat == Root) {
	u.dir.m.mtpt->mutated = 1;
	u.dir.m.mtpt->propagate();
	return;
    }

    /* Propagate flag up ALL paths. */
    assert(nUndeletedParents() <= 1 || type == DFS_IFREG);
    parent_iterator pnext(this);
    direntry *pde;
    while ((pde = pnext())!= NULL) {
	assert(!pde->deleted);

	fsobj *pf = FindFsobj(&pde->fid);
	assert(pf != 0);
	pf->mutated = 1;
	pf->propagate();
    }
}


/* Valid only when deletions have been purged. */
void fsobj::skeletize(char *path) {
    assert(!deleted);

    switch(type) {
	case DFS_IFREG:
	    {
	    /* First skeletize creates the file, subsequent skeletizes form hard links. */
	    if (u.file.path == 0) {
		fprintf(skeletonfp, "open %s %d -1\n",
			path, (O_RDWR | O_CREAT | O_EXCL));
		u.file.path = new char[strlen(path) + 1];
		strcpy(u.file.path, path);
	    }
	    else {
		fprintf(skeletonfp, "link %s %s\n", u.file.path, path);
	    }
	    }
	    break;

	case DFS_IFDIR:
	    {
	    /* Indirect to FS root if this is a mount point! */
	    if (u.dir.mstat == MountPoint) {
		u.dir.m.root->skeletize(path);
		break;
	    }

	    fprintf(skeletonfp, "mkdir %s\n", path);

	    /* Do children in a depth-first manner. */
	    child_iterator next(this);
	    direntry *d;
	    while ((d = next())!=NULL) {
		assert(!d->deleted);

		fsobj *cf = FindFsobj(&d->fid);
		assert(cf != 0);
		char cpath[MAXPATHLEN];
		strcpy(cpath, path);
		strcat(cpath, "/");
		strcat(cpath, d->comp);
		cf->skeletize(cpath);
	    }
	    }
	    break;

	case DFS_IFLNK:
	    {
	   const char *contents = u.symlink.contents
	      ? u.symlink.contents
	      : "unknown_contents";
	    fprintf(skeletonfp, "symlink %s %s\n", contents, path);
	    }
	    break;

	default:
	    assert(0);
    }
}


void fsobj::dump(int level) {
    switch(type) {
	case DFS_IFREG:
	    fprintf(stderr, "file %c%c%c\n",
		    initial ? 'i' : '-',
		    deleted ? 'd' : '-',
		    mutated ? 'm' : '-');
	    break;

	case DFS_IFDIR:
	    {
	    /* Indirect to FS root if this is a mount point! */
	    if (u.dir.mstat == MountPoint) {
		u.dir.m.root->dump(level);
		break;
	    }

	    fprintf(stderr, "dir %c%c%c\n",
		    initial ? 'i' : '-',
		    deleted ? 'd' : '-',
		    mutated ? 'm' : '-');

	    child_iterator next(this);
	    direntry *d;
	    while ((d = (direntry *)next())!= NULL)
		d->dump(level + 1);
	    }
	    break;

	case DFS_IFLNK:
	    fprintf(stderr, "symlink %c%c%c --> %s\n",
		    initial ? 'i' : '-',
		    deleted ? 'd' : '-',
		    mutated ? 'm' : '-',
		    u.symlink.contents ? u.symlink.contents : "???");
	    break;

	default:
	    assert(0);
    }
}


direntry::direntry(char *Comp, generic_fid *Fid, int Initial) {
    comp = new char[strlen(Comp) + 1];
    strcpy(comp, Comp);
    fid = *Fid;
    initial = Initial;
    deleted = 0;
}


direntry::~direntry() {
    assert(Pass == 2 || !initial);

    delete comp;
}


void direntry::dump(int level) {
    for (int i = 0; i < level; i++)
	fprintf(stderr, "   ");

    char buf[MAXNAMLEN];
    MakeFakeName(buf, &fid);
    fprintf(stderr, "%c%c %s %s : ",
	     initial ? 'i' : '-',
	     deleted ? 'd' : '-',
	     comp, buf);

    fsobj *f = FindFsobj(&fid);
    if (f == 0)
	fprintf(stderr, "--- object not found ---\n");
    else if (!ISDOTDOT(comp) && !ISDOT(comp))
	f->dump(level);
    else
	fprintf(stderr, "\n");
}


void GetPath(char *path, generic_fid *fid, char *trailer, int ignoreDeleted) {
    fsobj *f = FindFsobj(fid);
    assert(fid != 0);
    f->getpath(path, ignoreDeleted);

    if (trailer != 0) {
	strcat(path, "/");
	strcat(path, trailer);
    }
}


char *LastComp(char *path) {
    char *p = rindex(path, '/');
    return(p ? (p + 1) : path);
}


void MakeFakeName(char *buf, generic_fid *fid) {
    if (FID_EQ(RootFid, *fid)) {
	strcpy(buf, "root");
	return;
    }

    switch(fid->tag) {
	case DFS_ITYPE_UFS:
	    sprintf(buf, "ufs.%lx.%lx",
		    fid->value.local.device, fid->value.local.number);
	    break;

	case DFS_ITYPE_AFS:
	    sprintf(buf, "afs.%lx.%lx.%lx.%lx",
		    fid->value.afs.Cell, fid->value.afs.Fid.Volume,
		    fid->value.afs.Fid.Vnode, fid->value.afs.Fid.Unique);
	    break;

	case DFS_ITYPE_CFS:
	    sprintf(buf, "cfs.%lx.%lx.%lx",
		    fid->value.cfs.Volume, fid->value.cfs.Vnode,
		    fid->value.cfs.Unique);
	    break;

	default:
	    assert(0);
    }
}


int FidHash(generic_fid *fid) {
    return(((unsigned long *)&(fid->value))[0] +
	    ((unsigned long *)&(fid->value))[1] +
	    ((unsigned long *)&(fid->value))[2] +
	    ((unsigned long *)&(fid->value))[3]);
}


int DirHash(char *comp) {
    int hval = 0;
    char tc;
    while ((tc = (*comp++))!=  0){
	hval *= 173;
	hval += tc;
    }
    return(hval);
}


void DumpTree() {
    fprintf(stderr, "root : ");
    root->dump(0);
    fprintf(stderr, "\n");
}


void DumpRoots() {
    fsobj_iterator next;
    fsobj *f;
    while ((f = next())!= NULL)
	if (f->nParents() == 0) {
	    assert(f->type == DFS_IFDIR);
	    assert(f == root ||
		   (f->u.dir.mstat == Root && f->u.dir.m.mtpt != 0));

	    char fakename[MAXNAMLEN];
	    MakeFakeName(fakename, &f->fid);
	    fprintf(stderr, "%s", fakename);
	    if (f != root) {
		char mtptname[MAXNAMLEN];
		MakeFakeName(mtptname, &f->u.dir.m.mtpt->fid);
		fprintf(stderr, "  (mtpt = %s)", mtptname);
	    }
	    fprintf(stderr, "\n");

	    child_iterator cnext(f);
	    direntry *d;
	    while ( (d = cnext())!=NULL) {
		char childname[MAXNAMLEN];
		MakeFakeName(childname, &d->fid);
		fprintf(stderr, "%c%c %s %s\n",
			d->initial ? 'i' : '-',
			d->deleted ? 'd' : '-',
			d->comp, childname);
	    }

	    fprintf(stderr, "\n");
	}
}


void DumpStats() {
    fprintf(stderr, "Stats:\t\t\t\tPass 1\tPass 2\n");
    fprintf(stderr, "\tRecords Handled:\t%d\t%d\n",
	     Stats[0].RecordsHandled, Stats[1].RecordsHandled);
    fprintf(stderr, "\tCreates Inferred:\t%d\t%d\n",
	     Stats[0].CreatesInferred, Stats[1].CreatesInferred);
    fprintf(stderr, "\tDeletes Inferred:\t%d\t%d\n",
	     Stats[0].DeletesInferred, Stats[1].DeletesInferred);
    fprintf(stderr, "\tLinks Inferred:\t\t%d\t%d\n",
	     Stats[0].LinksInferred, Stats[1].LinksInferred);
    fprintf(stderr, "\tRenames Inferred:\t%d\t%d\n",
	     Stats[0].RenamesInferred, Stats[1].RenamesInferred);
}


void dprint(int dummy1, int dummy2, int dummy3, char *fmt ...) {
    if (!Verbose) return;

    va_list ap;

    char msg[240];
    char *cp = msg;

    /* Copy the message. */
    va_start(ap, fmt);
    vsprintf(cp, fmt, ap);
    va_end(ap);
    cp += strlen(cp);

    /* Append timestamp. */
    if (recPtr != 0) {
	struct tm *lt = localtime(&recPtr->time.tv_sec);
	sprintf(cp, " ( %02d:%02d:%02d )",
		lt->tm_hour, lt->tm_min, lt->tm_sec);
	cp += strlen(cp);
    }
    sprintf(cp, "\n");

    /* Write it to stderr. */
    fprintf(stderr, msg);
    fflush(stderr);
}


void Die(int dummy1, int dummy2, int dummy3, char *fmt ...) {
    static int dying = 0;

    if (!dying) {
	/* Avoid recursive death. */
	dying = 1;

	/* dprint the message, with an indication that it is fatal. */
	Verbose = 1;
	va_list ap;
	char msg[240];
	strcpy(msg, "fatal error -- ");
	va_start(ap, fmt);
	vsprintf(msg + strlen(msg), fmt, ap);
	va_end(ap);
	dprint(dummy1, dummy2, dummy3, msg);

	/* Dump system state to the log. */
	if (recPtr != 0)
	    Trace_DumpRecord(recPtr);
	DumpTree();
	DumpStats();
    }

    fflush(stderr);

exit(-1);
    /* Leave a core file. */
    kill(getpid(), SIGFPE);

    /* NOTREACHED */
}
