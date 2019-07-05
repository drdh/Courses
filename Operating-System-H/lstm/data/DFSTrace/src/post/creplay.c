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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/creplay.c,v 1.3 1998/10/13 21:14:31 tmkr Exp $";

*/
#endif _BLURB_


/*
 *
 *    Command Replay: replay skeleton or trace command file
 *
 *
 *    ToDo:
 *
 */


#ifdef __cplusplus
extern "C" {
#endif __cplusplus

#include <stdio.h>
#include <stdarg.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <utime.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifdef CMUCS
#include <libc.h>
#else
#include <stdlib.h>
#include <unistd.h>
#endif

#ifdef __cplusplus
}
#endif __cplusplus


const int MAXCMDLEN = 10 * MAXPATHLEN;	    /* ought to be sufficient */
const int MAXTYPELEN = 64;
const int nFindices = 65536;		    /* 2^16 */


char *CmdFile;
int Verbose = 0;
int Nogroup = 0;	/* is chowning of group id supported? */

char cmd[MAXCMDLEN];

int openfds;
int FindexMap[nFindices];


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

#define	STREQ(a, b) (strcmp((a), (b)) == 0)

void Open();
void Close();
void Chmod();
void Chown();
void Truncate();
void Utimes();
void Unlink();
void Link();
void Rename();
void Mkdir();
void Rmdir();
void Symlink();
void Stat();
void Lstat();
void Access();
void Readlink();
void Sleep();

void ParseArgs(int, char **);
void eprint(int, int, int, char * ...);
void Die(int, int, int, char * ...);


main(int argc, char *argv[]) {
        ParseArgs(argc, argv);

	FILE *cmdfp = fopen(CmdFile, "r");
	if (cmdfp == NULL) {
		fprintf(stderr, "cannot open %s\n", CmdFile);
		exit(-1);
	}

	for (int findex = 0; findex < nFindices; findex++)
		FindexMap[findex] = -1;

	while (fgets(cmd, MAXCMDLEN - 1, cmdfp) != NULL) {
		char type[MAXTYPELEN];
		if (sscanf(cmd, "%s ", type) != 1) {
			fprintf(stderr, "parse error: \'%s\'\n", cmd);
			exit(-1);
		}

		if (STREQ(type, "open"))
			Open();
		else if (STREQ(type, "close"))
			Close();
		else if (STREQ(type, "chmod"))
			Chmod();
		else if (STREQ(type, "chown"))
			Chown();
		else if (STREQ(type, "truncate"))
			Truncate();
		else if (STREQ(type, "utimes"))
			Utimes();
		else if (STREQ(type, "unlink"))
			Unlink();
		else if (STREQ(type, "link"))
			Link();
		else if (STREQ(type, "rename"))
			Rename();
		else if (STREQ(type, "mkdir"))
			Mkdir();
		else if (STREQ(type, "rmdir"))
			Rmdir();
		else if (STREQ(type, "symlink"))
			Symlink();
		else if (STREQ(type, "stat"))
			Stat();
		else if (STREQ(type, "lstat"))
			Lstat();
		else if (STREQ(type, "access"))
			Access();
		else if (STREQ(type, "readlink"))
			Readlink();
		else if (STREQ(type, "sleep"))
		        Sleep();
		else {
			fprintf(stderr, "unrecognized command: \'%s\'\n", type);
			exit(-1);
		}
	}

	if (openfds > 0) {
		fprintf(stderr, "openfds = %d\n", openfds);
		for (int findex = 0; findex < nFindices; findex++)
			if (FindexMap[findex] != -1)
				fprintf(stderr, "\t%d: %d\n", findex, FindexMap[findex]);
	}
}


void Open() {
	char path[MAXPATHLEN];
	int flags;
	int findex;
	assert(sscanf(cmd, "open %s %d %d", path, &flags, &findex) == 3);

	int fd = open(path, flags, 0777);
	assert(fd >= 0);
	if (findex == -1) {
		assert(ftruncate(fd, 0) == 0);
		assert(close(fd) == 0);
	}
	else {
		assert(FindexMap[findex] == -1);
		FindexMap[findex] = fd;
		openfds++;
	}
}


void Close() {
	int findex;
	int newsize;
	assert(sscanf(cmd, "close %d %d", &findex, &newsize) == 2);

	int fd = FindexMap[findex];
	assert(fd >= 0);
	assert(openfds > 0);
	FindexMap[findex] = -1;
	openfds--;

	if (newsize != -1) {
		struct stat tstat;
		assert(fstat(fd, &tstat) == 0);
		if (newsize > tstat.st_size) {
			assert(lseek(fd, tstat.st_size, SEEK_SET) == tstat.st_size);

			const int BLKSIZE = 4096;
			char buf[BLKSIZE];

			int nbytes = (int) (newsize - tstat.st_size);
			int nblocks = nbytes / BLKSIZE;
			for (int i = 0; i < nblocks; i++)
				assert(write(fd, buf, BLKSIZE) == BLKSIZE);

			int remainder = nbytes % BLKSIZE;
			if (remainder != 0)
				assert(write(fd, buf, remainder) == remainder);
		}
		else if (newsize < tstat.st_size) {
			assert(ftruncate(fd, newsize) == 0);
		}
	}
	assert(close(fd) == 0);
}


void Chmod() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "chmod %s", path) == 1);

	assert(chmod(path, 0777) == 0);
}


void Chown() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "chown %s", path) == 1);
	struct stat tstat;
	assert(stat(path, &tstat) == 0);
	uid_t owner = tstat.st_uid;
	gid_t group = tstat.st_gid;
	if (Nogroup) group =( unsigned int) -1;
	assert(chown(path, owner, group) == 0);
}


void Truncate() {
	char path[MAXPATHLEN];
	int newsize;
	assert(sscanf(cmd, "truncate %s %d", path, &newsize) == 2);

	assert(truncate(path, newsize) == 0);
}


void Utimes() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "utimes %s", path) == 1);
	struct stat tstat;
	assert(stat(path, &tstat) == 0);
	struct timeval tv[2];
	tv[0].tv_sec = tstat.st_atime;
	tv[0].tv_usec = 0;
	tv[1].tv_sec = tstat.st_mtime;
	tv[1].tv_usec = 0;

	assert(utimes(path, tv) == 0);
}


void Unlink() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "unlink %s", path) == 1);

	assert(unlink(path) == 0);
}


void Link() {
	char path1[MAXPATHLEN];
	char path2[MAXPATHLEN];
	assert(sscanf(cmd, "link %s %s", path1, path2) == 2);

	assert(link(path1, path2) == 0);
}


void Rename() {
	char path1[MAXPATHLEN];
	char path2[MAXPATHLEN];
	assert(sscanf(cmd, "rename %s %s", path1, path2) == 2);

	assert(rename(path1, path2) == 0);
}


void Mkdir() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "mkdir %s", path) == 1);

	assert(mkdir(path, 0777) == 0);
}


void Rmdir() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "rmdir %s", path) == 1);

	assert(rmdir(path) == 0);
}


void Symlink() {
	char path1[MAXPATHLEN];
	char path2[MAXPATHLEN];
	assert(sscanf(cmd, "symlink %s %s", path1, path2) == 2);

	assert(symlink(path1, path2) == 0);
}


void Stat() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "stat %s", path) == 1);

	struct stat tstat;
	assert(stat(path, &tstat) == 0);
}


void Lstat() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "lstat %s", path) == 1);

	struct stat tstat;
	assert(lstat(path, &tstat) == 0);
}


void Access() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "access %s", path) == 1);

	assert(access(path, F_OK) == 0);
}


void Readlink() {
	char path[MAXPATHLEN];
	assert(sscanf(cmd, "readlink %s", path) == 1);

	char contents[MAXPATHLEN];
	assert(readlink(path, contents, MAXPATHLEN) >= 0);
}


void Sleep() {
  unsigned seconds, useconds;

  assert(sscanf(cmd, "sleep %u.%u", &seconds, &useconds) == 2);

  useconds += 1000000*seconds;

  usleep(useconds);
}

void ParseArgs(int argc, char **argv) {
	int i;
	extern int optind;

	if (argc < 2 || argc > 3) goto usage;

	while ((i = getopt(argc, argv, "gv")) != EOF)
		switch (i) {
		case 'g':
			Nogroup = 1;
			break;
		case 'v':
			Verbose = 1;
			break;
		default:
			goto usage;
		}

	CmdFile = argv[optind];
	return;

 usage:
	fprintf(stderr, "usage: creplay [-g] [-v] cmdfile\n");
	exit(-1);
}


void eprint(int dummy1, int dummy2, int dummy3, char *fmt ...) {
	if (!Verbose) return;

	va_list ap;

	char msg[240];
	char *cp = msg;

	/* Copy the message. */
	va_start(ap, fmt);
	vsprintf(cp, fmt, ap);
	va_end(ap);
	cp += strlen(cp);

	/* Write it to stderr. */
	fprintf(stderr, msg);
	fflush(stderr);
}


void Die(int dummy1, int dummy2, int dummy3, char *fmt ...) {
	static int dying = 0;

	if (!dying) {
		/* Avoid recursive death. */
		dying = 1;

		/* eprint the message, with an indication that it is fatal. */
		Verbose = 1;
		va_list ap;
		char msg[240];
		strcpy(msg, "fatal error -- ");
		va_start(ap, fmt);
		vsprintf(msg + strlen(msg), fmt, ap);
		va_end(ap);
		eprint(dummy1, dummy2, dummy3, msg);

		/* Dump out the current command. */
		eprint(0, 0, 0, "cmd = %s\n", cmd);
	}

	fflush(stderr);

	exit(-1);
	/* Leave a core file. */
	kill(getpid(), SIGFPE);

	/* NOTREACHED */
}
