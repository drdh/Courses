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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/tm.c,v 1.4 1998/10/13 21:14:32 tmkr Exp $";
*/
#endif _BLURB_




/*
 * tm (tracemaker) -- scans traces, splitting and gluing where appropriate.
 */


#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>

#if __OSF1__ ||  __SOLARIS__
#include <arpa/inet.h>
#include <strings.h>
#endif


#if __LINUX__ || __NetBSD__
#include <netinet/in.h>
#include <string.h>
#endif

#include "tracelib.h"
#include "unpack_private.h"
#include "v1.h"
#include "v1_private.h"
#include "v2.h"
#include "v2_private.h"
#include "v3.h"
#include "v3_private.h"

/* chunk data block */
typedef struct cdb {
    u_short traceLevel;
    u_short serverTroubles;
    u_int   chunkSeq;
    u_int   bytesLost;
} cdb_t;

/* file data block */
typedef struct fdb {
    u_short type;
    u_short kernelVersion; /* v2 & v3 traces */
    u_short agentVersion;
    u_short collectorVersion;
    int     realChunkSize;
    int     filePreambleSize;
    int     chunkPreambleSize;
    int     desc;
    time_t  versionTime;  /* v1 traces */
    u_int   hostAddress;
    time_t  bootTime;  /* seconds only */
    time_t  agentBirthTime;
    struct timeval firstRecordTime;
    char    name[MAXPATHLEN];
} fdb_t;

char verb = 0;
char removeit = 0;
fdb_t toFDB, fromFDB;
cdb_t startCDB, endCDB;

int NewTrace();
int Decode();
int PreScan();
int PutFilePreamble();
void PrintUsage();
void UnpackChunkPreamble();

void main(argc, argv)
     int argc;
     char *argv[];
{
    int i, j, rc;
    extern int optind;
    extern char *optarg;
    char newOutputFile = 0;  /* need to write preamble for new files */
    u_int toEndChunkSeq = 0;
    char *buf;

    if (argc < 2)
	PrintUsage();

    /* Obtain invocation options */
    while ((i = getopt(argc, argv, "vr")) != EOF)
	switch (i) {
	case 'r':
	    removeit = 1;
	    break;
	case 'v':
	    verb = 1;
	    break;
	default:
	    PrintUsage();
	    break;
	}

    (void) strcpy(toFDB.name, argv[optind++]);
    if ((toFDB.desc = open(toFDB.name, (O_APPEND | O_RDWR), 0644)) < 0) {
	if (verb)
	    printf("Opening new output file %s\n", toFDB.name);
	if ((toFDB.desc = open(toFDB.name, (O_CREAT | O_WRONLY), 0644)) < 0) {
	    printf("tm: can't open file %s\n", toFDB.name);
	    exit(1);
	}
	newOutputFile = 1;
    } else {
	/* Decode and read preamble of existing trace. */
	if (Decode(&toFDB, &startCDB) < 0) {
	    printf("tm: can't decode %s\n", toFDB.name);
	    exit(1);
	}
	if (verb) {
	    printf("Appending to trace from %s, written %s",
		   Trace_NodeIdToStr(toFDB.hostAddress),
		   ctime((time_t *)&toFDB.firstRecordTime));
	    printf("Host booted at %s", ctime(&toFDB.bootTime));
	}

	/* prescan destination file */
	if (PreScan(&toFDB, &endCDB)) {
	    close(toFDB.desc);
	    exit(1);
	}
	if (endCDB.bytesLost) {
	    NewTrace(&toFDB);
	    close(toFDB.desc);
	    exit(1);
	}
	toEndChunkSeq = endCDB.chunkSeq;
    }

    for (; optind < argc; optind++) { /* loop through from files */
	(void) strcpy(fromFDB.name, argv[optind]);
	if (strcmp(toFDB.name, fromFDB.name) == 0) {
	    printf("Not appending file %s to itself.\n", fromFDB.name);
	    continue;
	}
	if ((fromFDB.desc = open(fromFDB.name, O_RDONLY)) < 0) {
	    printf("tm: can't open file %s.\n", fromFDB.name);
	    continue;
	}
	if (Decode(&fromFDB, &startCDB) < 0) {
	    printf("tm: can't decode %s\n", fromFDB.name);
	    exit(1);
	}

	if (newOutputFile) {
	    /* copy most fields, but not all, from from-file */
	    char oname[MAXPATHLEN];
	    int odesc = toFDB.desc;

	    (void) strcpy(oname, toFDB.name);
	    (void) bcopy((char *)&fromFDB, (char *)&toFDB, sizeof(fdb_t));
	    toFDB.desc = odesc;
	    (void) strcpy(toFDB.name, oname);
	    PutFilePreamble(&toFDB);
	}

	/* now check everything */
	if (fromFDB.type != toFDB.type) {
	    printf("Files %s and %s are of different versions. Not appending.\n",
		   toFDB.name, fromFDB.name);
	    close(fromFDB.desc);
	    continue;
	}
	assert(fromFDB.filePreambleSize == toFDB.filePreambleSize);
	assert(fromFDB.realChunkSize == toFDB.realChunkSize);

	if (fromFDB.hostAddress != toFDB.hostAddress) {
	    printf("Files %s and %s are not from the same host. Not appending.\n",
		   toFDB.name, fromFDB.name);
	    close(fromFDB.desc);
	    continue;
	}
	if (fromFDB.bootTime != toFDB.bootTime) {
	    printf("Boot times are not equal, files %s and %s. Not appending.\n",
		   toFDB.name, fromFDB.name);
	    close(fromFDB.desc);
	    continue;
	}
	if (fromFDB.agentBirthTime != toFDB.agentBirthTime) {
	    printf("Agent birth times not equal, files %s, %s. Not appending.\n",
		   toFDB.name, fromFDB.name);
	    close(fromFDB.desc);
	    continue;
	}
	if (timercmp(&fromFDB.firstRecordTime, &toFDB.firstRecordTime, <)) {
	    printf("Trace in file %s created before file %s. Not appending.\n",
		   fromFDB.name, toFDB.name);
	    close(fromFDB.desc);
	    continue;
	}
	if ((startCDB.bytesLost) && (!newOutputFile)) {
	    printf("Data lost at beginning of file %s. Not appending.\n",fromFDB.name);
	    close(fromFDB.desc);
	    continue;
	}
	if ((!newOutputFile) &&
	    (toEndChunkSeq+1 != startCDB.chunkSeq)) {
	    printf("Chunk numbers in files %s(%d), %s(%d) not continuous. Not appending.\n",
		   toFDB.name, toEndChunkSeq,
		   fromFDB.name, startCDB.chunkSeq);
	    close(fromFDB.desc);
	    continue;
	}
	if (PreScan(&fromFDB, &endCDB)) {
	    close(fromFDB.desc);
	    exit(1);
	}

	/* passed the tests. Now append to toFile. */
	/* check if we need to split the trace. */
	buf = (char *) malloc(toFDB.realChunkSize);

	if (endCDB.bytesLost == 0) { /* normal case */
	    j = 0;
	    while ((rc = read(fromFDB.desc, buf, fromFDB.realChunkSize))
		   == fromFDB.realChunkSize) {
		if ((rc = write(toFDB.desc, buf, toFDB.realChunkSize))
		    != toFDB.realChunkSize)
		    printf("Error %d while writing %s\n",
			   rc, toFDB.name);
		else
		    j++;
	    }
	} else {  /* lost data */
	    /* finish writing this trace */
	    /* first repos to first chunk, since we'll re-read */
	    if ((rc = lseek(fromFDB.desc, fromFDB.filePreambleSize, SEEK_SET))
		!= fromFDB.filePreambleSize) {
		printf("Couldn't reposition file.\n");
		exit(1);
	    }

	    j = startCDB.chunkSeq;
	    while (j < endCDB.chunkSeq) {
		if (((rc = read(fromFDB.desc, buf, fromFDB.realChunkSize))
		     != fromFDB.realChunkSize) ||
		    ((rc = write(toFDB.desc, buf, toFDB.realChunkSize))
		     != toFDB.realChunkSize))
		    printf("Error %d while moving data.\n", rc);
		j++;
	    }
	    if (verb) {
		printf("Added %d bytes from trace %s.\n",
		       (j - startCDB.chunkSeq)*fromFDB.realChunkSize,
		       fromFDB.name);
	    }
	    close(toFDB.desc);   /* added desc after error ---- tmk Oct 98*/
	    NewTrace(&fromFDB);
	    close(fromFDB.desc);
	    if (removeit)
		unlink(fromFDB.name);
	    exit(1);
	}
	toEndChunkSeq = endCDB.chunkSeq; /* update last chunk number */

	if (verb) {
	    printf("Added %d bytes from trace %s.\n",
		   j*fromFDB.realChunkSize, fromFDB.name);
	}

	newOutputFile = 0;
	close(fromFDB.desc);
	if (removeit)
	    (void) unlink(fromFDB.name);

    }
    close(toFDB.desc);
}

/*
 * PreScan -- prescan the file. Returns 0 is successful, 1 if not. Caller is
 * expected to check preamble returned for lost data!  Lost data at the
 * beginning of the trace is a special case, and is easy to check elsewhere.
 * If data is lost in the middle of the trace, this routine returns at that
 * point, leaving the file positioned at the troublesome chunk.
 */
int PreScan(fdbp, cdbp)
     fdb_t *fdbp;
     cdb_t *cdbp;
{
    int rc, size;
    char *buf;
    u_int ocs;
    u_int otl;

    if (verb) {
	printf("Scanning %s...", fdbp->name);
	fflush(stdout);
    }

    /* repos to first chunk */
    if ((rc = lseek(fdbp->desc, fdbp->filePreambleSize, SEEK_SET)
	 != fdbp->filePreambleSize)) {
	printf("Couldn't reposition file.\n");
	return(1);
    }

    buf = (char *) malloc(fdbp->realChunkSize);

    /* get past first chunk -- wouldn't want to return here. */
    if ((rc = read(fdbp->desc, buf, fdbp->realChunkSize)) != fdbp->realChunkSize) {
	printf("Couldn't read first chunk.\n");
	free(buf);
	return(1);
    }
    UnpackChunkPreamble(buf, fdbp, cdbp);

    size = fdbp->realChunkSize;
    while ((rc = read(fdbp->desc, buf, fdbp->realChunkSize)) == fdbp->realChunkSize) {
	otl = cdbp->traceLevel;
	ocs = cdbp->chunkSeq;
	UnpackChunkPreamble(buf, fdbp, cdbp);

	if (ocs != cdbp->chunkSeq-1) {
	    printf("\nFatal error: chunks %d and %d adjacent\n",
		   ocs, cdbp->chunkSeq-1);
	    free(buf);
	    return(1);
	}
	if (cdbp->bytesLost) {
	    printf("\nWarning: %d bytes lost at chunk %d.\n",
		   cdbp->bytesLost, cdbp->chunkSeq);

	    /* back up to beginning of this chunk */
	    (void) lseek(fdbp->desc, -fdbp->realChunkSize, SEEK_CUR);
	    free(buf);
	    return(0);  /* leave positioned at this chunk */
	}
	size += fdbp->realChunkSize;
    }

    /* repos to first chunk */
    if ((rc = lseek(fdbp->desc, fdbp->filePreambleSize, SEEK_SET)
	 != fdbp->filePreambleSize)) {
	printf("Couldn't reposition file.\n");
	free(buf);
	return(1);
    }

    if (verb)
	printf("%d bytes.\n", size);

    free(buf);
    return(0);
}

/*
 * NewTrace -- takes the portion of the trace file from the chunk
 * starting at the current position onward and writes it into a new
 * trace file.
 */
int NewTrace(fdbp)
     fdb_t *fdbp;
{
    int rc, i;
    fdb_t newFDB;
    dfs_header_t header;
    char *buf;
    char timeString[12], ctimeString[26];

    buf = (char *) malloc(fdbp->realChunkSize);
    (void) bcopy((char *)fdbp, (char *)&newFDB, sizeof(fdb_t));
    /*
     * new start time is time from first trace record, then back up
     * to start of chunk. File is positioned at beginning of chunk
     * that caused trouble.
     */
    (void) lseek(fdbp->desc, fdbp->chunkPreambleSize, SEEK_CUR);
    if ((rc = read(fdbp->desc, (char *) &header, sizeof(dfs_header_t))) < 0) {
	printf("Error %d reading header from %s\n", rc, fdbp->name);
	exit(1);
    }
    (void) lseek(fdbp->desc,
		 -(sizeof(dfs_header_t) + fdbp->chunkPreambleSize), SEEK_CUR);

    /* make the new file name */
    header.time.tv_sec = htonl(header.time.tv_sec);
    (void) strcpy(ctimeString, ctime(&header.time.tv_sec));
    for (i = 0; i < 3; i++) timeString[i] = ctimeString[i+4];
    for (i = 3; i < 11; i++) timeString[i] = ctimeString[i+5];
    timeString[11] = '\0';
    if (timeString[3] == ' ') timeString[3] = '0'; /* don't want blank */
    timeString[5] = '.';
    (void) strcpy(newFDB.name+strlen(newFDB.name)-strlen(timeString), timeString);
    if (verb)
	printf("Creating new trace %s\n", newFDB.name);
    if ((newFDB.desc = open(newFDB.name, (O_CREAT | O_WRONLY | O_EXCL), 0644)) < 0) {
	printf("tm: can't open new file %s\n", newFDB.name);
	exit(1);
    }

    /* write new file preamble */
    if ((rc = PutFilePreamble(&newFDB)) < 0) {
	printf("tm: Error writing preamble to %s\n", newFDB.name);
	exit(1);
    }

    /* copy remaining data to new trace file */
    i = 0;
    while ((rc = read(fdbp->desc, buf, fdbp->realChunkSize)) == fdbp->realChunkSize) {
	if ((rc = write(newFDB.desc, buf, fdbp->realChunkSize)) !=
	    fdbp->realChunkSize) {
	    printf("Error %d while writing %s\n", rc, newFDB.name);
	    exit(1);
	}
	i++;
    }
    if (verb)
	printf("Wrote %d bytes to new trace %s\n",
	       i * fdbp->realChunkSize, newFDB.name);

    return (close(newFDB.desc));
}

void PrintUsage()
{
    printf("Usage: tm [-v] [-r] to-file from-file1 [from-file2 ...]\n");
    exit(1);
}

/*
 * Decode -- figures out what sort of a trace we
 * have by reading the first DECODE_AMOUNT bytes of the
 * trace file.  Sets the version if it can figure
 * out what it is, also sets the traceProc structure and
 * returns 0. Otherwise, returns -1.
 */
int Decode(fdbp, cdbp)
     fdb_t *fdbp;
     cdb_t *cdbp;
{
    char buf[DECODE_AMOUNT];
    int rc = 0;
    int offset = 0;

    if (((rc = read(fdbp->desc, buf, DECODE_AMOUNT)) < 0) ||
	((rc = lseek(fdbp->desc, 0, SEEK_SET)) != 0)) /* repos to beginning */
	rc = -1;
    else {
	if (V3FilePreamble(buf)) {
	    fdbp->type = TRACE_VERSION_CMU3;
	    fdbp->realChunkSize = DFS_TRACE_CHUNK_SIZE_V3 +
		sizeof(dfs_chunk_preamble_v3_t);
	    fdbp->filePreambleSize = sizeof(dfs_file_preamble_v3_t);
	    fdbp->chunkPreambleSize = sizeof(dfs_chunk_preamble_v3_t);

	    /* unpack the file preamble */
	    DFS_LOG_UNPACK_FOUR(fdbp->hostAddress, buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->bootTime, buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->agentBirthTime, buf, offset);
	    DFS_LOG_UNPACK_TWO(fdbp->kernelVersion, buf, offset);
	    DFS_LOG_UNPACK_TWO(fdbp->agentVersion, buf, offset);
	    DFS_LOG_UNPACK_TWO(fdbp->collectorVersion, buf, offset);
	    offset += 2;

	    /* now the chunk preamble */
	    DFS_LOG_UNPACK_FOUR(cdbp->chunkSeq, buf, offset);
	    DFS_LOG_UNPACK_TWO(cdbp->traceLevel, buf, offset);
	    DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	    DFS_LOG_UNPACK_FOUR(cdbp->bytesLost, buf, offset);

	    /* now the first record time */
	    offset += 8;
	    DFS_LOG_UNPACK_FOUR(fdbp->firstRecordTime.tv_sec,
				buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->firstRecordTime.tv_usec,
				buf, offset);
	} else if (V2FilePreamble(buf)) {
	    fdbp->type = TRACE_VERSION_CMU2;
	    fdbp->realChunkSize = DFS_TRACE_CHUNK_SIZE_V2 +
		sizeof(dfs_chunk_preamble_v2_t);
	    fdbp->filePreambleSize = sizeof(dfs_file_preamble_v2_t);
	    fdbp->chunkPreambleSize = sizeof(dfs_chunk_preamble_v2_t);

	    /* unpack the file preamble */
	    DFS_LOG_UNPACK_FOUR(fdbp->hostAddress, buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->bootTime, buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->agentBirthTime, buf, offset);
	    DFS_LOG_UNPACK_TWO(fdbp->kernelVersion, buf, offset);
	    DFS_LOG_UNPACK_TWO(fdbp->agentVersion, buf, offset);
	    DFS_LOG_UNPACK_TWO(fdbp->collectorVersion, buf, offset);
	    offset += 2;

	    /* now the chunk preamble */
	    DFS_LOG_UNPACK_FOUR(cdbp->chunkSeq, buf, offset);
	    DFS_LOG_UNPACK_TWO(cdbp->traceLevel, buf, offset);
	    DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	    DFS_LOG_UNPACK_FOUR(cdbp->bytesLost, buf, offset);

	    /* now the first record time */
	    offset += 8;
	    DFS_LOG_UNPACK_FOUR(fdbp->firstRecordTime.tv_sec,
				buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->firstRecordTime.tv_usec,
				buf, offset);
	} else if (V1FilePreamble(buf)) {
	    fdbp->type = TRACE_VERSION_CMU1;
	    fdbp->realChunkSize = DFS_TRACE_CHUNK_SIZE_V1 +
		sizeof(dfs_chunk_preamble_v1_t);
	    fdbp->filePreambleSize = sizeof(dfs_file_preamble_v1_t);
	    fdbp->chunkPreambleSize = sizeof(dfs_chunk_preamble_v1_t);

	    /* unpack the file preamble */
	    DFS_LOG_UNPACK_FOUR(fdbp->hostAddress, buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->bootTime, buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->versionTime, buf, offset);

	    /* now the chunk preamble */
	    DFS_LOG_UNPACK_FOUR(fdbp->agentBirthTime, buf, offset);
	    DFS_LOG_UNPACK_FOUR(cdbp->chunkSeq, buf, offset);
	    DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	    DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	    DFS_LOG_UNPACK_FOUR(cdbp->bytesLost, buf, offset);

	    /* now the first record time */
	    offset += 8;
	    DFS_LOG_UNPACK_FOUR(fdbp->firstRecordTime.tv_sec,
				buf, offset);
	    DFS_LOG_UNPACK_FOUR(fdbp->firstRecordTime.tv_usec,
				buf, offset);
	} else
	    rc = -1;
    }
    return(rc);
}

void UnpackChunkPreamble(buf, fdbp, cdbp)
     char *buf;
     fdb_t *fdbp;
     cdb_t *cdbp;
{
    int offset = 0;

    switch (fdbp->type) {
    case TRACE_VERSION_CMU3:
    case TRACE_VERSION_CMU2:
	DFS_LOG_UNPACK_FOUR(cdbp->chunkSeq, buf, offset);
	DFS_LOG_UNPACK_TWO(cdbp->traceLevel, buf, offset);
	DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	DFS_LOG_UNPACK_FOUR(cdbp->bytesLost, buf, offset);
	break;
    case TRACE_VERSION_CMU1:
	DFS_LOG_UNPACK_FOUR(fdbp->agentBirthTime, buf, offset);
	DFS_LOG_UNPACK_FOUR(cdbp->chunkSeq, buf, offset);
	DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	DFS_LOG_UNPACK_TWO(cdbp->serverTroubles, buf, offset);
	DFS_LOG_UNPACK_FOUR(cdbp->bytesLost, buf, offset);
	break;
    default:
	break;
    }
}

int PutFilePreamble(fdbp)
     fdb_t *fdbp;
{
    int rc = 0;
    fdb_t flipped;

    flipped.kernelVersion = htons(fdbp->kernelVersion);
    flipped.agentVersion = htons(fdbp->agentVersion);
    flipped.collectorVersion = htons(fdbp->collectorVersion);
    flipped.versionTime = htonl(fdbp->versionTime);
    flipped.hostAddress = htonl(fdbp->hostAddress);
    flipped.bootTime = htonl(fdbp->bootTime);
    flipped.agentBirthTime = htonl(fdbp->agentBirthTime);

    switch (fdbp->type) {
    case TRACE_VERSION_CMU3:
    case TRACE_VERSION_CMU2:
	if ((write(fdbp->desc, &flipped.hostAddress, sizeof(u_int)) == -1) ||
	    (write(fdbp->desc, &flipped.bootTime, sizeof(time_t)) == -1) ||
	    (write(fdbp->desc, &flipped.agentBirthTime,
		   sizeof(time_t)) == -1) ||
	    (write(fdbp->desc, &flipped.kernelVersion,
		   sizeof(u_short)) == -1) ||
	    (write(fdbp->desc, &flipped.agentVersion,
		   sizeof(u_short)) == -1) ||
	    (write(fdbp->desc, &flipped.collectorVersion,
		   sizeof(u_short)) == -1) ||
	    (write(fdbp->desc, &flipped.collectorVersion,
		   sizeof(u_short)) == -1))
	    rc = -1;
	break;
    case TRACE_VERSION_CMU1:
	if ((write(fdbp->desc, &flipped.hostAddress, sizeof(u_int)) == -1) ||
	    (write(fdbp->desc, &flipped.bootTime, sizeof(time_t)) == -1) ||
	    (write(fdbp->desc, &flipped.versionTime, sizeof(time_t)) == -1))
	    rc = -1;
	break;
    default:
	rc = -1;
    }
    return(rc);
}
