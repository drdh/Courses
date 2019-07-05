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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/unpack.c,v 1.4 1998/10/07 15:00:36 tmkr Exp $";
*/
#endif _BLURB_


/* unpack.c -- figures out what sort of a trace we have, sets up unpacking structure */

#include <stdio.h>
#include <string.h>
#include "tracelib.h"
#include "trace.h"
#include "unpack_private.h"
#include "v1.h"
#include "v2.h"
#include "v3.h"
#include "v4.h"
#include "ucb.h"
#include "ur.h"

/*
 * DecodeTrace -- figures out what sort of a trace we
 * have by reading the first DECODE_AMOUNT bytes of the
 * trace file.  Sets the trace type if it can figure
 * out what it is, otherwise returns TRACE_BADVERSION.
 * If there are problems reading the file, it returns
 * TRACE_FILEREADERROR.  Also sets the unpacking operations.
 */
int DecodeTrace(tfPtr)
trace_file_t *tfPtr;
{
	char buf[DECODE_AMOUNT];
	int  rc;

	if ((rc=fread(buf, 1, DECODE_AMOUNT, tfPtr->fp) != DECODE_AMOUNT)) {
	    return(TRACE_FILEREADERROR);
	}

	/*We must now reset the trace pointer, since popen (for compressed
	 * files can't seek backwards so we must re-open the process */
	if (tfPtr-> compressed) {
	    pclose (tfPtr-> fp);
	    if ((tfPtr->fp =popen(tfPtr->command, "r")) == NULL) {
		FreeTraceFile(tfPtr);
		return(TRACE_FILEREADERROR);
	    }
	}
	/* if it is a normal file we can just seek to the beginning */
	else
	    if (fseek(tfPtr->fp, 0, 0)!=0) {
		return(TRACE_FILEREADERROR);
	    }

	if (V4FilePreamble(buf)) {
		tfPtr->unpackRecordProc = UnpackV4Record;
		tfPtr->version = TRACE_VERSION_UCSC;
		rc = UnpackV4FilePreamble(tfPtr);
		if (!rc) UnpackFirstV4RecordTime(tfPtr, buf);
	} else if (V3FilePreamble(buf)) {
		tfPtr->unpackRecordProc = UnpackV3Record;
		tfPtr->version = TRACE_VERSION_CMU3;
		rc = UnpackV3FilePreamble(tfPtr);
		if (!rc) UnpackFirstV3RecordTime(tfPtr, buf);
	} else if (V2FilePreamble(buf)) {
		tfPtr->unpackRecordProc = UnpackV2Record;
		tfPtr->version = TRACE_VERSION_CMU2;
		rc = UnpackV2FilePreamble(tfPtr);
		if (!rc) UnpackFirstV2RecordTime(tfPtr, buf);
	} else if (V1FilePreamble(buf)) {
		tfPtr->unpackRecordProc = UnpackV1Record;
		tfPtr->version = TRACE_VERSION_CMU1;
		rc = UnpackV1FilePreamble(tfPtr,buf);
		if (!rc) UnpackFirstV1RecordTime(tfPtr, buf);
	} else if (UCBFilePreamble(buf)) {
		tfPtr->unpackRecordProc = UnpackUCBRecord;
		tfPtr->version = TRACE_VERSION_UCB1;
		rc = UnpackUCBFilePreamble(tfPtr);
		if (!rc) UnpackFirstUCBRecordTime(tfPtr, buf);
	} else if (URFilePreamble(buf)) {
		tfPtr->unpackRecordProc = UnpackURRecord;
		tfPtr->version = TRACE_VERSION_UR;
		rc = UnpackURFilePreamble(tfPtr);
		if (!rc) UnpackFirstURRecordTime(tfPtr, buf);
	} else
		rc = TRACE_BADVERSION;

	return(rc);
}

void PrintPreamble(tfPtr)
trace_file_t *tfPtr;
{
	switch (tfPtr->version) {
	case TRACE_VERSION_CMU1:
		PrintV1FilePreamble(tfPtr);
		break;
	case TRACE_VERSION_CMU2:
		PrintV2FilePreamble(tfPtr);
		break;
	case TRACE_VERSION_CMU3:
		PrintV3FilePreamble(tfPtr);
		break;
	case TRACE_VERSION_UCSC:
		PrintV4FilePreamble(tfPtr);
		break;
	case TRACE_VERSION_UCB1:
		PrintUCBFilePreamble(tfPtr);
		break;
	case TRACE_VERSION_UR:
		PrintURFilePreamble(tfPtr);
		break;
	default:
		(void) printf("PrintPreamble: Unknown trace type!\n");
	}
}
