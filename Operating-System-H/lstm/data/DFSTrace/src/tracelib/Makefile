#ifndef _BLURB_
#define _BLURB_
#
#
#    DFStrace: an Experimental File Reference Tracing Package
#
#       Copyright (c) 1990-1995 Carnegie Mellon University
#                      All Rights Reserved.
#
#Permission  to use, copy, modify and distribute this software and
#its documentation is hereby granted (including for commercial  or
#for-profit use), provided that both the copyright notice and this
#permission  notice  appear  in  all  copies  of   the   software,
#derivative  works or modified versions, and any portions thereof,
#and that both notices appear  in  supporting  documentation,  and
#that  credit  is  given  to  Carnegie  Mellon  University  in all
#publications reporting on direct or indirect use of this code  or
#its derivatives.
#
#DFSTRACE IS AN EXPERIMENTAL SOFTWARE PACKAGE AND IS KNOWN TO HAVE
#BUGS, SOME OF WHICH MAY  HAVE  SERIOUS  CONSEQUENCES.    CARNEGIE
#MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS" CONDITION.
#CARNEGIE MELLON DISCLAIMS ANY  LIABILITY  OF  ANY  KIND  FOR  ANY
#DAMAGES  WHATSOEVER RESULTING DIRECTLY OR INDIRECTLY FROM THE USE
#OF THIS SOFTWARE OR OF ANY DERIVATIVE WORK.
#
#Carnegie Mellon encourages (but does not require) users  of  this
#software to return any improvements or extensions that they make,
#and to grant Carnegie Mellon the  rights  to  redistribute  these
#changes  without  encumbrance.   Such improvements and extensions
#should be returned to Software.Distribution@cs.cmu.edu.
#
#static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/tracelib/Makefile,v 1.7 1998/11/07 22:30:18 tmkr Exp $";
#
#endif _BLURB_


#Get the local OS and base directory
include ../../config.local
#Get settings specific for the operating system
include ../config.$(OS_NAME)

#get the main package configurations 
include ../config.main


CFLAGS = -g $(OS_FLAGS)

HEADERS = tracelib.h trace.h filter.h ftable.h split.h unpack.h pid.h v1.h v2.h v1_private.h v2_private.h v3_private.h v4_private.h v4.h
EXPORTED_HEADERS = tracelib.h

OBJECTS = trace.o filter.o ftable.o pid.o rec.o unpack.o split.o v1.o v2.o v3.o ucb.o ur.o v4.o
CSOBJECTS = atot.o parsedate.o nxtarg.o fold.o skipto.o

all: libtrace.a tm

libtrace.a: $(OBJECTS) $(CSOBJECTS)
	ar rv libtrace.a $(OBJECTS) $(CSOBJECTS)
	ranlib libtrace.a

filter.o: filter.c tracelib.h
	$(CC) $(CFLAGS) -c filter.c

ftable.o: ftable.c tracelib.h
	$(CC) $(CFLAGS) -c ftable.c

pid.o: pid.c pid.h tracelib.h
	$(CC) $(CFLAGS) -c pid.c

rec.o: rec.c tracelib.h
	$(CC) $(CFLAGS) -c rec.c

split.o: split.c split.h tracelib.h
	$(CC) $(CFLAGS) -c split.c

trace.o: trace.c trace.h tracelib.h filter.h split.h unpack.h
	$(CC) $(CFLAGS) -c trace.c

unpack.o: unpack.c tracelib.h unpack_private.h v1.h v2.h
	$(CC) $(CFLAGS) -c unpack.c

v1.o: v1.c tracelib.h trace.h unpack_private.h v1_private.h
	$(CC) $(CFLAGS) -c v1.c

v2.o: v2.c tracelib.h trace.h unpack_private.h v2_private.h
	$(CC) $(CFLAGS) -c v2.c

v3.o: v3.c tracelib.h trace.h unpack_private.h v3_private.h
	$(CC) $(CFLAGS) -c v3.c

ucb.o: ucb.c tracelib.h unpack_private.h ucb_private.h
	$(CC) $(CFLAGS) -c ucb.c

ur.o: ur.c tracelib.h unpack_private.h ur_private.h
	$(CC) $(CFLAGS) -c ur.c

tm: tm.o libtrace.a
	$(CC) $(CFLAGS) $(OS_LIB) -o tm tm.o libtrace.a

tm.o: tm.c
	$(CC) $(CFLAGS) -c tm.c

examine_raw:  examine_raw.o libtrace.a
	$(CC) $(CFLAGS) $(OS_LIB) -o examine_raw examine_raw.o libtrace.a

examine_raw.o: examine_raw.c
	$(CC) $(CFLAGS) -c examine_raw.c

parsedate.o: parsedate.c fold.h
	$(CC) $(CFLAGS) -c parsedate.c

install: installa installh

installa: libtrace.a 
	cp libtrace.a $(LIBDIR)/libtrace.a

installh: 
	cp tracelib.h ${INCLDIR}

clean:  
	rm -f tm *.o *.a *~



