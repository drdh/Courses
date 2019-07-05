#ifndef _BLURB_
#define _BLURB_
# /*
# 
#     DFStrace: an Experimental File Reference Tracing Package
# 
#        Copyright (c) 1990-1995 Carnegie Mellon University
#                       All Rights Reserved.
# 
# Permission  to use, copy, modify and distribute this software and
# its documentation is hereby granted (including for commercial  or
# for-profit use), provided that both the copyright notice and this
# permission  notice  appear  in  all  copies  of   the   software,
# derivative  works or modified versions, and any portions thereof,
# and that both notices appear  in  supporting  documentation,  and
# that  credit  is  given  to  Carnegie  Mellon  University  in all
# publications reporting on direct or indirect use of this code  or
# its derivatives.
# 
# DFSTRACE IS AN EXPERIMENTAL SOFTWARE PACKAGE AND IS KNOWN TO HAVE
# BUGS, SOME OF WHICH MAY  HAVE  SERIOUS  CONSEQUENCES.    CARNEGIE
# MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS" CONDITION.
# CARNEGIE MELLON DISCLAIMS ANY  LIABILITY  OF  ANY  KIND  FOR  ANY
# DAMAGES  WHATSOEVER RESULTING DIRECTLY OR INDIRECTLY FROM THE USE
# OF THIS SOFTWARE OR OF ANY DERIVATIVE WORK.
# 
# Carnegie Mellon encourages (but does not require) users  of  this
# software to return any improvements or extensions that they make,
# and to grant Carnegie Mellon the  rights  to  redistribute  these
# changes  without  encumbrance.   Such improvements and extensions
# should be returned to Software.Distribution@cs.cmu.edu.
# 
# */
# 
# static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/Makefile,v 1.6 1998/11/06 03:38:35 tmkr Exp $;
#endif _BLURB_

#Get the local OS and base directory
include ../../config.local
#Get settings specific for the operating system
include ../config.$(OS_NAME)

#get the main package configurations 
include ../config.main

LIBTRACE = $(LIBDIR)/libtrace.a 
CFLAGS = -I$(INCLDIR) -g $(OS_FLAGS)

all: files patterns replay sessions tstat users creplay untrace

dhash.o: dhash.c dhash.h
	$(CPLUSPLUS) ${CFLAGS} -c dhash.c

dlist.o: dlist.c dlist.h
	$(CPLUSPLUS) ${CFLAGS} -c dlist.c

untrace: untrace.o dlist.o dhash.o $(LIBTRACE)
	$(CPLUSPLUS) ${LIBFLAGS} -o untrace untrace.o dhash.o dlist.o $(LIBTRACE) 

untrace.o: untrace.c
	$(CPLUSPLUS) ${CFLAGS} -c untrace.c

creplay: creplay.o 
	$(CPLUSPLUS) ${LIBFLAGS} -o creplay creplay.o

creplay.o: creplay.c
	$(CPLUSPLUS) ${CFLAGS} -c creplay.c

replay: replay.o $(LIBTRACE)
	$(CC)   ${LIBFLAGS}  -o replay replay.o $(LIBTRACE)

tstat: tstat.o $(LIBTRACE)
	$(CC)  ${LIBFLAGS}  -o tstat tstat.o $(LIBTRACE)

files: files.o $(LIBTRACE)
	$(CC)  ${LIBFLAGS}  -o files files.o $(LIBTRACE)

patterns: patterns.o $(LIBTRACE)
	$(CC)  ${LIBFLAGS}  -o patterns patterns.o $(LIBTRACE)

sessions: sessions.o $(LIBTRACE)
	$(CC) ${LIBFLAGS}   -o sessions sessions.o $(LIBTRACE)

users: users.o $(LIBTRACE)
	$(CC)  ${LIBFLAGS}  -o users users.o $(LIBTRACE)

inter_arrival: inter_arrival.o histogram.o $(LIBTRACE)
	$(CC)  -o inter_arrival inter_arrival.o histogram.o $(LIBTRACE) -lm

clean:
	rm -f *.o *~ replay tstat patterns users sessions creplay untrace files

install: all
	cp -p replay tstat patterns users sessions creplay untrace files $(BINDIR)

