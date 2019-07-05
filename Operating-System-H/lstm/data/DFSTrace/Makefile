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
#static char *rcsid = "$Header: /home/cvs/root/DFSTrace/Makefile,v 1.4 1998/10/13 21:14:11 tmkr Exp $";
#
#endif _BLURB_

all: library processing

config: config.local

config.local: 
	echo  "TRACEDIR =`pwd`"  > config.local
	if (test -f "src/config.`uname -s`"); then \
		echo "OS_NAME = "`uname -s` >> config.local; \
	else \
		echo "OS_NAME = default" >> config.local; \
	fi

library: config.local
	if (test ! -d include); then  mkdir include; fi
	if (test ! -d lib); then  mkdir lib; fi
	cd src/tracelib ; make all install

processing: config.local library
	if (test ! -d bin); then  mkdir bin; fi
	cd src/post ; make all install

clean:  
	cd src/post ; make clean
	cd src/tracelib ; make clean

real_clean:
	cd src/post ; make clean
	cd src/tracelib ; make clean
	rm config.local



