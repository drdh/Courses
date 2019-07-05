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

*/

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/dlist.h,v 1.1.1.1 1998/09/29 18:39:18 tmkr Exp $";
#endif _BLURB_



/*
 *
 * dlist.h -- Specification of  doubly-linked list type where list elements
 * can be on only one list at a time and are kept in sorted order.
 *
 */

#ifndef _UTIL_DLIST_H_
#define _UTIL_DLIST_H_ 1

#ifdef __cplusplus
extern "C" {
#endif __cplusplus

#include <stdio.h>

#ifdef __cplusplus
}
#endif __cplusplus



class dlink;
class dlist;
class dhashtab;
typedef int (*CFN)(dlink *, dlink *);

enum DlGetType { DlGetMin, DlGetMax };

class dlist {
  friend class dhashtab;
    dlink *head;	    // head of list 
    int cnt;
    CFN	CmpFn;		    // function to order the elements 
  public:
    dlist();		    // default init with NULL compare func
    dlist(CFN);
    virtual ~dlist();
    void insert(dlink *);   // insert in sorted order 
    void prepend(dlink *);  // add at beginning of list 
    void append(dlink *);   // add at end of list 
    dlink *remove(dlink	*); // remove specified entry
    dlink *first();	    // return head of list
    dlink *last();	    // return tail of list
    dlink *get(DlGetType =DlGetMin);	// return and remove head or tail of list
    void clear();	    // remove all entries
    int count();
    int IsMember(dlink *);
    virtual void print();
    virtual void print(FILE *);
    virtual void print(int);
};

enum DlIterOrder { DlAscending, DlDescending };

class dlist_iterator {
    dlist *cdlist;	    // current dlist
    dlink *cdlink;	    // current dlink
    DlIterOrder	order;	    // iteration order
  public:
    dlist_iterator(dlist&, DlIterOrder =DlAscending);
    dlink *operator()();    // return next object or 0
};

class dlink {		    // objects are derived from this class
  friend class dlist;
  friend class dlist_iterator;
    dlink *next;
    dlink *prev;
  public:
    dlink();
    virtual ~dlink();
    virtual void print();
    virtual void print(FILE *);
    virtual void print(int);
};

#endif	not _UTIL_DLIST_H_
