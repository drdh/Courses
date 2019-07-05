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

static char *rcsid = "$Header: /home/cvs/root/DFSTrace/src/post/ht_hash_table.h,v 1.1.1.1 1998/09/29 18:39:18 tmkr Exp $;"
#endif _BLURB_

/*
 *  NOTE: The interface to this package and the lt_linear_table differ
 *  slightly.  First, the HT_GENERATE_ROUTINE_HEADERS macro takes an
 *  extra parameter for C++ compatibility.  Second, the hashFunctionName
 *  parameter to the HT_GENERATE_TABLE_ROUTINES macro must now return
 *  a 32-bit quantity.  It is no longer necessary to mod by the tableSize.
 *  The package now does a mod of the value that the hash function returns.
 *  This removes a potential source of error.  Third, the <prefix>-Init()
 *  function is no longer generated.
 */

/*
 *  The macros below expand to routines that implement a standard hash table
 *  interface.
 *
 *
 *  The HT_GENERATE_TABLE_HEADERS macro should be called in a .h file if the
 *  generated routines are exported to other .c files.  You must supply the
 *  prefix string for the routine names, the record_type name, and the
 *  key_type name (see below).  The calling conventions for the generated
 *  routines are described below.  Here is a sample invocation (sym_record_t
 *  is defined below):
 *
 *    HT_GENERATE_TABLE_HEADERS(Symbol,sym_record_t,char *)
 *
 *
 *  The HT_GENERATE_TABLE_ROUTINES macro should be called in the .c file
 *  that implements the hash table.  You must provide many arguments to
 *  this macro.  Each argument is described below, in the order in which
 *  it should be specified.
 *
 *    prefix - This parameter will be used as a prefix for all routines
 *		and global variables generated.  For example, Symbol.
 *
 *    tableSize - This is the size of the hash table.  It should be a
 *		prime number.  For example, 101.
 *
 *    record_type - The hash table will store pointers to records of this
 *		type.  This type must be a structure that contains key and
 *		link fields (see below).  For example, sym_record_t, where
 *		    typedef struct sym_record {
 *			char		symbol[MaxSymbolLength];
 *			long		myValue;
 *			struct sym_record *symLink;
 *		    } sym_record_t;
 *
 *    linkFieldName - The structure named by the record_type parameter must
 *		contain a field of type (record_type *).  This field will be
 *		used to link records together in the hash table.  The
 *		linkFieldName parameter specifies the name of this field.
 *		For example, symLink.
 *
 *    key_type - The name of the type specification used to pass keys
 *		around.  For example, char *.
 *
 *    keyExtractionFunctionName - This function (or macro) takes one argument
 *		of type (record_type *) and returns something of type
 *		(key_type).  For example, GIVE_KEY, where
 *		    #define GIVE_KEY(x) ((x)->symbol)
 *
 *    hashFunctionName - This function (or macro) takes one argument of type
 *		(key_type) and returns a hash value.  The value must be an
 *	        integer.  (It may be unsigned.)  For example, HASH_KEY,
 *	        where:
 *		    #define HASH_KEY(x) (x[0])
 *	        This would be a terrible hash funnction to use in practice.
 *
 *    equalFunctionName - This function (or macro) takes two arguments of type
 *		(key_type) and returns true if they are equal.  For example,
 *		EQUAL_KEY, where
 *		    #define EQUAL_KEY(x,y) (strcmp(x,y) == 0)
 *
 *  If we used the examples given above we would have the following:
 *
 *	#include <cam/ht_hash_table.h>
 *
 *	typedef struct sym_record {
 *		char		symbol[MaxSymbolLength];
 *		long		myValue;
 *		struct sym_record *symLink;
 *	} sym_record_t;
 *
 *	#define GIVE_KEY(x) ((x)->symbol)
 *	#define HASH_KEY(x) (x[0])
 *	#define EQUAL_KEY(x,y) (strcmp(x,y) == 0)
 *
 *	HT_GENERATE_TABLE_ROUTINES(Symbol,101,sym_record_t,symLink,
 *				   char *,GIVE_KEY,HASH_KEY,EQUAL_KEY)
 *
 *  The following routines would be generated:
 *
 *	sym_record_t *SymbolLookup(key)
 *	    char *key;
 *	...
 *
 *	void SymbolInsert(pRecord)
 *	    sym_record_t *pRecord;
 *	...
 *
 *	sym_record_t *SymbolDelete(key)
 *	    char *key;
 *	...
 *
 *	void SymbolForall(f)
 *	    void (*f)();
 *	...
 *
 *  These macros have been used before, so just ask if you have any
 *  questions, or need some real world examples.
 */

#ifndef	_HT_HASH_TABLE_

#define HT_IDENT(x) x
#define HT_CONCAT(a,b) HT_IDENT(a)b

#define HT_GENERATE_TABLE_HEADERS(prefix,record_type,key_type)		  \
									  \
extern record_type *HT_CONCAT(prefix,Lookup)();				  \
extern void HT_CONCAT(prefix,Insert)();					  \
extern record_type *HT_CONCAT(prefix,Delete)();				  \
typedef void *HT_CONCAT(prefix,ForAllFunctionType)();			  \
extern void HT_CONCAT(prefix,Forall)();

#define HT_GENERATE_TABLE_ROUTINES(prefix,tableSize,record_type,linkFieldName,key_type,keyExtractionFunctionName,hashFunctionName,equalFunctionName)	  \
									  \
record_type *HT_CONCAT(prefix,Table)[tableSize];		          \
									  \
									  \
record_type *HT_CONCAT(prefix,Lookup)(key)				  \
    key_type key;							  \
{                							  \
    record_type *current = HT_CONCAT(prefix,Table)[			  \
		  ((unsigned int) (hashFunctionName(key))) % tableSize];  \
									  \
    while (current != (record_type *) 0) {				  \
	if (equalFunctionName(key, keyExtractionFunctionName(current)))	  \
	    return(current);						  \
	current = current->linkFieldName;				  \
    }									  \
									  \
    return((record_type *) 0);						  \
}									  \
									  \
									  \
record_type *HT_CONCAT(prefix,Delete)(key)				  \
    key_type key;							  \
{                 							  \
    record_type **linkPtr = &HT_CONCAT(prefix,Table)[			  \
		  ((unsigned int) (hashFunctionName(key))) % tableSize];  \
									  \
    while (*linkPtr != (record_type *) 0) {				  \
	if (equalFunctionName(key,keyExtractionFunctionName((*linkPtr)))) { \
	    record_type *current = *linkPtr;				  \
	    (*linkPtr) = current->linkFieldName;			  \
	    return(current);						  \
	}								  \
	linkPtr = &(*linkPtr)->linkFieldName;				  \
    }									  \
									  \
    return((record_type *) 0);						  \
}									  \
									  \
									  \
void HT_CONCAT(prefix,Insert)(pRecord)					  \
    record_type *pRecord;						  \
{               							  \
    record_type **bucketPtr = &HT_CONCAT(prefix,Table)[			  \
	    ((unsigned int) (hashFunctionName(				  \
		keyExtractionFunctionName(pRecord)))) % tableSize];	  \
									  \
    pRecord->linkFieldName = *bucketPtr;				  \
    *bucketPtr = pRecord;						  \
									  \
}									  \
									  \
									  \
void HT_CONCAT(prefix,Forall)(f)					  \
    void (*f)();							  \
{               							  \
    unsigned int i;							  \
    record_type *current,*next;						  \
									  \
    for (i=0; i<tableSize; i++) {					  \
	current = HT_CONCAT(prefix,Table)[i];				  \
									  \
	while (current != (record_type *) 0) {				  \
	    /* Save next now, f may delete current! */			  \
	    next = current->linkFieldName;				  \
	    f(current);							  \
	    current = next;						  \
	}								  \
    }									  \
}

#endif	/* _HT_HASH_TABLE_ */
