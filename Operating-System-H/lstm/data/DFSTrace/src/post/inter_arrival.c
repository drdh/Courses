/*
 * inter_arrival.c  --  builds a histogram of the inter-arrival
 *  times  of trace events.
 *
 *note for older trace versions the best time resolution is
 *approximately 15msec
 *
 *  good exponentail params  -e -s 15000 -i 2 -b 20
 *
 * $Id:$
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <sys/param.h>
#include <limits.h>
#include <time.h>
#include <string.h>

#include "tracelib.h"
#include "histogram.h"

long subtract_time ();

void PrintUsage()
{
    printf("Usage:  inter_arrival [-v] [-e] [-d] [-s start] [-i incrament] [-b buckets] [-f filter] file\n");
    exit(1);
}

int main(argc, argv)
     int argc;
     char *argv[];
{
    extern int optind;
    extern char *optarg;
    FILE *inFile;
    dfs_header_t *record,*previous;
    unsigned long numb_buckets =  50;
    double start = 15000.0;
    double  increment = 15000.0;
    int numFRefs, i;
    int exponential = 0;
    char filterName[MAXPATHLEN];
    histogram_t arrivals;
    long difference;

    filterName [0] = 0;
    numFRefs =0;

    /* get filename */
    if (argc < 2)
	PrintUsage();

    /* Obtain invocation options */
    while ((i = getopt(argc, argv, "vdes:i:b:f:")) != EOF)
	switch (i) {
	case 'v':
	    verbose = 1;
	    break;
	case 'd':
	    debug = 1;
	    break;
	case 'e':
	    exponential = 1;
	    break;
	case 'i':
	    increment =atof(optarg);
	    break;
	case 'b':
	    numb_buckets = (unsigned long)atol(optarg);
	    break;
	case 's':
	    start =atof(optarg);
	    break;
	case 'f':
	    (void) strcpy(filterName, optarg);
	    break;
	default:
	    PrintUsage();
	}

    hist_initialize (& arrivals, numb_buckets, start, increment);
    if (exponential)
	hist_set_buckets_exponential (& arrivals, numb_buckets,
				      start, increment);

    if (((inFile = Trace_Open(argv[optind])) == NULL) ||
	(filterName[0] && Trace_SetFilter(inFile, filterName))) {
	printf("files: can't process file %s.\n", argv[optind]);
	exit(1);
    }

    previous =Trace_GetRecord (inFile);
    while ((record = Trace_GetRecord(inFile))!= NULL) {
	difference = subtract_time (previous, record);
	if (difference >= 0)
	    hist_add_sample (&arrivals,   (double) difference);
	(void) Trace_FreeRecord(inFile, previous);
	previous = record;
    }
    (void) Trace_Close(inFile);

    fprintf(stdout, "Inter-arrival data (times in usec) \n");
    hist_summary (&arrivals);
    hist_dump_buckets (&arrivals);
    return 0;
}

/**
 * subtract the time values for the two provided records
 *ignore overflows
 */
long subtract_time (dfs_header_t *previous,dfs_header_t *record) {
    long result;
    result = record-> time.tv_sec -  previous-> time.tv_sec;
    if (result > LONG_MAX/1000000)
	fprintf(stdout, "Warning time difference overflow\n");
    result *= 1000000;
    result +=  (record-> time.tv_usec -  previous-> time.tv_usec);
    return result;
}
