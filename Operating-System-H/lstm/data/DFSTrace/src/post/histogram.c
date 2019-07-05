/**
 * histogram.c
 *
 *  the header file for a class of routines used to maintain
 *statistical distributions of various data
 *
 *                                         tmk--October 98
 *$Id:$
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "histogram.h"

/**
 * clear all values and allocate buckets
 *and set them to a linear distribution starting at start and incrementing
 *by step
 *
 *  returns 1  if success 0 if failed
 */
int   hist_initialize( histogram_t *hist,
		       unsigned long buckets,
		       double start,
		       double step) {

    int index = 0;

    hist-> bucket_count =  (unsigned long*) malloc(sizeof (long)*buckets );
    if ( hist-> bucket_count== NULL) return 0;

    hist-> bucket_limit = (double*) malloc (sizeof (double)*buckets);
    if (hist-> bucket_limit== NULL) return 0;
    hist-> numb_buckets = buckets;

    for (index = 0;   index < buckets-1; index++ ) {
	hist-> bucket_limit[index] = start;
	start += step;
    }  /* end for ()  */
    hist-> bucket_limit [buckets-1] =DBL_MAX;
    hist_reset (hist);
    return 1;
}

/**
 * set the buckets of a histogram to exponential distribution
 *starting at start and multiplying by factor
 *
 *return one if successful 0 if failed
 */
int hist_set_buckets_exponential (histogram_t *hist,
				  unsigned long buckets,
				  double start,
				  double factor) {
    int index;

    free(hist->bucket_count);
    free(hist->bucket_limit);

    hist-> bucket_count =  (unsigned long*) malloc(sizeof (long)*buckets );
    if ( hist-> bucket_count== NULL) return 0;
    hist-> bucket_limit = (double*) malloc (sizeof (double)*buckets);
    if ( hist-> bucket_limit== NULL) return 0;
    hist-> numb_buckets = buckets;

    for (index = 0;   index < buckets-1; index++ ) {
	hist-> bucket_limit[index] = start;
	start *= factor;
	hist->bucket_count[index] =0;
    }  /* end for ()  */
    return 1;
}

/**
 *  clear the values of the provided histogram
 */
void  hist_reset(histogram_t *hist) {

    int index =0;

    hist->samples = 0;
    hist->sum = 0.0;
    hist->squared_sum = 0.0;
    hist-> minimum_value =DBL_MAX;
    hist-> maximum_value = DBL_MIN;

    for (index =0; index <  hist->numb_buckets; index++) {
	 hist->bucket_count [index] = 0;
    }  /* end for ()  */
}

/**
 *  free the memory for the given histogram
 */
void hist_free (histogram_t *hist) {
    free(hist->bucket_count);
    free(hist->bucket_limit);
    free (hist);
}

/**
 *  add a specific sample
 */
void hist_add_sample (histogram_t *hist, double sample) {
    int i;

    hist->samples++;
    hist->sum += sample;
    hist->squared_sum += (sample*sample);
    if (hist->minimum_value > sample) hist->minimum_value = sample;
    if (hist->maximum_value < sample) hist->maximum_value = sample;

    for (i = 0; i < hist-> numb_buckets; i++) {
        if ( sample <  hist->bucket_limit[i]) break;
    }
    hist->bucket_count[i]++;
}

/**
 * returns the number of samples
 */
unsigned long  hist_numb_samples(histogram_t *hist) {
    return hist-> samples;
}

/**
 * returns the mean
 */
double hist_mean(histogram_t *hist) {
    if (  hist-> samples > 0)
        return ( hist-> sum / hist-> samples);
    return(0.0);
}

/**
 *  returns the standard deviation
 */
double hist_stdDev(histogram_t *hist) {
    double var;
    var =  hist_var( hist);

    if ((hist-> samples==0) || (var <= 0))
        return(0.0);

    return( (double) sqrt( var ) );
}

/**
 *  returns the variance
 */
double hist_var(histogram_t *hist) {
    if (hist-> samples > 1)
	return ((hist-> squared_sum -((hist-> sum*hist-> sum)/hist-> samples))/
		 (hist-> samples-1));
    return(0.0);
}

/**
 *  returns the minimum and  maximum sample seen
 */
double hist_min(histogram_t *hist) {
    return hist-> minimum_value;
}

double hist_max(histogram_t *hist) {
    return hist-> maximum_value;
}

/**
 * returns the sum of all of the samples
 */
double hist_cumulative (histogram_t *hist) {
    return hist->sum;
}


/**
 * returns the width of the conference interval for the given level
 */
double  hist_confidence(histogram_t *hist, double p_value) {
     return 0.0;
}


/**
 *  print out the values of the histogram
 */
void  hist_summary (histogram_t *hist) {
    fprintf(stdout, "samples: %6ld  mean:%6.2f  max:%6.2f  min:%6.2f  std:%6.2f\n",
	    hist_numb_samples (hist), hist_mean (hist), hist_max (hist),
	    hist_min (hist), hist_stdDev (hist));
}

void hist_dump_buckets (histogram_t *hist) {
    unsigned long index;

    for (index = 0 ; index < (hist->numb_buckets) -1; index++ ) {
	fprintf(stdout, "<  %12.1f : %5ld\n",  hist-> bucket_limit [index],
		hist->bucket_count[index]);
    }  /* end for ()  */
    fprintf(stdout, "<           max : %5ld\n", hist->bucket_count[hist->numb_buckets-1] );
}
