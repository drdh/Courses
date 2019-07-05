/**
 * histogram.h
 *
 *  the header file for a class of routines used to maintain
 *statistical distributions of various data
 *
 *                                         tmk--October 98
 *$Id:$
 */


typedef struct {
    unsigned long samples;            /* the number of data samples */
    double sum;                     /* the sum of the data samples */
    double squared_sum;             /* the sum of the squares of the samples */
    double minimum_value;           /* the minimum value seen  */
    double maximum_value;           /* the maximum value seen */
    
    /*  histogram fields */
    unsigned long numb_buckets;     /* number of buckets  */
    unsigned long *bucket_count;    /* the count in each bucket */
    double *bucket_limit;           /* the limit on values stored in 
				     * each bucket */
}  histogram_t;



/**
 * various functions defined
 */

/**
 * clear all values and allocate buckets
 *and set them to a linear distribution starting at start and incrementing
 *by stdevep
 *
 *  returns 1  if success 0 if failed
 */
extern int   hist_initialize( histogram_t *hist, 
		       unsigned long buckets,
		       double start,
		       double step);

/**
 * set the buckets of a histogram to exponential distribution
 *starting at start and multiplying by factor
 *
 *return one if successful 0 if failed
 */
extern int hist_set_buckets_exponential (histogram_t *hist, 
		       unsigned long buckets,
		       double start,
		       double factor);

/**
 *  clear the values of the provided histogram
 */
extern void hist_reset(histogram_t *hist); 

/**
 *  free  the memory associated with the given histogram
 */
extern void hist_free (histogram_t *hist);

/**
 *  add a specific sample
 */
extern void hist_add_sample (histogram_t *hist, double sample);
 
/**
 * returns the number of samples
 */
extern unsigned long  hist_numb_samples(histogram_t *hist);

/**
 * returns the mean
 */
extern double hist_mean(histogram_t *hist);
/**
 *  returns the standard deviation
 */
extern double hist_stdDev(histogram_t *hist);

/**
 *  returns the variance
 */
extern double hist_var(histogram_t *hist);

/**
 *  returns the minimum and  maximum sample seen 
 */
extern double  hist_min(histogram_t *hist);
extern double hist_max(histogram_t *hist);

/**
 * returns the sum of all of the samples
 */
extern double hist_cumulative (histogram_t *hist);

/**
 * returns the width of the conference interval for the given level
 */
extern double hist_confidence(histogram_t *hist, double p_value);

/**
 *  print out the values of the histogram
 */
extern void  hist_summary (histogram_t *hist);
extern void  hist_dump_buckets (histogram_t *hist);
