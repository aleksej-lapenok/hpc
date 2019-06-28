#include <omp.h>

#include <stdio.h>
#include <stdlib.h> 


// #include <boost/math/quadrature/trapezoidal.hpp>

#define A ((double)_A)
#define B ((double)_B)
#define N ((double)_N)
#define PREC ((double)_PREC)


#ifndef INTEG
# define INTEG integrate0
#endif


#ifndef COMPUTE_SUM
# define COMPUTE_SUM compute_sum_reduction
#endif

#include <math.h>




double f(double x) {
	double a = (1./x) * sin(1./x);
	return pow(a, 2);
}

double f_int(double a, double b) {
	return (1./4.) * (  2. * (b-a)/(a*b) + sin(2/b) - sin(2/a) );
}

double F(double x) {
	return -1/(2*x) + (1/4) * sin(2/x);
}



double compute_sum_raw(size_t p, double h, double a){
	double sum = 0;

	// #pragma omp parallel for
	for(size_t j = 1; j < p; j += 1)
	{
		double y = f(a + j*h);
		sum += y;
	}
	return sum;
}

double compute_sum_atomic(size_t p, double h, double a){
	double sum = 0;

	#pragma omp parallel for
	for(size_t j = 1; j < p; j += 1)
	{
		double y = f(a + j*h);
		#pragma omp atomic
		sum += y;
	}
	return sum;
}

double compute_sum_critical(size_t p, double h, double a){
	double sum = 0;

	#pragma omp parallel for
	for(size_t j = 1; j < p; j += 1)
	{
		double y = f(a + j*h);
		#pragma omp critical
		sum += y;
	}
	return sum;
}

double compute_sum_reduction(size_t p, double h, double a){
	double sum = 0;

	#pragma omp parallel for reduction(+:sum)
	for(size_t j = 1; j < p; j += 1)
	{
		double y = f(a + j*h);
		sum += y;
	}
	return sum;
}

double compute_sum_locks(size_t p, double h, double a){
	double sum = 0;
	omp_lock_t lock;

	omp_init_lock(&lock);

	#pragma omp parallel for
	for(size_t j = 1; j < p ; j += 1)
	{
		double y = f(a + j*h);

		omp_set_lock(&lock);
		sum += y;
		omp_unset_lock(&lock);
	}
	omp_destroy_lock(&lock);

	return sum;
}


#define compute_sum compute_sum_atomic


double integrate0() {
	double a = A, b = B;
    double ya = f(a), yb = f(b);
    double h, error;
    double sum1 = 0, sum2 = 0, sum;

    int n = 4;

    for(size_t i = 0; i < 1e6; i += 50) {
    	

    	n += 1;
    	h = (b - a)/n;

    	sum = (ya + yb)*0.5;
    	sum += COMPUTE_SUM(n, h, a);
    	sum *= h;

		sum1 = sum2;
		sum2 = sum;
    	if(i != 0) {
    		error = fabs(sum2-sum1)/fabs(sum2);
    		if(fabs(error) <= fabs(PREC)) {
	    		break;
    		}
    	}
    }
    
    printf("points: %d\n", n);
    return sum2;
}

double integrate1() {
	double a = A, b = B;
    double ya = f(a), yb = f(b);
    double h, error;
    double sum1 = 0, sum_res = 0;
    double *sums, sum;
    int n = 4, end_work = 0;

    #pragma omp parallel shared(n, sums, sum, end_work, sum_res)
    {
    	int tid = omp_get_thread_num();
    	int n_threads = omp_get_num_threads();


    	#pragma omp single
    	{
    		sums = (double*)malloc(n_threads * sizeof(double));
    	}

    	// #pragma omp barrier


	    for(size_t i = 0; i < 1e6; ++i, n += n_threads) {
    		int __end_work;
	    	int _n = n + tid;
	    	
	    	double h = (b - a)/_n;

	    	sums[tid] = (ya + yb)*0.5;

			// #pragma omp for reduction(+:sum) nowait
			#pragma omp taskloop nowait
				for(size_t j = 1; j < _n; j += 1) {
					sums[tid] += f(a + j*h);
				}
	    	sums[tid] *= h;


	    	#pragma omp barrier




	    	#pragma omp single
    		{
		    	for(size_t j = 1; j < n_threads; ++j) {
		    		error = fabs(sums[j-1] - sums[j])/fabs(sums[j]);
		    		// printf("%d %.9lf\n", tid, error);
		    		if(fabs(error) <= fabs(PREC)) {
		    			// printf("DONE\n");

		    			#pragma omp atomic write
		    				end_work = 1;

		    			sum_res = sums[j];
		    			break;
	    			}
		    	}
		    }
		    // #pragma omp barrier


    		#pragma omp atomic read
				__end_work = end_work;
    		if(__end_work) break;
    		
	    }

    }
    printf("points: %d\n", n);
    return sum_res;
}


double integrate_boost(int max_refinements, double tol) {
	double a = A, b = B;
    double ya = f(a), yb = f(b);
    double h = (b - a)*0.5;
    double I0 = (ya + yb)*h;
    double IL0 = (abs(ya) + abs(yb))*h;

    double yh = f(a + h);
    double I1;
    I1 = I0*0.5 + yh*h;

    // The recursion is:
    // I_k = 1/2 I_{k-1} + 1/2^k \sum_{j=1; j odd, j < 2^k} f(a + j(b-a)/2^k)
    size_t k = 2;
    // We want to go through at least 4 levels so we have sampled the function at least 10 times.
    // Otherwise, we could terminate prematurely and miss essential features.
    // This is of course possible anyway, but 10 samples seems to be a reasonable compromise.
    double error = abs(I0 - I1);
    int points = 0;
    while (k < 4 || (k < max_refinements && error > tol*abs(I1)) )
    {
        I0 = I1;

        I1 = I0*0.5;
        size_t p = static_cast<size_t>(1u) << k;
        points += p/2;
        h *= 0.5;

        double sum = COMPUTE_SUM(h, p, a);
   
        I1 += sum*h;
        ++k;
        error = abs(I0 - I1);
    }
    // hmm
    printf("points: %d\n", points);
    return I1;
}


int main(int argc, char *argv[]) 
{

	double sum = INTEG();

	// using boost::math::quadrature::trapezoidal;
	// double I = trapezoidal(f, A, B, boost::math::tools::root_epsilon<double>(), 10);

	printf("res_prec: %f\n", f_int(A, B));
	printf("res_prec2: %f\n",F(B)-F(A));
	printf("res: %f\n", sum);
	// printf("res_boost: %f\n", I);
	// printf("precision: %f\n", fabs(sum - res_prec) / fabs(res_prec));

	return 0;
}