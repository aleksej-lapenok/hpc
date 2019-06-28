#include <omp.h>

#include <stdio.h>
#include <stdlib.h> 

#ifndef M
# define M 10
#endif

#ifndef N
# define N 14
#endif

int main(int argc, char *argv[]) 
{

	float **A, **B, **C;

	A = (float**)malloc(sizeof(float*)*M);
	for(int i=0; i < M; ++i) 
		A[i] = (float*)malloc(sizeof(float)*N);

	B = malloc(sizeof(float*)*N);
	for(int i=0; i < N; ++i) 
		B[i] = malloc(sizeof(float)*M);

	C = malloc(sizeof(float*)*M);
	for(int i=0; i < M; ++i) 
		C[i] = malloc(sizeof(float)*M);

	
	#pragma omp parallel for
	for(int i=0; i < M; ++i) {
		for(int j=0; j < N; ++j) {
			A[i][j] = i*j + 1; //rand();
			B[j][i] = i*100 + j; // rand();			
		}
	}

	#pragma omp parallel for
	for(int i=0; i < M; ++i) {
		for(int j=0; j < M; ++j) {
			C[j][i] = 0;
		}
	}


	
	
#define LOOP \
for(int i=0; i < M; ++i) { \
	for(int j=0; j < M; ++j) { \
		for(int k=0; k < N; ++k) { \
			C[i][j] += A[i][k] * B[k][j]; \
		} \
	}	\
}
	#pragma omp parallel
	{

#if defined _DYNAMIC
	//printf("# DYNAMIC SCHEDULE\n");
	#pragma omp for schedule(dynamic, CHUNK)
	LOOP
#elif defined _STATIC
	//printf("# STATIC SCHEDULE\n");
	#pragma omp for schedule(static,CHUNK)
	LOOP
#elif defined _GUIDED
	// printf("# GUIDED SCHEDULE\n");
	#pragma omp for schedule(guided)
	LOOP
#else
	//printf("# NO SCHEDULE\n");
	#pragma omp for
	LOOP
#endif

	}

	//printf("C: ");
	//print_raveled_mx(C, M, M);

	return 0;
}