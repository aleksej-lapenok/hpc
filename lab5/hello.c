#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <stdbool.h>

#ifndef MSG_LEN
# define MSG_LEN 32
#endif


#ifndef SEND_FN
# define SEND_FN MPI_Send
#endif

#if !defined(SYNC) && !defined(SEND_RECV) && !defined(ASYNC)
# define SYNC
#endif



void rand_str(char *str, size_t len) 
{
	for(size_t i = 0; i < len - 1; ++i) {
		str[i] = rand() % 26 + 64;
	}
	str[len] = 0;
}

int main(int argc,char **argv)
{
	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	MPI_Request req;
	MPI_Status status;
	bool wait = false;
	srand(rank+10);

	char buf[MSG_LEN], rbuf[MSG_LEN];

#ifdef SYNC
	printf("SYNC\n");
#endif

#ifdef SEND_RECV
	printf("SEND_RECV\n");
#endif

#ifdef ASYNC
	printf("ASYNC\n");
#endif

	for(size_t i = 0; i < 10; ++i) {
			
#ifdef SYNC
		if( (i + rank) % 2 == 0 ) {
			MPI_Recv(buf, MSG_LEN, MPI_CHAR, !rank, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			printf("%*sRECV(%d) : %s\n",rank*44, " ", rank, buf);
		} else {
			rand_str(buf, MSG_LEN);
			printf("%*sSEND(%d) : %s\n", rank*44, " ", rank, buf);
			SEND_FN(buf, MSG_LEN, MPI_CHAR, !rank, 0, MPI_COMM_WORLD);
		}
#endif

#ifdef SEND_RECV
		rand_str(buf, MSG_LEN);
		printf("%*sSEND(%d) : %s\n", rank*44, " ", rank, buf);
		MPI_Sendrecv(buf, MSG_LEN, MPI_CHAR, !rank, 0, 
						rbuf, MSG_LEN, MPI_CHAR, !rank, 0, 
							MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%*sRECV(%d) : %s\n",rank*44, " ", rank, rbuf);
#endif	


#ifdef ASYNC	

		if( (i + rank) % 2 == 0 ) {
			if(wait) {
				MPI_Wait(&req, &status);
				MPI_Irecv(buf, MSG_LEN, MPI_CHAR, !rank, 0, MPI_COMM_WORLD, &req);
				wait = true;
				printf("%*sRECV(%d) : %s\n",rank*44, " ", rank, buf);
			}
		} else {
			rand_str(buf, MSG_LEN);
			printf("%*sSEND(%d) : %s\n", rank*44, " ", rank, buf);
			if(wait) MPI_Wait(&req, &status);
			MPI_Isend(buf, MSG_LEN, MPI_CHAR, !rank, 0, MPI_COMM_WORLD, &req);
			wait = true;
		}
#endif	
		// sleep(rand() % 5);
	}

	MPI_Finalize();	
}
