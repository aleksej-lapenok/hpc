
CFLAGS=-g -O0 -fopenmp 
LDLIBS=-lm

all: $(EXE)

clean:
	@rm -f *.o $(EXE)

example1: $(OBJS) example1.o
	$(CC) $(CFLAGS) $^ -lm -o $@

# example2: $(OBJS) example2.o
# 	$(CC) $(CFLAGS) $^ -lm -o $@