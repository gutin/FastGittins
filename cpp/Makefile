CC = g++
CFLAGS=  -O3 -fopenmp -Wall -I ./boost_1_57_0/
CFLAGS_WO_OM =  -O3 -Wall -I ./boost_1_57_0/
LFLAGS = -L ./boost_1_57_0/stage/lib/ -lboost_program_options 
LFLAGS_WO_PO = -L ./boost_1_57_0/stage/lib/ 
OUTNAME = tom_exp
OUTNAME_RUNNER = tom_long
#TYPE = -DRANDARMS=1 -DSIMMULT=1 -DSIMIDS=1
#TYPE =  -DRANDARMS=1 -DSIMSHD=1 -DSIMTOM=1 -DSIMBUCB=1 -DSIMIDS=1 -DSIMUCB1=1 -DSIMNEW=1
#TYPE = -DSIMIDS=1 -DSIMSHD=1 -DSIMTOM=1 -DSIMBUCB=1 -DSIMUCB1=1 -DSIMNEW=1
TYPE = -DRANDARMS -DSIMSHD=1 -DSIMTOM=1 #-DSIMUCB1=1
#COMP_TYPE = -DSIMIDS=1 -DSIMIDS2=1

all: mab

stats.o : stats.cpp stats.hpp
	$(CC) $(CFLAGS_WO_OM) -c stats.cpp 

tex_table_writer.o : tex_table_writer.hpp tex_table_writer.cpp
	$(CC) $(CFLAGS_WO_OM) -c tex_table_writer.cpp

simulate.o : simulate.cpp simulate.hpp policy.hpp helpers.hpp
	$(CC) $(CFLAGS) -c simulate.cpp $(TYPE)

simulate_s.o : simulate.cpp simulate.hpp policy.hpp helpers.hpp
	$(CC) $(CFLAGS_WO_OM) -c simulate.cpp $(TYPE) -o simulate_s.o

simulate_c.o : simulate.cpp simulate.hpp policy.hpp helpers.hpp
	$(CC) $(CFLAGS) -c simulate.cpp $(COMP_TYPE)

mab.o : mab.cpp 
	$(CC) $(CFLAGS) -c mab.cpp


clean :
	rm *.o $(OUTNAME) $(OUTNAME_RUNNER)

test : test.cpp policy.hpp helpers.hpp
	$(CC) -g  -Wall -I ./boost_1_57_0/ test.cpp -o test

mab : mab.o stats.o simulate.o tex_table_writer.o
	$(CC) $(CFLAGS) $(LFLAGS) simulate.o mab.o stats.o tex_table_writer.o -o $(OUTNAME)

comp : mab.o stats.o simulate_c.o
	$(CC) $(CFLAGS) $(LFLAGS) simulate.o mab.o stats.o -o comp

runner.o : runner.cpp
	$(CC) $(CFLAGS_WO_OM) $(LFLAGS_WO_PO) -c runner.cpp

runner : runner.o stats.o simulate_s.o
	$(CC) $(CFLAGS) $(LFLAGS_WO_PO) simulate_s.o runner.o stats.o -o $(OUTNAME_RUNNER)
