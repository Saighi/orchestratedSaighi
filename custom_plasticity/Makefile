# Update the following lines to your system requirements
AURYNPATH=$(HOME)/auryn

# Path to Auryn include files (auryn.h)
AURYNINC=$(AURYNPATH)/src

# Path to Auryn library (libauryn.a)
AURYNLIB=$(AURYNPATH)/build/release/src



# The following should not require updating in most cases 
CC = mpic++
CFLAGS=-ansi -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I$(AURYNINC) -I/usr/local/include
LDFLAGS=$(AURYNLIB)/libauryn.a -lboost_filesystem -lboost_system -lboost_program_options -lboost_mpi -lboost_serialization 
SIMFILES=sim_custom_plas sim_one_BMC_2
OBJFILES=../src/GlobalPFConnection.o P10Connection.o Connection_BMC.o

.SECONDARY:

simfiles: $(SIMFILES)

sim_%: sim_%.o $(OBJFILES) 
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $(subst .o,,$<)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *~ *.o core a.out *.log *.a $(SIMFILES)

