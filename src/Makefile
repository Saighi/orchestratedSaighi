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
SIMFILES=sim_rc_P_simple sim_one_ex_inh_neuron sim_one_excitatory_neuron sim_rc_P_simple_working sim_rc_p10c_P sim_rc_p10c_P_dicted sim_rc_p10c sim_rc_p10d sim_rc_p11 sim_rc_p12 sim_one sim_preparation sim_rc_p10c_P_dicted_noplas sim_rc_p10c_P_dicted_noplas_at_all
OBJFILES=P10Connection.o P11Connection.o P12Connection.o GlobalPFConnection.o P10ConnectionP.o P10Connection_delayed.o P10Connection_nofas.o
#OBJFILES=P10Connection.o P11Connection.o P12Connection.o GlobalPFConnection.o P10ConnectionP.o P10Connection_delayed.o P10Connection_nocons_dirty.o

.SECONDARY:

simfiles: $(SIMFILES)

sim_%: sim_%.o $(OBJFILES) 
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $(subst .o,,$<)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *~ *.o core a.out *.log *.a $(SIMFILES)

