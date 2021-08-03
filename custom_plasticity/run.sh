#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \


OUTDIR="/mnt/data1/data_paul/custom_plas/sim_test"
SIMTIME=200
mkdir -p $OUTDIR
RATE=4
WIXT=0.2
WIE=0.1 #0.1
WEI=1 #1 
WII=0.08
WEE=0.2
GAMMA_EXT=0.06 
SIGMA_EXT=0.3 
NB_STIM=16384
NB_EXC=1024

# python ../src/generate_spiketrains_final.py -rate $RATE -timesim $SIMTIME -nbneurons $NB_STIM \
# 	-nbsegment 1 -outdir $OUTDIR

make -C . -j8 sim_custom_plas && mpirun -n 4 ./sim_custom_plas \
	--simtime $SIMTIME \
	--dir $OUTDIR \
	--nb_stim $NB_STIM \
	--nb_exc $NB_EXC \
	--wixt $WIXT \
	--wie $WIE \
	--wei $WEI \
	--wii $WII \
	--wee $WEE \
	--gamma_ext $GAMMA_EXT \
	--sigma_ext $SIGMA_EXT \
	--sparseness_se 0.1 \
	--sparseness_ie 0.025 \
	--sparseness_si 0.05 \
	--sparseness_ee 0.1 \
	--rate_one $RATE 
	#-sparseness_ie 0.0025 \
	#--sparseness_si 0.05 \
cp $0 $OUTDIR

#--wee 0.20 --wext 0.20 --wei 1 --wii 0.08 --wie 0.1 \