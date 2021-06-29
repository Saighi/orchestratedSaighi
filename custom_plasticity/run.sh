#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \
. ./globalvars.sh


OUTDIR="/mnt/data1/data_paul/custom_plas/sim_test"
SIMTIME=10
mkdir -p $OUTDIR
RATE=4
INIT_W


make -C $DIR -j8 sim_custom_plas && mpirun -n $NP $DIR/sim_custom_plas \
	--simtime $SIMTIME \
	--bgrate 4 \
	--init_w $INIT_W \
	--rate_one $RATE \
	--rate_two $RATE

cp $0 $OUTDIR