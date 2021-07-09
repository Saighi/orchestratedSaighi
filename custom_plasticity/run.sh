#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \


OUTDIR="/mnt/data1/data_paul/custom_plas/sim_test"
SIMTIME=10
mkdir -p $OUTDIR
RATE=4
WEXT=0.05
WIXT=0.025
WIE=1
WEI=0.5
WEE=1
NB_STIM=16384
NB_EXC=1024


make -C . -j8 sim_custom_plas && ./sim_custom_plas \
	--simtime $SIMTIME \
	--dir $OUTDIR \
	--nb_stim $NB_STIM \
	--nb_exc $NB_EXC \
	--wext $WEXT \
	--wixt $WIXT \
	--wie $WIE \
	--wei $WEI \
	--wee $WEE \
	--sparseness_se 0.1 \
	--sparseness_ie 0.05 \
	--sparseness_ee 0.1 \
	--rate_one $RATE 

cp $0 $OUTDIR