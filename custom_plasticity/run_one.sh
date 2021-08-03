#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \


NB_STIM=1000
SIMTIME=2000
OUTDIR="/mnt/data1/data_paul/custom_plas/one_neuron/test_28jul"
mkdir -p $OUTDIR

# python ../src/generate_spiketrains_final.py -rate 4 -timesim $SIMTIME -nbneurons $NB_STIM \
# 	-nbsegment 1 -outdir $OUTDIR 

python ../src/generate_spiketrains_final.py -rate 4 -timesim $SIMTIME -nbneurons $NB_STIM \
	-nbsegment 1 -outdir $OUTDIR -pattern -nbpattern 1 -patternsize 0.1 \
	-patternfrequency 4 -sparsitypattern 1 -refpattern 0.05 -starttime 500

make -C . -j8 sim_one_BMC_2 && ./sim_one_BMC_2 \
	--simtime $SIMTIME \
	--dir $OUTDIR \
	--nb_stim $NB_STIM \

cp $0 $OUTDIR

#--wee 0.20 --wext 0.20 --wei 1 --wii 0.08 --wie 0.1 \