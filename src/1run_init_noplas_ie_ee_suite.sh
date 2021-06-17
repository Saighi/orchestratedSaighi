#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \
. ./globalvars.sh

NB_PATTERN=1
PATTERNSIZE=0.1
PATTERNFREQUENCY=5
SPARSITYPATTERN=1
REFPAT=0.05
OUTDIR="/mnt/data1/data_paul/new_formula/sim_nocons_dirty_1pat_oscillating"
SIMTIME=38000
SPIKETRAINS_FILE="spiketrains"
LOADDIR="/mnt/data1/data_paul/new_formula/sim_nocons_dirty_oscillating"
mkdir -p $OUTDIR

NB_SEGMENT=10
RATE=4

mkdir -p $OUTDIR


python generate_spiketrains_final.py -rate $RATE -timesim $SIMTIME -nbneurons 2048 \
	-nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -nbpattern $NB_PATTERN -patternsize $PATTERNSIZE \
	-patternfrequency $PATTERNFREQUENCY -sparsitypattern $SPARSITYPATTERN -refpattern $REFPAT


make -C $DIR -j8 sim_rc_p10c_P_dicted_noplas_rec_inh && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_noplas_rec_inh \
	--load $LOADDIR/rf1 \
	--dir $OUTDIR \
	--prefix rf1 --size 2048 --save \
	--xi $XI \
	--wee 0.15 --wext 0.15 --wei 0.4 --wii 0.1 --wie 0.7 \
	--simtime $SIMTIME --tauf $TAUF --taud $TAUD \
	--intsparse 0.1 \
	--extsparse 0.2 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate 4 --scale $SCALE --weight_a $WEIGHTA --alpha 1 --delta 0.02 \
	--nocons \
	--input_spiketrains $SPIKETRAINS_FILE \
	--nb_segment $NB_SEGMENT


cp $0 $OUTDIR