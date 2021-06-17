#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \
. ./globalvars.sh


OUTDIR="/mnt/data1/data_paul/new_formula_plas_inh/sim_oscillating_more"
SPIKETRAINS_FILE="spiketrains"
SIMTIME=1000

NB_SEGMENT=1
RATE=4

mkdir -p $OUTDIR


python generate_spiketrains_final.py -rate $RATE -timesim $SIMTIME -nbneurons 2048 -nbsegment $NB_SEGMENT -outdir $OUTDIR

make -C $DIR -j8 sim_rc_p10c_P_dicted_plas_inh.cpp && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_plas_inh.cpp \
	--dir $OUTDIR \
	--prefix rf1 --size 2048 --save \
	--xi $XI \
	--wee 0.15 --wext 0.15 --wei 1.2 --wii 0.08 --wie 0.1 \
	--simtime $SIMTIME --tauf $TAUF --taud $TAUD \
	--intsparse 0.1 \
	--extsparse 0.2 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate 4 --scale $SCALE --weight_a $WEIGHTA --alpha 1 --delta 0.02 \
	--nocons \
	--input_spiketrains $SPIKETRAINS_FILE \
	--nb_segment $NB_SEGMENT

cp $0 $OUTDIR