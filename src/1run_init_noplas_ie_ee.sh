#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \
. ./globalvars.sh


OUTDIR="/mnt/data1/data_paul/new_formula/sim_10HZ"
SPIKETRAINS_FILE="spiketrains"
SIMTIME=300

NB_SEGMENT=1
RATE=10

mkdir -p $OUTDIR

#--wie 0.7 --wee 0.12 --wext 0.12 --wei 0.4 --wii 0.1  \ #modifié 
#python generate_spiketrains_final.py -rate $RATE -timesim $SIMTIME -nbneurons 2048 -nbsegment $NB_SEGMENT -outdir $OUTDIR

make -C $DIR -j8 sim_rc_p10c_P_dicted_noplas_rec_inh && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_noplas_rec_inh \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--xi $XI \
	--wie 0.08 --wee 0.1 --wext 0.2 --wei 0.72 --wii 0.08 \
	--simtime $SIMTIME --tauf $TAUF --taud $TAUD \
	--intsparse 0.05 \
	--extsparse 0.1 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate 4 --scale $SCALE --weight_a $WEIGHTA --delta 0.02 --alpha $ALPHA\
	--nocons \
	--input_spiketrains $SPIKETRAINS_FILE \
	--nb_segment $NB_SEGMENT