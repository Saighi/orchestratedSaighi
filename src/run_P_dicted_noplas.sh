#!/bin/sh
{
	. ./globalvars.sh

	TIME=21800
	SIZE=4096
	NB_SEGMENT=10
	SIZE_EXT=1024

	OUTDIR="/mnt/data1/data_paul/sim_less_stim_neuron_nocons_noplas"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR

	#python generate_spiketrains.py 10 0 20 5 $TIME 3 $SIZE $NB_SEGMENT $OUTDIR false
	python generate_spiketrains_final.py -rate 10 -timesim $TIME -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR

	make -C $DIR -j8 sim_rc_p10c_P_dicted_noplas && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_noplas \
		--dir $OUTDIR \
		--prefix rf1 --size $SIZE --save \
		--wie 0.08 --wee 0.1 --wext 0.2 --wei 0.72 --wii 0.08 \
		--simtime $TIME --tauf $TAUF --taud $TAUD \
		--intsparse 0.05 \
		--extsparse 0.2 \
		--off 2.0 --on 1.0 \
		--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
		--input_spiketrains $SPIKETRAINS_FILE \
		--nb_segment $NB_SEGMENT \
		--size_ext $SIZE_EXT \
		--nocons

	exit
}