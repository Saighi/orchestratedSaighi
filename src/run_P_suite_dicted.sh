#!/bin/sh
{
	. ./globalvars.sh

	TIME=21800
	SIZE=4096
	SIZE_EXT=1024
	NB_SEGMENT=10
	NB_PATTERN=4
	PATTERNSIZE=0.05
	PATTERNFREQUENCY=10
	SPARSITYPATTERN=0.75
	REFPAT=0.025

	#_dicted_pat_6h
	OUTDIR="/mnt/data1/data_paul/sim_less_stim_neurons_nocons_suite_6h_pat4_ref_0.1_spars0.75_size0.05_intspars0.07_wei0.52_wii0.06_wee0.8_wie0.06"
	LOADDIR="/mnt/data1/data_paul/sim_less_stim_neurons_nocons"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR

	#-inhomogeneous -maxrate 20 -minrate 0 -speedchange 5 -samplingvar 3
	python generate_spiketrains_final.py -rate 10  -timesim $TIME -nbneurons $SIZE_EXT \
	 -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -nbpattern $NB_PATTERN -patternsize $PATTERNSIZE \
	 -patternfrequency $PATTERNFREQUENCY -sparsitypattern $SPARSITYPATTERN -refpattern $REFPAT

	make -C $DIR -j8 sim_rc_p10c_P_dicted && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted \
		--load $LOADDIR/rf1 \
		--dir $OUTDIR \
		--prefix rf1 --size $SIZE --save \
		--wie 0.06 --wee 0.8 --wext 0.2 --wei 0.52 --wii 0.06 \
		--simtime $TIME --tauf $TAUF --taud $TAUD \
		--intsparse 0.07 \
		--extsparse 0.2 \
		--off 2.0 --on 1.0 \
		--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
		--input_spiketrains $SPIKETRAINS_FILE \
		--nb_segment $NB_SEGMENT \
		--size_ext $SIZE_EXT \
		--nocons
	exit
}