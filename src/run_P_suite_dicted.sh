#!/bin/sh
{
	. ./globalvars.sh

	TIME=38000
	SIZE=4096
	SIZE_EXT=1024
	NB_SEGMENT=10
	NB_PATTERN=2
	PATTERNSIZE=0.1
	PATTERNFREQUENCY=5
	SPARSITYPATTERN=1
	REFPAT=0.05
	RATE=10

	OUTDIR="/mnt/data1/data_paul/different_Hz/sim_less_stim_neurons_nocons_corrected_2dif_10h45"
	LOADDIR="/mnt/data1/data_paul/Sim_old/USEFUL_sim_less_stim_neurons_nocons"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR

	#-inhomogeneous -maxrate 20 -minrate 0 -speedchange 5 -samplingvar 3
	python generate_spiketrains_final.py -rate $RATE -timesim $TIME -nbneurons $SIZE_EXT \
	-nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -nbpattern $NB_PATTERN -patternsize $PATTERNSIZE \
	-patternfrequency $PATTERNFREQUENCY -sparsitypattern $SPARSITYPATTERN -refpattern $REFPAT

	make -C $DIR -j8 sim_rc_p10c_P_dicted && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted \
		--load $LOADDIR/rf1 \
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
