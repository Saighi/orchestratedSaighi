#!/bin/sh
{
	. ./globalvars.sh

	TIME=1500
	SIZE=4096
	SIZE_EXT=1024
	NB_SEGMENT=1
	RATE=10
	NB_PATTERN=1
	PATTERNSIZE=0.1
	PATTERNFREQUENCY=3
	SPARSITYPATTERN=1
	REFPAT=0.05
	VARRATE=$RATE
	FREQUENCY=10
	SAMPLINGRATE=10

	OUTDIR="/mnt/data1/data_paul/sim_oscillation30_nopattern"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR
	#python generate_spiketrains_probability_use.py -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -timepattern $PATTERNSIZE -nbpattern $NB_PATTERN -sparsitypattern $SPARSITYPATTERN -patternfrequency $PATTERNFREQUENCY -refpattern $REFPAT
 	python generate_spiketrains_probability_use.py -oscillation -varrate $VARRATE -frequency $FREQUENCY -samplingrate $SAMPLINGRATE -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR 

	make -C $DIR -j8 sim_rc_p10c_P_dicted && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted \
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
		#--consolidateweights

	exit
}