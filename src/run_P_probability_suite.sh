#!/bin/sh
{
	. ./globalvars.sh

	TIME=10000
	SIZE=4096
	SIZE_EXT=1024
	NB_SEGMENT=5
	RATE=10
	NB_PATTERN=1
	PATTERNSIZE=0.1
	PATTERNFREQUENCY=4
	SPARSITYPATTERN=1
	REFPAT=0.05
	VARRATE=$RATE
	FREQUENCY=40
	SAMPLINGRATE=10
	MEMBRANTIME=0.001

	OUTDIR="/mnt/data1/data_paul/sim_probability_pat_10Hz_0.001jit"
	LOADDIR="/mnt/data1/data_paul/sim_10Hz_nopattern_morcoext"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR
	#python generate_spiketrains_probability_use.py -membrantime $MEMBRANTIME -oscillation -varrate $VARRATE -frequency $FREQUENCY -samplingrate $SAMPLINGRATE -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -timepattern $PATTERNSIZE -nbpattern $NB_PATTERN -sparsitypattern $SPARSITYPATTERN -patternfrequency $PATTERNFREQUENCY -refpattern $REFPAT
	python generate_spiketrains_probability_use.py -membrantime $MEMBRANTIME -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -timepattern $PATTERNSIZE -nbpattern $NB_PATTERN -sparsitypattern $SPARSITYPATTERN -patternfrequency $PATTERNFREQUENCY -refpattern $REFPAT

	#python generate_spiketrains_probability_use2.py -membrantime $MEMBRANTIME -oscillation -varrate $VARRATE -frequency $FREQUENCY -samplingrate $SAMPLINGRATE -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -timepattern $PATTERNSIZE -nbpattern $NB_PATTERN -sparsitypattern $SPARSITYPATTERN -patternfrequency $PATTERNFREQUENCY -refpattern $REFPAT

	make -C $DIR -j8 sim_rc_p10c_P_dicted && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted \
		--load $LOADDIR/rf1 \
		--dir $OUTDIR \
		--prefix rf1 --size $SIZE --save \
		--wie 0.08 --wee 0.1 --wext 0.05 --wei 0.72 --wii 0.08 \
		--simtime $TIME --tauf $TAUF --taud $TAUD \
		--intsparse 0.05 \
		--extsparse 0.8 \
		--off 2.0 --on 1.0 \
		--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
		--input_spiketrains $SPIKETRAINS_FILE \
		--nb_segment $NB_SEGMENT \
		--size_ext $SIZE_EXT \
		--nocons
		#--consolidateweights


	exit
}