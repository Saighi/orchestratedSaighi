#!/bin/sh
{
	. ./globalvars.sh

	TIME=2000
	SIZE=4096
	SIZE_EXT=4096
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
	MEMBRANTIME=0.01


	OUTDIR="/mnt/data1/data_paul/return_old/sim_return"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR 
	#python generate_spiketrains_probability_use.py -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -timepattern $PATTERNSIZE -nbpattern $NB_PATTERN -sparsitypattern $SPARSITYPATTERN -patternfrequency $PATTERNFREQUENCY -refpattern $REFPAT
 	#python generate_spiketrains_probability_use2_shift.py -shift -rateshift 4 -startshift 1000 -membrantime $MEMBRANTIME -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR 
 	python generate_spiketrains_probability_use.py -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR 
	#python generate_spiketrains_probability_use.py -membrantime $MEMBRANTIME -oscillation -varrate $VARRATE -frequency $FREQUENCY -samplingrate $SAMPLINGRATE -rate $RATE -timesim $TIME -samplingrate 10 -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR


	make -C $DIR -j8 sim_rc_p10c_P_dicted_noplas_rec && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_noplas_rec \
		--dir $OUTDIR \
		--prefix rf1 --size $SIZE --save \
		--wie 0.2 --wee 0.1 --wext 0.2 \
		--simtime $TIME --tauf $TAUF --taud $TAUD \
		--intsparse 0.05 \
		--extsparse 0.05 \
		--off 2.0 --on 1.0 \
		--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
		--input_spiketrains $SPIKETRAINS_FILE \
		--nb_segment $NB_SEGMENT \
		--size_ext $SIZE_EXT \
		--nocons
		#--consolidateweights

	exit
}