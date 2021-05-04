#!/bin/sh
{
	. ./globalvars.sh

	TIME=5
	SIZE_EXT=4096
	NB_SEGMENT=2
	NB_PATTERN=1
	PATTERNSIZE=0.1
	PATTERNFREQUENCY=1
	SPARSITYPATTERN=0.80
	REFPAT=0.5

	#_dicted_pat_6h
	OUTDIR="."
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR

	#-inhomogeneous -maxrate 20 -minrate 0 -speedchange 5 -samplingvar 3
	#python generate_spiketrains.py 10 0 20 5 $TIME 3 $SIZE $NB_SEGMENT $OUTDIR false
	python generate_spiketrains_final_modified.py -rate 10  -timesim $TIME -nbneurons $SIZE_EXT \
	 -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -nbpattern $NB_PATTERN -patternsize $PATTERNSIZE \
	 -patternfrequency $PATTERNFREQUENCY -sparsitypattern $SPARSITYPATTERN -refpattern $REFPAT

	exit
}