#!/bin/sh
{
	. ./globalvars.sh

	# TIME=2000
	# SIZE=4096
	# SIZE_EXT=1026
	# NB_SEGMENT=2
	# RATE=10


	# OUTDIR="/mnt/data1/data_paul/sim_noplas_rec"
	# SPIKETRAINS_FILE="spiketrains"
	# mkdir -p $OUTDIR

	# python generate_spiketrains_final.py -rate $RATE -timesim $TIME -nbneurons $SIZE_EXT -nbsegment $NB_SEGMENT -outdir $OUTDIR
	TIME=1200
	SIZE=4096
	SIZE_EXT=4096
	NB_SEGMENT=1
	NB_PATTERN=2
	PATTERNSIZE=0.1
	PATTERNFREQUENCY=4
	SPARSITYPATTERN=1
	REFPAT=0.05

	#_dicted_pat_6h
	OUTDIR="/mnt/data1/data_paul/sim_fourrier_frec_noplas_rec_more_wee_more_wse"
	SPIKETRAINS_FILE="spiketrains"
	mkdir -p $OUTDIR

	#-inhomogeneous -maxrate 10 -minrate 0 -speedchange 5 -samplingvar 3
	python generate_spiketrains_final.py -rate 10  -timesim $TIME -nbneurons $SIZE_EXT \
	 -nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -nbpattern $NB_PATTERN -patternsize $PATTERNSIZE \
	 -patternfrequency $PATTERNFREQUENCY -sparsitypattern $SPARSITYPATTERN -refpattern $REFPAT


	make -C $DIR -j8 sim_rc_p10c_P_dicted_noplas_rec && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_noplas_rec \
		--dir $OUTDIR \
		--prefix rf1 --size $SIZE --save \
		--wie 0.08 --wee 0.3 --wext 0.2 --wei 0.72 --wii 0.08 \
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