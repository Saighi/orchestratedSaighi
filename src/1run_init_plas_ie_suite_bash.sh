#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
#--stim ./data/shapes.pat \
{
	. ./globalvars.sh

	NB_PATTERN=2
	PATTERNSIZE=0.1
	PATTERNFREQUENCY=5
	SPARSITYPATTERN=1
	REFPAT=0.05
	BASHSIZE=10000
	OUTDIR="/mnt/data1/data_paul/new_formula_plas_inh/sim_2pat_oscillating_0.1_bash10000_20h"
	SIMTIME=76000
	SPIKETRAINS_FILE="spiketrains"
	LOADDIR="/mnt/data1/data_paul/new_formula_plas_inh/sim_oscillating"
	mkdir -p $OUTDIR

	cp $0 $OUTDIR

	NB_SEGMENT=20
	RATE=4

	mkdir -p $OUTDIR


	python generate_spiketrains_final_bash.py -rate $RATE -timesim $SIMTIME -nbneurons 2048 \
		-nbsegment $NB_SEGMENT -outdir $OUTDIR -pattern -nbpattern $NB_PATTERN -patternsize $PATTERNSIZE \
		-patternfrequency $PATTERNFREQUENCY -sparsitypattern $SPARSITYPATTERN -refpattern $REFPAT -bashsize $BASHSIZE


	make -C $DIR -j8 sim_rc_p10c_P_dicted_plas_inh && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted_plas_inh \
		--load $LOADDIR/rf1 \
		--dir $OUTDIR \
		--prefix rf1 --size 2048 --save \
		--xi $XI \
		--wee 0.20 --wext 0.20 --wei 1 --wii 0.08 --wie 0.1 \
		--simtime $SIMTIME --tauf $TAUF --taud $TAUD \
		--intsparse 0.1 \
		--extsparse 0.2 \
		--off 2.0 --on 1.0 \
		--beta $BETA --eta $ETA --bgrate 4 --scale $SCALE --weight_a $WEIGHTA --alpha 1 --delta 0.02 \
		--nocons \
		--input_spiketrains $SPIKETRAINS_FILE \
		--nb_segment $NB_SEGMENT



	exit
}