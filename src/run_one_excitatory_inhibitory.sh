#!/bin/sh
. ./globalvars.sh

TIME=5

OUTDIR="$HOME/data/sim_network/sim_one_ex_inh_neuron_many_input"


mkdir -p $OUTDIR


make -C $DIR -j8 sim_one_ex_inh_neuron && ./$DIR/sim_one_ex_inh_neuron \
	--dir $OUTDIR \
	--prefix rf1 --size 1 --save \
	--wie 0.2 --wext 0.2 \
	--simtime $TIME --tauf $TAUF --taud $TAUD \
	--intsparse 1 \
	--extsparse 1 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
