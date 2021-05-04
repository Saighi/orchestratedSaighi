#!/bin/sh
. ./globalvars.sh

TIME=100

OUTDIR="$HOME/data/sim_network/sim_one_excitatory_neuron_many_input"

#python recurrent_zenke_one_neuron.py 1 one_neuron_1_many_input
#OUTDIR="$HOME/data/sim_network/sim_one_excitatory_neuron_many_input_smaller_time_step"
#python recurrent_zenke_one_neuron.py 1 one_neuron_smaller_time_step

mkdir -p $OUTDIR


make -C $DIR -j8 sim_one_excitatory_neuron && ./$DIR/sim_one_excitatory_neuron \
	--dir $OUTDIR \
	--prefix rf1 --size 1 --save \
	--wext 0.2 \
	--simtime $TIME --tauf $TAUF --taud $TAUD \
	--intsparse 1 \
	--extsparse 1 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
