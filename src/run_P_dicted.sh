#!/bin/sh
. ./globalvars.sh

TIME=7200

OUTDIR="$HOME/data/sim_network/sim_dicted_npat_2h"
SPIKETRAINS_DIR="$HOME/data/sim_network/sim_10Hz_cons_2h_npat_mrco/e_spikes.txt"
mkdir -p $OUTDIR



make -C $DIR -j8 sim_rc_p10c_P_dicted && mpirun -n $NP $DIR/sim_rc_p10c_P_dicted \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--wie 0.2 --wee 0.1 --wext  0.2 \
	--simtime $TIME --tauf $TAUF --taud $TAUD \
	--intsparse 0.1 \
	--extsparse 0.05 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
  --input_spiketrains $SPIKETRAINS_DIR
