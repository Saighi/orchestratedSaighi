#!/bin/sh
. ./globalvars.sh

TIME=100
BG=10
OUTDIR="$HOME/data/sim_network/sim_10Hz_simple_se0.2"

mkdir -p $OUTDIR


# make -C $DIR
# ./sim_preparation --bgrate $BG --begin 5 --pattern_rate 6 \
#                   --time_pattern 0.05 --nb_neurons 4096 --dir $OUTDIR --simtime $TIME --nb_pattern $NB_PATTERN

make -C $DIR -j8 sim_rc_p10c_P_simple && mpirun -n $NP $DIR/sim_rc_p10c_P_simple \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--wie 0.2 --wee 0.1 --wext 0.2 \
	--simtime $TIME --tauf $TAUF --taud $TAUD \
	--intsparse $INTSPARSENESS \
	--extsparse 0.05 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate $BG --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02
