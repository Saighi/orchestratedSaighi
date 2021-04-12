#!/bin/sh
. ./globalvars.sh

TIME=14400
BG=10
NB_PATTERN=1

LOADDIR="$HOME/data/sim_network/sim_10Hz_cons_2h_npat_mrco"
OUTDIR="$HOME/data/sim_network/sim_10Hz_cons_2h_1pat_mrco_7_1.5ms"
mkdir -p $OUTDIR


make -C $DIR
./sim_preparation --bgrate $BG --begin 100 --pattern_rate 2 \
                  --time_pattern 0.15 --nb_neurons 4096 --dir $OUTDIR --simtime $TIME --nb_pattern $NB_PATTERN

make -C $DIR -j8 sim_rc_p10c_P && mpirun -n $NP $DIR/sim_rc_p10c_P \
	--load $LOADDIR/rf1 \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--wie 0.2 --wee 0.1 --wext  0.2 \
	--simtime $TIME --tauf $TAUF --taud $TAUD \
	--intsparse 0.1 \
	--extsparse 0.05 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate $BG --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02 \
    --nb_pattern $NB_PATTERN
