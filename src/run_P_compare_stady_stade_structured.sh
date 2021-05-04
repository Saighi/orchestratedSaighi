#!/bin/sh
. ./globalvars.sh

TIME=21800
BG=10

#OUTDIR="$HOME/data/sim_network/sim_stady_state_wii0.08_wie0.08_wei0.72_wext0.2"
OUTDIR="/mnt/data1/data_paul/sim_stady_state_wii0.08_wie0.08_wei0.72_structured"
#OUTDIR="$HOME/data/sim_network/sim_stady_state_wii0.08_wie0.08_wei0.72_suite_2"
#LOADDIR="/mnt/data2/paul_data/Auryn_archives/sim_stady_state_wii0.08_wie0.08_wei0.72"
# --load $LOADDIR/rf1 \
# --noconsolidation

mkdir -p $OUTDIR


# make -C $DIR
# ./sim_preparation --bgrate $BG --begin 5 --pattern_rate 6 \
#                   --time_pattern 0.05 --nb_neurons 4096 --dir $OUTDIR --simtime $TIME --nb_pattern $NB_PATTERN

make -C $DIR -j8 sim_rc_p10c && mpirun -n $NP $DIR/sim_rc_p10c \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--wie 0.08 --wee 0.1 --wext 0.2 --wei 0.72 --wii 0.08 \
	--simtime $TIME --tauf $TAUF --taud $TAUD \
	--intsparse 0.05 \
	--extsparse 0.05 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate $BG --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02