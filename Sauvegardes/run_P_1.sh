#!/bin/sh

. ./globalvars.sh

make -C $DIR
./sim_preparation --bgrate 30 --pattern_rate 4 --time_pattern 0.1 --nb_neurons 4096 --dir $OUTDIR --simtime 7200 --begin 3600 #Dans global

make -C $DIR -j8 sim_rc_p10c_P && mpirun -n $NP $DIR/sim_rc_p10c_P \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--recfile $RECFILE --xi $XI \
	--wie 0.2 --wee 0.1 --wext  0.05 \
	--simtime 720 --tauf $TAUF --taud $TAUD \
	--intsparse $INTSPARSENESS \
	--extsparse 0.10 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate 30 --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02
