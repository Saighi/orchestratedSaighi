#!/bin/sh
# --stim ./data/shapes.pat \
# --monf ./data/rf1.pat \
#--recfile $RECFILE --xi $XI \

OUTDIR="$HOME/data/sim_network/sim"
mkdir -p $OUTDIR

. ./globalvars.sh

make -C $DIR -j8 $BIN && mpirun -n $NP $DIR/$BIN \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--wie 0.2 --wee 0.1 --wext 0.2 \
	--simtime $SIMTIME --tauf $TAUF --taud $TAUD \
	--intsparse $INTSPARSENESS \
	--extsparse 0.05 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate 10 --scale $SCALE --weight_a $WEIGHTA --alpha $ALPHA --delta 0.02

