#!/bin/sh

#--recfile $RECFILE
#--monf ./data/rf1.pat \
. ./globalvars.sh

OUTDIR="/users/nsr/saighi/data/sim_network/sim"
SIMTIME=1000

make -C $DIR -j8 $BIN && mpirun -n $NP $DIR/$BIN \
	--dir $OUTDIR \
	--prefix rf1 --size 4096 --save \
	--xi $XI \
	--stim ./data/shapes.pat \
	--wie 0.2 --wee 0.1 --wext 0.5 \
	--simtime $SIMTIME --tauf $TAUF --taud $TAUD \
	--intsparse $INTSPARSENESS \
	--extsparse 0.10 \
	--off 2.0 --on 1.0 \
	--beta $BETA --eta $ETA --bgrate 10 --scale $SCALE --weight_a $WEIGHTA --alpha 10 --delta 0.02