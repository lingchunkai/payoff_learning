#!/bin/bash

mkdir ./data/rps1_weights/
for ((i=1;i<5;i+=1))
do
	for ((j=1;j<11;j+=1))
	do
		fname=./data/rps1_weights/data$i-$j.h5
		if [ -f $fname ]; then
			echo $fname exists already
			continue
		fi
		echo Generating dataset $i-$j
		echo $fname
		python -m src.rps.rps_datagen \
			--type RCP_weights \
			--softmax 0\
			--size 3 3 \
			--nSamples 10000 \
			--nFeats 2 \
			--scale 10.0 \
			--seed $i \
			--savePath ./data/rps1_weights/data$i-$j.h5
	done
done

