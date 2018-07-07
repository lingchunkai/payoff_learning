#!/bin/bash


mkdir ./data/SeqSecurityGameOne_scale2_unifsuc/
for ((i=1;i<5;i+=1))
do
	for ((j=1;j<11;j+=1))
	do
		fname=./data/SeqSecurityGameOne_scale2_unifsuc/data$i-$j.h5
		if [ -f $fname ]; then
			echo $fname exists already
			continue
		fi
		echo Generating dataset $i-$j
		echo $fname
		python -m src.sec.security_datagen \
			--type SecurityGameOne \
			--nDef 5\
			--SecurityGameScale 2\
			--SecurityGameStaticRewards 1\
			--SecurityGameDefProbs 0.5,0.5,0.5,0.5\
			--nSamples 10000 \
			--seed $i \
			--savePath ./data/SeqSecurityGameOne_scale2_unifsuc/data$i-$j.h5
	done
done
