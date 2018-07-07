#!/bin/bash

mkdir ./data/SeqOneCardPokerWNHDMiniExp_10_10_1/
for ((i=1;i<5;i+=1))
do
	for ((j=1;j<11;j+=1))
	do
		fname=./data/SeqOneCardPokerWNHDMiniExp_10_10_1/data$i-$j.h5
		if [ -f $fname ]; then
			echo $fname exists already
			continue
		fi
		echo Generating dataset $i-$j
		echo $fname
		python -m src.poker.poker_datagen \
			--type OneCardPokerWNHD \
			--nCards 4\
			--initial_bet 10.0\
			--raiseval 10.0\
			--dist dirichlet\
			--stdev 1.0\
			--nSamples 10000 \
			--seed $i \
			--savePath ./data/SeqOneCardPokerWNHDMiniExp_10_10_1/data$i-$j.h5
	done
done

