# What game are we playing? End-to-end learning in normal and extensive form games
This repo contains the code for reproducing experiments for [differentiably learning 2-player zero-sum games](https://arxiv.org/abs/1805.02777) 
(to appear in IJCAI 2018).

There are 4 main experiments.
1. One-card-poker with 4 cards, abbreviated as `OCP(4)`
2. Security game with a time horizon of 1, abbreviated as `secgameone`
3. Security game with a time horizon of 2, abbreviated as `secgame`
4. Rock-paper-scissors, abbreviated as `RPS`.

In order to reduce clutter, I have tried to remove all traces of other experiments (e.g. other card distributions
for poker) and auxillary scripts used to debug the solver. 

Any feedback or bug reports are welcome.

## Installation:
```shell
cd path/to/repo/
conda env create -f environment.yml -p path/to/repo/paynet_env
source activate paynet_env
```

## Generate dataset
```shell
./seq_datagen_WNHD.sh
./seq_datagen_security.sh
./seq_datagen_security_one.sh
./datagen_rps.sh
```
This will need some time, the data-generation code is not quite optimized.

## Run experiments
1. Ensure that the relevant datasets have been generated.
2. Add the relevant folders to PYTHONPATH.
```shell
export PYTHONPATH=$PYTHONPATH:/run_expts_poker
export PYTHONPATH=$PYTHONPATH:/run_expts_sec
export PYTHONPATH=$PYTHONPATH:/run_expts_rps
```
3. Run the following scripts. Each may be run concurrently. 
The slowest-running dataset should be OCP(4).

#### OCP(4)
```shell
python ./run_expts_poker/bsize128_expt_console_WNHD_dsize500_mini_SLOWrmsprop_BATCH.py
python ./run_expts_poker/bsize128_expt_console_WNHD_dsize1000_mini_SLOWrmsprop_BATCH.py
python ./run_expts_poker/bsize128_expt_console_WNHD_dsize2000_mini_SLOWrmsprop_BATCH.py
python ./run_expts_poker/bsize128_expt_console_WNHD_dsize5000_mini_SLOWrmsprop_BATCH.py
```

#### Security Game, t=1
```shell
python ./run_expts_sec/expt_console_secgame_dsize100_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgame_dsize200_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgame_dsize500_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgame_dsize1000_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgame_dsize2000_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgame_dsize5000_mini_SLOWrmsprop_BATCH_vonly.py
```

#### Security Game, t=2
```shell
python ./run_expts_sec/expt_console_secgameone_dsize200_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgameone_dsize500_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgameone_dsize1000_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgameone_dsize2000_mini_SLOWrmsprop_BATCH_vonly.py
python ./run_expts_sec/expt_console_secgameone_dsize5000_mini_SLOWrmsprop_BATCH_vonly.py
```

#### Rock Paper Scissors
```shell
python ./run_expts_rps/rps_500.py
python ./run_expts_rps/rps_1000.py
python ./run_expts_rps/rps_2000.py
python ./run_expts_rps/rps_5000.py
```

## Extract statistics and plot figures
1. Open jupyter notebook in the root folder.
```shell
jupyter notebook
```
2. Open and run the appropriate notebook. Take care to point the target folder to the saved results. The notebooks are:
	1. vis-OCP.ipynb
	2. vis-rps.ipynb
	3. vis-secgame.ipynb
	4. vis-secgameone.ipynb

## General questions and issues
##### How large is the training data?
The numbers {500, 1000, 2000, 5000} are the sizes of the *entire*
dataset. 
By default, we use a 70-30 split, which means that a dataset of 1000 contains
just 700 training samples. 
##### What and how are the metrics computed?
For most of the experiments, we take the MSE over parameters. Because the parameter values can fluctuate quite
a bit throughout the course of training, we return the model which gives us the best testing *log-loss*.

Furthermore, we are taking the position that each experiment is run 
independently of the rest, as is typical in practice. Because of the 70-30 split, the test sets are of different
sizes, which depends on the overall size of the dataset. 
Strictly speaking, it is not exactly
appropriate to compare MSEs over different test sets for evaluation. However, we believe 
that the overall trend should still remain similar to what was reported.
##### Why do the experiments for RPS run the slowest when the payoff matrix is much smaller?
For OCP and security games, there are no features 
, i.e. the payoff matrix is fixed across datapoints (in practice, we set the feature to be a single constant with value 1). 
Thus, solutions for the QRE may be reused within a single minibatch, dramatically improving runtime.
For RPS, we need to solve for the QRE for each game individually, which is much more 
time consuming despite the small game matrix.
##### What does WNHD in one-card-poker stand for?
[Wallenius' noncentral hypergeometric distribution](https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution). 
(Very) informally, this is a distribution of weighted balls
without replacement.
##### Why use WNHD instead of actual card distributions from a standard deck (allowing a varying number of each card)?
This was indeed experimented with, and in general, the joint *distribution* of cards was learned accurately.
This was verified using the MSE of the P matrix itself, or the KL divergence of the joint distribution.
However, the MSE of *parameters* was extremely high, 
and it was observed that the learned total number of cards becomes extremely large. 
This is because of the near-identifiablity issues 
discussed in the paper; it is possible for 2 very different parameterizations (in terms of MSE) to give nearly the same 
distribution of cards.  
##### GPU support?
We only support CPU cycles for now.

## Known issues
##### Random seeds
Unfortunately, when generating data, we do not have a random seed for the sampled actions of each player. This means that
the exact dataset being generated will differ each time. However, the parameters to be learned *are* seeded appropriately.
Hence, this should not cause a huge discrepency in the results as long as enough trials are run.
