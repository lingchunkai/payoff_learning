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
```

## Activate virtualenv
```shell
source activate paynet_env
```

## Generate dataset
```shell
./scripts/generate_data/datagen_rps.sh
./scripts/generate_data/seq_datagen_scurity_one.sh
./scripts/generate_data/seq_datagen_security_one.sh
./scripts/generate_data/seq_datagen_security.sh
./scripts/generate_data/seq_datagen_WNHD.sh
```
This will need some time, the data-generation code is not quite optimized.

## Run experiments
1. Ensure that the relevant datasets have been generated.
2. Make sure that the virtualenv has been activated.
3. Run the following scripts. Each may be run concurrently. 
The slowest-running dataset should be OCP(4).

#### OCP(4)
```shell
./scripts/run/OCP.sh
```

#### Security Game, t=1
```shell
./scripts/run/secgameone.sh
```

#### Security Game, t=2
```shell
./scripts/run/secgame.sh
```

#### Rock Paper Scissors
```shell
./scripts/run/rps.sh
```

## Extract statistics and plot figures
1. Open jupyter notebook in the root folder.
```shell
jupyter notebook
```
2. Open and run the appropriate notebook. Take care to point the target folder to the saved results. The notebooks are:
	1. ./scripts/visualize/vis-OCP.ipynb
	2. ./scripts/visualize/vis-rps.ipynb
	3. ./scripts/visualize/vis-secgame.ipynb
	4. ./scripts/visualize/vis-secgameone.ipynb

## General questions and issues
##### How large is the training data?
The numbers {500, 1000, 2000, 5000} are the sizes of the *entire*
dataset. 
For all experiments, the network is evaluated is every 5-20 epochs. 
For all experiments other than RPS, only the true parameters (and u, v) are evaluated; the log loss of the test/validation set is not really used.
Only in the RPS experiment do we actually use the test set of size 2000 to evaluate the MSE of (u, v), as there are different features each time.

Unlike previously, we no longer recommend using early stopping as this tends to make the results overly dependent on initialization of parameters.
##### What and how are the metrics computed?
For most of the experiments, we take the MSE over parameters. This was chosen for convenience and 
is not always the best measure of `distance' from the true parameters, i.e. 
2 parameters may differ greatly but have nearly the same equilibria (u, v). 
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
Unfortunately, when generating data, we all for the specification of the random seed for the sampled actions of each player. This means that
the exact dataset being generated will differ each time. However, the parameters to be learned *are* seeded appropriately.
Hence, this should not cause a huge discrepency in the results as long as enough trials are run.

Furthermore, it is known that for RPS, the features 1-x, for all x are all equivalent due to equal seeding. This is a bug.
However, the trend obtained for a single experiment (e.g. what was reported in the paper) should not be affected.
