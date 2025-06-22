# Dynamic Conditional Optimal Transport through Simulation-Free Flows

This repo contains an implementation of COT-FM, as described in our publication [Dynamic Conditional Optimal Transport through Simulation-Free Flows](https://arxiv.org/abs/2404.04240).

Our code is heavily based on the (very nice) [torchcfm](https://github.com/atong01/conditional-flow-matching) package, which is required as a dependency.

## Environment

Clone the repo and create a conda environment:

`conda env create -f environment.yml`

which can be activated with

`conda activate cotfm`. In order to generate data for the Darcy Flow exeperiments, a separate environment is required for FEniCSx due to compatability issues. Please see [this link](https://github.com/TADSGroup/ConditionalOT2023/tree/main) for further details.

## Structure and Usage

The directory `cotfm/` contains an implementation of COT-FM. The file `cotfm/trainer.py` is a basic script for training and tuning models on the 2D and LV datasets considered in the paper. To launch training (using a random search over hyperparameters), run e.g.
`python ./cotfm/trainer.py --savedir ./2moons/ --dataset 2moons --cotmode batch`
changing the `--dataset` as necessary.

As the Darcy Flow dataset is of a different nature than the other datasets considered, a somewhat independent implementation for this dataset is contained in the subdirectory `cotfm/darcy_flow/`.

The folder `mcmc/` contains the necessary code for running MCMC on the Lotka-Voltera and Darcy Flow datasets, and all models are implemented in `models/`. The `data/` directory contains various files for generating and loading the datasets considered in our paper.

## Citation

If you found our paper or code useful, please consider citing our work as follows:

```
@misc{kerrigan2024dynamic,
      title={Dynamic Conditional Optimal Transport through Simulation-Free Flows}, 
      author={Gavin Kerrigan and Giosue Migliorini and Padhraic Smyth},
      year={2024},
      eprint={2404.04240},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
