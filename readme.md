# Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion

[Alan N Amin](https://alannawzadamin.github.io), [Nate Gruver](https://ngruver.github.io), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

[arXiv](?). Presenting at ? workshop at ICLR 2025.

### Description

Masking discrete diffusion makes use of a fundamental difference between continuous and discrete Markov processes: discrete Markov processes evolve by discontinuous jumps at a fixed rate and, unlike other discrete diffusion models, masking diffusion *builds in the known distribution of jump times* and only learns where to jump to. We show that we can similarly bake in the known distribution of jump times into *any* discrete diffusion model. The resulting models -- schedule-conditioned diffusion (SCUD) -- generalize classical discrete diffusion and masking diffusion. By applying SCUD to models with noising processes that incorporate inductive biases on images, text, and protein data, we build diffusion models that outperform masking.

This codebase implements schedule-conditioned diffusion (**SCUD**). We provide instructions to train models on image and protein data. We also include code to train masking diffusion or classical diffusion models.

----

### Installation

Install dependencies by running ```pip install .``` with a recent version of Python.

### Usage

#### Data

Running image or language models automatically installs the relevant data.
To train protein models, you can download Uniref50 data from [here](https://zenodo.org/records/6564798). Place this data in ```data/uniref_2020/uniref50/```.

#### Noise rate function

We choose our function $\beta(t)$ to linearly decrease the mutual information in time.
As described in the paper, this involves finding a zero using Newton's method, which can slow down training when there are too many states.
Thus when there are $B>200$ states, we precompute the values $\beta(t)$ at a resulution of $10^{-6}$ and save them in ```data/save_alphas``` before begining training.
Be sure to account for this taking up to an hour the first time you train on a new set of data (calculating this schedule for SEDD can take much longer).

#### Training a model

?

#### Sampling from a model

?
