# Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion

[Alan N Amin](https://alannawzadamin.github.io), [Nate Gruver](https://ngruver.github.io), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

[Paper](https://arxiv.org/abs/2506.08316)

<p align="center">
  <img width="321" alt="concept" src="https://github.com/user-attachments/assets/40daf76e-8381-4986-b8f9-fc9d93571bf9" />
</p>

### Description

Masking discrete diffusion makes use of a fundamental difference between continuous and discrete Markov processes: discrete Markov processes evolve by discontinuous jumps at a fixed rate and, unlike other discrete diffusion models, masking diffusion *builds in the known distribution of jump times* and only learns where to jump to. We show that we can similarly bake in the known distribution of jump times into *any* discrete diffusion model. The resulting models -- schedule-conditioned diffusion (SCUD) -- generalize classical discrete diffusion and masking diffusion. By applying SCUD to models with noising processes that incorporate inductive biases on images, text, and protein data, we build diffusion models that outperform masking.

This codebase implements schedule-conditioned diffusion (**SCUD**). We provide instructions to train models on image and protein data. We also include code to train masking diffusion or classical diffusion models.

----

### Installation

Install dependencies by running ```pip install .``` with a recent version of Python.

### Basic usage

To train a small U-net on CIFAR10 with 64 states, run ```python3 train.py```.

#### Training protein models

To train protein models, you can download Uniref50 data from [here](https://zenodo.org/records/6564798). Place this data in ```data/uniref_2020/uniref50/```.
Then you can train a SCUD model with a small CARP architecture by running ```python3 train.py --config-name=basic_protein```.

#### Other usage

You can change the hyperparameters of the model, and even train masking and classical discrete diffusion models by modifying the config file ```configs/basic.cfg```.
To change the training parameters, modify the ```train``` parameters.
To change the model architecture, modify ```architecture``` parameters.
Changing ```data.N``` for image data changes the number of states in CIFAR images (up to 256).

```model.model``` can be set to ```SCUD```, ```Masking```, or ```Classical```.
```model.gamma``` controls the conditioning parameter $\gamma$ for SCUD.
```model.schedule_type``` controls the noise rate function $\beta(t)$ and can be set to ```linear```, ```cos```, or our choice from the paper: ```mutual_information``` (note below for this choice).
```model.forward_kwargs``` controls the forward process; note the ```get_inf_gen``` function in ```scud/utils.py``` for choices.
```model.logistic_pars``` toggles the logistic parametersation for image data.
Set ```model.restart``` to the folder of a checkpoint to restart training.

#### Note on noise rate function

We choose our function $\beta(t)$ to linearly decrease the mutual information in time.
As described in the paper, this involves finding a zero using Newton's method, which can slow down training when there are too many states.
Thus when there are $B>200$ states, we precompute the values $\beta(t)$ at a resolution of $10^{-6}$ and save them in ```data/save_alphas``` before begining training.
Be sure to account for this taking up to an hour the first time you train on a new set of data (calculating this schedule for classical discrete diffusion can take much longer).
