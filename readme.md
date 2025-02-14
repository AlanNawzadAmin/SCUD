# Why Masking Diffusion Works: Condition on the Jump \\ Schedule for Improved Discrete Diffusion

[Alan N Amin](https://alannawzadamin.github.io), [Nate Gruver](https://ngruver.github.io), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

[arXiv](?). Presenting at ? workshop at ICLR 2025.

### Description

Masking discrete diffusion makes use of a fundamental difference between continuous and discrete Markov processes: discrete Markov processes evolve by discontinuous jumps at a fixed rate and, unlike other discrete diffusion models, masking diffusion *builds in the known distribution of jump times* and only learns where to jump to. We show that we can similarly bake in the known distribution of jump times into *any* discrete diffusion model. The resulting models -- schedule-conditioned diffusion (SCUD) -- generalize classical discrete diffusion and masking diffusion. By applying SCUD to models with noising processes that incorporate inductive biases on images, text, and protein data, we build diffusion models that outperform masking.

This codebase implements schedule-conditioned diffusion (**SCUD**).

----

### Installation

Install dependencies by running ```pip install .``` with a recent version of Python.

### Pretrained models

?

### Usage

Get data? Training? Sampling?
