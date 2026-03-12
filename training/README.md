# Training data and code

## Download datasets

The file `Simulations.jld2` can be downloaded via: https://www.staff.uni-bayreuth.de/~bt306964/data/meta-pair/Simulations.jld2,
The file `matchinput.npz` can be downloaded via: https://www.staff.uni-bayreuth.de/~bt306964/data/meta-pair/matchinput.npy

## Prepare simulation data

The file `Simulations.jld2` contains raw simulation data from grand canonical Monte Carlo simulations of inhomogeneous 1D fluids.
This file needs to be preprocessed prior to training, see `prepare_simulation_data.ipynb` for instructions.

## Training

Training is done in Python (version 3.12 was used for development) using the Keras/Tensorflow ecosystem.
To set up a suitable environment, you may use `uv`:
```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternatively, make sure to install the correct versions of Python and all required packages manually.

Running `train.py` trains a neural functional from scratch.
The settings in this script, in particular the parameter `alpha_c2x`, determine if metadirect pair-correlation regularization is used.

For further utilization in Julia, transfer the neural network weights to a Flux.jl model by running `julia --project convert-keras-to-flux.jl`.

## Creating testparticle data for metadirect regularization

In `create_testparticle_data.jl`, we demonstrate how to use an unregularized neural functional for generating testparticle data.
These may be used for the regularization of a second neural functional.
For convenience, a pregenerated dataset of testparticle result is provided in `matchinput.npz`.
