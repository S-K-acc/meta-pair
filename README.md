# Metadensity functional learning for classical fluids: Regularizing with pair correlations

This repository contains code, data and neural models for the methods presented in:

**Metadensity functional learning for classical fluids: Regularizing with pair correlations**  
*Stefanie M. Kampa, Florian Sammüller and Matthias Schmidt


## Instructions

### Setup

A recent version of [Julia](https://julialang.org/downloads/) needs to be installed on your system (Julia 1.12 was used for development).
Launch the Julia interpreter within this directory and type `]` to enter the package manager.
Activate the environment and install the required packages as follows:

```julia
activate .
instantiate
```

Type backspace to exit the package manager.
Start a Jupyter server:

```julia
using IJulia
jupyterlab()
```

### Usage

#### Training

The directory `training` contains data and code for training the required neural functionals from scratch, see the README and notebooks in this directory for further instructions.

#### Application

In `meta_OZ.ipynb`, we demonstrate how to utilize the neural functional in the meta-Ornstein-Zernike equation to obtain the central results of Figs. 2 and 3 in the manuscript.
