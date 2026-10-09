# Monomer infiltration saturation time surrogate model
## Summary
The code trains a multilayer perceptron to infer a mapping between the physical parameters of the delignified transparent wood composites and their monomer infiltration saturation time. Full saturation of the porous wood is necessary for obtaining bot the optical transparency and mechanical integrity of the final material.

## Input and output
The input of the model consists of six parameters characterizing the transparent wood sample:
- Sample length [m] - L
- Fluid dynamic viscosity [Pa$\cdot$s] - $\mu$
- Longitudinal permeability [$m^2$] - $k_{long}$
- Mean pore radius [m] - $r_{\mu}$
- Surface tension [N/m] - $\gamma$
- Porosity - $\phi$

The output is the saturation time measured in seconds.

## Dataset
The dataset for training the surrogate model is provided in the file **surrogate_data.csv**. Each row corresponds to one simulation carried out using a Darcy-flow-based infiltration model implmented in FEniCSx 0.9. The columns correspond to the physical parameters in the same order as described above followed by the saturation time.

## Surrogate model
The surrogate model is a multilayer perceptron implemented in PyTorch. It consists of 3 hidden layers, each with 50 neurons. ReLU is used as the activation function for every layer, and sigmoid is used in the output.

Before training both the input and output values are transformed using the logarithm function and scaled to the range (0,1).
