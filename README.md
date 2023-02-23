# Federated_comparison_cttc
Code for "How Much Does it Cost to Train a Machine Learning Model over Distributed Data Sources?"

## Repository description
This repository contains the resources used to generate the results included in the paper entitled "How Much Does it Cost to Train a Machine Learning Model over Distributed Data Sources?". The files included are:

1) CFL: code to reproduce centralized federated learning simulations 
2) GFL: code to reproduce gossip federated learning simulations
3) BFL: code to reproduce blockchain enabled federated learing simulations

The file TFF.yml can be used to create the conda environment for the simulations:
```
conda env create -f TFF.yml
```