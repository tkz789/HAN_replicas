Requiered modules: torch (it was used with 2.3.1), numpy

# Training

To train a new model of single replica (rectangular sample with periodic boundary conditions), use:
```
python one_replica.py -L L -k K --beta BETA
```
where L - horizontal size, K - defines vertical size as a multiplicity of the horizontal size, BETA - reversed temperature

Additional options can be checked using -h.

Two replicas can be trained similarly:
```
python two_replicas -L L -k K --beta BETA --shift SHIFT
```
where SHIFT $\in \mathbb{N} \cap (-L/2, L/2)$ is a value defining a shift between two subsystems A and B. If 0, then they are of the same size.

Models trained in this way will be saved by default to ./saved_states, controlled by --path_saved parameter. 
# Generation

```
python generate_one_replica.py -L L -k K --beta BETA
```
```
python generate_two_replicas.py -L L -k K --beta BETA
```
Generated data will be saved by default to ./data, controlled by --path_data parameter.

# Additional informations
## Parameters printed out during training:
- ESS - effective sample size, ESS $\in [0, 1]$. It is $\approx$ 1 for well trained networks.

# Acknowledgement
This research was funded in part by National Science Centre, Poland, grant No. 2021/43/D/ST2/03375.
