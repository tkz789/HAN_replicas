import os
import numpy as np
import torch
from scipy.special import logsumexp
from pathlib import Path
import time

import my_ising
import one_replica
import two_replicas
from my_parameters import (
    max_step,
    beta_final,
    batch_size,
    net_depth,
    default_dtype_torch,
    device,
    ham,
)
from rectangular_sampling_file import build_2_replicas, calc_log_prob
from my_args import my_args

step_range = max_step
loss_cont = np.empty([step_range, batch_size])
energy_cont = np.empty([step_range, batch_size])
print(f"k: {one_replica.k}, L: {one_replica.L}, shift: {my_args.shift}")
k: int = my_args.k
L: int = two_replicas.L


shift: int = my_args.shift

net_b, int_nets, _ = two_replicas.create_networks()
real_size = two_replicas.real_size
try:
    checkpoint = torch.load(
        my_args.path_save
        + f"/2replicas{beta_final=}{L=}{k=}{shift=}depth{net_depth}.out",
        map_location=device,
    )
    net_b.load_state_dict(checkpoint["net_b"])
    print("last step:", checkpoint["last_step"])
    for i, net in enumerate(int_nets):
        net.load_state_dict(checkpoint[f"int_net{i}"])
        net.eval()
    print("Starting generation from the loaded state")
except FileNotFoundError:
    print("No starting point")
    exit()

time_start = time.time()
for i in range(step_range):
    with torch.inference_mode():
        sample, list_args_for_nets, log_prob_chess = build_2_replicas(
            net_b,
            int_nets,
            beta_final,
            two_replicas.k,
            two_replicas.real_size,
            batch_size,
            device,
            default_dtype_torch,
            shift=shift,
        )

        log_prob = calc_log_prob(
            one_replica.z2, net_b, int_nets, list_args_for_nets, log_prob_chess
        )
        energy = my_ising.energy(sample, ham, alpha_replicas=2)
        loss: torch.Tensor = log_prob + beta_final * energy


        loss_cont[i, :] = loss.cpu().numpy()
        energy_cont[i, :] = energy.cpu().numpy()
        if (i + 1) % 100 == 0:
            print(i + 1)
            log_ess = 2 * torch.logsumexp(-loss, 0) - torch.logsumexp(-2 * loss, 0)
            ess = torch.exp(log_ess) / batch_size
            ess = ess.data.cpu().numpy()
            print(f"{ess=}")
print(f"--- {time.time() - time_start} seconds ---")


path = Path(
    my_args.path_data
    + f"/2replicas{beta_final=}{L=}{k=}{shift=}depth{net_depth}.npz"
)
if not os.path.exists(my_args.path_data):
    os.makedirs(my_args.path_data)
if (not my_args.overload) and path.is_file():
    print("Concatenating with previously generated samples")
    temp_data = np.load(path)
    loss_cont = np.concatenate((temp_data["loss_cont"].ravel(), loss_cont.ravel()))
    energy_cont = np.concatenate(
        (temp_data["energy_cont"].ravel(), energy_cont.ravel())
    )
else:
    loss_cont = loss_cont.ravel()
    energy_cont.ravel()
np.savez_compressed(
    path,
    loss_cont=loss_cont.astype("float32"),
    energy_cont=energy_cont.astype("float32"),
)
