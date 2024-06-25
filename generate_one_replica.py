import os
import numpy as np
import time
import torch
from pathlib import Path
import my_ising
import one_replica
from my_parameters import (
    max_step,
    batch_size,
    beta_final,
    device,
    default_dtype_torch,
    ham,
    net_depth,
)
from rectangular_sampling_file import (
    build_replica,
    calc_log_prob,
)
from my_args import my_args

step_range = max_step
loss_cont = np.empty([step_range, batch_size])
energy_cont = np.empty([step_range, batch_size])
shift = my_args.shift
print(f"k: {one_replica.k}, L: {one_replica.L} shift: {shift}")
k = one_replica.k




net_b, int_nets, _ = one_replica.create_networks(net_depth)
real_size = one_replica.real_size
try:
    checkpoint = torch.load(one_replica.path)
    net_b.load_state_dict(checkpoint["net_b"])
    for i, net in enumerate(int_nets):
        net.load_state_dict(checkpoint[f"int_net{i}"])
        net.eval()
    print("Starting evaluation from the loaded state\n")
except FileNotFoundError:
    print("No starting point")
    exit()

start_time = time.time()
for i in range(step_range):
    with torch.inference_mode():
        sample, list_args_for_nets, log_prob_chess = build_replica(
            net_b,
            int_nets,
            beta_final,
            one_replica.L * k,
            one_replica.L,
            batch_size,
            device,
            default_dtype_torch,
        )

        log_prob = calc_log_prob(
            one_replica.z2, net_b, int_nets, list_args_for_nets, log_prob_chess
        )
        energy = my_ising.energy(sample, ham)
        loss: torch.Tensor = log_prob + beta_final * energy


        loss_cont[i, :] = loss.cpu().numpy()
        energy_cont[i, :] = energy.cpu().numpy()
        if (i + 1) % 100 == 0:
            print(i + 1)
            log_ess = 2 * torch.logsumexp(-loss, 0) - torch.logsumexp(-2 * loss, 0)
            ess = torch.exp(log_ess) / batch_size
            ess = ess.data.cpu().numpy()
            print(f"ess: {ess}")
print(f"--- {time.time() - start_time} seconds ---")


path = Path(
    my_args.path_data
    + f"/one_replica_{beta_final=}L={one_replica.L}{k=}{shift=}.npz"
)

if not os.path.exists(my_args.path_data):
    os.makedirs(my_args.path_data)


if (not my_args.overload) and path.is_file():
    print("Concatenating with previously generated samples")
    temp = np.load(path)
    loss_cont = np.concatenate(
        (temp["loss_cont"].ravel(), loss_cont.ravel().astype("float32"))
    )
    energy_cont = np.concatenate(
        (temp["energy_cont"].ravel(), energy_cont.ravel().astype("float32"))
    )
else:
    loss_cont = loss_cont.ravel().astype("float32")
    energy_cont = energy_cont.ravel().astype("float32")
np.savez_compressed(
    path,
    loss_cont=loss_cont,
    energy_cont=energy_cont,
)
