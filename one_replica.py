import os
from pathlib import Path
import torch
from my_dense_rect import AutoregressiveBoundary, AutoregressiveInternal
import time
import matplotlib.pyplot as plt
import my_ising
from rectangular_sampling_file import build_replica, calc_log_prob
import numpy as np
from my_args import my_args
from my_parameters import (
    batch_size,
    beta_final,
    beta_anneal,
    max_step,
    lr,
    epsilon,
    net_depth,
    net_width,
    z2,
    device,
    bias,
    print_step,
    default_dtype_torch,
    ham
)


def get_n_nets_horizontal(k: int) -> int:
    n_nets_horizontal = 0
    while k != 1:
        if k % 2 == 1:
            n_nets_horizontal += 1
        n_nets_horizontal += 1
        k >>= 1
    return n_nets_horizontal


k: int = my_args.k  # number of blocks
n_int_nets = int(np.log2(my_args.L)) - 2
assert (my_args.L & (2**(n_int_nets + 2) - 1)) == 0 and (
    my_args.L >> (n_int_nets + 2)
) == 1, "The argument (L) needs to be a power of 2"


L = my_args.L
n_nets_horizontal = get_n_nets_horizontal(k)

n_spins_border = [2 * L for _ in range(n_nets_horizontal + 1)] + [
    4 * (L // (2 ** (i + 1)) - 1) for i in range(n_int_nets)
]

n_spins_interior = (
    [L for _ in range(n_nets_horizontal)]
    + [3 * L - 4]
    + [2 * L // (2 ** (i + 1)) - 3 for i in range(n_int_nets)]
)

n_spins_VAN = L
real_size = (L * k, L)  # size of the whole sample
path = Path(
    my_args.path_save + f"/1replica{beta_final=}{L=}{k=}depth{net_depth}.out"
)
if not os.path.exists(my_args.path_save):
    os.makedirs(my_args.path_save)


def create_networks(net_depth: int) -> tuple[AutoregressiveBoundary, torch.nn.ModuleList, list[torch.nn.Parameter]]:
    net_b = AutoregressiveBoundary(
        n_spins_VAN,
        net_depth=net_depth,
        net_width=net_width,
        bias=bias,
        z2=z2,
        epsilon=epsilon,
        device=device,
    )
    net_b.to(device)
    params_b = list(net_b.parameters())

    n_params_b = int(sum([np.prod(p.shape) for p in params_b]))
    print("Boundary net - number of trainable parameters:", n_params_b, "\n")

    params = params_b

    int_nets = torch.nn.ModuleList()
    for i_int, i_border in zip(n_spins_interior, n_spins_border):
        net_i = AutoregressiveInternal(
            i_int,
            i_border,
            net_depth=net_depth,
            net_width=net_width,
            bias=bias,
            z2=z2,
            epsilon=epsilon,
            device=device,
        )
        net_i.to(device)
        params_i = list(net_i.parameters())
        nparams_i = int(sum([np.prod(p.shape) for p in params_i]))
        print("Interior net - number of trainable parameters:", nparams_i, "\n")
        int_nets.append(net_i)
        params += params_i
    return net_b, int_nets, params


def main():

    print(f"{beta_final=}")
    print(f"size: {L=} {k=}")
    print(f"{real_size=}")
    print(f"{z2=}")

    print(f"{n_spins_border=}")
    print("n_nets_horizontal ", n_nets_horizontal)
    print(f"{n_spins_interior=}")
    print(f"{max_step=}")
    print(f"{beta_anneal=}")
    print(f"{batch_size=}")
    print(f"{lr=}")

    net_b, int_nets, params = create_networks(net_depth)
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    first_step = 0
    step_cont = []
    F_cont = []
    F_err_cont = []
    ess_cont = []
    if (not my_args.overload) and (not my_args.import_pretrained):
        # loading state from file if it exists
        try:
            checkpoint = torch.load(path, map_location=device)

            net_b.load_state_dict(checkpoint["net_b"])
            net_b.train()
            first_step = checkpoint["last_step"]
            step_cont = checkpoint["step_cont"]
            F_cont = checkpoint["F_cont"]
            F_err_cont = checkpoint["F_err_cont"]
            for i, net in enumerate(int_nets):
                net.load_state_dict(checkpoint[f"int_net{i}"])
                net.train()
            ess_cont = checkpoint["ess_cont"]
            print("Starting from the loaded state\n")
        except FileNotFoundError:
            print("No starting point\n")

    elif my_args.import_pretrained:
        if k != 1:
            try:
                checkpoint = torch.load(
                    my_args.path_save
                    + f"/1replica{beta_final=}{L=}k={k-1}depth{net_depth}.out",
                    map_location=device,
                )
                n_nets_horizontal_1 = get_n_nets_horizontal(k - 1)
                for i, net in enumerate(int_nets[n_nets_horizontal:]): # type: ignore
                    net.load_state_dict(
                        checkpoint[f"int_net{n_nets_horizontal_1 + i}"]
                    )
                    print(f"Pretrained int_net{i+1} loaded from k = {k-1}")
            except FileNotFoundError:
                print("Pretrained network not found")
                exit(1)
        else:
            try:
                checkpoint = torch.load(
                    my_args.path_save
                    + f"/1replica{beta_final=}L={L//2}k=1depth{net_depth}.out",
                    map_location=device,
                )

                for i, net in enumerate(int_nets[2:]):  # type: ignore
                    net.load_state_dict(checkpoint[f"int_net{i+1}"])
                    print(f"Pretrained int_net{i+1} loaded from L/2")
            except FileNotFoundError:
                print("Pretrained network not found")
                exit(1)
    else:
        print("Loading set to overwrite any existing state")
    U_cont = []
    U_err_cont = []
    

    start_time = time.time()

    print("training")

    for step in range(1 + first_step, first_step + max_step + 1):
        last_step = step
        optimizer.zero_grad()

        beta = beta_final * (1 - beta_anneal**step)
        with torch.no_grad():
            sample, list_args_for_nets, log_prob_chess = build_replica(
                net_b,
                int_nets,
                beta,
                L * k,
                L,
                batch_size,
                device,
                default_dtype_torch,
            )


        log_prob = calc_log_prob(
            z2, net_b, int_nets, list_args_for_nets, log_prob_chess
        )

        with torch.no_grad():
            energy = my_ising.energy(sample, ham, alpha_replicas=1)
            loss: torch.Tensor = log_prob + beta * energy

        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        loss_reinforce.backward()
        optimizer.step()

        if step % print_step == 0:
            free_energy_mean = loss.mean() / beta / real_size[0] / real_size[1]
            free_energy_error = (
                loss.std()
                / np.sqrt(batch_size)
                / beta
                / real_size[0]
                / real_size[1]
            )
            free_energy_mean = free_energy_mean.data.cpu().numpy()
            free_energy_error = free_energy_error.data.cpu().numpy()

            energy_mean = energy.mean() / real_size[0] / real_size[1]
            energy_error = (
                energy.std()
                / np.sqrt(batch_size)
                / real_size[0]
                / real_size[1]
            )
            energy_mean = energy_mean.data.cpu().numpy()
            energy_error = energy_error.data.cpu().numpy()

            abs_mag = torch.abs(sample.sum(dim=(1, 2, 3))) / real_size[0] / real_size[1]
            abs_mag_mean = abs_mag.mean()
            abs_mag_error = (
                abs_mag.std()
                / np.sqrt(batch_size)
            )
            abs_mag_mean = abs_mag_mean.data.cpu().numpy()
            abs_mag_error = abs_mag_error.data.cpu().numpy()

            log_ess = 2 * torch.logsumexp(-loss, 0) - torch.logsumexp(-2 * loss, 0)
            ess = torch.exp(log_ess) / batch_size
            ess = ess.data.cpu().numpy()

            loss_m = torch.min(loss)
            l_obs = -torch.exp(loss_m - loss)  # l_obs>=-1 since loss_m < loss
            eigenv_batch = 1 + l_obs
            eigenv_batch = torch.mean(eigenv_batch)
            tau_batch = -1 / torch.log(eigenv_batch)
            eigenv_batch = eigenv_batch.data.cpu().numpy()
            tau_batch = tau_batch.data.cpu().numpy()

            print("beta/beta_final= ", beta / beta_final)

            print(
                f"training step: {step} F_b: {free_energy_mean} ({free_energy_error})",
                f"U_B: {energy_mean} ({energy_error}) tau: {tau_batch.item()} ess: {ess.item()} <|M|>: {abs_mag_mean} ({abs_mag_error})",
            )

            F_cont.append(free_energy_mean)
            F_err_cont.append(free_energy_error)
            U_cont.append(energy_mean)
            U_err_cont.append(energy_error)
            step_cont.append(step)
            ess_cont.append(ess)


    print("--- %s seconds ---" % (time.time() - start_time))

    state = dict(
        F_cont=F_cont,
        last_step=last_step,
        step_cont=step_cont,
        net_b=net_b.state_dict(),
        F_err_cont=F_err_cont,
        ess_cont=ess_cont,
    )
    state.update(
        {"int_net" + str(i): net.state_dict() for i, net in enumerate(int_nets)}
    )

    torch.save(state, path)

    plt.errorbar(step_cont[5:], F_cont[5:], F_err_cont[5:], fmt=".r")
    plt.xlabel("step")
    plt.ylabel("Free energy/spin")
    plt.title(f"1replica {beta_final=} {L=} {k=}")
    if not os.path.exists(my_args.path_plots):
        os.makedirs(my_args.path_plots)
    plt.savefig(my_args.path_plots + f"/1replica{beta_final=}{L=}{k=}.pdf")
    if my_args.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
