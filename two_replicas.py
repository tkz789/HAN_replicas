import os
from pathlib import Path
import torch
from my_dense_rect import AutoregressiveBoundary, AutoregressiveInternal
import matplotlib.pyplot as plt
import time
import my_ising
from rectangular_sampling_file import build_2_replicas, calc_log_prob
import numpy as np
from my_args import my_args
from my_parameters import (
    batch_size,
    device,
    default_dtype_torch,
    net_width,
    net_depth,
    beta_final,
    beta_anneal,
    z2,
    bias,
    epsilon,
    lr,
    max_step,
    ham,
    print_step,
)

def get_n_nets_horizontal(k: int) -> int:
    n_nets_horizontal = 0
    while k != 1:
        if k % 2 == 1:
            n_nets_horizontal += 1
        n_nets_horizontal += 1
        k >>= 1
    return n_nets_horizontal

k: int = my_args.k
shift: int = my_args.shift
n_int_nets = int(np.log2(my_args.L)) - 2
assert (my_args.L & (2**(n_int_nets + 2) - 1)) == 0 and (
    my_args.L >> (n_int_nets + 2)
) == 1, "The argument (L) needs to be a power of 2"

L: int = my_args.L
real_size = (L * k * 2, L)
L = my_args.L

n_nets_horizontal = get_n_nets_horizontal(k)

n_spins_border: list[int] = [2 * L for _ in range(n_nets_horizontal + 1)] + [
    4 * (L // (2 ** (i + 1)) - 1) for i in range(n_int_nets)
]

n_spins_interior: list[int] = (
    [L for _ in range(n_nets_horizontal)]
    + [3 * L - 4]
    + [2 * L // (2 ** (i + 1)) - 3 for i in range(n_int_nets)]
)

n_spins_VAN = L * 2



path = Path(
    my_args.path_save
    + f"/2replicas{beta_final=}{L=}{k=}{shift=}depth{net_depth}.out"
)
if not os.path.exists(my_args.path_save):
    os.makedirs(my_args.path_save)


def create_networks() -> tuple[AutoregressiveBoundary, torch.nn.ModuleList, list[torch.nn.Parameter]]:
    # creation of neural networks
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
        params = params + params_i
    return net_b, int_nets, params


def main():
    print(f"{beta_final=}")
    print(f"size: {L=} {k=}")
    print(f"{real_size=}")
    print(f"{z2=}")

    print(f"{n_spins_border=}")
    print(f"{n_spins_interior=}")
    print(f"{max_step=}")
    print(f"{beta_anneal=}")
    print(f"{batch_size=}")
    print(f"{lr=}")

    net_b, int_nets, params = create_networks()

    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    first_step = 0
    step_cont: list[int] = []
    F_cont: list[float] = []
    F_err_cont: list[float] = []
    ess_cont: list[float] = []
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
            del checkpoint
            print("Starting from the loaded state\n")
        except FileNotFoundError:
            print("No starting point")
    elif my_args.import_pretrained:
        try:
            print("Trying to import pretrained network...")
            if my_args.pretrained_beta_shift != 0:
                checkpoint = torch.load(
                    my_args.path_save
                    + f"/2replicasbeta_final={round(beta_final + my_args.pretrained_beta_shift, 2)}{L=}{k=}{shift=}depth{net_depth}.out",
                    map_location=device,
                )
                print("Loading with shifted beta")
                net_b.load_state_dict(checkpoint["net_b"])
            elif shift == 0:
                checkpoint = torch.load(
                    my_args.path_save
                    + f"/1replica{beta_final=}{L=}{k=}depth{net_depth}.out",
                    map_location=device,
                )
            else:
                checkpoint = torch.load(
                    my_args.path_save
                    + f"/2replicas{beta_final=}{L=}{k=}shift=0depth{net_depth}.out",
                    map_location=device,
                )
                net_b.load_state_dict(checkpoint["net_b"])
            for i, net in enumerate(int_nets):
                net.load_state_dict(checkpoint[f"int_net{i}"])
                net.train()
                print(f"Loaded pretrained int_net{i}")
            del checkpoint
        except FileNotFoundError:
            print("Pretrained network not found")
            exit(1)
    else:
        print("Saved network will be overloaded")

    U_cont = []
    U_err_cont = []


    start_time = time.time()

    print("training")

    for step in range(1 + first_step, first_step + max_step + 1):
        last_step = step
        optimizer.zero_grad()

        beta = beta_final * (1 - beta_anneal**step)
        with torch.no_grad():
            sample, list_args_for_nets, log_prob_chess = build_2_replicas(
                net_b,
                int_nets,
                beta,
                k,
                real_size,
                batch_size,
                device,
                default_dtype_torch,
                shift=shift,
            )


        log_prob = calc_log_prob(
            z2, net_b, int_nets, list_args_for_nets, log_prob_chess
        )

        with torch.no_grad():
            energy = my_ising.energy(sample, ham, alpha_replicas=2)
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
                f"U_B {energy_mean} ({energy_error}) tau: {tau_batch} ess: {ess} <|M|>: {abs_mag_mean} ({abs_mag_error})",
            )

            F_cont.append(free_energy_mean)
            F_err_cont.append(free_energy_error)
            U_cont.append(energy_mean)
            U_err_cont.append(energy_error)
            step_cont.append(step)
            ess_cont.append(ess)

    print("--- %s seconds ---" % (time.time() - start_time))

    # saving state
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
    plt.title(
        f"2 replicas {beta_final=} {L=} {k=} {'' if shift == 0 else f'{shift=}'}"
    )
    if not os.path.exists(my_args.path_plots):
        os.makedirs(my_args.path_plots)
    plt.savefig(
        my_args.path_plots
        + f"/2replicas{beta_final=}{L=}{k=}{shift=}depth{net_depth}.pdf"
    )
    if my_args.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
