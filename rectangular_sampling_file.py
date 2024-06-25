from typing import Sequence
import numpy as np
from torch import Tensor
import torch

from my_dense_rect import AutoregressiveBoundary


def find_borders(rectangle: Tensor) -> Tensor:
    """It returns a tensor consisting of the boundary cells neighbouring with the interior"""
    borders = torch.cat(
        [
            rectangle[:, :, 0, 1:-1],
            rectangle[:, :, 1:-1, -1],
            rectangle[:, :, -1, 1:-1],
            rectangle[:, :, 1:-1, 0],
        ],
        dim=2,
    )

    return borders


def make_cross(cross: Tensor, X: int, Y: int, device: torch.device) -> Tensor:
    """It shapes the given sample into cross and inputs it into the otherwise empty rectangle (X x Y)"""
    interior = torch.zeros(
        [cross.shape[0], 1, X, Y], device=device, dtype=torch.float32
    )
    if X % 2 == 1 and Y % 2 == 1:
        split = [X - 2, (Y - 1) // 2 - 1, (Y - 1) // 2 - 1]
        list_of_spins = torch.split(cross, split, dim=2)
        interior[:, :, 1 : (X - 1), Y // 2] = list_of_spins[0]
        interior[:, :, X // 2, 1 : (Y - 1) // 2] = list_of_spins[1]
        interior[:, :, X // 2, ((Y - 1) // 2 + 1) : (Y - 1)] = list_of_spins[2]

    elif X % 2 == 1 and Y % 2 == 0:
        split = [X - 2, X - 2, (Y - 4) // 2, (Y - 4) // 2]
        list_of_spins = torch.split(cross, split, dim=2)
        interior[:, :, 1:-1, Y // 2 - 1] = list_of_spins[0]
        interior[:, :, 1:-1, Y // 2] = list_of_spins[1]
        interior[:, :, (X - 1) // 2, 1 : Y // 2 - 1] = list_of_spins[2]
        interior[:, :, (X - 1) // 2, Y // 2 + 1 : -1] = list_of_spins[3]

    return interior


def divide_into_squares(square: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """It divides a square with cross inside into 4 smaller squares with the border"""

    X = square.shape[2]
    Y = square.shape[3]
    if square.shape[2] == square.shape[3] and square.shape[2] % 2 == 1:
        Li = square.shape[2] - 2

        B11 = square[:, :, 0 : (Li + 3) // 2, 0 : (Li + 3) // 2]

        B12 = square[:, :, 0 : (Li + 3) // 2, (Li + 1) // 2 :]

        B21 = square[:, :, (Li + 1) // 2 :, 0 : (Li + 3) // 2]

        B22 = square[:, :, (Li + 1) // 2 :, (Li + 1) // 2 :]

    elif square.shape[2] + 1 == square.shape[3]: #almost square
        B11 = square[:, :, : (X // 2) + 1, : Y // 2]
        B12 = square[:, :, : (X // 2) + 1, Y // 2 :]
        B21 = square[:, :, (X // 2) :, : Y // 2]
        B22 = square[:, :, (X // 2) :, Y // 2 :]

    else:
        raise AssertionError

    return (B11, B12, B21, B22)


def add_into_squares(four_of_squares: Sequence[Tensor], *, repeated_Y=True) -> Tensor:
    """It reverses the effect of divide_into_squares"""
    assert four_of_squares[0].shape[2] == four_of_squares[0].shape[3]
    assert len(four_of_squares) == 4
    if repeated_Y:
        b11 = four_of_squares[0]
        b12 = four_of_squares[1][:, :, :, 1:]
        b21 = four_of_squares[2]
        b22 = four_of_squares[3][:, :, :, 1:]
    else:
        b11 = four_of_squares[0]
        b12 = four_of_squares[1]
        b21 = four_of_squares[2]
        b22 = four_of_squares[3]

    b11 = torch.cat([b11, b12], dim=3)
    b22 = torch.cat([b21, b22], dim=3)
    b22 = b22[:, :, 1:, :]

    b11 = torch.cat([b11, b22], dim=2)

    return b11


def add_log_into_squares(list_of_tensor: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    b11 = list_of_tensor[0]
    b12 = list_of_tensor[1]
    b21 = list_of_tensor[2]
    b22 = list_of_tensor[3]

    b11 = torch.cat([b11, b12], dim=2)
    b22 = torch.cat([b21, b22], dim=2)
    b11 = torch.cat([b11, b22], dim=1)
    return b11


def build_sample(
    sample: Tensor,
    list_args_for_nets: list[Tensor | list[Tensor]],
    int_nets: torch.nn.ModuleList,
    beta: float,
    X: int,
    Y: int,
    device: torch.device,
    default_dtype_torch=torch.float32,
) -> tuple[Tensor, list[Tensor | list[Tensor]], Tensor]:
    """Receives the sample with set boundary conditions (in temporal direction) and fill its interior"""
    k = X // Y
    net = iter(int_nets)
    ready_to_go_samples = []

    # dividing vertically recursively
    while k != 1:
        if (
            k % 2 == 1
        ):  # not possible to simply divide into two pieces, one square generated separately
            border = torch.cat([sample[:, :, -1, :-1], sample[:, :, 0, :-1]], dim=2)
            horizontal = next(net).sample(border)
            list_args_for_nets.append([horizontal, border])
            sample[:, :, Y, :-1] = horizontal
            ready_to_go_samples.append(sample[:, :, :Y, :])  # saving for later
            sample = sample[:, :, Y:, :]
        border = torch.cat([sample[:, :, -1, :-1], sample[:, :, 0, :-1]], dim=2)
        horizontal = next(net).sample(border)
        list_args_for_nets.append([horizontal, border])
        sample[:, :, sample.shape[2] // 2, :-1] = horizontal
        sample = torch.cat(
            [
                sample[:, :, : (sample.shape[2] // 2 + 1), :],
                sample[:, :, sample.shape[2] // 2 :, :],
            ]
        )
        k >>= 1
    k = X // Y
    # for simplicity, back to original sample shape, could be done later
    for i in bin(
        k
    )[
        3:
    ]:  # iterating over binary representation backward starting from 2nd most-left number
        sample = torch.cat(
            [
                sample[: sample.shape[0] // 2, :, :, :],
                sample[sample.shape[0] // 2 :, :, 1:, :],
            ],
            dim=2,
        )
        if i == "1":
            sample_to_add = ready_to_go_samples.pop()
            sample = torch.cat([sample_to_add, sample[:, :, :, :]], dim=2)
        # else:
        #     sample = torch.cat([sample, torch.unsqueeze(sample[:, :, 0, :], dim=2)], dim=2)

    # squarification
    sample = torch.cat(
        [sample[:, :, (Y * i) : (Y * (i + 1) + 1), :] for i in range(X // Y)], dim=0
    )
    # first cross (with additional vertical line for full division
    border = torch.cat([sample[:, :, -1, :-1], sample[:, :, 0, :-1]], dim=2)
    cross_border = next(net).sample(border)
    list_args_for_nets.append([cross_border, border])
    sample[:, :, 1:-1, 0] = sample[:, :, 1:-1, -1] = cross_border[:, :, : (Y - 1)]
    sample += make_cross(
        cross_border[:, :, (Y - 1) :], sample.shape[2], sample.shape[3], device
    )
    sample = torch.cat(divide_into_squares(sample), dim=0)
    divisions_into_squares = 1

    # crosses recursively
    for i_net in net:
        border = find_borders(sample)
        cross = i_net.sample(border)
        list_args_for_nets.append([cross, border])
        sample += make_cross(cross, sample.shape[2], sample.shape[3], device)
        sample = torch.cat(divide_into_squares(sample), dim=0)
        divisions_into_squares += 1

    # heat bath
    border = find_borders(sample)
    prob_spin_up = torch.sum(border, 2)
    prob_spin_up = 1 / (1 + torch.exp(-2 * beta * prob_spin_up))
    cross = torch.bernoulli(prob_spin_up).to(default_dtype_torch) * 2 - 1
    sample[:, :, 1, 1] = cross
    log_prob_chess = (
        torch.log(prob_spin_up[:, 0]) * (cross[:, 0] + 1) / 2
        + torch.log(1 - prob_spin_up[:, 0]) * (1 - cross[:, 0]) / 2
    )
    log_prob_chess = log_prob_chess.view(-1, 1, 1)

    # recreation of the initial sample
    for _ in range(divisions_into_squares):
        sample = add_into_squares(torch.chunk(sample, 4, dim=0))
        log_prob_chess = torch.chunk(log_prob_chess, 4, dim=0)
        log_prob_chess = add_log_into_squares(log_prob_chess)

    return sample, list_args_for_nets, log_prob_chess


def build_replica(
    net_b: AutoregressiveBoundary,
    int_nets: torch.nn.ModuleList,
    beta: float,
    X: int,
    Y: int,
    batch_size: int,
    device: torch.device,
    default_dtype_torch=torch.float32,
) -> tuple[Tensor, list[Tensor | list[Tensor]], Tensor]:
    sample_b = net_b.sample(batch_size)
    list_args_for_nets: list[Tensor | list[Tensor]] = [sample_b]
    sample: Tensor = torch.zeros(
        [batch_size, 1, X + 1, Y + 1], dtype=default_dtype_torch, device=device
    )
    sample[:, :, 0, :-1] = sample[:, :, -1, :-1] = sample_b

    del sample_b
    torch.cuda.empty_cache()
    sample, list_args_for_nets, log_prob_chess = build_sample(
        sample,
        list_args_for_nets,
        int_nets,
        beta,
        X,
        Y,
        device,
        default_dtype_torch=torch.float32,
    )
    # desquarification
    sample = torch.cat(torch.chunk(sample[:, :, 1:, :], X // Y, dim=0), dim=2)
    log_prob_chess = torch.chunk(log_prob_chess, X // Y, dim=0)
    log_prob_chess = torch.cat(log_prob_chess, dim=2)
    log_prob_chess = torch.sum(log_prob_chess, dim=(1, 2))
    sample = sample[:, :, :, :-1]  # the last column is a duplication of the first one
    return sample, list_args_for_nets, log_prob_chess



def make_2_replicas(
    sample_b: Tensor,
    rX: int,
    rY: int,
    device: torch.device,
    shift: int = 0,
) -> Tensor:
    """Size: X x Y, it creates 2 connected replicas for calculations of entropy, periodic boundary conditions
    in space (Y), it's complicated in time (X)"""

    X = rX + 2
    Y = rY + 1
    list_samples = torch.split(
        sample_b,
        [Y // 2 + shift] * 2 + [Y // 2 - shift] * 2,
        dim=2,
    )

    sample = torch.zeros(
        (sample_b.shape[0], 1, X, Y), device=device, dtype=torch.float32
    )
    # left part, periodic time
    sample[:, :, 0, : (Y // 2 + shift)] = list_samples[0]
    sample[:, :, -1, : (Y // 2 + shift)] = list_samples[0]
    sample[:, :, X // 2 - 1, : (Y // 2 + shift)] = list_samples[1]
    sample[:, :, X // 2, : (Y // 2 + shift)] = list_samples[1]

    # right part, periodic but upper and lower parts separate
    sample[:, :, 0, (Y // 2 + shift) : -1] = list_samples[2]
    sample[:, :, X // 2 - 1, (Y // 2 + shift) : -1] = list_samples[2]
    sample[:, :, X // 2, (Y // 2 + shift) : -1] = list_samples[3]
    sample[:, :, -1, (Y // 2 + shift) : -1] = list_samples[3]

    return sample


def build_2_replicas(
    net_b: AutoregressiveBoundary,
    int_nets: torch.nn.ModuleList,
    beta: float,
    k: int,
    replica_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    default_dtype_torch=torch.float32,
    shift: int = 0,
) -> tuple[Tensor, list[Tensor | list[Tensor]], Tensor]:
    sample_b = net_b.sample(batch_size)
    list_args_for_nets: list[Tensor | list[Tensor]] = [sample_b]
    sample = make_2_replicas(
        sample_b, replica_size[0], replica_size[1], device=device, shift=shift
    )
    sample = [
        sample[:, :, : sample.shape[2] // 2, :],
        sample[:, :, (sample.shape[2] // 2) :, :],
    ]
    sample = torch.cat(sample, dim=0)

    del sample_b

    sample, list_args_for_nets, log_prob_chess = build_sample(
        sample,
        list_args_for_nets,
        int_nets,
        beta,
        replica_size[0] // 2,
        replica_size[1],
        device,
        default_dtype_torch,
    )

    # reverting squarification
    if k != 1:
        sample = torch.chunk(sample, k, dim=0)
        sample = [
            chunk[:, :, 1:, :] if i != 0 else chunk for i, chunk in enumerate(sample)
        ]
        sample = torch.cat(sample, dim=2)
        log_prob_chess = torch.chunk(log_prob_chess, k, dim=0)
        log_prob_chess = torch.cat(log_prob_chess, dim=2)
    # 2 replicas back
    sample = torch.chunk(sample, 2, dim=0)
    sample = torch.cat(sample, dim=2)
    log_prob_chess = torch.chunk(log_prob_chess, 2, dim=0)
    log_prob_chess = torch.cat(log_prob_chess, dim=2)
    log_prob_chess = torch.sum(log_prob_chess, dim=(1, 2))
    sample = sample[:, :, :, :-1]
    # assert sample.shape == (batch_size, 1, *replica_size)
    return sample, list_args_for_nets, log_prob_chess


def _calc_log_prob_net(
    net_b: AutoregressiveBoundary,
    int_nets: torch.nn.ModuleList,
    sign: int,
    list_args_for_nets: list[Tensor | list[Tensor]],
):
    log_prob_net = 0
    for index, net_arg in enumerate(list_args_for_nets):
        if index == 0:
            log_prob_net = net_b.log_prob(sign * net_arg) # type: ignore
        else:
            # sum over the batches corresponding to the same sample due to parallelization
            log_prob_net += torch.sum(
                int_nets[index - 1]
                .log_prob(sign * net_arg[0], sign * net_arg[1])
                .reshape([-1, list_args_for_nets[0].shape[0]]), # type: ignore
                dim=0,
            )

    return log_prob_net


def calc_log_prob(
    z2: bool,
    net_b: AutoregressiveBoundary,
    int_nets: torch.nn.ModuleList,
    list_args_for_nets: list[Tensor | list[Tensor]],
    log_prob_chess: Tensor,
) -> Tensor:
    log_prob_net = _calc_log_prob_net(net_b, int_nets, 1, list_args_for_nets)

    log_prob: Tensor = log_prob_net + log_prob_chess

    if z2:
        log_prob_net_inv = _calc_log_prob_net(net_b, int_nets, -1, list_args_for_nets)

        log_prob_inv = log_prob_chess + log_prob_net_inv

        log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv]), dim=0)
        log_prob = log_prob - np.log(2)
    return log_prob
