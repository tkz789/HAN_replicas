# 2D classical Ising model
from torch import Tensor


def energy(sample: Tensor, ham: str, *, alpha_replicas=1) -> Tensor:
    # Counting connections only to

    term: Tensor = (
        sample[:, :, 1:, :] * sample[:, :, :-1, :]
    )  # multiply all S_[i,j]*S[i-1,j] and later sum over i and j
    # (summation over dim=1 is trivial since sample is of length 1 in this direction)
    term = term.sum(dim=(1, 2, 3))
    output = term
    term = (
        sample[:, :, :, 1:] * sample[:, :, :, :-1]
    )  # multiply all S_[i,j]*S[i,j-1] and later sum over i and j
    term = term.sum(dim=(1, 2, 3))
    output += term

    if alpha_replicas == 1:
        term = sample[:, :, 0, :] * sample[:, :, -1, :]
        term = term.sum(dim=(1, 2))
        output += term
        term = sample[:, :, :, 0] * sample[:, :, :, -1]
        term = term.sum(dim=(1, 2))
        output += term

    if alpha_replicas == 2:
        X = sample.shape[2]
        term = sample[:, :, X//2-1, :] * sample[:, :, X//2, :]  # vertical connections in the middle of
        output -= term.sum(dim=(1, 2))
        term = sample[:, :, X//2 - 1, 1:] * sample[:, :, X//2 - 1, :-1]  # horizontal connections in the middle
        output -= term.sum(dim=(1, 2))
        term = sample[:, :, -1, 1:] * sample[:, :, -1, :-1]  # horizontal connections lower part
        output -= term.sum(dim=(1, 2))
        term = sample[:, :, :, 0] * sample[:, :, :, -1]
        output += term.sum(dim=(1, 2))
        output -= (sample[:, :, X//2 - 1, 0] * sample[:, :, X//2 - 1, -1]).sum(dim=1)
        output -= (sample[:, :, -1, 0] * sample[:, :, -1, -1]).sum(dim=1)

    if ham == "fm":
        output *= -1

    return output
