# MADE: Masked Autoencoder for Distribution Estimation

import torch
from torch import nn


class MaskedLinear(nn.Linear):
    def __init__(
        self, in_channels: int, out_channels: int, n: int, bias: bool, exclusive: bool
    ):  # n=L^2 =number of spins
        super(MaskedLinear, self).__init__(
            in_channels * n,
            out_channels
            * n,  # parameters for nn.Linear class - we increase n times input channels and n times output channels
            bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        self.register_buffer("mask", torch.ones([self.n] * 2))
        if self.exclusive:
            self.mask = 1 - torch.triu(
                self.mask
            )  # mask for n=3:  [[0,0,0],[1,0,0],[1,1,0]]
        else:
            self.mask = torch.tril(
                self.mask
            )  # mask for n=3:  [[1,0,0],[1,1,0],[1,1,1]]
        self.mask = torch.cat(
            [self.mask] * in_channels, dim=1
        )  # replicate mask in_channels times in dim 1
        self.mask = torch.cat(
            [self.mask] * out_channels, dim=0
        )  # replicate mask out_channels times in dim 0
        self.weight.data *= self.mask


        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return super(
            MaskedLinear, self
        ).extra_repr() + ", exclusive={exclusive}".format(**self.__dict__)


class MaskedLinear_nonsquare(nn.Linear):
    def __init__(
        self, in_channels: int, out_channels: int, n: int, m: int, bias: bool, exclusive: bool
    ):  # n=number of spins interior + borders, m= numer of spins interior
        super(MaskedLinear_nonsquare, self).__init__(
            in_channels * n,
            out_channels
            * m,  # parameters for nn.Linear class - we increase n times input channels and m times output channels
            bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.m = m
        self.exclusive = exclusive

        self.register_buffer("mask", torch.ones([self.m, self.m]))
        self.mask = torch.tril(self.mask)  # mask for n=3:  [[1,0,0],[1,1,0],[1,1,1]]
        self.mask = torch.cat([torch.ones([self.m, self.n - self.m]), self.mask], dim=1)

        self.mask = torch.cat(
            [self.mask] * in_channels, dim=1
        )  # replicate mask in_channels times in dim 1
        self.mask = torch.cat(
            [self.mask] * out_channels, dim=0
        )  # replicate mask out_channels times in dim 0
        self.weight.data *= self.mask

        # print(self.mask)

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class MaskedLinear_border(nn.Linear):
    def __init__(
        self, in_channels: int, out_channels: int, n: int, m: int, bias: bool, exclusive: bool
    ):  # n=number of spins interior + borders, m= numer of spins interior
        super(MaskedLinear_border, self).__init__(
            in_channels * n,
            out_channels
            * n,  # parameters for nn.Linear class - we increase n times input channels and m times output channels
            bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.m = m
        self.exclusive = exclusive

        self.register_buffer("mask", torch.ones([self.m, self.m]))
        if self.exclusive:
            self.mask = 1 - torch.triu(
                self.mask
            )  # mask for n=3:  [[0,0,0],[1,0,0],[1,1,0]]
        else:
            self.mask = torch.tril(
                self.mask
            )  # mask for n=3:  [[1,0,0],[1,1,0],[1,1,1]]
        self.mask = torch.cat(
            [torch.zeros([self.n - self.m, self.m]), self.mask], dim=0
        )
        self.mask = torch.cat([torch.ones([self.n, self.n - self.m]), self.mask], dim=1)

        self.mask = torch.cat(
            [self.mask] * in_channels, dim=1
        )  # replicate mask in_channels times in dim 1
        self.mask = torch.cat(
            [self.mask] * out_channels, dim=0
        )  # replicate mask out_channels times in dim 0
        self.weight.data *= self.mask

        # print(self.mask)

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class ChannelLinear(nn.Linear):
    def __init__(self, in_channels: int, out_channels: int, n: int, bias: bool):
        super(ChannelLinear, self).__init__(in_channels * n, out_channels * n, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.register_buffer(
            "mask", torch.eye(self.n)
        )  # matrix 1 of size n x n (diagonal with ones at diagonal)
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x: torch.Tensor):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class AutoregressiveBoundary(nn.Module):
    '''
    Network with spins independent of any outside connections.
    '''
    def __init__(
        self,
            n: int,
            net_depth: int,
            net_width: int,
            bias: bool,
            z2: bool,
            epsilon: float,
            device: torch.device,
    ) -> None:
        super(AutoregressiveBoundary, self).__init__()
        self.n = n  # Number of boundary sites
        self.net_depth = net_depth
        self.net_width = net_width
        self.bias = bias
        self.z2 = z2
        self.epsilon = epsilon
        self.device = device

        self.default_dtype_torch = torch.float32

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer("x_hat_mask", torch.ones([self.n]))
            self.x_hat_mask[0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.n]))
            self.x_hat_bias[0] = 0.5

        layers: list[nn.Module] = [
            MaskedLinear(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.n,
                self.bias,
                exclusive=True,
            )
        ]
        for count in range(self.net_depth - 2):
            layers.append(self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(self._build_simple_block(self.net_width, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, in_channels: int, out_channels: int):
        layers = [
            nn.PReLU(in_channels * self.n, init=0.5),
            MaskedLinear(in_channels, out_channels, self.n, self.bias, exclusive=False),
        ]
        block = nn.Sequential(*layers)
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x_hat: torch.Tensor = self.net(x)
        x_hat = x_hat.view(x_hat.shape[0], 1, self.n)


        return x_hat

    # sample = +/-1, +1 = up = white, -1 = down = black\n
    # sample.dtype == default_dtype_torch\n
    # x_hat = p(x_{i, j} == +1 | x_{0, 0}, ..., x_{i, j - 1})\n
    # 0 < x_hat < 1\n
    # x_hat will not be flipped by z2\n
    def sample(self, batch_size: int) -> torch.Tensor:
        
        sample = torch.zeros(
            [batch_size, 1, self.n], dtype=self.default_dtype_torch, device=self.device
        )
        for i in range(self.n):
            x_hat = self.__call__(sample)
            sample[:, :, i] = (
                torch.bernoulli(x_hat[:, :, i]).to(self.default_dtype_torch) * 2 - 1
            )

        return sample

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        x_hat = self.__call__(sample)
        mask = (sample + 1) / 2
        log_prob: torch.Tensor = torch.log(x_hat + self.epsilon) * mask + torch.log(
            1 - x_hat + self.epsilon
        ) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)

        return log_prob


class AutoregressiveInternal(nn.Module):
    '''
    Network with spins depending on outside states.
    '''
    def __init__(
        self,
        n_out: int,
        n_in: int,
        net_depth: int,
        net_width: int,
        bias: bool,
        z2: bool,
        epsilon: float,
        device: torch.device,
    ):
        super(AutoregressiveInternal, self).__init__()
        self.m = n_out  # number of spins in "cross"
        self.n = (
            self.m + n_in
        )  # Number of all spins = interior spins + spins at border
        self.net_depth = net_depth
        self.net_width = net_width
        self.bias = bias
        self.z2 = z2
        self.epsilon = epsilon
        self.device = device

        self.default_dtype_torch = torch.float32

        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer("x_hat_mask", torch.ones([self.n]))
            self.x_hat_mask[0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.n]))
            self.x_hat_bias[0] = 0.5

        layers: list[nn.Module] = [
            MaskedLinear_border(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.n,
                self.m,
                self.bias,
                exclusive=True,
            )
        ]
        for count in range(self.net_depth - 2):
            layers.append(
                self._build_simple_block_border(self.net_width, self.net_width)
            )
        if self.net_depth >= 2:
            layers.append(self._build_simple_block_nonsquare(self.net_width, 1))

        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _build_simple_block_border(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.PReLU(in_channels * self.n, init=0.5),
            MaskedLinear_border(
                in_channels, out_channels, self.n, self.m, self.bias, exclusive=False
            ),
        ]
        block = nn.Sequential(*layers)
        return block

    def _build_simple_block_nonsquare(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.PReLU(in_channels * self.n, init=0.5),
            MaskedLinear_nonsquare(
                in_channels, out_channels, self.n, self.m, self.bias, exclusive=False
            ),
        ]
        block = nn.Sequential(*layers)
        return block

    def forward(self, x: torch.Tensor, border: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        border = border.view(border.shape[0], -1)
        x = torch.cat([border, x], dim=1)
        x_hat: torch.Tensor = self.net(x)
        x_hat = x_hat.view(x_hat.shape[0], 1, self.m)

        return x_hat

    def sample(self, border: torch.Tensor) -> torch.Tensor:
        batch_size = border.shape[0]
        sample = torch.zeros(
            [batch_size, 1, self.m], dtype=self.default_dtype_torch, device=self.device
        )
        for i in range(self.m):
            x_hat = self.__call__(sample, border)
            sample[:, :, i] = (
                torch.bernoulli(x_hat[:, :, i]).to(self.default_dtype_torch) * 2 - 1
            )

        return sample

    def log_prob(self, sample: torch.Tensor, border: torch.Tensor) -> torch.Tensor:
        x_hat = self.__call__(sample, border)
        mask = (sample + 1) / 2
        log_prob: torch.Tensor = torch.log(x_hat + self.epsilon) * mask + torch.log(
            1 - x_hat + self.epsilon
        ) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

