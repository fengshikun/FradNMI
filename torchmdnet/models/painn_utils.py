from typing import Sequence, Callable, Union, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_


class Dense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        A custom dense (fully connected) layer with optional activation and initialization.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
            activation (Union[Callable, nn.Module], optional): Activation function to apply after the linear transformation. Defaults to None, which uses nn.Identity().
            weight_init (Callable, optional): Initialization function for the weights. Defaults to xavier_uniform_.
            bias_init (Callable, optional): Initialization function for the bias. Defaults to zeros_.
        """
        
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
) -> nn.Module:
    """
    Build a Multi-Layer Perceptron (MLP) neural network.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden (Optional[Union[int, Sequence[int]]], optional): Number of neurons in the hidden layers. If None, it will automatically create a decreasing number of neurons for each layer. Defaults to None.
        n_layers (int, optional): Number of layers in the network. Defaults to 2.
        activation (Callable, optional): Activation function to apply to each hidden layer. Defaults to F.silu.

    Returns:
        nn.Module: A neural network model consisting of the specified number of layers and neurons, with the specified activation function applied to each hidden layer.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation)
        for i in range(n_layers - 1)
    ]
    # assign a Dense layer (without activation function) to the output layer
    layers.append(Dense(n_neurons[-2], n_neurons[-1], activation=None))
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    return _scatter_add(x, idx_i, dim_size, dim)


def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    """
    Computes the Gaussian Radial Basis Function (RBF) for the given inputs, offsets, and widths.

    Args:
        inputs (torch.Tensor): The input tensor.
        offsets (torch.Tensor): The offsets for the RBF centers.
        widths (torch.Tensor): The widths (standard deviations) for the RBFs.

    Returns:
        torch.Tensor: The resulting tensor after applying the Gaussian RBF.
    """
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.
        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}
        Args:
            cutoff (float, optional): cutoff radius.
        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.
    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)

