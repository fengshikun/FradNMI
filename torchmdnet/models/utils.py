import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph


def visualize_basis(basis_type, num_rbf=50, cutoff_lower=0, cutoff_upper=5):
    """
    Function for quickly visualizing a specific basis. This is useful for inspecting
    the distance coverage of basis functions for non-default lower and upper cutoffs.

    Args:
        basis_type (str): Specifies the type of basis functions used. Can be one of
            ['gauss',expnorm']
        num_rbf (int, optional): The number of basis functions.
            (default: :obj:`50`)
        cutoff_lower (float, optional): The lower cutoff of the basis.
            (default: :obj:`0`)
        cutoff_upper (float, optional): The upper cutoff of the basis.
            (default: :obj:`5`)
    """
    import matplotlib.pyplot as plt

    distances = torch.linspace(cutoff_lower - 1, cutoff_upper + 1, 1000)
    basis_kwargs = {
        "num_rbf": num_rbf,
        "cutoff_lower": cutoff_lower,
        "cutoff_upper": cutoff_upper,
    }
    basis_expansion = rbf_class_mapping[basis_type](**basis_kwargs)
    expanded_distances = basis_expansion(distances)

    for i in range(expanded_distances.shape[-1]):
        plt.plot(distances.numpy(), expanded_distances[:, i].detach().numpy())
    plt.show()


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        """
        NeighborEmbedding module for message passing in graph neural networks.

        Args:
        - hidden_channels (int): Size of hidden embeddings.
        - num_rbf (int): Number of radial basis functions.
        - cutoff_lower (float): Lower cutoff distance for interactions.
        - cutoff_upper (float): Upper cutoff distance for interactions.
        - max_z (int, optional): Maximum atomic number for embeddings. (default: 100)
        """
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        """
        Forward pass for the NeighborEmbedding module.

        Args:
        - z (torch.Tensor): Atomic numbers tensor.
        - x (torch.Tensor): Node feature tensor.
        - edge_index (torch.Tensor): Edge indices tensor.
        - edge_weight (torch.Tensor): Edge weights tensor.
        - edge_attr (torch.Tensor): Edge attributes tensor.

        Returns:
        - torch.Tensor: Updated node features after neighbor embedding.
        """
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        """
        Gaussian smearing module for expanding distances into a series of Gaussian basis functions.

        Args:
        - cutoff_lower (float, optional): Lower cutoff distance. (default: 0.0)
        - cutoff_upper (float, optional): Upper cutoff distance. (default: 5.0)
        - num_rbf (int, optional): Number of radial basis functions. (default: 50)
        - trainable (bool, optional): Whether to make the parameters trainable. (default: True)
        """
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        """
        Exponential normal smearing module for expanding distances into a series of exponential normal basis functions.

        Args:
        - cutoff_lower (float, optional): Lower cutoff distance. (default: 0.0)
        - cutoff_upper (float, optional): Upper cutoff distance. (default: 5.0)
        - num_rbf (int, optional): Number of radial basis functions. (default: 50)
        - trainable (bool, optional): Whether to make the parameters trainable. (default: True)
        """
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        """
        Shifted Softplus activation function.

        The Shifted Softplus is a variant of the Softplus activation function with a constant shift to ensure it has zero mean at zero input.

        Attributes:
        - shift (float): The constant shift value, equal to log(2).

        Methods:
        - forward(x): Applies the Shifted Softplus activation function to the input tensor `x`.
        """
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        """
        Cosine Cutoff function for interatomic distances.

        This function applies a cosine cutoff to smooth the transition to zero for interatomic distances 
        near the specified cutoff values. It ensures that interactions smoothly go to zero as distances 
        approach the upper cutoff.

        Args:
            cutoff_lower (float): Lower cutoff distance. Default is 0.0.
            cutoff_upper (float): Upper cutoff distance. Default is 5.0.
        """
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        """
        Applies a cosine cutoff function to the given distances. The function ensures that interactions smoothly 
        transition to zero as distances approach the cutoff values. 

        If `cutoff_lower` is greater than 0, contributions below the lower cutoff radius and beyond the upper cutoff 
        radius are removed.

        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of cutoff values, with the same shape as the input distances, where each value has 
            been modified by the cosine cutoff function.
        """
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    """
    Module for computing interatomic distances and corresponding vectors based on position data.

    Args:
        cutoff_lower (float): Lower cutoff distance for interatomic interactions.
        cutoff_upper (float): Upper cutoff distance for interatomic interactions.
        max_num_neighbors (int, optional): Maximum number of neighbors to return for each atom.
            Defaults to 32.
        return_vecs (bool, optional): Whether to return distance vectors. Defaults to False.
        loop (bool, optional): Whether to include self-loops in the graph. Defaults to False.

    Inputs:
        pos (torch.Tensor): Tensor of atom positions with shape (num_atoms, num_dimensions).
        batch (torch.Tensor): Tensor indicating the batch index for each atom.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.LongTensor): Index tensor of shape (2, num_edges) representing
            edges between atoms.
            - edge_weight (torch.Tensor): Tensor of shape (num_edges,) representing edge weights
            (distances or norms of distance vectors).
            - edge_vec (torch.Tensor or None): Tensor of shape (num_edges, num_dimensions) representing
            distance vectors if `return_vecs` is True, otherwise None.
    """
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    Args:
    hidden_channels (int): Number of input channels or hidden dimensions.
    out_channels (int): Number of output channels.
    intermediate_channels (int, optional): Number of channels in the intermediate layer.
        Defaults to None, which sets it to `hidden_channels`.
    activation (str, optional): Type of activation function to use. Defaults to "silu".
    scalar_activation (bool, optional): Whether to apply activation function to scalar output `x`.
        Defaults to False.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_channels).
        v (torch.Tensor): Input tensor of shape (batch_size, num_edges, hidden_channels).

    Returns:
        tuple: A tuple containing:
            - x (torch.Tensor): Output tensor of shape (batch_size, out_channels).
            - v (torch.Tensor): Output tensor of shape (batch_size, num_edges, out_channels),
            representing updated edge features.
    """
    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}
