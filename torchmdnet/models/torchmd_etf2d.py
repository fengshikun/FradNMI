from typing import Optional, Tuple
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)
from torch.nn.parameter import Parameter
from torch.nn import Linear

from torchmdnet.models.feats import dist_emb, angle_emb, torsion_emb, xyz_to_dat
import torch.nn.functional as F


# init edge features

class EdgeFeatureInit(nn.Module):
    r"""init the edge feature of torchmd net"""
    def __init__(self, distance_exp, activation, num_radial, hidden_channels) -> None:
        """
        Initialize the edge features for a TorchMD_ETF2D.

        Parameters:
        - distance_exp (float): Exponent for distance computation.
        - activation (nn.Module): Activation function for the layer.
        - num_radial (int): Number of radial channels.
        - hidden_channels (int): Number of hidden channels.

        Returns:
        - None
        """
        super().__init__()
        self.distance_exp = distance_exp
        self.act = activation()
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        # self.lin_rbf_1.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    # TODO add 2D edge information
    def forward(self, node_embs, edge_index, edge_weight):
        """
        Forward pass method for EdgeFeatureInit module.

        Parameters:
        - node_embs (torch.Tensor): Node embeddings tensor.
        - edge_index (torch.Tensor): Tensor representing edge indices.
        - edge_weight (torch.Tensor): Tensor representing edge weights.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors e1 and e2.
        """
        rbf = self.distance_exp(edge_weight)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([node_embs[edge_index[0]], node_embs[edge_index[1]], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act):
        """
        Residual layer module with two linear transformations and activation function.

        Parameters:
        - hidden_channels (int): Number of hidden channels.
        - act (torch.nn.Module): Activation function.

        Returns:
        - torch.Tensor: Output tensor after residual connection and transformations.
        """
        super(ResidualLayer, self).__init__()
        self.act = act()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))



class UpdateE(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act):
        """
        Update module for edge features.

        Parameters:
        - hidden_channels (int): Number of hidden channels.
        - int_emb_size (int): Size of intermediate embeddings.
        - basis_emb_size_dist (int): Size of distance basis embeddings.
        - basis_emb_size_angle (int): Size of angle basis embeddings.
        - basis_emb_size_torsion (int): Size of torsion basis embeddings.
        - num_spherical (int): Number of spherical channels.
        - num_radial (int): Number of radial channels.
        - num_before_skip (int): Number of residual layers before skip connection.
        - num_after_skip (int): Number of residual layers after skip connection.
        - act (torch.nn.Module): Activation function.

        Returns:
        - None
        """
        super(UpdateE, self).__init__()
        self.act = act()
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.e0_norm = nn.LayerNorm(hidden_channels)
        self.e1_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        """
        Forward pass method for UpdateE module.

        Parameters:
        - x (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors x0 and some other tensor.
        - emb (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple of tensors rbf0, sbf, and t.
        - idx_kj (torch.Tensor): Tensor representing indices for x_kj.
        - idx_ji (torch.Tensor): Tensor representing indices for x_ji.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors e1 and e2.
        """
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

    
        # NOTE layernorm for large value in e1 and e2
        e1 = self.e0_norm(e1)
        e2 = self.e1_norm(e2)

        return e1, e2


def check_for_nan(module, gin, gout):
    if gin[0].isnan().any():
        gin.isnan().any()
        print("NaN values found in gradients!")

class EMB(torch.nn.Module):
    """
    EMB is a PyTorch module that encapsulates the embedding of distance, angle, and torsion features.

    Args:
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float): Cutoff distance for embeddings.
        envelope_exponent (int): Exponent for the envelope function.

    Functions:
        reset_parameters():
            Resets the parameters of the distance embedding to their initial values.

        forward(dist, angle, torsion, idx_kj):
            Forward pass through the embedding layers.

            Args:
                dist (torch.Tensor): Distance tensor.
                angle (torch.Tensor): Angle tensor.
                torsion (torch.Tensor): Torsion tensor.
                idx_kj (torch.Tensor): Index tensor for angular and torsional embeddings.

            Returns:
                tuple: A tuple containing the distance embedding, angle embedding, and torsion embedding.
    """

    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(EMB, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb




# fuse 2D infomation inside
class TorchMD_ETF2D(nn.Module):
    r"""The Special TorchMD equivariant Transformer architecture For LBA task.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
        layernorm_on_vec (str or None, optional): Layer normalization type. (default: None)
        md17 (bool, optional): Whether using on MD17 Tasks, will disable some layernorm layers. (default: False)
        seperate_noise (bool, optional): Separate noise handling. (default: False)
        num_spherical (int, optional): Number of spherical channels. (default: 3)
        num_radial (int, optional): Number of radial channels. (default: 6)
        envelope_exponent (int, optional): Exponent for envelope function. (default: 5)
        int_emb_size (int, optional): Size of intermediate embeddings. (default: 64)
        basis_emb_size_dist (int, optional): Size of distance basis embeddings. (default: 8)
        basis_emb_size_angle (int, optional): Size of angle basis embeddings. (default: 8)
        basis_emb_size_torsion (int, optional): Size of torsion basis embeddings. (default: 8)
        num_before_skip (int, optional): Number of layers before skip connection. (default: 1)
        num_after_skip (int, optional): Number of layers after skip connection. (default: 2)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        layernorm_on_vec=None,
        md17=False,
        seperate_noise=False,
        num_spherical=3, num_radial=6, envelope_exponent=5,
        int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
        num_before_skip=1, num_after_skip=2
    ):
        super(TorchMD_ETF2D, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.layernorm_on_vec = layernorm_on_vec

        act_class = act_class_mapping[activation]

        if self.max_z > 200: # lba
            max_z = self.max_z // 2
            self.embedding = nn.Embedding(max_z, hidden_channels // 2)
            self.type_embedding = nn.Embedding(2, hidden_channels // 2)
        else:
            self.embedding = nn.Embedding(self.max_z, hidden_channels)
        # self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()

        self.md17 = md17
        if not self.md17:
            self.vec_norms = nn.ModuleList()
            self.x_norms = nn.ModuleList()
            
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
            ).jittable()
            self.attention_layers.append(layer)
            if not self.md17:
                self.vec_norms.append(EquivariantLayerNorm(hidden_channels))
                self.x_norms.append(nn.LayerNorm(hidden_channels))

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.seperate_noise = seperate_noise
        if self.seperate_noise:
            assert not self.layernorm_on_vec
            self.out_norm_vec = EquivariantLayerNorm(hidden_channels)

        


        self.init_e = EdgeFeatureInit(self.distance_expansion, act_class, num_rbf, hidden_channels)
        
        #  + 1
        self.update_es = torch.nn.ModuleList([
            UpdateE(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip, act_class) for _ in range(num_layers)])
        self.emb = EMB(num_spherical, num_radial, cutoff_upper, envelope_exponent)

        h = self.embedding.register_backward_hook(check_for_nan)


        
        if self.layernorm_on_vec:
            if self.layernorm_on_vec == "whitened":
                self.out_norm_vec = EquivariantLayerNorm(hidden_channels)
            else:
                raise ValueError(f"{self.layernorm_on_vec} not recognized.")
            
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        if self.layernorm_on_vec:
            self.out_norm_vec.reset_parameters()




    def forward(self, z, pos, batch, return_e=False, type_idx=None):
        """
        Forward pass method for the TorchMD_ETF2D module.

        Parameters:
        - z (torch.Tensor): Atomic numbers tensor.
        - pos (torch.Tensor): Positions tensor.
        - batch (torch.Tensor): Batch tensor.
        - return_e (bool, optional): Flag to return edge features. (default: False)
        - type_idx (torch.Tensor, optional): Type indices tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Updated node features, vector features, atomic numbers, positions, and batch indices.
        If return_e is True, also returns edge features and edge indices.
        """
        x = self.embedding(z)
        if type_idx is not None:
            type_embedding = self.type_embedding(type_idx)
            x = torch.concat([x, type_embedding], dim=1)

        num_nodes=z.size(0)

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)

        # init edge feature
        e = self.init_e(x, edge_index, edge_weight) # contains e1 and e2  


        # mask the self loop edge
        mask = edge_index[0] != edge_index[1]
        no_loop_edge_index = edge_index[:, mask]


        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, no_loop_edge_index, num_nodes, use_torsion=True)
        emb = self.emb(dist, angle, torsion, idx_kj)
        org_emb = (emb[0].clone(),emb[1].clone(),emb[2].clone())


        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight) # replace with f2d
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        update_e0, epdate_e1 = e[0][mask], e[1][mask]

        for lidx, attn in enumerate(self.attention_layers):
            
            # update edge feature
            # update e without self loop
            update_e0, epdate_e1 = self.update_es[lidx]((update_e0, epdate_e1), emb, idx_kj, idx_ji)
            # replace edge_attr with the edge feature
            # NOTE: use e2 to update node feature
            edge_e1 = torch.clone(e[1]) # This function is differentiable, so gradients will flow back from the result of this operation to e[1]
            edge_e1[mask] = epdate_e1

            x_before_attn = x.clone()
            vec_before_attn = vec.clone()
            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_e1, edge_vec)
            # if not self.md17:
            #     dx = self.x_norms[lidx](dx)
            x = x + dx # may be nan
            # if not self.md17:
            #     x = self.x_norms[lidx](x)
            # if not self.md17:
            #     dvec = self.vec_norms[lidx](dvec)
            vec = vec + dvec
            if not self.md17:
                vec = self.vec_norms[lidx](vec)
            if torch.isnan(x).sum():
                print('nan happens1111')
        if torch.isnan(x).sum():
            print('nan happens11112222')
        # x = torch.clip(x, min=-1e+7, max=1e+7)
        xnew = self.out_norm(x)
        if torch.isnan(xnew).sum():
            print('nan happens111122223333')
            # import pdb; pdb.set_trace()# print('nan happens2222')
        if self.layernorm_on_vec:
            vec = self.out_norm_vec(vec)
        if self.seperate_noise:
            nvec = self.out_norm_vec(vec)
            return xnew, vec, nvec, z, pos, batch

        if return_e:
            e_clone_0 = torch.clone(e[0]) # clone is differentiable
            e_clone_1 = torch.clone(e[1])
            e_clone_0[mask], e_clone_1[mask] = update_e0, epdate_e1
            return xnew, vec, z, pos, batch, e, edge_index

        return xnew, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            # self.dk_proj = nn.Linear(num_rbf, hidden_channels)
            self.dk_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            # self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)
            self.dv_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        # x = x * attn.unsqueeze(2)
        x = x * F.softmax(attn, dim=-1).unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None) # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.

        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = (
            s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        )
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(
            -2, -1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64) # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(
            self.weight.dtype
        ) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, "
            "elementwise_linear={elementwise_linear}".format(**self.__dict__)
        )
