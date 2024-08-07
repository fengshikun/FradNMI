import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors
import warnings


def create_model(args, prior_model=None, mean=None, std=None):
    """
    Create a TorchMD model based on specified arguments.

    Args:
        args (dict): Dictionary containing model configuration parameters.
        prior_model (torch.nn.Module, optional): Prior model to incorporate.
        mean (float, optional): Mean value for normalization.
        std (float, optional): Standard deviation value for normalization.

    Returns:
        torch.nn.Module: Created TorchMD model.

    Raises:
        ValueError: If the specified architecture in `args["model"]` is unknown or invalid.
        AssertionError: If `args["prior_model"]` is requested but not provided in arguments.

    Notes:
        This function dynamically creates different types of models based on the `args["model"]` parameter.
        It supports various configurations including graph-networks, transformers, equivariant transformers,
        EGNNs, PaiNNs.

    """
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            md17=args["md17"],
            seperate_noise=args['seperate_noise'],
            **shared_args,
        )
    elif args["model"] == "egnn":
        from torchmdnet.models.egnn_clean import EGNN_finetune_last
        is_equivariant = True    
        # representation_model = EGNN_finetune_last(
        #     in_node_nf=300, hidden_nf=128, in_edge_nf=4, act_fn=nn.SiLU(), n_layers=7, residual=True, attention=True, normalize=False, tanh=False, use_layer_norm=False
        # )
        representation_model = EGNN_finetune_last(
            in_node_nf=300, hidden_nf=args['hidden_nf'], in_edge_nf=4, act_fn=nn.SiLU(), n_layers=args['n_layers'], residual=True, attention=True, normalize=False, tanh=False, use_layer_norm=False
        )
    elif args["model"] == "painn":
        from torchmdnet.models.painn import PaiNN
        is_equivariant = True
        representation_model = PaiNN(
            n_atom_basis=128,
            n_interactions=3,
            n_rbf=20,
            cutoff=5.0,
            max_z=100,
            n_out=1,
            readout='add',
        )
    elif args["model"] == "equivariant-transformerf2d":
        from torchmdnet.models.torchmd_etf2d import TorchMD_ETF2D

        is_equivariant = True
        representation_model = TorchMD_ETF2D(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            md17=args["md17"],
            seperate_noise=args['seperate_noise'],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # create the denoising output network
    output_model_noise = None
    if args['output_model_noise'] is not None:
        if args['bond_length_scale']:
            # output_bond_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            #     args["embedding_dimension"] * 2, args["activation"],
            # )
            # output_angle_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            #     args["embedding_dimension"] * 2, args["activation"],
            # )
            # output_dihedral_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            #     args["embedding_dimension"], args["activation"],
            # )


            # SIMPLE MLP Scalar head
            scalar_output_prefix = ''
            output_bond_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
            output_angle_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                args["embedding_dimension"] * 3, args["activation"],
            )
            output_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                args["embedding_dimension"] * 4, args["activation"],
            )
            output_rotate_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                args["embedding_dimension"] * 4, args["activation"],
            )


            output_model_noise = nn.ModuleList([output_bond_noise, output_angle_noise, output_dihedral_noise, output_rotate_dihedral_noise])

        else:
            output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"], args["activation"],
            )
    
    output_model_mask_atom = None 
    if args['mask_atom']:
        output_model_mask_atom = getattr(output_modules, "MaskHead", args["embedding_dimension"])(args["embedding_dimension"], args["activation"],) 
    
    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        output_model_mask_atom=output_model_mask_atom,
        position_noise_scale=args['position_noise_scale'],
        no_target_mean=args['no_target_mean'],
        seperate_noise=args['seperate_noise'],
        # bond length scale
        bond_length_scale=args['bond_length_scale'],
        
    )
    return model


# to copy the embedding of atom
# todo z start from 0 or 1(H)
pcq_with_h = {'name': 'pcq', 
              'atom_encoder': {'H': 0, 'He': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Mg': 8, 'Si': 9, 'P': 10, 'S': 11, 'Cl': 12, 'Ar': 13, 'Ca': 14, 'Ti': 15, 'Zn': 16, 'Ga': 17, 'Ge': 18, 'As': 19, 'Se': 20, 'Br': 21}, 
              'atomic_nb': [1, 2, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 20, 22, 30, 31, 32, 33, 34, 35], 
              'atom_decoder': ['H', 'He', 'Be', 'B', 'C', 'N', 'O', 'F', 'Mg', 'Si', 'P', 'S', 'Cl', 'Ar', 'Ca', 'Ti', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br'], 
              'max_n_nodes': 53, 
              'n_nodes': {2: 77, 3: 62, 4: 162, 5: 439, 6: 721, 7: 1154, 8: 1879, 9: 2758, 10: 4419, 11: 6189, 12: 9283, 13: 12620, 14: 18275, 15: 24340, 16: 31938, 17: 40477, 18: 51301, 19: 62211, 20: 74714, 21: 87453, 22: 103873, 23: 121040, 24: 135340, 25: 148497, 26: 165882, 27: 177003, 28: 185152, 29: 187492, 30: 204544, 31: 183114, 32: 183603, 33: 177381, 34: 174403, 35: 147153, 36: 129541, 37: 113794, 38: 99679, 39: 79646, 40: 59481, 41: 46282, 42: 36100, 43: 26546, 44: 17533, 45: 15672, 46: 13709, 47: 7774, 48: 1256, 49: 5445, 50: 955, 51: 118, 52: 1, 53: 125}, 
              'atom_types': {0: 51915415, 1: 5, 2: 2, 3: 17730, 4: 35554802, 5: 5667122, 6: 4981302, 7: 561570, 8: 2, 9: 33336, 10: 40407, 11: 506659, 12: 310138, 13: 3, 14: 2, 15: 4, 16: 4, 17: 4, 18: 369, 19: 299, 20: 1459, 21: 36399}, 
              'colors_dic': ['#FFFFFF99', 'C2', 'C7', 'C0', 'C3', 'C1', 'C5', 'C6', 'C4', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20'], 'radius_dic': [0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], 'with_h': True}


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    """
    Load a TorchMD model from a checkpoint file.

    Args:
        filepath (str): Filepath to the checkpoint file.
        args (dict, optional): Dictionary containing model configuration parameters.
        device (str, optional): Device to load the model onto (default is "cpu").
        mean (float, optional): Mean value for normalization.
        std (float, optional): Standard deviation value for normalization.
        **kwargs: Additional keyword arguments to update `args` with.

    Returns:
        torch.nn.Module: Loaded TorchMD model.
    """
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)
    
    if "state_dict" not in ckpt: # load diffusion model
        new_state_dict = {}
        for k, v in ckpt.items():
            # if 'pos_normalizer' not in k:
            if 'dynamics.gnn' in k:
                k = k.replace('dynamics.gnn.', '')
                new_state_dict[k] = v
        
        current_model_dict = model.representation_model.state_dict()
        # ommit mismatching shape
        new_state_dict2 = {}
        
        embedding_keys = []
        for k in current_model_dict:
            if k in new_state_dict:
                if current_model_dict[k].size() == new_state_dict[k].size():
                    new_state_dict2[k] = new_state_dict[k]
                else:
                    print(f'warning {k} shape mismatching, not loaded')
                    new_state_dict2[k] = current_model_dict[k]
                    new_state_dict2[k][pcq_with_h['atomic_nb']] = new_state_dict[k].T[:-1] + new_state_dict[k.split('.weight')[0] + '.bias']
                    embedding_keys.append(k)
        
        loading_return = model.representation_model.load_state_dict(new_state_dict2, strict=False)
        

    else:

        state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
        loading_return = model.load_state_dict(state_dict, strict=False)
    
    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        # assert all(("output_model_noise" in k or "pos_normalizer" in k) for k in loading_return.unexpected_keys)
        pass
    # assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"
    if len(loading_return.missing_keys) > 0:
        print(f'warning:  load model missing keys {loading_return.missing_keys}')

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.,
        no_target_mean=False,
        seperate_noise=False,
        output_model_mask_atom=None,
        bond_length_scale=0.,
    ):
        """
        TorchMD Network module combining a representation model and an output model for molecular dynamics.

        Args:
            representation_model (torch.nn.Module): Model for extracting molecular representations.
            output_model (torch.nn.Module): Model for predicting outputs based on representations.
            prior_model (torch.nn.Module, optional): Prior model for additional constraints (default: None).
            reduce_op (str, optional): Reduction operation for aggregating embeddings (default: "add").
            mean (float or torch.Tensor, optional): Mean value for normalization (default: None).
            std (float or torch.Tensor, optional): Standard deviation value for normalization (default: None).
            derivative (bool, optional): Whether to compute derivatives for md tasks (default: False).
            output_model_noise (torch.nn.ModuleList, optional): Models for predicting noisy outputs (default: None).
            position_noise_scale (float, optional): Scale for positional noise (default: 0.).
            no_target_mean (bool, optional): Whether to subtract mean from target (default: False).
            seperate_noise (bool, optional): Whether apply improved noisy nodes for md17 (default: False).
            output_model_mask_atom (torch.nn.Module, optional): Model for masking atoms type prediction (default: None).
            bond_length_scale (float, optional): Scale for bond length (default: 0.).
        """
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise        
        self.position_noise_scale = position_noise_scale
        self.no_target_mean = no_target_mean
        self.seperate_noise = seperate_noise

        self.bond_length_scale = bond_length_scale

        self.output_model_mask_atom = output_model_mask_atom

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        if self.position_noise_scale > 0 and not self.no_target_mean:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
        else:
            self.pos_normalizer = None

        if self.bond_length_scale > 0 and not self.no_target_mean:
            self.bond_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.angle_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.dihedral_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.rotate_dihedral_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            # TODO: self.output_model_noise: List
            hidden_channels = self.representation_model.hidden_channels

            self.angle_ijk_proj = nn.Linear(hidden_channels * 3, hidden_channels * 2)
            self.dihedral_jk_proj = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.bond_pos_normalizer = None
            self.angle_pos_normalizer = None
            self.dihedral_pos_normalizer = None

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None, batch_org = None, egnn_dict=None, radius_edge_index=None):
        """
        Forward pass of the TorchMD_Net module.

        Args:
            z (torch.Tensor): Input tensor representing atomic indices.
            pos (torch.Tensor): Input tensor representing atomic positions.
            batch (Optional[torch.Tensor], optional): Batch tensor for grouped nodes (default: None).
            batch_org (None, optional): Default: None.
            egnn_dict (None, optional): Dictionary of arguments for the EGNN model (default: None).
            radius_edge_index (None, optional): Edge index for radius (default: None).

        Returns:
            torch.Tensor: Predicted outputs.
            torch.Tensor or None: Predicted noise outputs.
            torch.Tensor or None: Mask logits for masking atoms in outputs.
        """
        if egnn_dict is not None:
            pred, noise_pred = self.representation_model(h=z, x=pos, edges=egnn_dict['edges'], edge_attr=egnn_dict['edge_attr'], node_mask=egnn_dict['node_mask'], n_nodes=egnn_dict['n_nodes'], mean=self.mean, std=self.std)
            return pred, noise_pred
        
        
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        if self.seperate_noise:
            x, v, nv, z, pos, batch = self.representation_model(z, pos, batch=batch)
        else:
            # run the potentially wrapped representation model
            if radius_edge_index is not None:
                x, v = self.representation_model(x=z, positions=pos, batch=batch, radius_edge_index=radius_edge_index)
            else:
                x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
            nv = None


        # whether mask or not
        mask_logits = None
        if self.output_model_mask_atom is not None:
            mask_logits = self.output_model_mask_atom.pre_reduce(x)


        
        if self.bond_length_scale > 0:
            # collect bond featrue
            bond_idx = batch_org.bond_target[:, :2].to(torch.long)
            bond_i_x = x[bond_idx[:, 0]]
            bond_j_x = x[bond_idx[:, 1]]
            bond_i_v = v[bond_idx[:, 0]]
            bond_j_v = v[bond_idx[:, 1]]

            # concat i and j
            bond_x = torch.cat([bond_i_x, bond_j_x], axis=1) # X * 512
            bond_v = torch.cat([bond_i_v, bond_j_v], axis=2) # X * 512
            
            # collect angle featrue
            angle_idx = batch_org.angle_target[:, :3].to(torch.long)
            angle_i_x = x[angle_idx[:, 0]]
            angle_j_x = x[angle_idx[:, 1]]
            angle_k_x = x[angle_idx[:, 2]]
            # angle_x = self.angle_ijk_proj(torch.cat([angle_i_x, angle_j_x, angle_k_x], axis=1))
            angle_x = torch.cat([angle_i_x, angle_j_x, angle_k_x], axis=1) 
            
            angle_i_v = v[angle_idx[:, 0]]
            angle_j_v = v[angle_idx[:, 1]]
            angle_k_v = v[angle_idx[:, 2]]

            angle_ji_v = angle_i_v - angle_j_v # TODO direction?
            angle_jk_v = angle_k_v - angle_j_v # TODO direction?
            angle_v = torch.cat([angle_ji_v, angle_jk_v], axis=2)
        
            # collect dihedral featrue
            dihedral_idx = batch_org.dihedral_target[:, :4].to(torch.long)
            # only pick j,k
            dihedral_i_x = x[dihedral_idx[:, 0]]
            dihedral_j_x = x[dihedral_idx[:, 1]]
            dihedral_k_x = x[dihedral_idx[:, 2]]
            dihedral_l_x = x[dihedral_idx[:, 3]]
            # dihedral_x = self.dihedral_jk_proj(torch.cat([dihedral_j_x, dihedral_k_x], axis=1))
            dihedral_x = torch.cat([dihedral_i_x, dihedral_j_x, dihedral_k_x, dihedral_l_x], axis=1)


            dihedral_j_v = v[dihedral_idx[:, 0]]
            dihedral_k_v = v[dihedral_idx[:, 1]]
            dihedral_v = dihedral_k_v - dihedral_j_v # TODO direction?


            rotate_dihedral_idx = batch_org.rotate_dihedral_target[:, :4].to(torch.long)
            # only pick j,k
            rotate_dihedral_i_x = x[rotate_dihedral_idx[:, 0]]
            rotate_dihedral_j_x = x[rotate_dihedral_idx[:, 1]]
            rotate_dihedral_k_x = x[rotate_dihedral_idx[:, 2]]
            rotate_dihedral_l_x = x[rotate_dihedral_idx[:, 3]]
            # dihedral_x = self.dihedral_jk_proj(torch.cat([dihedral_j_x, dihedral_k_x], axis=1))
            rotate_dihedral_x = torch.cat([rotate_dihedral_i_x, rotate_dihedral_j_x, rotate_dihedral_k_x, rotate_dihedral_l_x], axis=1)


            rotate_dihedral_j_v = v[rotate_dihedral_idx[:, 0]]
            rotate_dihedral_k_v = v[rotate_dihedral_idx[:, 1]]
            rotate_dihedral_v = rotate_dihedral_k_v - rotate_dihedral_j_v # TODO direction?

            


        # predict noise
        noise_pred = None
        if self.output_model_noise is not None:
            if nv is not None:
                noise_pred = self.output_model_noise.pre_reduce(x, nv, z, pos, batch)
            else:
                if self.bond_length_scale > 0:
                    bond_noise_pred = self.output_model_noise[0].pre_reduce(bond_x, bond_v, z, pos, batch).mean(axis=1)
                    angle_noise_pred = self.output_model_noise[1].pre_reduce(angle_x, angle_v, z, pos, batch).mean(axis=1)
                    dihedral_noise_pred = self.output_model_noise[2].pre_reduce(dihedral_x, dihedral_v, z, pos, batch).mean(axis=1)
                    rotate_dihedral_noise_pred = self.output_model_noise[3].pre_reduce(rotate_dihedral_x, rotate_dihedral_v, z, pos, batch).mean(axis=1)
                    
                    noise_pred = [bond_noise_pred, angle_noise_pred, dihedral_noise_pred, rotate_dihedral_noise_pred]
                else:
                    noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # shift by data mean
        if self.mean is not None:
            out = out + self.mean

        # apply output model after reduction
        out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, noise_pred, -dy
        # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        # return out, noise_pred, None
        return out, noise_pred, mask_logits


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        """
        AccumulatedNormalization module for running normalization of a tensor.

        Args:
            accumulator_shape (Tuple[int, ...]): Shape of the accumulator tensors.
            epsilon (float, optional): Small value added to denominator for numerical stability (default: 1e-8).

        """
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        """
        Update statistics for AccumulatedNormalization module.

        Args:
            batch (torch.Tensor): Batch tensor to update statistics.

        """
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        """
        Forward pass of the AccumulatedNormalization module.

        Args:
            batch (torch.Tensor): Batch tensor to be normalized.

        Returns:
            torch.Tensor: Normalized batch tensor.

        """
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)

