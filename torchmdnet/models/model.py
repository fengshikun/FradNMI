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
            output_bond_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
            output_angle_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
            output_dihedral_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"], args["activation"],
            )
            output_model_noise = nn.ModuleList([output_bond_noise, output_angle_noise, output_dihedral_noise])

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


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)

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

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None, batch_org = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        if self.seperate_noise:
            x, v, nv, z, pos, batch = self.representation_model(z, pos, batch=batch)
        else:
            # run the potentially wrapped representation model
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
            angle_x = self.angle_ijk_proj(torch.cat([angle_i_x, angle_j_x, angle_k_x], axis=1))    
            
            angle_i_v = v[angle_idx[:, 0]]
            angle_j_v = v[angle_idx[:, 1]]
            angle_k_v = v[angle_idx[:, 2]]

            angle_ji_v = angle_i_v - angle_j_v # TODO direction?
            angle_jk_v = angle_k_v - angle_j_v # TODO direction?
            angle_v = torch.cat([angle_ji_v, angle_jk_v], axis=2)
        
            # collect dihedral featrue
            dihedral_idx = batch_org.dihedral_target[:, 1:3].to(torch.long)
            # only pick j,k
            dihedral_j_x = x[dihedral_idx[:, 0]]
            dihedral_k_x = x[dihedral_idx[:, 1]]
            dihedral_x = self.dihedral_jk_proj(torch.cat([dihedral_j_x, dihedral_k_x], axis=1))
            
            dihedral_j_v = v[dihedral_idx[:, 0]]
            dihedral_k_v = v[dihedral_idx[:, 1]]
            dihedral_v = dihedral_k_v - dihedral_j_v # TODO direction?

            


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
                    noise_pred = [bond_noise_pred, angle_noise_pred, dihedral_noise_pred]
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
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
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
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)

