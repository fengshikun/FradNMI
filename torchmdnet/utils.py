import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from typing import Any, Optional, Tuple, Union
from torch import Tensor


# code from equibind:

def isRingAromatic(mol, bondRing):
    """
    Check if all bonds in a ring are aromatic.

    Args:
        mol (rdkit.Chem.Mol): The molecule to analyze.
        bondRing (list): List of bond indices representing the ring.

    Returns:
        bool: True if all bonds in the specified ring are aromatic, False otherwise.
    """
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True

def get_geometry_graph_ring(lig, only_atom_ring=False):
    """
    Constructs a geometry graph for a molecular ligand.

    Args:
        lig (Chem.Mol): The molecular ligand object.
        only_atom_ring (bool, optional): If True, include only atoms in rings as destinations. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - edges_src (list): List of source indices for edges.
            - edges_dst (list): List of destination indices for edges.
            - feat (torch.Tensor): Tensor of edge features representing bond lengths.
    """

    rings = lig.GetRingInfo().AtomRings()
    bond_rings = lig.GetRingInfo().BondRings()
    edges_src = []
    edges_dst = []
    for i, atom in enumerate(lig.GetAtoms()):
        src_idx = atom.GetIdx()
        assert src_idx == i
        if not only_atom_ring:
            one_hop_dsts = [neighbor for neighbor in list(atom.GetNeighbors())]
            two_and_one_hop_idx = [neighbor.GetIdx() for neighbor in one_hop_dsts]
            for one_hop_dst in one_hop_dsts:
                for two_hop_dst in one_hop_dst.GetNeighbors():
                    two_and_one_hop_idx.append(two_hop_dst.GetIdx())
            all_dst_idx = list(set(two_and_one_hop_idx))
        else:
            all_dst_idx = []
        for ring_idx, ring in enumerate(rings):
            if src_idx in ring and isRingAromatic(lig,bond_rings[ring_idx]):
                all_dst_idx.extend(list(ring))
        all_dst_idx = list(set(all_dst_idx))
        if len(all_dst_idx) == 0: continue
        all_dst_idx.remove(src_idx)
        all_src_idx = [src_idx] *len(all_dst_idx)
        edges_src.extend(all_src_idx)
        edges_dst.extend(all_dst_idx)
    
    coords = lig.GetConformer().GetPositions()
    # graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=lig.GetNumAtoms(), idtype=torch.long)
    feat = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    return edges_src, edges_dst, feat
    # return {'edges_src': edges_src, 'edges_dst': edges_dst, 'feat': feat}
    # return graph



def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    """
    Split a dataset into training, validation, and test sets.

    Args:
        dset_len (int): Total number of samples in the dataset.
        train_size (float or int): Size of the training set. If float, interpreted as a proportion.
        val_size (float or int): Size of the validation set. If float, interpreted as a proportion.
        test_size (float or int): Size of the test set. If float, interpreted as a proportion.
        seed (int): Seed for random number generation.
        order (list or None): Optional list specifying the order of indices to use.

    Returns:
        tuple: A tuple containing:
            - np.array: Indices for the training set.
            - np.array: Indices for the validation set.
            - np.array: Indices for the test set.
    """

    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,
    splits=None,
    order=None,
):
    """
    Generate train, validation, and test splits for a dataset.

    Args:
        dataset_len (int): Total number of samples in the dataset.
        train_size (float): Proportion of the dataset to include in the training set.
        val_size (float): Proportion of the dataset to include in the validation set.
        test_size (float): Proportion of the dataset to include in the test set.
        seed (int): Random seed for reproducibility.
        filename (str, optional): If specified, save the splits to a NPZ file with this name.
        splits (str, optional): Path to a NPZ file containing pre-defined splits.
        order (str, optional): Optional ordering of the dataset if splits are not provided.

    Returns:
        tuple: A tuple containing torch tensors of the indices for train, validation, and test sets.
            - torch.Tensor: Indices for the training set.
            - torch.Tensor: Indices for the validation set.
            - torch.Tensor: Indices for the test set.

    Notes:
        - If `splits` is provided, it loads the splits from the NPZ file.
        - If `filename` is specified, it saves the generated splits to the specified NPZ file.
    """

    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


class LoadFromFile(argparse.Action):
    """
    Custom argparse action to load configuration from a YAML file.

    Usage:
        parser.add_argument('--file', type=open, action=LoadFromFile)

    This action reads a YAML configuration file and updates the argparse namespace
    with the contents of the file. It verifies that all keys in the YAML file are valid
    arguments for the parser, raising an error if an unknown key is encountered.

    Raises:
        ValueError: If the configuration file does not end with '.yaml' or '.yml',
                    or if there are unknown arguments in the configuration file.

    Attributes:
        parser: The ArgumentParser object.
        namespace: The namespace containing parsed arguments.
        values: The values from the command line.
        option_string: The option string used to invoke this action.

    """
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    """
    Action to load model configuration and path from a checkpoint.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.
        namespace (argparse.Namespace): The namespace containing parsed arguments.
        values (str): The value associated with the argument.
        option_string (str, optional): The option string passed to the argument.

    Raises:
        ValueError: If there are unknown arguments in the loaded configuration.

    """
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    """
    Save argparse arguments to a YAML file.

    Args:
        args (argparse.Namespace): The parsed arguments object.
        filename (str): The name of the file to save to.
        exclude (str or list, optional): Arguments to exclude from saving.

    Raises:
        ValueError: If the filename extension is not yaml or yml.

    """
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    """
    Converts a textual representation of a number to an appropriate numeric type.

    Args:
        text (str): The textual representation of the number.

    Returns:
        int or float or None: The converted numeric value or None if text is invalid.

    """
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float


class MissingEnergyException(Exception):
    pass


from torch_geometric.data import Data

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        # cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                # elif key  == 'connected_edge_indices':
                #     item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            # cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)



def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


def collate_fn(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    
    aggate_keys = batch[0].keys

    if 'name' in aggate_keys:
        aggate_keys.remove('name')
        aggate_keys.remove('edge_index')
        aggate_keys.remove('edge_attr')
        
        
        
        
    # for key in aggate_keys:
    #     batch[key] = batch_stack([mol[key] for mol in batch])
    
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in aggate_keys}

    to_keep = (batch['z'].sum(0) > 0)

    # batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
    
    aggate_keys = ['z', 'pos_target', 'pos']
    for key in aggate_keys:
        if key in batch:
            batch[key] = drop_zeros(batch[key], to_keep)
    
    # batch = {key: drop_zeros(batch[key], to_keep) for key in batch.keys()}

    atom_mask = batch['z'] > 0
    batch['atom_mask'] = atom_mask

    #Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    #mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    if 'y' in batch:
        batch['y'] = batch['y'].reshape(-1, 1)

    batch_res = Data()
    
    for k, v in batch.items():
        batch_res[k] = v

    return batch_res

def process_input(atom_type,max_atom_type=100, charge_power=2):
    one_hot = nn.functional.one_hot(atom_type, max_atom_type)
    charge_tensor = (atom_type.unsqueeze(-1) / max_atom_type).pow(
        torch.arange(charge_power + 1., dtype=torch.float32).to(atom_type))
    charge_tensor = charge_tensor.view(atom_type.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
    return atom_scalars

def binarize(x):
    return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj, order):
    """
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    """

    adj_mats = [torch.eye(adj.size(0), device=adj.device), \
                binarize(adj + torch.eye(adj.size(0), device=adj.device))]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj).float()

    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i
    return order_mat.long()

def dense_to_sparse(adj: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> # Forr a single adjacency matrix
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be 2- or "
                         f"3-dimensional (got {adj.dim()} dimensions)")

    edge_index = adj.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
        row = edge_index[1] + adj.size(-2) * edge_index[0]
        col = edge_index[2] + adj.size(-1) * edge_index[0]
        return torch.stack([row, col], dim=0), edge_attr




def gen_fully_connected_with_hop(pos, mask):
    """
    Generates fully connected graph edges with hop constraints based on position and mask inputs.

    Args:
        pos (torch.Tensor): Tensor of shape (batch_size, num_nodes, num_dims) representing node positions.
        mask (torch.Tensor): Binary mask of shape (batch_size, num_nodes) indicating node existence.

    Returns:
        edge_index (torch.Tensor): Sparse edge index tensor of shape (2, num_edges) representing graph edges.
        edge_type (torch.Tensor): Tensor of shape (num_edges,) indicating edge types after dense-to-sparse conversion.
    """

    batch, nodes = mask.shape
    batch_adj = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), p=2, dim=-1) # batch * n * n
    batch_mask_fc = mask[:, :, None] * mask[:, None, :] # batch * n * n
    # 1.6 is an empirically reasonable cutoff to distinguish the existence of bonds for stable small molecules
    batch_mask = batch_mask_fc.bool() & (batch_adj <= 1.6) & (~torch.eye(nodes).to(mask).bool()) # batch * n * n
    batch_mask = torch.block_diag(*batch_mask)
    adj_order = get_higher_order_adj_matrix(batch_mask,3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    fc_mask = batch_mask_fc.bool() & (~torch.eye(nodes).to(mask).bool())
    fc_mask = torch.block_diag(*fc_mask)
    type_new = batch_mask + type_highorder + fc_mask
    edge_index, edge_type = dense_to_sparse(type_new)
    return edge_index, edge_type - 1