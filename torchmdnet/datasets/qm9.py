import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
import os
import numpy as np

from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
from rdkit import Chem

from torsion_utils import get_torsions, GetDihedral, apply_changes
import copy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph


class QM9(QM9_geometric):
    def __init__(self, root, transform=None, dataset_arg=None, add_radius_edge=False):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([self._filter_label, transform])

        self.add_radius_edge = add_radius_edge
        if self.add_radius_edge:
            self.radius = 5.0
        super(QM9, self).__init__(root, transform=transform)

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()

    # for debug
    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        """
        Retrieves and processes a data item at the specified index.

        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            Data: Processed data item.
        """
        org_data = super().__getitem__(idx)
        if self.add_radius_edge: # mimic the painn
            radius_edge_index = radius_graph(org_data.pos, r=self.radius, loop=False)
            org_data.radius_edge_index = radius_edge_index
        return org_data