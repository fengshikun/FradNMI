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
from torch_geometric.data import Data

class QM9(QM9_geometric):
    def __init__(self, root, transform=None, dataset_arg=None):
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



# Globle variable
MOL_LST = None

class QM9A(QM9_geometric):
    def __init__(self, root, transform=None, dataset_arg=None, dihedral_angle_noise_scale=0.1, position_noise_scale=0.005, composition=False, transform_y=None):
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
        
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition
        self.transform_y = transform_y

        raw_sdf_file = os.path.join(root, 'processed/qm9_mols.npy')

        global MOL_LST
        if MOL_LST is None:
            MOL_LST = np.load(raw_sdf_file, allow_pickle=True)

        super(QM9A, self).__init__(root, transform=transform)

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

    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise
    
    def default_noise(self, org_data):
        pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
        org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
        org_data.org_pos = org_data.pos
        org_data.pos = torch.tensor(pos_noise_coords)

    
        return org_data


    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol
        if self.transform_y is not None:
            org_data.y = self.transform_y(org_data)


        mol = copy.copy(MOL_LST[idx.item()])
        atom_num = mol.GetNumAtoms()
        

        # return self.default_noise(org_data)
        # get rotate bond
        try:
            Chem.SanitizeMol(mol)
            no_h_mol = Chem.RemoveHs(mol)
        except:
            print(f'Chem.RemoveHs failed on idx {idx.item()}')
            return self.default_noise(org_data)
        # rotable_bonds = get_torsions([mol])
        rotable_bonds = get_torsions([no_h_mol])

        

        # prob = random.random()
        assert atom_num == org_atom_num
        if len(rotable_bonds) == 0: # or prob < self.random_pos_prb:
            return self.default_noise(org_data)

        # else angel random
        # if len(rotable_bonds):
        try:
            org_angle = []
            for rot_bond in rotable_bonds:
                org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
            org_angle = np.array(org_angle)        
            noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
            new_mol = apply_changes(mol, noise_angle, rotable_bonds)
            
            coord_conf = new_mol.GetConformer()
            pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
            # pos_noise_coords = new_mol.GetConformer().GetPositions()
            for idx in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(idx)
                pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

            # coords = np.zeros((atom_num, 3), dtype=np.float32)
            # coord_conf = mol.GetConformer()
            # for idx in range(atom_num):
            #     c_pos = coord_conf.GetAtomPosition(idx)
            #     coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
            # coords = mol.GetConformer().GetPositions()

            pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
            
            
            # if self.composition or not len(rotable_bonds):
            #     pos_noise_coords = self.transform_noise(coords, self.position_noise_scale)
            #     if len(rotable_bonds): # set coords into the mol
            #         conf = mol.GetConformer()
            #         for i in range(mol.GetNumAtoms()):
            #             x,y,z = pos_noise_coords[i]
            #             conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            
            
            # org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            if self.composition:
                org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
                org_data.org_pos = org_data.pos
                org_data.pos = torch.tensor(pos_noise_coords)
            else:
                org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
                org_data.pos = torch.tensor(pos_noise_coords)
            
            return org_data
        except:
            print(f'rotate failed on idx {idx.item()}')
            return self.default_noise(org_data)


    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()
