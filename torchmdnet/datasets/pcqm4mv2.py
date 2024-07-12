from typing import Optional, Callable, List

import os
from tqdm import tqdm
import glob
import ase
import numpy as np
from rdkit import Chem
from torchmdnet.utils import isRingAromatic, get_geometry_graph_ring
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
import random
import torch.nn.functional as F
import copy
import lmdb
import pickle

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

from torsion_utils import get_torsions, GetDihedral, apply_changes, get_rotate_order_info, add_equi_noise, add_equi_noise_new
from rdkit.Geometry import Point3D
from torch_geometric.nn import radius_graph


class PCQM4MV2_XYZ(InMemoryDataset):
    r"""3D coordinates for molecules in the PCQM4Mv2 dataset (from zip).
    """

    raw_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2 does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['pcqm4m-v2_xyz']

    @property
    def processed_file_names(self) -> str:
        return 'pcqm4mv2__xyz.pt'

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self):
        dataset = PCQM4MV2_3D(self.raw_paths[0])
        
        data_list = []
        for i, mol in enumerate(tqdm(dataset)):
            pos = mol['coords']
            pos = torch.tensor(pos, dtype=torch.float)
            z = torch.tensor(mol['atom_type'], dtype=torch.long)

            data = Data(z=z, pos=pos, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


# Globle variable
MOL_LST = None
MOL_DEBUG_LST = None
debug = False
debug_cnt = 0





# use force filed definition
# bond length, angle ,dihedral angel
# equilibrium
EQ_MOL_LST = None
EQ_EN_LST = None

class PCQM4MV2_Dihedral2(PCQM4MV2_XYZ):
    '''
    We process the data by adding noise to atomic coordinates and providing denoising targets for denoising pre-training.
    '''
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, decay=False, decay_coe=0.2, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, equilibrium=False, eq_weight=False, cod_denoise=False, integrate_coord=False, addh=False, mask_atom=False, mask_ratio=0.15, bat_noise=False, add_radius_edge=False):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # angle noise as the start

        self.decay = decay
        self.decay_coe = decay_coe

        self.random_pos_prb = 0.5
        self.equilibrium = equilibrium # equilibrium settings
        self.eq_weight = eq_weight
        self.cod_denoise = cod_denoise # reverse to coordinate denoise

        self.integrate_coord = integrate_coord
        self.addh = addh

        self.mask_atom = mask_atom
        self.mask_ratio = mask_ratio
        self.num_atom_type = 119

        self.bat_noise = bat_noise
        
        global MOL_LST
        global EQ_MOL_LST
        global EQ_EN_LST
        if self.equilibrium and EQ_MOL_LST is None:
            # debug
            EQ_MOL_LST = np.load('MG_MOL_All.npy', allow_pickle=True) # mol lst
            EQ_EN_LST = np.load('MG_All.npy', allow_pickle=True) # energy lst
        else:
            if MOL_LST is None:
            # import pickle
            # with open(sdf_path, 'rb') as handle:
            #     MOL_LST = pickle.load(handle)
            # MOL_LST = np.load("mol_iter_all.npy", allow_pickle=True)
                # MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
                MOL_LST = lmdb.open(f'{root}/MOL_LMDB', readonly=True, subdir=True, lock=False)
            
        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
        self.add_radius_edge = add_radius_edge
        if self.add_radius_edge:
            self.radius = 5.0
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise
    
    def transform_noise_decay(self, data, position_noise_scale, decay_coe_lst):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale * torch.tensor(decay_coe_lst)
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        """
        Retrieves and processes a data item at the specified index, adding noise to atomic coordinates and providing
        denoising targets for denoising pre-training.

        Args:
            idx (Union[int, np.integer, IndexType]): Index of the data item to retrieve.

        Returns:
            Union['Dataset', Data]: Processed data item with original and noisy coordinates, and denoising targets.

        Notes:
            When processing data, if `bat_noise` is enabled, the 'bond angle torsion noise' is added to the molecule's equilibrium 
            conformation. Otherwise, dihedral angle noise is applied. Gaussian coordinate noise is subsequently added.
        """
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol

        # check whether mask or not
        if self.mask_atom:
            num_atoms = org_data.z.size(0)
            sample_size = int(num_atoms * self.mask_ratio + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
            org_data.mask_node_label = org_data.z[masked_atom_indices]
            org_data.z[masked_atom_indices] = self.num_atom_type
            org_data.masked_atom_indices = torch.tensor(masked_atom_indices)

        if self.equilibrium:
            # for debug
            # max_len = 422325 - 1
            # idx = idx.item() % max_len
            idx = idx.item()
            mol = copy.copy(EQ_MOL_LST[idx])
            energy_lst = EQ_EN_LST[idx]
            eq_confs = len(energy_lst)
            conf_num = mol.GetNumConformers()
            assert conf_num == (eq_confs + 1)
            if eq_confs:
                weights = F.softmax(-torch.tensor(energy_lst))
                # random pick one
                pick_lst = [idx for idx in range(conf_num)]
                p_idx = random.choice(pick_lst)
                
                for conf_id in range(conf_num):
                    if conf_id != p_idx:
                        mol.RemoveConformer(conf_id)
                # only left p_idx
                if p_idx == 0:
                    weight = 1
                else:
                    if self.eq_weight:
                        weight = 1
                    else:
                        weight = weights[p_idx - 1].item()
                        
            else:
                weight = 1
            
        else:
            ky = str(idx.item()).encode()
            serialized_data = MOL_LST.begin().get(ky)
            mol = pickle.loads(serialized_data)
            # mol = MOL_LST[idx.item()]


        atom_num = mol.GetNumAtoms()

        # get rotate bond
        if self.addh:
            rotable_bonds = get_torsions([mol])
        else:
            no_h_mol = Chem.RemoveHs(mol)
            rotable_bonds = get_torsions([no_h_mol])
        

        # prob = random.random()
        cod_denoise = self.cod_denoise
        if self.integrate_coord:
            assert not self.cod_denoise
            prob = random.random()
            if prob < 0.5:
                cod_denoise = True
            else:
                cod_denoise = False

        if atom_num != org_atom_num or len(rotable_bonds) == 0 or cod_denoise: # or prob < self.random_pos_prb:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)

            
            if self.add_radius_edge: # mimic the painn
                radius_edge_index = radius_graph(org_data.pos, r=self.radius, loop=False)
                org_data.radius_edge_index = radius_edge_index
            
            if self.equilibrium:
                org_data.w1 = weight
                org_data.wg = torch.tensor([weight for _ in range(org_atom_num)], dtype=torch.float32)
            return org_data

        # else angel random
        # if len(rotable_bonds):
        org_angle = []
        if self.decay:
            rotate_bonds_order, rb_depth = get_rotate_order_info(mol, rotable_bonds)
            decay_coe_lst = []
            for i, rot_bond in enumerate(rotate_bonds_order):
                org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
                decay_scale = (self.decay_coe) ** (rb_depth[i] - 1)    
                decay_coe_lst.append(self.dihedral_angle_noise_scale*decay_scale)
            noise_angle = self.transform_noise_decay(org_angle, self.dihedral_angle_noise_scale, decay_coe_lst)
            new_mol = apply_changes(mol, noise_angle, rotate_bonds_order)
        else:
            if self.bat_noise:
                new_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst, specific_var_lst = add_equi_noise_new(mol, add_ring_noise=False)
            else:
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

        if self.bat_noise:
            # check nan
            if torch.tensor(pos_noise_coords_angle).isnan().sum().item():# contains nan
                print('--------bat nan, revert back to org coord-----------')
                pos_noise_coords_angle = org_data.pos.numpy()



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
            org_data.pos = torch.tensor(pos_noise_coords)
        else:
            # denoise angle + guassian noise
            # org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            # org_data.pos = torch.tensor(pos_noise_coords)
            
            # only denoise angle
            org_data.pos_target = torch.tensor(pos_noise_coords_angle - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords_angle)
        
        if self.equilibrium:
            org_data.w1 = weight
            org_data.wg = torch.tensor([weight for _ in range(atom_num)], dtype=torch.float32)

        if self.add_radius_edge: # mimic the painn
            radius_edge_index = radius_graph(org_data.pos, r=self.radius, loop=False)
            org_data.radius_edge_index = radius_edge_index
        
        return org_data

