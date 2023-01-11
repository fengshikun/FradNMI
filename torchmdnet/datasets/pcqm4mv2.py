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


import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

from torsion_utils import get_torsions, GetDihedral, apply_changes, get_rotate_order_info
from rdkit.Geometry import Point3D


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
class PCQM4MV2_XYZ_BIAS(PCQM4MV2_XYZ):
    #  sdf path: pcqm4m-v2-train.sdf
    # set the transform to None
    def __init__(self, root: str, sdf_path: str, position_noise_scale: float, sample_number: int, violate: bool, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2_XYZ_BIAS does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.position_noise_scale = position_noise_scale
        self.sample_number = sample_number
        self.violate = violate
        global MOL_LST
        if MOL_LST is None:
            import pickle
            with open(sdf_path, 'rb') as handle:
                MOL_LST = pickle.load(handle)
        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
        # import pickle
        # with open(sdf_path, 'rb') as handle:
        #     self.mol_lst = pickle.load(handle)

        print('PCQM4MV2_XYZ_BIAS Initialization finished')

    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(data) * position_noise_scale
        data_noise = data + noise
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        # change org_data coordinate
        # get mol
        molinfo = MOL_LST[idx.item()]
        edges_src, edges_dst, org_coordinate = molinfo
        atom_woh_number = org_coordinate.shape[0]
        
        coords = org_data.pos

        repeat_coords = coords.unsqueeze(0).repeat(self.sample_number, 1, 1)
        noise_coords = self.transform_noise(repeat_coords, self.position_noise_scale)
        noise_feat = torch.linalg.norm(noise_coords[:,edges_src] - noise_coords[:,edges_dst], dim=2)
        feat = torch.linalg.norm(coords[edges_src] - coords[edges_dst], dim=1)
        loss_lst = torch.mean((noise_feat**2 - feat ** 2)**2, dim=1)
        # sorted_value, sorted_idx = torch.sort(loss_lst)
        
        # min_violate_idx, max_violate_idx = sorted_idx[0], sorted_idx[-1]
        
        if self.violate:
            # new_coords = noise_coords[max_violate_idx]
            new_coords = noise_coords[torch.argmax(loss_lst)]
        else:
            # new_coords = noise_coords[min_violate_idx]
            new_coords = noise_coords[torch.argmin(loss_lst)]
        
        org_data.pos_target = new_coords - coords
        org_data.pos = new_coords
        
        global debug_cnt
        if debug:
            import copy
            from rdkit.Geometry import Point3D
            mol = MOL_DEBUG_LST[idx.item()]
            violate_coords = noise_coords[torch.argmax(loss_lst)].cpu().numpy()
            n_violate_coords = noise_coords[torch.argmin(loss_lst)].cpu().numpy()
            mol_cpy = copy.copy(mol)
            conf = mol_cpy.GetConformer()
            for i in range(mol.GetNumAtoms()):
                x,y,z = n_violate_coords[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            

            writer = Chem.SDWriter(f'org_{debug_cnt}.sdf')
            writer.write(mol)
            writer.close()

            # supplier = Chem.SDMolSupplier('v3000.sdf')
            writer = Chem.SDWriter(f'min_noise_{debug_cnt}.sdf')
            writer.write(mol_cpy)
            writer.close()
            # show mol coordinate
            mol_cpy = copy.copy(mol)
            conf = mol_cpy.GetConformer()
            for i in range(mol.GetNumAtoms()):
                x,y,z = violate_coords[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

            writer = Chem.SDWriter(f'max_noise_{debug_cnt}.sdf')
            writer.write(mol_cpy)
            writer.close()
            debug_cnt += 1
            if debug_cnt > 10:
                exit(0)

        return org_data


class PCQM4MV2_Dihedral(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # compose dihedral angle and position noise
        global MOL_LST
        if MOL_LST is None:
            # import pickle
            # with open(sdf_path, 'rb') as handle:
            #     MOL_LST = pickle.load(handle)
            # MOL_LST = np.load("mol_iter_all.npy", allow_pickle=True)
                MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)

        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol
        mol = MOL_LST[idx.item()]
        atom_num = mol.GetNumAtoms()
        if atom_num != org_atom_num:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos)
            org_data.pos = torch.tensor(pos_noise_coords)
        
            return org_data


        coords = np.zeros((atom_num, 3), dtype=np.float32)
        coord_conf = mol.GetConformer()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # coords = mol.GetConformer().GetPositions()

        
        # get rotate bond
        rotable_bonds = get_torsions([mol])
        
        if self.composition or not len(rotable_bonds):
            pos_noise_coords = self.transform_noise(coords, self.position_noise_scale)
            if len(rotable_bonds): # set coords into the mol
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    x,y,z = pos_noise_coords[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        

        if len(rotable_bonds):
            org_angle = []
            for rot_bond in rotable_bonds:
                org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
            org_angle = np.array(org_angle)        
            noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
            new_mol = apply_changes(mol, noise_angle, rotable_bonds)
            
            coord_conf = new_mol.GetConformer()
            pos_noise_coords = np.zeros((atom_num, 3), dtype=np.float32)
            # pos_noise_coords = new_mol.GetConformer().GetPositions()
            for idx in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(idx)
                pos_noise_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

        
        org_data.pos_target = torch.tensor(pos_noise_coords - coords)
        org_data.pos = torch.tensor(pos_noise_coords)
        
        return org_data


# equilibrium
EQ_MOL_LST = None
EQ_EN_LST = None

class PCQM4MV2_Dihedral2(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, decay=False, decay_coe=0.2, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, equilibrium=False, eq_weight=False, cod_denoise=False):
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
                MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
            
        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise
    
    def transform_noise_decay(self, data, position_noise_scale, decay_coe_lst):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale * torch.tensor(decay_coe_lst)
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol

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
            mol = MOL_LST[idx.item()]


        atom_num = mol.GetNumAtoms()

        # get rotate bond
        no_h_mol = Chem.RemoveHs(mol)
        # rotable_bonds = get_torsions([mol])
        rotable_bonds = get_torsions([no_h_mol])

        # prob = random.random()
        if atom_num != org_atom_num or len(rotable_bonds) == 0 or self.cod_denoise: # or prob < self.random_pos_prb:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)

            
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
            org_data.pos = torch.tensor(pos_noise_coords)
        else:
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)
        
        if self.equilibrium:
            org_data.w1 = weight
            org_data.wg = torch.tensor([weight for _ in range(atom_num)], dtype=torch.float32)

        return org_data



class PCQM4MV2_3D:
    """Data loader for PCQM4MV2 from raw xyz files.
    
    Loads data given a path with .xyz files.
    """
    
    def __init__(self, path) -> None:
        self.path = path
        self.xyz_files = glob.glob(path + '/*/*.xyz')
        self.xyz_files = sorted(self.xyz_files, key=self._molecule_id_from_file)
        self.num_molecules = len(self.xyz_files)
        
    def read_xyz_file(self, file_path):
        atom_types = np.genfromtxt(file_path, skip_header=1, usecols=range(1), dtype=str)
        atom_types = np.array([ase.Atom(sym).number for sym in atom_types])
        atom_positions = np.genfromtxt(file_path, skip_header=1, usecols=range(1, 4), dtype=np.float32)        
        return {'atom_type': atom_types, 'coords': atom_positions}
    
    def _molecule_id_from_file(self, file_path):
        return int(os.path.splitext(os.path.basename(file_path))[0])
    
    def __len__(self):
        return self.num_molecules
    
    def __getitem__(self, idx):
        return self.read_xyz_file(self.xyz_files[idx])