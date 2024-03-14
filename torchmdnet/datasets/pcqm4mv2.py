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

from torsion_utils import get_torsions, GetDihedral, apply_changes, get_rotate_order_info, add_equi_noise
# , add_equi_noise_new
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




# use force filed definition
# bond length, angle ,dihedral angel
class PCQM4MV2_Force(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, angle_noise_scale: float, bond_length_scale: float, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.angle_noise_scale = angle_noise_scale
        self.bond_length_scale = bond_length_scale
        
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

        # add noise to mol with different types of noise
        noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst = add_equi_noise(mol, bond_var=self.bond_length_scale, angle_var=self.angle_noise_scale, torsion_var=self.dihedral_angle_noise_scale)

        # get noise_mol coordinate
        atom_num = mol.GetNumAtoms()
        # assert atom_num == org_atom_num # todo, we may need handle such situation

        if atom_num != org_atom_num:
            print('assert atom_num == org_atom_num failed')
            atoms = mol.GetAtoms()
            z_lst = []
            for i in range(atom_num):
                atom = atoms[i]
                z_lst.append(atom.GetAtomicNum()) # atomic num start from 1
            
            org_data.z = torch.tensor(z_lst) # atomic num start from 1


        # get coordinates
        coords = np.zeros((atom_num, 3), dtype=np.float32)
        coord_conf = noise_mol.GetConformer()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # coords = mol.GetConformer().GetPositions()

        # set coordinate to atom
        org_data.pos = torch.tensor(coords)
        org_data.bond_target = torch.tensor(bond_label_lst)
        org_data.angle_target = torch.tensor(angle_label_lst)
        org_data.dihedral_target = torch.tensor(dihedral_label_lst)
        org_data.rotate_dihedral_target = torch.tensor(rotate_dihedral_label_lst)

        return org_data

# equilibrium
EQ_MOL_LST = None
EQ_EN_LST = None

class PCQM4MV2_Dihedral2(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, decay=False, decay_coe=0.2, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, equilibrium=False, eq_weight=False, cod_denoise=False, integrate_coord=False, addh=False, mask_atom=False, mask_ratio=0.15, bat_noise=False):
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
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)
        
        if self.equilibrium:
            org_data.w1 = weight
            org_data.wg = torch.tensor([weight for _ in range(atom_num)], dtype=torch.float32)

        return org_data



# learn force field exp
ORG_MOLS = None
SAMPLE_POS = None
FORCES_LABEL = None

# exp1: noisy node --> dft force feild
# exp2: noisy node --> noisy
# exp3: frad noisy node


# exp4: use rkdit conformation: frad or coord denoise
class PCQM4MV2_DihedralF(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, force_field: bool=False, pred_noise: bool=False, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, cod_denoise=False, rdkit_conf=False):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # compose dihedral angle and position noise
        self.force_field = force_field
        self.pred_noise = pred_noise

        self.rdkit_conf = rdkit_conf

        
        global ORG_MOLS
        global SAMPLE_POS
        global FORCES_LABEL
        if ORG_MOLS is None:
            if self.rdkit_conf:
                ORG_MOLS = np.load('/home/fengshikun/Pretraining-Denoising/rdkit_mols_conf_lst.npy', allow_pickle=True)    
            else:
                ORG_MOLS = np.load('/home/fengshikun/Backup/Denoising/data/dft/head_1w/mols_head_1w.npy', allow_pickle=True)
                SAMPLE_POS = np.load('/home/fengshikun/Backup/Denoising/data/dft/head_1w/mols_head_1w_pos.npy', allow_pickle=True)
                FORCES_LABEL = np.load('/home/fengshikun/Backup/Denoising/data/dft/head_1w/mols_head_1w_force.npy', allow_pickle=True)
        self.mol_num = len(ORG_MOLS)
        self.cod_denoise = cod_denoise
        print(f'load PCQM4MV2_DihedralF complete, mol num is {self.mol_num}')
    
    def process_data(self, max_node_num=30):
        pass


    def __len__(self):
        return self.mol_num

    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_mol = ORG_MOLS[idx.item()]
        # get org pos

        org_atom_num = org_data.pos.shape[0]

        atom_num = org_mol.GetNumAtoms()

        
        # assert org_atom_num == atom_num
        org_pos = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        
        # check the conformers number(when use the rdkit generated conformers, conf number may be zero)
        conf_num = org_mol.GetNumConformers()
        if not conf_num:
            # use the orginal pos
            assert self.rdkit_conf # this only happen when use rdkit generated conf
            org_pos = org_data.pos
            coord_conf = Chem.Conformer(org_mol.GetNumAtoms())

            if org_atom_num != atom_num:
                pos_noise_coords = self.transform_noise(org_pos, self.position_noise_scale)
                org_data.pos_target = torch.tensor(pos_noise_coords - org_pos)
                org_data.pos = torch.tensor(pos_noise_coords)
                return org_data

            for i in range(atom_num):
                coord_conf.SetAtomPosition(i, (org_pos[i][0].item(), org_pos[i][1].item(), org_pos[i][2].item()))
            org_mol.AddConformer(coord_conf)
        else:
            coord_conf = org_mol.GetConformer()
            atoms = org_mol.GetAtoms()
            z_lst = [] # the force filed data may not consistant with the original data with same index. we only pick mols which have less atoms than 30 atoms.
            for i in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(i)
                org_pos[i] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
                atom = atoms[i]
                z_lst.append(atom.GetAtomicNum()) # atomic num start from 1
            
            org_data.z = torch.tensor(z_lst) # atomic num start from 1
        # random sample one pos
        if self.force_field or self.pred_noise:
            sample_poses = SAMPLE_POS[idx]
            sample_pos_num = len(sample_poses)
            random_idx = random.randint(0, sample_pos_num - 1)
            sample_pos = sample_poses[random_idx]
            
            force_label = FORCES_LABEL[idx][random_idx]

        if self.force_field:
            org_data.pos_target = torch.tensor(force_label)
            org_data.pos = torch.tensor(sample_pos)
        elif self.pred_noise:
            org_data.pos_target = torch.tensor(sample_pos - org_pos)
            org_data.pos = torch.tensor(sample_pos)
        elif self.composition:
            rotable_bonds = get_torsions([org_mol])
            if len(rotable_bonds) == 0 or self.cod_denoise:
                pos_noise_coords = self.transform_noise(org_pos, self.position_noise_scale)
                org_data.pos_target = torch.tensor(pos_noise_coords - org_pos)
                org_data.pos = torch.tensor(pos_noise_coords)
                return org_data

            org_angle = []
            for rot_bond in rotable_bonds:
                org_angle.append(GetDihedral(org_mol.GetConformer(), rot_bond))
            org_angle = np.array(org_angle)        
            noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
            new_mol = apply_changes(org_mol, noise_angle, rotable_bonds)
        
            coord_conf = new_mol.GetConformer()
            pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
            # pos_noise_coords = new_mol.GetConformer().GetPositions()
            for idx in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(idx)
                pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

            pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
            org_data.pos = torch.tensor(pos_noise_coords)
        else:
            raise Exception('Not implemented situation, one of pred_noise, composition and force_filed should be true')
        
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