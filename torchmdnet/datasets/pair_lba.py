import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)
import pandas as pd
from Bio.PDB import PDBParser
import pickle
import os
import lmdb
import numpy as np


atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118, 'D': 119}

# captialize
atomic_number_cap = {}
for key in atomic_number:
    atomic_number_cap[key.upper()] = atomic_number[key]
atomic_number = atomic_number_cap

def process_one_pdb(pdbfile):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('pdb_file', pdbfile)[0]
    recpt = list(structure['R'].get_atoms())
    pocket_atom_type = [x.element for x in recpt]
    pocket_coord = [x.coord for x in recpt]
    real_lig = list(structure['L'].get_atoms())
    real_lig_atom_type = [x.element for x in real_lig]
    real_lig_coord = [x.coord for x in real_lig]
    return {
            'pocket_atoms': pocket_atom_type, 
            'pocket_coordinates': np.array(pocket_coord), 
            'lig_atoms_real': real_lig_atom_type, 
            'lig_coord_real': np.array(real_lig_coord), 
    }

# pairA - pairB = diff
class ChEMBL(InMemoryDataset):
    def __init__(self, pd_file='train_Pairs_0.5t.csv', data_col_name='pairA', threshold=1.5, with_label=False, feat_dict='/data/protein/SKData/Uni-Mol/unimol_tools/PDB_feat_idx_dict.pkl', feat_path='/data/protein/SKData/PocData/old_data/feat_path', use_lig_feat=False):
        df = pd.read_csv(pd_file)
        tdf = df[df['diff'] > threshold]
        self.pdb_lst = tdf[data_col_name].to_numpy()
        self.with_label = with_label
        if with_label:
            self.aff_lst = tdf['diff'].to_numpy()
        self.length = len(self.pdb_lst)
        print(f'load ChEMBL dataset, pair diff threshold: {threshold}, column: {data_col_name}, length: {self.length}')
        self.use_lig_feat = use_lig_feat
        if use_lig_feat:
            with open(feat_dict, 'rb') as fp:
                self.feat_idx_dict = pickle.load(fp)
        
            self.env = lmdb.open('/data/protein/SKData/PocData/old_data/feat_path', readonly=True, subdir=True, lock=False)
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
    
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx, drop_atom_lst=['H', 'D']):
        data = Data()
        # get z, pos, and y
        # read element
        pdb_file = self.pdb_lst[idx]
        org_data = process_one_pdb(pdb_file)
        
        pocket_atoms = org_data['pocket_atoms']
        pocket_coordinates = org_data['pocket_coordinates']
        # pocket_z = [atomic_number.index(ele) for ele in pocket_atoms]
        pocket_z = [atomic_number[ele] for ele in pocket_atoms]
        
        lig_atoms_real = org_data['lig_atoms_real']
        lig_coord_real = org_data['lig_coord_real']
        # lig_z = [atomic_number.index(ele) for ele in lig_atoms_real]
        lig_z = [atomic_number[ele] for ele in lig_atoms_real]


        num_atoms = len(pocket_atoms) + len(lig_atoms_real)

        

        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[len(pocket_atoms): ] = 1 # lig 1
        
        # concat z and pos
        pocket_atoms.extend(lig_atoms_real)
        pocket_z.extend(lig_z)
        all_pos = np.concatenate([pocket_coordinates, lig_coord_real])

        data.z = torch.tensor(pocket_z, dtype=torch.long) 
        data.pos = torch.tensor(all_pos, dtype=torch.float32)
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        
        if self.use_lig_feat:
            ky = self.feat_idx_dict[pdb_file]
            ky = str(ky).encode()
            feat = deserialize_array(self.env.begin().get(ky)).reshape(-1, 512)
            data.lig_feat = feat[1:-1]
        
        if len(drop_atom_lst): # erase H
            pocket_atoms = np.array(pocket_atoms)
            mask_idx = np.in1d(pocket_atoms, drop_atom_lst)
            data.z = data.z[~mask_idx]
            data.pos = data.pos[~mask_idx]
            data.type_mask = data.type_mask[~mask_idx]


        # if self.transform_noise is not None:
        #     data = self.transform_noise(data) # noisy node
        if self.with_label:
            data.diff = self.aff_lst[idx]
        
        return data



class ChEMBLReg(InMemoryDataset):
    def __init__(self, pdb_lst_file='/data/protein/SKData/ChEMBL_dataset_sample/train_pdb_lst.pkl', oracle_label_dict='/data/protein/SKData/ChEMBL_dataset_sample/PDB_Exp_values_dict.pkl'):
        with open(pdb_lst_file, 'rb') as fp:
            self.pdb_lst = pickle.load(fp)
        with open(oracle_label_dict, 'rb') as fp:
            self.label_dict = pickle.load(fp)
        self.length = len(self.pdb_lst)
            
        
    
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx, drop_atom_lst=['H', 'D']):
        data = Data()
        # get z, pos, and y
        # read element
        pdb_file = self.pdb_lst[idx]
        org_data = process_one_pdb(pdb_file)
        
        pocket_atoms = org_data['pocket_atoms']
        pocket_coordinates = org_data['pocket_coordinates']
        # pocket_z = [atomic_number.index(ele) for ele in pocket_atoms]
        pocket_z = [atomic_number[ele] for ele in pocket_atoms]
        
        lig_atoms_real = org_data['lig_atoms_real']
        lig_coord_real = org_data['lig_coord_real']
        # lig_z = [atomic_number.index(ele) for ele in lig_atoms_real]
        lig_z = [atomic_number[ele] for ele in lig_atoms_real]


        num_atoms = len(pocket_atoms) + len(lig_atoms_real)

        # concat z and pos
        pocket_atoms.extend(lig_atoms_real)
        pocket_z.extend(lig_z)
        all_pos = np.concatenate([pocket_coordinates, lig_coord_real])

        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[len(pocket_atoms): ] = 1 # lig 1

        data.z = torch.tensor(pocket_z, dtype=torch.long) 
        data.pos = torch.tensor(all_pos, dtype=torch.float32)
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        if len(drop_atom_lst): # erase H
            pocket_atoms = np.array(pocket_atoms)
            mask_idx = np.in1d(pocket_atoms, drop_atom_lst)
            data.z = data.z[~mask_idx]
            data.pos = data.pos[~mask_idx]
            data.type_mask = data.type_mask[~mask_idx]

        
        

        data.y = float(np.mean(self.label_dict[pdb_file]))
        
        return data

class ChEMBLTest(InMemoryDataset):
    def __init__(self, pk_file='/data/protein/SKData/ChEMBL_dataset_sample/val_assays_sorted_lst.pkl', feat_dict='/data/protein/SKData/Uni-Mol/unimol_tools/PDB_feat_idx_dict.pkl', feat_path='/data/protein/SKData/PocData/old_data/feat_path', use_lig_feat=False):
        with open(pk_file, 'rb') as file_handler:
            self.pk_lst = pickle.load(file_handler)
        
        # get the pdb lst
        pdb_lst = []
        label_lst = []
        for assay_item in self.pk_lst:
            asssy_path = assay_item[0] # xxx.aff
            assay_dir = os.path.dirname(asssy_path)
            all_pdbs = assay_item[1][0]
            all_pdb_path = [os.path.join(assay_dir, ele[0]) for ele in all_pdbs]
            all_pdb_labels = [ele[1] for ele in all_pdbs]
            pdb_lst.extend(all_pdb_path)
            label_lst.extend(all_pdb_labels)
        self.pdb_lst = pdb_lst
        self.label_lst = label_lst
        self.length = len(self.pdb_lst)
        
        self.use_lig_feat = use_lig_feat
        if use_lig_feat:
            with open(feat_dict, 'rb') as fp:
                self.feat_idx_dict = pickle.load(fp)
        
            self.env = lmdb.open('/data/protein/SKData/PocData/old_data/feat_path', readonly=True, subdir=True, lock=False)
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        
        
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, idx, drop_atom_lst=['H', 'D']):
        data = Data()
        # get z, pos, and y
        # read element
        pdb_file = self.pdb_lst[idx]
        org_data = process_one_pdb(pdb_file)
        
        pocket_atoms = org_data['pocket_atoms']
        pocket_coordinates = org_data['pocket_coordinates']
        # pocket_z = [atomic_number.index(ele) for ele in pocket_atoms]
        pocket_z = [atomic_number[ele] for ele in pocket_atoms]
        
        lig_atoms_real = org_data['lig_atoms_real']
        lig_coord_real = org_data['lig_coord_real']
        # lig_z = [atomic_number.index(ele) for ele in lig_atoms_real]
        lig_z = [atomic_number[ele] for ele in lig_atoms_real]


        num_atoms = len(pocket_atoms) + len(lig_atoms_real)

        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[len(pocket_atoms): ] = 1 # lig 1
        
        # concat z and pos
        pocket_atoms.extend(lig_atoms_real)
        pocket_z.extend(lig_z)
        all_pos = np.concatenate([pocket_coordinates, lig_coord_real])

        

        data.z = torch.tensor(pocket_z, dtype=torch.long) 
        data.pos = torch.tensor(all_pos, dtype=torch.float32)
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        if len(drop_atom_lst): # erase H
            pocket_atoms = np.array(pocket_atoms)
            mask_idx = np.in1d(pocket_atoms, drop_atom_lst)
            data.z = data.z[~mask_idx]
            data.pos = data.pos[~mask_idx]
            data.type_mask = data.type_mask[~mask_idx]

        if self.use_lig_feat:
            ky = self.feat_idx_dict[pdb_file]
            ky = str(ky).encode()
            feat = deserialize_array(self.env.begin().get(ky)).reshape(-1, 512)
            data.lig_feat = feat[1:-1]
        

        # if self.transform_noise is not None:
        #     data = self.transform_noise(data) # noisy node
        data.y = self.label_lst[idx]
        
        return data
  

        # df = pd.read_csv(pd_file)
        # tdf = df[df['diff'] > 0.5]
        # self.pdb_lst = tdf['pairA'].to_numpy()
        # self.with_label = with_label
        # if with_label:
        #     self.aff_lst = tdf['diff'].to_numpy()
        # self.length = len(self.pdb_lst)
        # print(f'load ChEMBL dataset, pair diff threshold: {threshold}, column: {data_col_name}, length: {self.length}')



class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def make_ChEMBL_pair_dataset(pd_file_path='/data/protein/SKData/ChEMBL_dataset_sample/train_Pairs_0.5t.csv', threshold=1.5, use_uni_feat=False):
    chem_dataA = ChEMBL(pd_file_path, data_col_name='pairA', with_label=True, threshold=threshold, use_lig_feat=use_uni_feat)
    chem_dataB = ChEMBL(pd_file_path, data_col_name='pairB', threshold=threshold, use_lig_feat=use_uni_feat)
    return ConcatDataset(chem_dataA, chem_dataB)

def deserialize_array(serialized):
    return np.frombuffer(serialized, dtype=np.float32)

if __name__ == '__main__':
    # env = lmdb.open('/data/protein/SKData/PocData/old_data/feat_path', readonly=True)
    # with env.begin() as txn:
    #     cursor = txn.cursor()
    #     for key, value in cursor:
    #         array = deserialize_array(value)
    #         # 在这里对读取到的数组进行处理或使用
    #         print(key.decode('ascii'), array)
    # env.close()
    cue = ChEMBL(pd_file='/data/protein/SKData/ChEMBL_dataset_sample/train_Pairs_0.5t.csv', use_feat=True)
    for i in range(100):
        data = cue[i]