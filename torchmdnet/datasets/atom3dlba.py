from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)

import pickle as pk
import json, io
import msgpack
import gzip, logging
import pandas as pd
from pathlib import Path
import lmdb
import random
import dgl
from .pair_lba import deserialize_array
import os


def coord2graph(coord:torch.tensor,cutoff=6,daul_graph=True) -> dgl.graph:
    vector = coord[None,:,:] - coord[:,None,:]
    distance = torch.linalg.norm(vector,dim=-1,keepdim=True)
    vector = vector / distance
    torch.nan_to_num_(vector,0)
    distance.squeeze_()
    start_node,end_node = torch.where(distance<cutoff)
    g = dgl.graph((start_node, end_node))
    edge_dist = distance[(start_node,end_node)]
    
    if daul_graph:
        start_edge,end_edge = torch.where((start_node[None,:]-start_node[:,None])*(end_node[None,:]-end_node[:,None])==0)
        dg = dgl.graph((start_edge,end_edge))
        edge_vec = vector[(start_node,end_node)]
        dedge_cos = (edge_vec[start_edge][:,None,:] @ edge_vec[end_edge][:,:,None]).squeeze_()
    else:
        dg = None
        dedge_cos = None
    return g,dg,edge_dist,dedge_cos


def deserialize(x, serialization_format):
    """
    Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == 'pkl':
        return pk.loads(x)
    elif serialization_format == 'json':
        serialized = json.loads(x)
    elif serialization_format == 'msgpack':
        serialized = msgpack.unpackb(x)
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized

class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, data_file, transform=None):
        """constructor

        """
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        env = lmdb.open(str(self.data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()
            self._id_to_idx = deserialize(
                txn.get(b'id_to_idx'), self._serialization_format)

        self._env = env
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def get(self, id: str):
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        if id not in self._id_to_idx:
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids):
        return [self.id_to_idx(id) for id in ids]

    def ids(self):
        return list(self._id_to_idx.keys())

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        with self._env.begin(write=False) as txn:

            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            try:
                item = deserialize(serialized, self._serialization_format)
            except:
                return None
        # Recover special data types (currently only pandas dataframes).
        if 'types' in item.keys():
            for x in item.keys():
                if (self._serialization_format=='json') and (item['types'][x] == str(pd.DataFrame)):
                    item[x] = pd.DataFrame(**item[x])
        else:
            logging.warning('Data types in item %i not defined. Will use basic types only.'%index)

        if 'file_path' not in item:
            item['file_path'] = str(self.data_file)
        if 'id' not in item:
            item['id'] = str(index)
        if self._transform:
            item = self._transform(item)
        return item


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
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}

class LBADataset(InMemoryDataset):
    def __init__(self, data_path, transform_noise=None, lp_sep=False, feat_path='/data/protein/SKData/LBADATA/', use_lig_feat=False):
        
        self.transform_noise = transform_noise
        self.data_dict = np.load(data_path, allow_pickle=True).item() # dict
        self.length = len(self.data_dict['index'])
        self.lp_sep = lp_sep
        self.pocket_atom_offset = 120
        self.use_lig_feat = use_lig_feat
        if self.use_lig_feat:
            lmdb_folder = os.path.basename(data_path)[:-4]
            lmdb_path = os.path.join(feat_path, f'{lmdb_folder}feat')
            self.env = lmdb.open(lmdb_path, readonly=True, subdir=True, lock=False)
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx):

        data = Data()
        # get z, pos, and y
        num_atoms = self.data_dict['num_atoms'][idx]
        pocket_atomsnum = self.data_dict['pocket_atoms'][idx]
        ligand_atomsnum = self.data_dict['ligand_atoms'][idx]
        assert (pocket_atomsnum + ligand_atomsnum) == num_atoms
        data.z = torch.tensor(self.data_dict['charges'][idx][:num_atoms], dtype=torch.long)
        # data.z[:pocket_atomsnum] += self.pocket_atom_offset
        data.pos = torch.tensor(self.data_dict['positions'][idx][:num_atoms], dtype=torch.float32)
        if self.transform_noise is not None:
            data = self.transform_noise(data) # noisy node
        
        data.y =  torch.tensor(self.data_dict['neglog_aff'][idx], dtype=torch.float32)

        # NOTE: for debug:
        data.org_pos = torch.tensor(self.data_dict['positions'][idx][:num_atoms], dtype=torch.float32)

        # data.type_mask = torch.zeros(num_atoms, dtype=torch.long)
        # data.type_mask[:pocket_atomsnum] = 1
        data.pocket_atomsnum = pocket_atomsnum
        
        if self.use_lig_feat:
            ky = idx
            ky = str(ky).encode()
            feat = deserialize_array(self.env.begin().get(ky)).reshape(-1, 512)
            data.lig_feat = feat[1:-1]
            assert data.lig_feat.shape[0] == ligand_atomsnum
            
        # type mask
        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[pocket_atomsnum: ] = 1 # lig 1
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)
        
        # data.ligand_atomsnum = ligand_atomsnum
        return data

class LBADataset3(InMemoryDataset):

    def __init__(self,
                 data,
                 max_len=256,
                 seed=42,
                 daul_graph=True,
                 rm_H = True) -> None:
        super().__init__()
        random.seed(seed)
        self.data = LMDBDataset(data)
        self.seed = seed
        self.max_len = max_len
        #self.atom_type = ["C","N","O","S","H","Cl","F","Br","I","Si","P","B","Na","K","Al","Ca","Sn","As","Hg","Fe","Zn","Cr","Se","Gd","Au","Li"]
        self.atom_type = ["C","N","O","S","H","CL","F","BR","I","SI","P","B","NA","K","AL","CA","SN","AS","HG","FE","ZN","CR","SE","GD","AU","LI","MG",'NI','MN','CO','CU','SR','CS','CD']
        self.daul_graph = daul_graph
        self.rm_H = rm_H
    def __getitem__(self, index):
        datay = Data()
        data = self.data[index]
        pocket_atomtype = data['atoms_pocket']['element'].tolist()
        pocket_atomtype = torch.tensor([self.atom_type.index(i) for i in pocket_atomtype]).long()

        ligand_atomtype = data['atoms_ligand']['element'].tolist()
        ligand_atomtype = torch.tensor([self.atom_type.index(i) for i in ligand_atomtype]).long()

        pocket_coord = data['atoms_pocket'][['x','y','z']].to_numpy()
        pocket_coord = torch.tensor(pocket_coord).float()
        ligand_coord = data['atoms_ligand'][['x','y','z']].to_numpy()
        ligand_coord = torch.tensor(ligand_coord).float()     
 
        if self.rm_H:
            pocket_coord = pocket_coord[pocket_atomtype!=4]
            ligand_coord = ligand_coord[ligand_atomtype!=4]
            pocket_atomtype = pocket_atomtype[pocket_atomtype!=4]
            ligand_atomtype = ligand_atomtype[ligand_atomtype!=4]

        poc_lig_id = torch.cat([torch.zeros_like(pocket_atomtype),torch.ones_like(ligand_atomtype)])
        atomtype = torch.cat([pocket_atomtype, ligand_atomtype])
        coord = torch.cat([pocket_coord, ligand_coord])
        
        datay.z = atomtype
        # data.z[:pocket_atomsnum] += self.pocket_atom_offset
        datay.pos = coord
        datay.type_mask = poc_lig_id
    
        
        affinity = torch.tensor([data['scores']['neglog_aff']])
        datay.y = affinity

        return datay
        # g,dg,edge_dist,dedge_cos = coord2graph(coord,daul_graph = self.daul_graph)     
        
        # return g,dg,poc_lig_id,atomtype,edge_dist,dedge_cos,affinity

    def __len__(self):
        return len(self.data)


class LBADataset2(InMemoryDataset):

    def __init__(self,
                 data,
                 max_len=256,
                 seed=42,
                 daul_graph=False,
                 Hbond=False,
                 ) -> None:
        super().__init__()
        random.seed(seed)
        with open(data,'rb') as f:
            self.data = pk.load(f)
        self.seed = seed
        self.Hbond = Hbond
        self.max_len = max_len
        #self.atom_type = ["C","N","O","S","H","Cl","F","Br","I","Si","P","B","Na","K","Al","Ca","Sn","As","Hg","Fe","Zn","Cr","Se","Gd","Au","Li"]
        # self.atom_type = ["A","C","HD","N","NA","OA","SA",'H','O','S',"P","HS",'NS','OS','F','Cl',"Br",'I']
        # self.atom_type = ["C","C","H","N","NA","OA","SA",'H','O','S',"P","HS",'NS','OS','F','Cl',"Br",'I']
        self.atom_type = ['C', 'O', 'S', 'N', 'H', 'I', 'Br', 'Cl', 'F', 'P']
        self.atom_degerate = {
            'A':'C',
            'C':'C',
            'HD':'H',
            'N':'N',
            'NA':'N',
            'OA':'O',
            'SA':'S',
            'H':'H',
            'O':'O',
            'S':'S',
            'P':'P',
            'HS':'H',
            'NS':'N',
            'OS':'O',
            'F':'F',
            'Cl':'Cl',
            'Br':'Br',
            'I':'I'
        }
        self.daul_graph = daul_graph

    def __getitem__(self, index):
        datay = Data()
        data = self.data[index]
        pocket_atomtype = data['pocket_atoms'].tolist()
        if not self.Hbond:
            pocket_atomtype = [self.atom_degerate[i] for i in pocket_atomtype]
        pocket_atomtype = torch.tensor([self.atom_type.index(i) for i in pocket_atomtype]).long()

        ligand_atomtype = data['atoms'].tolist()
        if not self.Hbond:
            ligand_atomtype = [self.atom_degerate[i] for i in ligand_atomtype]
        ligand_atomtype = torch.tensor([self.atom_type.index(i) for i in ligand_atomtype]).long()

        pocket_coord = data['pocket_coordinates']
        pocket_coord = torch.tensor(pocket_coord).squeeze_().float()
        ligand_coord = data['coordinates']
        ligand_coord = torch.tensor(ligand_coord).squeeze_().float()     

        pocket_charge = data['pocket_charges']
        pocket_charge = torch.tensor(pocket_charge).float()
        ligand_charge = data['charge']
        ligand_charge = torch.tensor(ligand_charge).float()

        if not self.Hbond:
            pocket_coord = pocket_coord[pocket_atomtype!=4]
            ligand_coord = ligand_coord[ligand_atomtype!=4]
            pocket_atomtype = pocket_atomtype[pocket_atomtype!=4]
            ligand_atomtype = ligand_atomtype[ligand_atomtype!=4]


        poc_lig_id = torch.cat([torch.zeros_like(pocket_atomtype),torch.ones_like(ligand_atomtype)])
        atomtype = torch.cat([pocket_atomtype,ligand_atomtype])
        coord = torch.cat([pocket_coord,ligand_coord])  
        charge = torch.cat([pocket_charge,ligand_charge])
        g,dg,edge_dist,dedge_cos = coord2graph(coord,daul_graph = self.daul_graph)     
        affinity = torch.tensor([data['affinity']])

        datay.z = atomtype
        # data.z[:pocket_atomsnum] += self.pocket_atom_offset
        datay.pos = coord
        datay.type_mask = poc_lig_id
        datay.y = affinity

        return datay

    def __len__(self):
        return len(self.data)
