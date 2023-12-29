import os
from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.utils import MissingEnergyException
from torch_scatter import scatter
import h5py
import numpy as np
import os
from tqdm import tqdm
from ase.db import connect


# ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
#             ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
#             ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
#             ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
ELE_TO_NUM = {'H':1 , 'C': 6, 'N': 7, 'O':8}
ATOM_DICT = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}



class MD22(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.smiles = smiles
        self.forces = forces

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        dy = self.forces[index]

        x = []
        for atom in atoms:
            x.append(ELE_TO_NUM[atom])
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1,-1)
        dy = torch.tensor(dy, dtype=torch.float)
        
        if self.smiles is None:
            data = Data(z=x, pos=pos, y=y, dy=dy)
        else:
            data = Data(z=x, pos=pos, y=y, dy=dy, smi=self.smiles[index])

        return data

    def __len__(self):
        return len(self.positions)


class MD22A(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None, dihedral_angle_noise_scale=0.1, position_noise_scale=0.005):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.forces = forces
        self.smiles = smiles
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        dy = self.forces[index]

        x = []
        for atom in atoms:
            x.append(ELE_TO_NUM[atom])
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1,-1)
        dy = torch.tensor(dy, dtype=torch.float)
        
        if self.smiles is None:
            org_data = Data(z=x, pos=pos, y=y, dy=dy)
        else:
            org_data = Data(z=x, pos=pos, y=y, dy=dy, smi=self.smiles[index])
        
        
        # random noise only
        pos_noise_coords = self.transform_noise(org_data.pos)
        org_data.pos_target = (pos_noise_coords - org_data.pos.numpy()).clone().detach()
        org_data.org_pos = org_data.pos
        org_data.pos = pos_noise_coords.clone().detach()
        return org_data


    def transform_noise(self, data):
        try:
            if type(data) == np.ndarray:
                data = torch.from_numpy(data)
        except:
            print(type(data))
        noise = torch.randn_like(data.clone().detach()) * self.position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise


    def __len__(self):
        return len(self.positions)



class MD22DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(MD22DataModule, self).__init__()
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

        self.args = hparams
        self.mask_atom = hparams.mask_atom
        self.data_dir = hparams.dataset_root
        self.dataset_arg = hparams.dataset_arg
        self.seed = hparams.seed
        self.num_workers = hparams.num_workers
        
        self.batch_size = hparams.batch_size
        self.inference_batch_size = hparams.inference_batch_size
        self.valid_size = hparams.val_size
        self.test_size = hparams.test_size
    
    def _read_xyz(self, xyz_path):
        species, coordinates, energies = [], [], []
        elements, coords = [], []
        forces, fs = [], []

        with open(xyz_path, 'r') as f:
            for i, line in enumerate(f):
                l_list = line.strip().split()
                if len(l_list) == 2:
                    if len(elements) > 0:
                        species.append(elements)
                        coordinates.append(coords)
                        forces.append(fs)
                    elements = []
                    coords = []
                    fs = []

                    e = l_list[0].replace('Energy=', '')
                    energies.append(float(e))

                elif len(l_list) == 7:
                    ele, x, y, z, f_x, f_y, f_z = l_list
                    point = [float(x), float(y), float(z)]
                    force = [float(f_x), float(f_y), float(f_z)]
                    elements.append(ele)
                    coords.append(point)
                    fs.append(force)

        species.append(elements)
        coordinates.append(coords)
        forces.append(fs)

        return species, coordinates, np.array(energies), forces


    def setup(self, stage):
        if not os.path.exists(os.path.join(self.data_dir, self.dataset_arg)):
            os.makedirs(os.path.join(self.data_dir, self.dataset_arg))
        self.train_processed_data = os.path.join(self.data_dir, self.dataset_arg, "train.pt")
        self.valid_processed_data = os.path.join(self.data_dir, self.dataset_arg, "valid.pt")
        self.test_processed_data = os.path.join(self.data_dir, self.dataset_arg, "test.pt")
        if os.path.exists(self.train_processed_data):
            self.train_clean = torch.load(self.train_processed_data)
            self.valid_clean = torch.load(self.valid_processed_data)
            self.test_clean = torch.load(self.test_processed_data)
            print("Load preprocessed data!")
        else:
            print("Preprocessing data...")
            
            random_state = np.random.RandomState(seed=self.seed)

            # self.species, self.positions, self.energies = [], [], []
            self.train_species, self.valid_species, self.test_species = [], [], []
            self.train_positions, self.valid_positions, self.test_positions = [], [], []
            self.train_energies, self.valid_energies, self.test_energies = [], [], []
            self.train_smiles, self.valid_smiles, self.test_smiles = [], [], []
            self.train_forces, self.valid_forces, self.test_forces = [], [], []
            
            species, coordinates, energies, forces = self._read_xyz(os.path.join(self.data_dir, f'md22_{self.dataset_arg}.xyz'))

            assert len(species) == len(coordinates) == len(energies)
            n_conf = len(species)
            indices = list(range(n_conf))
            random_state.shuffle(indices)
            split1 = int(np.floor(self.valid_size * n_conf))
            split2 = int(np.floor(self.test_size * n_conf))
            valid_idx, test_idx, train_idx = \
                indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

            self.train_species.extend([species[idx] for idx in train_idx])
            self.train_energies = energies[train_idx]
            for i in train_idx:
                self.train_positions.append(coordinates[i])
                self.train_forces.append(forces[i])
            
            self.valid_species.extend([species[idx] for idx in valid_idx])
            self.valid_energies = energies[valid_idx]
            for i in valid_idx:
                self.valid_positions.append(coordinates[i])
                self.valid_forces.append(forces[i])
            
            self.test_species.extend([species[idx] for idx in test_idx])
            self.test_energies = energies[test_idx]
            for i in test_idx:
                self.test_positions.append(coordinates[i])
                self.test_forces.append(forces[i])
            self.test_smiles.extend([self.data_dir.split('/')[-1]] * len(test_idx))

            n_conf = len(self.test_species) + len(self.train_species) + len(self.valid_species)

            print("# conformations:", len(species))
            print("# train conformations:", len(self.train_species))
            print("# valid conformations:", len(self.valid_species))
            print("# test conformations:", len(self.test_species))

            self.train_clean = MD22(
                self.data_dir, species=self.train_species,
                positions=self.train_positions, energies=self.train_energies,
                forces=self.train_forces,
                # smiles=self.train_smiles
            )
            self.valid_clean = MD22(
                self.data_dir, species=self.valid_species,
                positions=self.valid_positions, energies=self.valid_energies,
                forces=self.valid_forces,
                # smiles=self.valid_smiles
            )
            self.test_clean = MD22(
                self.data_dir, species=self.test_species, 
                positions=self.test_positions, energies=self.test_energies, 
                forces=self.test_forces,
                smiles=self.test_smiles
            )

            torch.save(self.train_clean, self.train_processed_data)
            torch.save(self.valid_clean, self.valid_processed_data)
            torch.save(self.test_clean, self.test_processed_data)
            print(f'# Preprocess MD22-{self.dataset_arg} clean dataset and save locally successfully!')

            del self.train_species, self.valid_species, self.test_species
            del self.train_positions, self.valid_positions, self.test_positions
            del self.train_energies, self.valid_energies, self.test_energies
            del self.train_forces, self.valid_forces, self.test_forces
            del self.train_smiles, self.valid_smiles, self.test_smiles

        if 'MD22A' not in self.args.dataset:
            self.train_dataset = self.train_clean
            self.valid_dataset = self.valid_clean
            self.test_dataset = self.test_clean
        else:
            self.train_dataset = MD22A(
                self.train_clean.data_dir, species=self.train_clean.species,
                positions=self.train_clean.positions, energies=self.train_clean.energies,
                forces=self.train_clean.forces, smiles=self.train_clean.smiles,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.valid_dataset = MD22A(
                self.valid_clean.data_dir, species=self.valid_clean.species,
                positions=self.valid_clean.positions, energies=self.valid_clean.energies,
                forces=self.valid_clean.forces, smiles=self.valid_clean.smiles,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.test_dataset = MD22A(
                self.test_clean.data_dir, species=self.test_clean.species, 
                positions=self.test_clean.positions, energies=self.test_clean.energies, 
                forces=self.test_clean.forces, smiles=self.test_clean.smiles,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )


        self.train_loader = PyGDataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=True, drop_last=True, 
            pin_memory=False, persistent_workers=False
        )
        self.valid_loader = PyGDataLoader(
            self.valid_dataset, batch_size=self.inference_batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=False, persistent_workers=False
        )
        self.test_loader = PyGDataLoader(
            self.test_dataset, batch_size=self.inference_batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=False
        )

        if self.args.standardize:
            self._standardize()


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        loaders = [self.valid_loader]
        if (
            len(self.test_dataset) > 0
            and self.trainer.current_epoch % self.args.test_interval == 0 and self.trainer.current_epoch != 0
        ):
            loaders.append(self.test_loader)
        return loaders

    def test_dataloader(self):
        return self.test_loader

    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return self._std

    def _standardize(self, ):
        loader =  PyGDataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=False, persistent_workers=True
        )
        data = tqdm(
            loader,
            desc="computing mean and std",
        )
        try:
            ys = torch.cat([batch.y.clone() for batch in data])
        except:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return
        
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
            
