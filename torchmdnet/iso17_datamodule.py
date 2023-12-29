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
import json

# ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
#             ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
#             ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
#             ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
ELE_TO_NUM = {'H':1 , 'C': 6, 'N': 7, 'O':8}
ATOM_DICT = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}

# energyScale = 23.0605
# forceScale = 23.0605
energyScale = 1
forceScale = 1

class ISO17(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = [e * energyScale for e in energies]
        self.forces = [f * forceScale for f in forces]
        self.smiles = smiles

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        force = self.forces[index]
        y = self.energies[index]

        x = []
        for atom in atoms:
            x.append(ELE_TO_NUM[atom])
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1,-1)
        force = torch.tensor(force, dtype=torch.float)
        
        if self.smiles is None:
            data = Data(z=x, pos=pos, y=y, dy=force)
        else:
            data = Data(z=x, pos=pos, y=y, dy=force, smi=self.smiles[index])

        return data

    def __len__(self):
        return len(self.positions)


class ISO17A(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None, dihedral_angle_noise_scale=0.1, position_noise_scale=0.005):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = [e * energyScale for e in energies]
        self.forces = [f * forceScale for f in forces]
        self.smiles = smiles
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        force = self.forces[index]

        x = []
        for atom in atoms:
            x.append(ELE_TO_NUM[atom])
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1,-1)
        force = torch.tensor(force, dtype=torch.float)

        if self.smiles is None:
            org_data = Data(z=x, pos=pos, y=y, dy=force)
        else:
            org_data = Data(z=x, pos=pos, y=y, dy=force, smi=self.smiles[index])
        
        
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



class ISO17DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(ISO17DataModule, self).__init__()
        self._mean, self._std = None, None
        self._dy_mean, self._dy_std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

        self.args = hparams
        self.mask_atom = hparams.mask_atom
        self.data_dir = hparams.dataset_root
        self.seed = hparams.seed
        self.num_workers = hparams.num_workers
        
        self.batch_size = hparams.batch_size
        self.inference_batch_size = hparams.inference_batch_size

    
    def _read_db(self, db_path):
        species, coordinates = [], []
        energies = []
        forces = []
        with connect(db_path) as conn:
            for row in conn.select():
                atoms = row.toatoms()
                elements = atoms.get_atomic_numbers()
                elements = [ATOM_DICT[i] for i in elements]
                pos = atoms.get_positions()
                e = float(row['total_energy'])
                species.append(elements)
                coordinates.append(pos)
                energies.append(e)
                force = np.array(json.loads(row._data)['atomic_forces'])
                forces.append(force)
        return species, coordinates, np.array(energies), forces


    def setup(self, stage):
        if not os.path.exists(os.path.join(self.data_dir, "processed")):
            os.makedirs(os.path.join(self.data_dir, "processed"))
        self.train_processed_data = os.path.join(self.data_dir, "processed", "train.pt")
        self.valid_processed_data = os.path.join(self.data_dir, "processed", "valid.pt")
        self.test_processed_data = os.path.join(self.data_dir, "processed", "test.pt")
        if os.path.exists(self.train_processed_data):
            self.train_clean = torch.load(self.train_processed_data)
            self.valid_clean = torch.load(self.valid_processed_data)
            self.test_clean = torch.load(self.test_processed_data)
            print("Load preprocessed data!")
        else:
            print("Preprocessing data...")
            
            train_idx = []
            with open(os.path.join(self.data_dir, 'train_ids.txt'), 'r') as f:
                for line in f:
                    train_idx.append(int(line.strip()) - 1)
            valid_idx = []
            with open(os.path.join(self.data_dir, 'validation_ids.txt'), 'r') as f:
                for line in f:
                    valid_idx.append(int(line.strip()) - 1)

            species, coordinates, energies, forces = self._read_db(os.path.join(self.data_dir, 'reference.db'))
            print(len(species), len(coordinates), len(energies), len(train_idx), len(valid_idx))
            self.train_species = [species[idx] for idx in train_idx]
            self.train_positions = [coordinates[idx] for idx in train_idx]
            self.train_energies = energies[train_idx]
            self.train_forces = [forces[idx] for idx in train_idx]
            self.valid_species = [species[idx] for idx in valid_idx]
            self.valid_positions = [coordinates[idx] for idx in valid_idx]
            self.valid_energies = energies[valid_idx]
            self.valid_forces = [forces[idx] for idx in valid_idx]

            self.test_species, self.test_positions, self.test_energies, self.test_forces = \
                self._read_db(os.path.join(self.data_dir, 'test_other.db'))
            self.test_smiles = ['C7O2H10'] * len(self.test_energies)

            n_conf = len(self.test_species) + len(self.train_species) + len(self.valid_species)

            print("# molecules:", n_conf)
            print("# train conformations:", len(self.train_species))
            print("# valid conformations:", len(self.valid_species))
            print("# test conformations:", len(self.test_species))

            self.train_clean = ISO17(
                self.data_dir, species=self.train_species,
                positions=self.train_positions, energies=self.train_energies,
                forces=self.train_forces,
            )
            self.valid_clean = ISO17(
                self.data_dir, species=self.valid_species,
                positions=self.valid_positions, energies=self.valid_energies,
                forces=self.valid_forces
            )
            self.test_clean = ISO17(
                self.data_dir, species=self.test_species, 
                positions=self.test_positions, energies=self.test_energies, 
                smiles=self.test_smiles,
                forces=self.test_forces
            )
            torch.save(self.train_clean, self.train_processed_data)
            torch.save(self.valid_clean, self.valid_processed_data)
            torch.save(self.test_clean, self.test_processed_data)
            print('# Preprocess ISO clean dataset and save locally successfully!')
            
            del self.train_species, self.valid_species, self.test_species
            del self.train_positions, self.valid_positions, self.test_positions
            del self.train_energies, self.valid_energies, self.test_energies
            del self.train_forces, self.valid_forces, self.test_forces
            del self.test_smiles


        if 'ISO17A' not in self.args.dataset:
            self.train_dataset = self.train_clean
            self.valid_dataset = self.valid_clean
            self.test_dataset = self.test_clean
        else:
            self.train_dataset = ISO17A(
                self.train_clean.data_dir, species=self.train_clean.species,
                positions=self.train_clean.positions, energies=self.train_clean.energies,
                forces=self.train_clean.forces,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.valid_dataset = ISO17A(
                self.valid_clean.data_dir, species=self.valid_clean.species,
                positions=self.valid_clean.positions, energies=self.valid_clean.energies,
                forces=self.valid_clean.forces,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.test_dataset = ISO17A(
                self.test_clean.data_dir, species=self.test_clean.species, 
                positions=self.test_clean.positions, energies=self.test_clean.energies, 
                forces=self.test_clean.forces,
                smiles=self.test_clean.smiles,
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
    
    @property
    def dy_mean(self):
        return self._dy_mean
    
    @property
    def dy_std(self):
        return self._dy_std

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
            self._mean = ys.mean(dim=0)
            self._std = ys.std(dim=0)
        except:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        try:
            dys = torch.cat([batch.dy.clone() for batch in data])
            self._dy_mean = dys.mean(dim=0)
            self._dy_std = dys.std(dim=0)
        except:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset does not contain forces."
            )
            return