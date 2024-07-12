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


# ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
#             ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
#             ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
#             ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
ATOM_DICT = {'H':1 , 'C': 6, 'N': 7, 'O':8}
INDEX_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
SELF_INTER_ENERGY = {
    'H': -0.500607632585, 
    'C': -37.8302333826,
    'N': -54.5680045287,
    'O': -75.0362229210
}



def ani1x_iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file. 
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for smi, grp in f.items():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            d['smi'] = smi
            yield d 


class ANI1X(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None):
        """
        ANI1X Dataset for loading atomic data from directories.

        Args:
            data_dir (str): Directory where the dataset is stored.
            species (list): List of atomic species for each data point.
            positions (list): List of atomic positions for each data point.
            energies (list): List of energy values for each data point.
            forces (list): List of force values for each data point.
            smiles (list, optional): List of SMILES representations for each data point. Defaults to None.

        Methods:
            __getitem__(index):
                Retrieves the data point at the specified index.
                
                Args:
                    index (int): Index of the data point to retrieve.
                
                Returns:
                    Data: A Data object containing the following attributes:
                        - z (torch.Tensor): Atomic numbers of shape [Na,], encoded as integers.
                        - pos (torch.Tensor): Atomic positions of shape [Na, 3].
                        - y (torch.Tensor): Energy values of shape [1,], converted to kcal/mol and adjusted for self energy.
                        - self_energy (torch.Tensor): Self energy values of shape [1,], in kcal/mol.
                        - dy (torch.Tensor): Force values of shape [Na, 3], converted to kcal/mol/Å.
                        - smi (str, optional): SMILES representation, if provided.

            __len__():
                Returns the number of data points in the dataset.
                
                Returns:
                    int: The number of data points.
        """
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.forces = forces
        self.smiles = smiles

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        dy = self.forces[index]

        x = []
        self_energy = 0.0
        for atom in atoms:
            # x.append(ATOM_DICT[(atom, 0)])
            x.append(ATOM_DICT[atom])
            self_energy += SELF_INTER_ENERGY[atom]
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        # Hartree to kcal/mol
        y = torch.tensor(y, dtype=torch.float).view(1,-1) * 627.5
        # Hartree to kcal/mol
        self_energy = torch.tensor(self_energy, dtype=torch.float).view(1,-1) * 627.5
        # Hartree/A to kcal/mol/A
        dy = torch.tensor(dy, dtype=torch.float) * 627.5
        
        if self.smiles is None:
            data = Data(z=x, pos=pos, y=y-self_energy, self_energy=self_energy, dy=dy)
        else:
            data = Data(z=x, pos=pos, y=y-self_energy, self_energy=self_energy, dy=dy, smi=self.smiles[index])

        return data

    def __len__(self):
        return len(self.positions)


class ANI1XA(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None, position_noise_scale=0.005):
        """
        ANI1X Dataset for loading atomic data from directories, applying the noisy node.

        Args:
            data_dir (str): Directory where the dataset is stored.
            species (list): List of atomic species for each data point.
            positions (list): List of atomic positions for each data point.
            energies (list): List of energy values for each data point.
            forces (list): List of force values for each data point.
            smiles (list, optional): List of SMILES representations for each data point. Defaults to None.

        Methods:
            __getitem__(index):
                Retrieves the data point at the specified index.
                
                Args:
                    index (int): Index of the data point to retrieve.
                
                Returns:
                    Data: A Data object containing the following attributes:
                        - z (torch.Tensor): Atomic numbers of shape [Na,], encoded as integers.
                        - pos (torch.Tensor): Atomic positions of shape [Na, 3].
                        - y (torch.Tensor): Energy values of shape [1,], converted to kcal/mol and adjusted for self energy.
                        - self_energy (torch.Tensor): Self energy values of shape [1,], in kcal/mol.
                        - dy (torch.Tensor): Force values of shape [Na, 3], converted to kcal/mol/Å.
                        - smi (str, optional): SMILES representation, if provided.
                        - position_noise_scale (float): Scale of the random noise applied to atomic positions.

            __len__():
                Returns the number of data points in the dataset.
                
                Returns:
                    int: The number of data points.
        """
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.smiles = smiles
        self.forces = forces
        # self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        dy = self.forces[index]

        x = []
        self_energy = 0.0
        for atom in atoms:
            # x.append(ATOM_DICT[(atom, 0)])
            x.append(ATOM_DICT[atom])
            self_energy += SELF_INTER_ENERGY[atom]
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        # Hartree to kcal/mol
        y = torch.tensor(y, dtype=torch.float).view(1,-1) * 627.5
        # Hartree to kcal/mol
        self_energy = torch.tensor(self_energy, dtype=torch.float).view(1,-1) * 627.5
        # Hartree/A to kcal/mol/A
        dy = torch.tensor(dy, dtype=torch.float) * 627.5
        
        if self.smiles is None:
            org_data = Data(z=x, pos=pos, y=y-self_energy, dy=dy, self_energy=self_energy)
        else:
            org_data = Data(z=x, pos=pos, y=y-self_energy, dy=dy, self_energy=self_energy, smi=self.smiles[index])

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



class ANIXDataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        """
        ANIX Data Module for managing ANI-1x dataset loading and processing using PyTorch Lightning.

        Args:
            hparams (Namespace): Hyperparameters for the data module.
            dataset (Dataset, optional): Custom dataset to use. Defaults to None.

        Attributes:
            dataset (Dataset): The dataset object.
            args (Namespace): Hyperparameters passed to the data module.
            mask_atom (str): Mask atom type specified in hyperparameters.
            dir (str): Directory where the dataset is stored.
            seed (int): Seed for random operations.
            num_workers (int): Number of workers for data loading.
            batch_size (int): Batch size for training.
            inference_batch_size (int): Batch size for inference.
            valid_size (float): Proportion of data for validation.
            test_size (float): Proportion of data for testing.
            _mean (torch.Tensor, optional): Mean of the dataset for standardization. Defaults to None.
            _std (torch.Tensor, optional): Standard deviation of the dataset for standardization. Defaults to None.
            _saved_dataloaders (dict): Dictionary to store data loaders.

        Methods:
            setup(stage):
                Sets up the data module for training, validation, and testing.
                
                Args:
                    stage (str): The stage for which to set up the data module ('fit', 'validate', 'test', or 'predict').

            train_dataloader():
                Returns the data loader for the training dataset.
                
                Returns:
                    DataLoader: Data loader for the training dataset.

            val_dataloader():
                Returns the data loader for the validation dataset.
                
                Returns:
                    list of DataLoader: List containing the data loader for the validation dataset,
                    and optionally the data loader for the test dataset based on the training epoch.

            test_dataloader():
                Returns the data loader for the test dataset.
                
                Returns:
                    DataLoader: Data loader for the test dataset.

            mean:
                Returns the mean of the dataset for standardization.
                
                Returns:
                    torch.Tensor: Mean of the dataset.

            std:
                Returns the standard deviation of the dataset for standardization.
                
                Returns:
                    torch.Tensor: Standard deviation of the dataset.

            _standardize():
                Computes and sets the mean and standard deviation of the training dataset for standardization.
        """
        super(ANIXDataModule, self).__init__()
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

        self.args = hparams
        self.mask_atom = hparams.mask_atom
        self.dir = hparams.dataset_root
        self.seed = hparams.seed
        self.num_workers = hparams.num_workers
        
        self.batch_size = hparams.batch_size
        self.inference_batch_size = hparams.inference_batch_size
        self.valid_size = hparams.val_size
        self.test_size = hparams.test_size

    
    def setup(self, stage):
        if not os.path.exists(os.path.join(self.dir, "processed")):
            os.makedirs(os.path.join(self.dir, "processed"))
        self.train_processed_data = os.path.join(self.dir, "processed", "train.pt")
        self.valid_processed_data = os.path.join(self.dir, "processed", "valid.pt")
        self.test_processed_data = os.path.join(self.dir, "processed", "test.pt")
        if os.path.exists(self.train_processed_data):
            self.train_clean = torch.load(self.train_processed_data)
            self.valid_clean = torch.load(self.valid_processed_data)
            self.test_clean = torch.load(self.test_processed_data)
            print("Load preprocessed data!")
        else:
            print("Preprocessing data...")
            self.raw_data_dir = os.path.join(self.dir, "raw")

            random_state = np.random.RandomState(seed=self.seed)
            # read the data
            hdf5files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.h5')]
            data_path = os.path.join(self.raw_data_dir, hdf5files[0])

            curr_idx, n_mol = 0, 0
            self.train_species, self.valid_species, self.test_species = [], [], []
            self.train_positions, self.valid_positions, self.test_positions = [], [], []
            self.train_energies, self.valid_energies, self.test_energies = [], [], []
            self.train_smiles, self.valid_smiles, self.test_smiles = [], [], []
            self.train_forces, self.valid_forces, self.test_forces = [], [], []
            
            data_keys = ['wb97x_dz.energy','wb97x_dz.forces'] # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
            
            # extracting DFT/DZ energies and forces
            for data in ani1x_iter_data_buckets(data_path, keys=data_keys):
                X = data['coordinates']
                S = data['atomic_numbers']
                E = data['wb97x_dz.energy']
                F = data['wb97x_dz.forces']
                S = [INDEX_TO_SYMBOL[c] for c in S]
                 
                smi = data['smi']
                n_conf = E.shape[0]
                indices = list(range(n_conf))
                random_state.shuffle(indices)
                split1 = int(np.floor(self.valid_size * n_conf))
                split2 = int(np.floor(self.test_size * n_conf))
                valid_idx, test_idx, train_idx = \
                    indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

                self.train_species.extend([S] * len(train_idx))
                self.train_energies.append(E[train_idx])
                for i in train_idx:
                    self.train_positions.append(X[i])
                    self.train_forces.append(F[i])
                self.train_smiles.extend([smi] * len(train_idx))

                self.valid_species.extend([S] * len(valid_idx))
                self.valid_energies.append(E[valid_idx])
                for i in valid_idx:
                    self.valid_positions.append(X[i])
                    self.valid_forces.append(F[i])
                self.valid_smiles.extend([smi] * len(valid_idx))

                self.test_species.extend([S] * len(test_idx))
                self.test_energies.append(E[test_idx])
                for i in test_idx:
                    self.test_positions.append(X[i])
                    self.test_forces.append(F[i])
                self.test_smiles.extend([smi] * len(test_idx))

                n_mol += 1
            
            self.train_energies = np.concatenate(self.train_energies, axis=0)
            self.valid_energies = np.concatenate(self.valid_energies, axis=0)
            self.test_energies = np.concatenate(self.test_energies, axis=0)
            print("# molecules:", n_mol)

            self.train_clean = ANI1X(
                self.dir, species=self.train_species,
                positions=self.train_positions, energies=self.train_energies,
                forces=self.train_forces,
                smiles=self.train_smiles,
            )
            self.valid_clean = ANI1X(
                self.dir, species=self.valid_species,
                positions=self.valid_positions, energies=self.valid_energies,
                forces=self.valid_forces,
                smiles=self.valid_smiles
            )
            self.test_clean = ANI1X(
                self.dir, species=self.test_species, 
                positions=self.test_positions, energies=self.test_energies, 
                forces=self.test_forces,
                smiles=self.test_smiles
            )

            print("# train conformations:", len(self.train_clean))
            print("# valid conformations:", len(self.valid_clean))
            print("# test conformations:", len(self.test_clean))

            torch.save(self.train_clean, self.train_processed_data)
            torch.save(self.valid_clean, self.valid_processed_data)
            torch.save(self.test_clean, self.test_processed_data)
            print('# Preprocess ANI1 dataset and save locally successfully!')

            del self.train_species, self.valid_species, self.test_species
            del self.train_positions, self.valid_positions, self.test_positions
            del self.train_energies, self.valid_energies, self.test_energies
            del self.train_forces, self.valid_forces, self.test_forces
            del self.train_smiles, self.valid_smiles, self.test_smiles

        if 'ANI1XA' not in self.args.dataset:
            self.train_dataset = self.train_clean
            self.valid_dataset = self.valid_clean
            self.test_dataset = self.test_clean
        else:
            self.train_dataset = ANI1XA(
                self.train_clean.data_dir, species=self.train_clean.species,
                positions=self.train_clean.positions, energies=self.train_clean.energies,
                forces=self.train_clean.forces,
                smiles=self.train_clean.smiles,
                # dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.valid_dataset = ANI1XA(
                self.valid_clean.data_dir, species=self.valid_clean.species,
                positions=self.valid_clean.positions, energies=self.valid_clean.energies,
                forces=self.valid_clean.forces,
                smiles=self.valid_clean.smiles,
                # dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.test_dataset = ANI1XA(
                self.test_clean.data_dir, species=self.test_clean.species, 
                positions=self.test_clean.positions, energies=self.test_clean.energies, 
                forces=self.test_clean.forces,
                smiles=self.test_clean.smiles,
                # dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )



        self.train_loader = PyGDataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=True, drop_last=False, 
            pin_memory=False, persistent_workers=False
        )
        self.valid_loader = PyGDataLoader(
            self.valid_dataset, batch_size=self.inference_batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=False, 
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
            
