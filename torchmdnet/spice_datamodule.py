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
from rdkit import Chem


# ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
#             ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
#             ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
#             ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}

# posScale = 1*bohr/angstrom
# to Angstrom
posScale = 0.529177210903
# energyScale = 1*hartree/item/(kilojoules_per_mole)
# to KJ/mol
energyScale = 2625.499639479826
# to Kcal/mol
# energyScale = 627.5
# forceScale = energyScale/posScale
# to KJ/mol/Angstrom
forceScale = 4961.475258920568
# to Kcal/mol/Angstrom
# forceScale = 1185.8031432026708

ELE_TO_NUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'P': 15, 'S': 16, 'K': 19, 'Ca': 20, 'Cl': 17, 'Br': 35, 'I': 53, 'Li': 3}
ATOM_DICT = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}


class SPICE(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None):
        """
        SPICE dataset class for molecular data.

        Args:
            data_dir (str): Directory path to the dataset.
            species (list): List of atomic species data for each sample.
            positions (list): List of atomic positions for each sample.
            energies (list): List of energy values for each sample.
            forces (list): List of force values for each sample.
            smiles (list, optional): List of SMILES representations for each sample. Default is None.

        Returns:
            __getitem__:
                torch_geometric.data.Data: A single data sample containing:
                    - z (torch.Tensor): Atomic species as long tensor.
                    - pos (torch.Tensor): Atomic positions as float tensor.
                    - y (torch.Tensor): Energy values as float tensor, reshaped to (1, -1).
                    - dy (torch.Tensor): Force values as float tensor.
                    - smi (str, optional): SMILES representation, included if self.smiles is not None.
            __len__:
                int: Total number of samples in the dataset.
        """
        self.data_dir = data_dir
        self.species = species
        self.positions = [p * posScale for p in positions]
        self.energies = [e * energyScale for e in energies]
        self.forces = [f * forceScale for f in forces]
        self.smiles = smiles

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        dy = self.forces[index]

        x = torch.tensor(atoms, dtype=torch.long)
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


class SPICEA(Dataset):
    def __init__(self, data_dir, species, positions, energies, forces, smiles=None, position_noise_scale=0.005):
        """
        Class `SPICEA` inherits from `Dataset` and represents a dataset for SPICEA, applying Noise node.
        Args:
            data_dir (str): Directory path to the dataset.
            species (list): List of atomic species.
            positions (list): List of atomic positions.
            energies (list): List of energies associated with each configuration.
            forces (list): List of forces acting on each atom in each configuration.
            smiles (list or None, optional): List of SMILES strings for each configuration (default: None).
            position_noise_scale (float, optional): Noisy scale for position noise (default: 0.005).

        Methods:
            __getitem__(index):
                Retrieves and preprocesses data at a specified index.
                
                Args:
                    index (int): Index of the data point to retrieve.
                
                Returns:
                    org_data (Data): Processed data point with atomic species, positions, energies, forces,
                                    and optionally SMILES information.
            
            transform_noise(data):
                Applies random noise transformation to input data.
                
                Args:
                    data (numpy.ndarray or torch.Tensor): Input data to add noise to.
                
                Returns:
                    data_noise (torch.Tensor): Noisy version of the input data.
            
            __len__():
                Returns the number of data points in the dataset.
                
                Returns:
                    int: Number of data points in the dataset.
        
        """
        self.data_dir = data_dir
        self.species = species
        self.positions = [p * posScale for p in positions]
        self.energies = [e * energyScale for e in energies]
        self.forces = [f * forceScale for f in forces]
        self.smiles = smiles
        self.position_noise_scale = position_noise_scale

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]
        dy = self.forces[index]

        x = torch.tensor(atoms, dtype=torch.long)
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



class SPICEDataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        """
        SPICEDataModule is a PyTorch Lightning DataModule for handling datasets
        related to SPICE (Solvation Parameters Interface Calculation Environment).
        It preprocesses the data into train, validation, and test sets, optionally
        applying standardization. It provides DataLoader instances for training,
        validation, and testing phases.

        Args:
            hparams (argparse.Namespace): Hyperparameters and configuration options.
                Contains attributes:
                    - mask_atom (bool): Whether to mask atoms.
                    - dataset_root (str): Root directory of the dataset.
                    - seed (int): Random seed for reproducibility.
                    - num_workers (int): Number of workers for data loading.
                    - batch_size (int): Batch size for training.
                    - inference_batch_size (int): Batch size for inference.
                    - val_size (float): Fraction of data to use for validation.
                    - test_size (float): Fraction of data to use for testing.

        Returns:
            None

        Methods:
            setup(stage):
                Preprocesses the dataset into train, validation, and test sets,
                saving preprocessed data locally if not already processed.

            train_dataloader():
                Returns a PyTorch DataLoader for the training set.

            val_dataloader():
                Returns a list of PyTorch DataLoader instances for validation sets.
                Includes test DataLoader based on epoch and test_interval.

            test_dataloader():
                Returns a PyTorch DataLoader for the test set.

            mean:
                Property returning the mean computed during standardization.

            std:
                Property returning the standard deviation computed during standardization.

            _standardize():
                Computes and sets the mean and standard deviation of the dataset,
                used for standardizing the dataset during training.

        """

        super(SPICEDataModule, self).__init__()
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
            
            random_state = np.random.RandomState(seed=self.seed)

            n_mol = 0
            self.train_species, self.valid_species, self.test_species = [], [], []
            self.train_positions, self.valid_positions, self.test_positions = [], [], []
            self.train_energies, self.valid_energies, self.test_energies = [], [], []
            self.train_forces, self.valid_forces, self.test_forces = [], [], []
            self.train_smiles, self.valid_smiles, self.test_smiles = [], [], []

            with h5py.File(os.path.join(self.dir, 'SPICE-1.1.2.hdf5'), 'r') as raw_data:
                for name in raw_data:
                    n_mol += 1
                    if n_mol % 1000 == 0:
                        print('Loading # molecule %d' % n_mol)
                    
                    g = raw_data[name]
                    smi = g['smiles'][0].decode('ascii')
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    species = []
                    for atom in mol.GetAtoms():
                        ele = atom.GetSymbol()
                        # charge = atom.GetFormalCharge()
                        # species.append(ATOM_DICT[(ele, charge)])
                        species.append(ELE_TO_NUM[ele])
                    
                    n_conf = g['conformations'].shape[0]
                    indices = list(range(n_conf))
                    random_state.shuffle(indices)
                    split1 = int(np.floor(self.valid_size * n_conf))
                    split2 = int(np.floor(self.test_size * n_conf))
                    valid_idx, test_idx, train_idx = \
                        indices[:split1], indices[split1:split1+split2], indices[split1+split2:]
                    
                    for i in train_idx:
                        self.train_positions.append(g['conformations'][i])
                        self.train_energies.append(g['formation_energy'][i])
                        self.train_forces.append(g['dft_total_gradient'][i])
                        self.train_species.append(species)
                    self.train_smiles.extend([smi] * len(train_idx))

                    for i in valid_idx:
                        self.valid_positions.append(g['conformations'][i])
                        self.valid_energies.append(g['formation_energy'][i])
                        self.valid_forces.append(g['dft_total_gradient'][i])
                        self.valid_species.append(species)
                    self.valid_smiles.extend([smi] * len(valid_idx))

                    for i in test_idx:
                        self.test_positions.append(g['conformations'][i])
                        self.test_energies.append(g['formation_energy'][i])
                        self.test_forces.append(g['dft_total_gradient'][i])
                        self.test_species.append(species)
                    self.test_smiles.extend([smi] * len(test_idx))

            n_conf = len(self.test_species) + len(self.train_species) + len(self.valid_species)

            print("# molecules:", n_conf)
            print("# train conformations:", len(self.train_species))
            print("# valid conformations:", len(self.valid_species))
            print("# test conformations:", len(self.test_species))

            self.train_clean = SPICE(
                self.dir, species=self.train_species,
                positions=self.train_positions, energies=self.train_energies,
                forces=self.train_forces,
                smiles=self.train_smiles,
            )
            self.valid_clean = SPICE(
                self.dir, species=self.valid_species,
                positions=self.valid_positions, energies=self.valid_energies,
                forces=self.valid_forces,
                smiles=self.valid_smiles,
            )
            self.test_clean = SPICE(
                self.dir, species=self.test_species, 
                positions=self.test_positions, energies=self.test_energies, 
                forces=self.test_forces,
                smiles=self.test_smiles,
            )

            torch.save(self.train_clean, self.train_processed_data)
            torch.save(self.valid_clean, self.valid_processed_data)
            torch.save(self.test_clean, self.test_processed_data)
            print('# Preprocess SPICE clean dataset and save locally successfully!')
            
            del self.train_species, self.valid_species, self.test_species
            del self.train_positions, self.valid_positions, self.test_positions
            del self.train_energies, self.valid_energies, self.test_energies
            del self.train_forces, self.valid_forces, self.test_forces
            del self.train_smiles, self.valid_smiles, self.test_smiles

        if 'SPICEA' not in self.args.dataset:
            self.train_dataset = self.train_clean
            self.valid_dataset = self.valid_clean
            self.test_dataset = self.test_clean
        else:
            self.train_dataset = SPICEA(
                self.train_clean.data_dir, species=self.train_clean.species,
                positions=self.train_clean.positions, energies=self.train_clean.energies,
                forces=self.train_clean.forces,
                smiles=self.train_clean.smiles,
                # dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.valid_dataset = SPICEA(
                self.valid_clean.data_dir, species=self.valid_clean.species,
                positions=self.valid_clean.positions, energies=self.valid_clean.energies,
                forces=self.valid_clean.forces,
                smiles=self.valid_clean.smiles,
                # dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.test_dataset = SPICEA(
                self.test_clean.data_dir, species=self.test_clean.species, 
                positions=self.test_clean.positions, energies=self.test_clean.energies, 
                forces=self.test_clean.forces,
                smiles=self.test_clean.smiles,
                # dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
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
            
