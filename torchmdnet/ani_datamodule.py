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
SELF_INTER_ENERGY = {
    'H': -0.500607632585, 
    'C': -37.8302333826,
    'N': -54.5680045287,
    'O': -75.0362229210
}


class anidataloader(object):

    ''' Contructor '''
    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - '+store_file)
        self.store = h5py.File(store_file)

    ''' Group recursive iterator (iterate through all groups in all branches and return datasets in dicts) '''
    def h5py_dataset_iterator(self,g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset): # test for dataset
                data = {'path':path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k])

                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if type(dataset[0]) is np.bytes_:
                                    dataset = [a.decode('ascii') for a in dataset]

                        data.update({k:dataset})

                yield data
            else: # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    ''' Returns a list of all groups in the file '''
    def get_group_list(self):
        return [g for g in self.store.values()]

    ''' Allows interation through the data in a given group '''
    def iter_group(self,g):
        for data in self.h5py_dataset_iterator(g):
            yield data

    ''' Returns the requested dataset '''
    def get_data(self, path, prefix=''):
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        # print(path)
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k].value)

                if type(dataset) is np.ndarray:
                    if dataset.size != 0:
                        if type(dataset[0]) is np.bytes_:
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    ''' Returns the number of groups '''
    def group_size(self):
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    ''' Close the HDF5 file '''
    def cleanup(self):
        self.store.close()


class ANI1(Dataset):
    def __init__(self, data_dir, species, positions, energies, smiles=None):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.smiles = smiles

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]

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
        
        if self.smiles is None:
            data = Data(z=x, pos=pos, y=y-self_energy, self_energy=self_energy)
        else:
            data = Data(z=x, pos=pos, y=y-self_energy, self_energy=self_energy, smi=self.smiles[index])

        return data

    def __len__(self):
        return len(self.positions)


class ANI1A(Dataset):
    def __init__(self, data_dir, species, positions, energies, smiles=None, dihedral_angle_noise_scale=0.1, position_noise_scale=0.005):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.smiles = smiles
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]

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
        

        if self.smiles is None:
            org_data = Data(z=x, pos=pos, y=y-self_energy, self_energy=self_energy)
        else:
            org_data = Data(z=x, pos=pos, y=y-self_energy, self_energy=self_energy, smi=self.smiles[index])

        
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



class ANIDataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(ANIDataModule, self).__init__()
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
            self.raw_data_dir = os.path.join(self.dir, "raw", "ANI-1_release")

            random_state = np.random.RandomState(seed=self.seed)
            # read the data
            hdf5files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.h5')]

            curr_idx, n_mol = 0, 0
            self.train_species, self.valid_species, self.test_species = [], [], []
            self.train_positions, self.valid_positions, self.test_positions = [], [], []
            self.train_energies, self.valid_energies, self.test_energies = [], [], []
            self.train_smiles, self.valid_smiles, self.test_smiles = [], [], []
            
            for f in hdf5files:
                print('reading:', f)
                h5_loader = anidataloader(os.path.join(self.raw_data_dir, f))
                for data in h5_loader:
                    X = data['coordinates']
                    S = data['species']
                    E = data['energies']
                    smi = ''.join(data['smiles'])
                    
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
                    self.train_smiles.extend([smi] * len(train_idx))

                    self.valid_species.extend([S] * len(valid_idx))
                    self.valid_energies.append(E[valid_idx])
                    for i in valid_idx:
                        self.valid_positions.append(X[i])
                    self.valid_smiles.extend([smi] * len(valid_idx))

                    self.test_species.extend([S] * len(test_idx))
                    self.test_energies.append(E[test_idx])
                    for i in test_idx:
                        self.test_positions.append(X[i])
                    self.test_smiles.extend([smi] * len(test_idx))

                    n_mol += 1
                
                h5_loader.cleanup()
            
            self.train_energies = np.concatenate(self.train_energies, axis=0)
            self.valid_energies = np.concatenate(self.valid_energies, axis=0)
            self.test_energies = np.concatenate(self.test_energies, axis=0)
            print("# molecules:", n_mol)


            self.train_clean = ANI1(
                self.raw_data_dir, species=self.train_species,
                positions=self.train_positions, energies=self.train_energies,
                smiles=self.train_smiles,
            )
            self.valid_clean = ANI1(
                self.raw_data_dir, species=self.valid_species,
                positions=self.valid_positions, energies=self.valid_energies,
                smiles=self.valid_smiles
            )
            self.test_clean = ANI1(
                self.raw_data_dir, species=self.test_species, 
                positions=self.test_positions, energies=self.test_energies, 
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
            del self.train_smiles, self.valid_smiles, self.test_smiles

        if 'ANI1A' not in self.args.dataset:
            self.train_dataset = self.train_clean
            self.valid_dataset = self.valid_clean
            self.test_dataset = self.test_clean
        else:
            self.train_dataset = ANI1A(
                self.train_clean.data_dir, species=self.train_clean.species,
                positions=self.train_clean.positions, energies=self.train_clean.energies,
                smiles=self.train_clean.smiles,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.valid_dataset = ANI1A(
                self.valid_clean.data_dir, species=self.valid_clean.species,
                positions=self.valid_clean.positions, energies=self.valid_clean.energies,
                smiles=self.valid_clean.smiles,
                dihedral_angle_noise_scale=self.args.dihedral_angle_noise_scale,
                position_noise_scale=self.args.position_noise_scale,
            )
            self.test_dataset = ANI1A(
                self.test_clean.data_dir, species=self.test_clean.species, 
                positions=self.test_clean.positions, energies=self.test_clean.energies, 
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
            
