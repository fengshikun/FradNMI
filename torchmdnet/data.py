from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException, DataLoaderMasking
from torch_scatter import scatter
from torchmdnet.utils import collate_fn
import os

class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        """
        DataModule class for managing dataset loading and transformations in a PyTorch Lightning project.

        Args:
            hparams: Hyperparameters and configurations for the data module.
            dataset: Optional pre-loaded dataset.

        Attributes:
            mask_atom: Boolean indicating if masking is applied to atoms.
            model: The model to be used.
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            test_dataset: The test dataset.
            dataset_maybe_noisy: Dataset with optional noise transformations.
            _mean: Mean of the dataset features.
            _std: Standard deviation of the dataset features.
            _saved_dataloaders: Dictionary to store data loaders.

        Methods:
            setup(stage):
                Sets up the dataset based on the stage (e.g., 'train', 'val', 'test').
                
                Args:
                    stage (str): Stage of the setup process.

            train_dataloader():
                Returns the data loader for the training dataset.
                
                Returns:
                    DataLoader: PyTorch DataLoader for the training dataset.

            val_dataloader():
                Returns the data loader(s) for the validation dataset.
                
                Returns:
                    list[DataLoader]: List containing PyTorch DataLoader(s) for the validation dataset.

            test_dataloader():
                Returns the data loader for the test dataset.
                
                Returns:
                    DataLoader: PyTorch DataLoader for the test dataset.

            atomref:
                Returns the atom reference values if available.
                
                Returns:
                    Tensor or None: Atom reference tensor or None if not available.

            mean:
                Returns the mean of the dataset features.
                
                Returns:
                    Tensor: Mean of the dataset features.

            std:
                Returns the standard deviation of the dataset features.
                
                Returns:
                    Tensor: Standard deviation of the dataset features.

            _get_dataloader(dataset, stage, store_dataloader=True):
                Creates and returns a data loader for the specified dataset and stage.
                
                Args:
                    dataset: The dataset to create the DataLoader for.
                    stage (str): Stage of the setup process.
                    store_dataloader (bool): Whether to store the DataLoader for future use.
                
                Returns:
                    DataLoader: PyTorch DataLoader for the specified dataset and stage.

            get_energy_data(data):
                Returns energy data after removing atom reference energies.
                
                Args:
                    data: Input data containing atomic positions and energy.
                
                Returns:
                    Tensor: Energy data with atom reference energies removed.

            _standardize():
                Computes and sets the mean and standard deviation of the dataset.
        """

        super(DataModule, self).__init__()
        self._set_hparams(hparams.__dict__ if hasattr(hparams, "__dict__") else hparams)
        # self.hparams = hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

        self.mask_atom = hparams.mask_atom
        
        self.model = hparams.model

    def setup(self, stage):
        if self.dataset is None:
            if "LBADataset" in self.hparams["dataset"]:
                # special for the atom3d LBA task
                if self.hparams['position_noise_scale'] > 0.:
                    def transform(data):
                            noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                            data.pos_target = noise
                            data.pos = data.pos + noise
                            if self.hparams["prior_model"] == "Atomref":
                                data.y = self.get_energy_data(data)
                            return data
                else:
                    transform = None

                
                dataset_factory = getattr(datasets, self.hparams["dataset"])
                
                if self.hparams["dataset"] == 'LBADataset':
                    self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "lba_train.npy"), transform_noise=transform, lp_sep=self.hparams['lp_sep'], use_lig_feat=self.hparams['use_uni_feat'])
                    self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "lba_valid.npy"), transform_noise=None, lp_sep=self.hparams['lp_sep'], use_lig_feat=self.hparams['use_uni_feat'])
                    self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "lba_test.npy"), transform_noise=None, lp_sep=self.hparams['lp_sep'], use_lig_feat=self.hparams['use_uni_feat'])
                else:
                    # self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "train"))
                    # self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "val"))
                    # self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "test"))
                    self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "train_data.pk"))
                    self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "val_data.pk"))
                    self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "test_data.pk"))

                # normalize
                if self.hparams["standardize"]:
                    self._standardize()                


                return
            
            
            if self.hparams["dataset"] == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams["coord_files"],
                    self.hparams["embed_files"],
                    self.hparams["energy_files"],
                    self.hparams["force_files"],
                )
            else:
                if self.hparams['position_noise_scale'] > 0. and 'BIAS' not in self.hparams['dataset'] and 'Dihedral' not in self.hparams['dataset'] and 'QM9A' not in self.hparams['dataset']:
                    def transform(data):
                        noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                        data.pos_target = noise
                        data.pos = data.pos + noise
                        if self.hparams["prior_model"] == "Atomref":
                            data.y = self.get_energy_data(data)
                        return data
                else:
                    transform = None

                if 'BIAS' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['position_noise_scale'], self.hparams['sample_number'], self.hparams['violate'], dataset_arg=self.hparams["dataset_arg"], transform=t)
                elif 'PCQM4MV2_Force' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['angle_noise_scale'], self.hparams['bond_length_scale'], dataset_arg=self.hparams["dataset_arg"], transform=t)
                elif 'Dihedral2' in self.hparams['dataset'] or 'Dihedral3' in self.hparams['dataset'] or 'Dihedral4' in self.hparams['dataset']:
                    if self.hparams.model == 'painn':
                        add_radius_edge = True
                    else:
                        add_radius_edge = False
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'], self.hparams['composition'], self.hparams['decay'], self.hparams['decay_coe'], dataset_arg=self.hparams["dataset_arg"], equilibrium=self.hparams['equilibrium'], eq_weight=self.hparams['eq_weight'], cod_denoise=self.hparams['cod_denoise'], integrate_coord=self.hparams['integrate_coord'], addh=self.hparams['addh'], mask_atom=self.hparams['mask_atom'], mask_ratio=self.hparams['mask_ratio'], bat_noise=self.hparams['bat_noise'], transform=t, add_radius_edge=add_radius_edge)
                elif 'DihedralF' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'], self.hparams['composition'], self.hparams['force_field'], self.hparams['pred_noise'], cod_denoise=self.hparams['cod_denoise'], rdkit_conf=self.hparams['rdkit_conf'])
                elif 'Dihedral' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'], self.hparams['composition'], dataset_arg=self.hparams["dataset_arg"], transform=t)
                elif 'QM9A' in self.hparams['dataset'] or 'MD17A' in self.hparams['dataset']:
                    if 'QM9A' in self.hparams['dataset']:
                        if self.hparams["prior_model"] == "Atomref":
                            transform_y = self.get_energy_data
                        else:
                            transform_y = None
                        dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=None, dihedral_angle_noise_scale=self.hparams['dihedral_angle_noise_scale'], position_noise_scale=self.hparams['position_noise_scale'], composition=self.hparams['composition'], transform_y=transform_y)
                    else: # MD17A
                        dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=None, dihedral_angle_noise_scale=self.hparams['dihedral_angle_noise_scale'], position_noise_scale=self.hparams['position_noise_scale'], composition=self.hparams['composition'], reverse_half=self.hparams['reverse_half'], addh=self.hparams['addh'], cod_denoise=self.hparams['cod_denoise'])
                elif 'TestData' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=None)
                else:
                    if self.hparams.model == 'painn':
                        add_radius_edge = True
                    else:
                        add_radius_edge = False
                    
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=t, add_radius_edge=add_radius_edge)

                # Noisy version of dataset
                self.dataset_maybe_noisy = dataset_factory(transform)
                # Clean version of dataset
                if self.hparams["prior_model"] == "Atomref":
                    def transform_atomref(data):
                        data.y = self.get_energy_data(data)
                        return data
                    self.dataset = dataset_factory(transform_atomref)
                else:
                    self.dataset = dataset_factory(None)

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset_maybe_noisy, self.idx_train)

        # If denoising is the only task, test/val datasets are also used for measuring denoising performance.
        if self.hparams['denoising_only']:
            self.val_dataset = Subset(self.dataset_maybe_noisy, self.idx_val)
            self.test_dataset = Subset(self.dataset_maybe_noisy, self.idx_test)            
        else:
            self.val_dataset = Subset(self.dataset, self.idx_val)
            self.test_dataset = Subset(self.dataset, self.idx_test)

        if hasattr(self.hparams, "infer_mode"):
            self.test_dataset = self.dataset
        
        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
            len(self.test_dataset) > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0 and self.trainer.current_epoch != 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
            store_dataloader and not self.trainer.reload_dataloaders_every_n_epochs
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        if self.mask_atom:
            dl = DataLoaderMasking(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.hparams["num_workers"],
                pin_memory=True,
            )
        else:
            if self.model == 'egnn':
                from torch.utils.data import DataLoader
                dl = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=self.hparams["num_workers"],
                    pin_memory=True,
                    collate_fn=collate_fn,
                )
            else:
                from torch_geometric.data import DataLoader
                dl = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=self.hparams["num_workers"],
                    pin_memory=True,
                )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def get_energy_data(self, data):
        if data.y is None:
            raise MissingEnergyException()


        # remove atomref energies from the target energy
        atomref_energy = self.atomref.squeeze()[data.z].sum()
        return (data.y.squeeze() - atomref_energy).unsqueeze(dim=0).unsqueeze(dim=1)


    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            # atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            atomref = None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
