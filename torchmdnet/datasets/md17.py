import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
from rdkit import Chem

from torsion_utils import get_torsions, GetDihedral, apply_changes



class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="aspirin_dft.npz",
        benzene="benzene2017_dft.npz",
        ethanol="ethanol_dft.npz",
        malonaldehyde="malonaldehyde_dft.npz",
        naphthalene="naphthalene_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        uracil="uracil_dft.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(MD17, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(MD17, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return [MD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md17-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(MD17.raw_url + file_name, self.raw_dir)

    def process(self):
        for path in self.raw_paths:
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()

            samples = []
            for pos, y, dy in zip(positions, energies, forces):
                samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy))

            if self.pre_filter is not None:
                samples = [data for data in samples if self.pre_filter(data)]

            if self.pre_transform is not None:
                samples = [self.pre_transform(data) for data in samples]

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])





MOL_LST = None
class MD17A(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="aspirin_dft.npz",
        benzene="benzene2017_dft.npz",
        ethanol="ethanol_dft.npz",
        malonaldehyde="malonaldehyde_dft.npz",
        naphthalene="naphthalene_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        uracil="uracil_dft.npz",
    )


    mol_npy_files = dict(
        aspirin="aspirin.npy",
        benzene="benzene2017.npy",
        ethanol="ethanol.npy",
        malonaldehyde="malonaldehyde.npy",
        naphthalene="naphthalene.npy",
        salicylic_acid="salicylic.npy",
        toluene="toluene.npy",
        uracil="uracil.npy",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None, dihedral_angle_noise_scale=0.1, position_noise_scale=0.005, composition=False, reverse_half=False, addh=False, cod_denoise=False):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(MD17A, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )
        
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition
        self.reverse_half = reverse_half
        self.addh = addh
        self.cod_denoise = cod_denoise

        global MOL_LST
        if MOL_LST is None:
            MOL_LST = np.load(f"{root}/processed/{MD17A.mol_npy_files[dataset_arg]}", allow_pickle=True)
    
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

        # get rotate bond
        if self.addh: # default: False
            no_h_mol = mol
        else:
            no_h_mol = Chem.RemoveHs(mol)
        # rotable_bonds = get_torsions([mol])
        rotable_bonds = get_torsions([no_h_mol])

        
        if self.reverse_half:
            reverse_bonds = []
            for rb in rotable_bonds:
                l_rb = list(rb)
                l_rb.reverse()
                reverse_bonds.append(l_rb)

        assert atom_num == org_atom_num
        if len(rotable_bonds) == 0 or self.cod_denoise:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.org_pos = org_data.pos
            org_data.pos = torch.tensor(pos_noise_coords)
            return org_data



        # else angel random
        # if len(rotable_bonds):
        org_angle = []
        for rot_bond in rotable_bonds:
            org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
        org_angle = np.array(org_angle)        
        noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
        if self.reverse_half:
            noise_diff = noise_angle - org_angle
            half_noise_angle = org_angle + noise_diff / 2
            new_mol = apply_changes(mol, half_noise_angle, reverse_bonds)
            new_mol = apply_changes(new_mol, noise_angle, rotable_bonds)
        else:
            new_mol = apply_changes(mol, noise_angle, rotable_bonds)
        
        coord_conf = new_mol.GetConformer()
        pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

        
        # only angle denoise
        
        
        if not self.composition:
            org_data.pos_target = torch.tensor(pos_noise_coords_angle - org_data.pos.numpy())
            org_data.org_pos = org_data.pos
            org_data.pos = torch.tensor(pos_noise_coords_angle)
            return org_data


        # composition
        pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
        
        org_data.org_pos = org_data.pos
        org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
        org_data.pos = torch.tensor(pos_noise_coords)
        
        
        # if self.composition:
        #     org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
        #     org_data.pos = torch.tensor(pos_noise_coords)
        # else:
        #     org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
        #     org_data.pos = torch.tensor(pos_noise_coords)
        
        return org_data


    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(MD17A, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return [MD17A.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md17-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(MD17A.raw_url + file_name, self.raw_dir)

    def process(self):
        for path in self.raw_paths:
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()

            samples = []
            for pos, y, dy in zip(positions, energies, forces):
                samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy))

            if self.pre_filter is not None:
                samples = [data for data in samples if self.pre_filter(data)]

            if self.pre_transform is not None:
                samples = [self.pre_transform(data) for data in samples]

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])
