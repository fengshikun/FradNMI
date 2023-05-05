import h5py
import os
filename = "./pubchem-0-100.hdf5"
# filename = 'pubchem-1-2500.hdf5'
n_mol = 0
from openmm.unit import *

from openff.toolkit.topology import Molecule, Topology
from rdkit import Chem
import numpy as np

ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}

# posScale = 1*bohr/angstrom
posScale = 10
energyScale = 1*hartree/item/(kilojoules_per_mole)
mol_lst = []
energy_lst = []

begin_cnt = 10000
begin_cnt = 0

with h5py.File(filename, 'r') as raw_data:
    for name in raw_data:
        n_mol += 1
        # if n_mol % 1000 == 0:
        #     print('Loading # molecule %d' % n_mol)
        
        if n_mol < begin_cnt:
            continue
        
        g = raw_data[name]
        smi = g['smiles'][0].decode('ascii')
        mol = Chem.MolFromSmiles(smi)
        # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        mol = Chem.AddHs(mol)
        # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        
        # org_mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
        # mol = org_mol.to_rdkit()

        # species = []
        atom_sym_lst = []
        # for atom in mol.GetAtoms():
        #     ele = atom.GetSymbol()
        #     atom_sym_lst.append(ele)
        #     charge = atom.GetFormalCharge()
        #     species.append(ATOM_DICT[(ele, charge)])
        
        


        n_conf = g['conformations'].shape[0]
        # atomic_numbers = list(g['atomic_numbers'])

        mol = Molecule.from_mapped_smiles(smi, allow_undefined_stereo=True)
        # for i, atom in enumerate(mol.atoms):
        #     assert atom.atomic_number == g['atomic_numbers'][i]


        rd_mol = mol.to_rdkit()
        rd_atom_sym_lst = []
        for atom in rd_mol.GetAtoms():
            ele = atom.GetSymbol()
            rd_atom_sym_lst.append(ele)            

        mol = rd_mol

        conformations = g['conformations']
        energy_mol = []

        atom_num = mol.GetNumAtoms()
        # save multiple coordinate into mol
        for i in range(n_conf):
            coord_conf = Chem.Conformer(atom_num)
            pos = conformations[i] * posScale
            for j in range(atom_num):
                coord_conf.SetAtomPosition(j, (float(pos[j][0]), float(pos[j][1]), float(pos[j][2])))
            coord_conf.SetId(i)
            mol.AddConformer(coord_conf)
            # energy_mol.append(float(g['formation_energy'][i]) * energyScale)
            i += 1
        mol_lst.append(mol)
        energy_lst.append(energy_mol)
        
        if n_mol > (begin_cnt + 100):
            break

np.save("SPICE_mols.npy", mol_lst)
np.save("SPICE_mols_energy.npy", energy_lst)