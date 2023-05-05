import numpy as np
from openff.toolkit.topology import Molecule
from openmm.unit import *
from collections import defaultdict
import h5py

typeDict = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}

infile = h5py.File('/home/fengshikun/SPICE-1.1.2.hdf5')

# First pass: group the samples by total number of atoms.

groupsByAtomCount = defaultdict(list)
for name in infile:
    group = infile[name]
    count = len(group['atomic_numbers'])
    groupsByAtomCount[count].append(group)

# Create the output file.

filename = 'SPICE-processed.hdf5'
outfile = h5py.File(filename, 'w')

# One pass for each number of atoms, creating a group for it.

print(sorted(list(groupsByAtomCount.keys())))
posScale = 1*bohr/angstrom
energyScale = 1*hartree/item/(kilojoules_per_mole)
forceScale = energyScale/posScale
for count in sorted(groupsByAtomCount.keys()):
    print(count)
    smiles = []
    pos = []
    types = []
    energy = []
    forces = []
    for g in groupsByAtomCount[count]:
        molSmiles = g['smiles'][0]
        mol = Molecule.from_mapped_smiles(molSmiles, allow_undefined_stereo=True)
        molTypes = [typeDict[(atom.symbol, (atom.formal_charge/elementary_charge).magnitude._value)] for atom in mol.atoms]
        assert len(molTypes) == count
        for i, atom in enumerate(mol.atoms):
            assert atom.atomic_number == g['atomic_numbers'][i]
        numConfs = g['conformations'].shape[0]
        for i in range(numConfs):
            smiles.append(molSmiles)
            pos.append(g['conformations'][i])
            types.append(molTypes)
            energy.append(g['formation_energy'][i])
            forces.append(g['dft_total_gradient'][i])
    group = outfile.create_group(f'samples{count}')
    group.create_dataset('smiles', data=smiles, dtype=h5py.string_dtype())
    group.create_dataset('types', data=np.array(types), dtype=np.int8)
    group.create_dataset('pos', data=np.array(pos)*posScale, dtype=np.float32)
    group.create_dataset('energy', data=np.array(energy)*energyScale, dtype=np.float32)
    group.create_dataset('forces', data=-np.array(forces)*forceScale, dtype=np.float32)