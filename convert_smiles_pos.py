from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=False):
    """
    Generate atomic coordinates from a SMILES string representation of a molecule.

    Args:
        smi (str): SMILES representation of the molecule.
        seed (int, optional): Random seed for conformer generation. Default is 42.
        mode (str, optional): Mode for conformer generation ('fast' or 'heavy'). Default is 'fast'.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from results. Default is False.

    Returns:
        tuple: If remove_hs=True, returns a tuple containing lists of atomic numbers and corresponding coordinates (numpy array). 
            If remove_hs=False, returns a tuple containing lists of atomic numbers and corresponding coordinates (numpy array) including hydrogen atoms.
            
    Raises:
        AssertionError: If the number of atoms and coordinates do not match.
        AssertionError: If attempting to remove hydrogen atoms but the resulting lists of atoms and coordinates do not match.

    """
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    # get Atom number by GetAtomicNumber


    
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    # atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms),3))
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates


# parse smiles file from args
import argparse
parser = argparse.ArgumentParser(description='user sample')
parser.add_argument('--smiles_file', type=str, default='smiles.lst', help='input smiles list file')
parser.add_argument('--output_file', type=str, default='smiles_coord.lst', help='output save result file')


args = parser.parse_args()
smiles_file = args.smiles_file
output_file = args.output_file


with open(smiles_file, 'r') as f:
    smiles = f.readlines()[1:]
    smiles = [smi.strip() for smi in smiles]

from tqdm import tqdm

# generate atoms, coordinates for each molecule
atoms_list = []
coordinates_list = []
for smi in tqdm(smiles):
    atoms, coordinates = inner_smi2coords(smi, mode='heavy', remove_hs=False)
    atoms_list.append(atoms)
    coordinates_list.append(coordinates)

with open(args.output_file, 'w') as f:
    f.write('atom_list \t pos_list\n')
    save_num = len(atoms_list)
    for i in range(save_num):
        f.write(f'{str(atoms_list[i])}\t{str(coordinates_list[i].tolist())}\n')
