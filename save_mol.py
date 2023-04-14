import numpy as np
from rdkit import Chem
MOL_LST2 = np.load("h_mol_lst.npy", allow_pickle=True)
test_mol = MOL_LST2[0]

writer = Chem.SDWriter(f'test_mol.sdf')
writer.write(test_mol)
writer.close()