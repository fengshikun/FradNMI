from pyscf import gto, dft
mol_hf = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = '6-31g', symmetry = True)
mf_hf = dft.RKS(mol_hf)
mf_hf.xc = 'b3lyp'
energy = mf_hf.kernel()
g_2 = mf_hf.nuc_grad_method() 
force = g_2.kernel()
print(force)


from tqdm import tqdm
from ase.build import molecule
from ase import Atoms
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import numpy as np
import time
MOL_LST2 = np.load("h_mol_lst.npy", allow_pickle=True)

sample_num = 100

energy_lst = []
forces_lst = []

start = time.time()
for i in tqdm(range(sample_num)):
    test_mol = MOL_LST2[i]
    elements = [atom.GetSymbol() for atom in test_mol.GetAtoms()]
    coordinates = test_mol.GetConformer().GetPositions()
    # position = 
    atoms = [(element, coordinate) for element, coordinate in zip(elements, coordinates)]
    # d = 1.1
    # atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])
    atoms = Atoms(''.join(elements), coordinates)
    atoms = Atoms('HF', [[0, 0, 0], [0, 0, 1.1]])
    # atoms = molecule("CH3CH2OCH3")
    # device="cuda:0" for fast GPU computation.
    calc = TorchDFTD3Calculator(atoms=atoms, device="cuda", damping="bj")

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    energy_lst.append(energy)
    forces_lst.append(forces)
    # print(f"energy {energy} eV")
    # print(f"forces {forces}")
end = time.time()
print(f"execute time is {end-start}s") 
exit(0)


import numpy as np
MOL_LST2 = np.load("h_mol_lst.npy", allow_pickle=True)
test_mol = MOL_LST2[0]
elements = [atom.GetSymbol() for atom in test_mol.GetAtoms()]
coordinates = test_mol.GetConformer().GetPositions()
atoms = [(element, coordinate) for element, coordinate in zip(elements, coordinates)]

pyscf_mole = gto.Mole(basis="sto-3g")
pyscf_mole.atom = atoms
pyscf_mole.build()

mf_hf = dft.RKS(pyscf_mole)
mf_hf.xc = 'b3lyp'
energy = mf_hf.kernel()
g_2 = mf_hf.nuc_grad_method() 
force = g_2.kernel()


print(force)