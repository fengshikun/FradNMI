# from pyscf import gto, dft
# # mol_hf = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = '6-31g', symmetry = True)
# mol_hf = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz')
# mf_hf = dft.RKS(mol_hf)
# mf_hf.xc = 'b3lyp'
# energy = mf_hf.kernel()
# g_2 = mf_hf.nuc_grad_method() 
# force = g_2.kernel()
# print(force)

import pyscf
from pyscf import gto, scf
# Load the gpu4pyscf to enable GPU mode
import sys
sys.path.insert(0, "/home/fengshikun/gpu4pyscf")
from gpu4pyscf import patch_pyscf

from tqdm import tqdm
from ase.build import molecule
from ase import Atoms
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import numpy as np
import time
from torchmdnet.datasets import MD17
from scipy.stats import pearsonr

npy_file = '/share/project/sharefs-skfeng/xyz2mol/aspirin.npy'
asp_mols = np.load(npy_file, allow_pickle=True)
equi_mol = asp_mols[0]
md17_data = MD17('/share/project/sharefs-skfeng/MD17', dataset_arg='aspirin')



elements = [atom.GetSymbol() for atom in equi_mol.GetAtoms()]

sample_force = []
pear_lst = []
energy_lst = []

samples_num = 1000
for i, md_ele in enumerate(tqdm(md17_data)):
    coordinates = md_ele.pos.numpy()
    force_dft = md_ele.dy.numpy().flatten()

    print(f"md17 label force: {force_dft}")
    # sample_force.append()
    atoms = Atoms(''.join(elements), coordinates)
    # atoms = molecule("CH3CH2OCH3")
    # device="cuda:0" for fast GPU computation.
    # atoms = Atoms('OHH', [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    calc = TorchDFTD3Calculator(atoms=atoms, device="cuda", damping="bjm", xc="b3-lyp")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f'pytorch dft forces: \n {forces}, energy: {energy}')

    atoms = [(element, coordinate) for element, coordinate in zip(elements, coordinates)]
    pyscf_mole = gto.Mole(basis="6-31g")
    pyscf_mole.atom = atoms
    pyscf_mole.build()
    # mf_hf = dft.RKS(pyscf_mole)
    # mf_hf.xc = 'b3lyp'
    # energy = mf_hf.kernel()
    # g_2 = mf_hf.nuc_grad_method() 
    # force_pyscf = g_2.kernel()
    mf = scf.RHF(pyscf_mole)
    res = mf.kernel()
    g2 = mf.nuc_grad_method()
    force_pyscf = g2.kernel()


    print(f'pyscf forces: \n {forces}')


    res = pearsonr(force_pyscf.flatten(), forces.flatten())
    energy_lst.append(energy)
    pear_lst.append(res[0])
    if i > samples_num:
        break



print(f'mean pearson value is {np.mean(pear_lst)}')




