import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torsion_utils import get_torsions, GetDihedral
import copy
import argparse
from tqdm import tqdm
import multiprocessing




def find_minima_id(energy_lst, window_size=5):
    idx = window_size // 2
    minima_idx = []
    e_len = len(energy_lst)
    while idx + 2 < e_len:
        if (energy_lst[idx - 2] > energy_lst[idx - 1]) and (energy_lst[idx - 1] > energy_lst[idx]) \
           and (energy_lst[idx] < energy_lst[idx + 1]) and (energy_lst[idx + 1] < energy_lst[idx + 2]):
            minima_idx.append(idx)
        idx += 1
    return minima_idx


def pre_check(equi_mol):
    no_h_mol = Chem.RemoveHs(equi_mol)
    rotable_bonds = get_torsions([no_h_mol])
    if not len(rotable_bonds):
        return -1
    mp = AllChem.MMFFGetMoleculeProperties(equi_mol)
    ff2 = AllChem.MMFFGetMoleculeForceField(equi_mol, mp)
    return 0
    # check finished



def find_local_minima(equi_mol):
    no_h_mol = Chem.RemoveHs(equi_mol)
    rotable_bonds = get_torsions([no_h_mol])
    org_angle = []

    local_minma_energy = []
    try:
        for rot_bond in rotable_bonds:
            org_angle.append(GetDihedral(equi_mol.GetConformer(), rot_bond))

            m2=copy.deepcopy(equi_mol)
            m3 = copy.deepcopy(m2)
            # for each roateble bond, change on m3, add energy to the m2
            mp = AllChem.MMFFGetMoleculeProperties(m3)
            # mp.SetMMFFOopTerm(False)    # That's the critical bit here - switch off out of plane terms for MMFF
            # ffm = AllChem.MMFFGetMoleculeForceField(m3, mp)
            energy = []
            confid = 0
            angles = range(0, 370, 5)
            for angle in angles:
                confid += 1
                ff2 = AllChem.MMFFGetMoleculeForceField(m3, mp)
                if ff2 is None:
                    continue
                ff2.MMFFAddTorsionConstraint(rot_bond[0], rot_bond[1], rot_bond[2], rot_bond[3], False, angle - .1, angle + .1, 5.0)
                ff2.MMFFAddTorsionConstraint(rot_bond[0], rot_bond[1], rot_bond[2], rot_bond[3], False, angle - .1, angle + .1, 5.0)
                ff2.Minimize()
                energy.append(ff2.CalcEnergy())
                # xyz=ff2.Positions()
                new_conf = Chem.Conformer(equi_mol.GetNumAtoms())
                for i in range(equi_mol.GetNumAtoms()):
                    new_conf.SetAtomPosition(i, (m3.GetConformer(-1).GetAtomPosition(i)))
                new_conf.SetId(confid)
                m2.AddConformer(new_conf)

            min_conf_ids = find_minima_id(energy)
            confid = equi_mol.GetNumConformers()
            for min_id in min_conf_ids:
                new_conf = m2.GetConformer(min_id + 1)
                new_conf.SetId(confid)
                equi_mol.AddConformer(new_conf)
                confid += 1
                local_minma_energy.append(energy[min_id])
    except:
        print('Error happens')
    
    return equi_mol, local_minma_energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mix of Guassian Data augmentation of PCQM4MV2')
    parser.add_argument("--all_h_mol", type=str, default="/share/project/sharefs-skfeng/pre-training-via-denoising/h_mol_lst.npy")

    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()
    
    all_h_mol = args.all_h_mol
    MOL_LST = np.load(all_h_mol, allow_pickle=True)

    all_mol_len = len(MOL_LST)
    all_mol_len = 3378606

    split_num = all_mol_len // 8
    check_idx = [1, 2, 3, 4, 5, 6, 7, 8]
    
    empty_idx_lst = []
    # get empty index
    for idx in check_idx:
        mg_e_file = f'MG_E_part_{idx}.npy'
        MG_E = np.load(mg_e_file, allow_pickle=True)
        
        start_idx = split_num * (idx - 1)
        for idx, ele in enumerate(MG_E):
            if not len(ele):
                empty_idx_lst.append(idx + start_idx)

    np.save('empty_idx_lst.npy', empty_idx_lst)
    empty_idx_lst = np.load('empty_idx_lst.npy', allow_pickle=True)

    MG_MOL_LST = []
    MG_E = []
    for idx in tqdm(empty_idx_lst):
        mol = MOL_LST[idx]
        equi_mol = copy.copy(mol)
        new_mol, minima_energy = find_local_minima(equi_mol)
        # print(minima_energy)
        MG_MOL_LST.append(new_mol)
        MG_E.append(minima_energy)
    
    np.save(f'MG_MOL_LST_empty.npy', MG_MOL_LST)
    np.save(f'MG_E_part_empty.npy', MG_E)
    MG_E = np.load('MG_E_part_empty.npy', allow_pickle=True)

    print('finished')


    # check mol
    # res_lst = []
    # for mol in tqdm(MOL_LST):
    #     try:
    #         res = pre_check(mol)
    #         res_lst.append(res)
    #     except:
    #         res_lst.append(-2)
    # np.save('res_lst.npy', res_lst)
    # exit(0)