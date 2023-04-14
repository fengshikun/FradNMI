import numpy as np
MOL_LST2 = np.load("h_mol_lst.npy", allow_pickle=True)

mol_len = len(MOL_LST2)

from rdkit import Chem
import copy
from rdkit.Chem import AllChem
# from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.ipython_useSVG = True

import multiprocessing
from tqdm import tqdm
rdkit_failed_cnt = 0
rdkit_conf_mol_lst = []

error_idx = 2662420

cnt = 1

if __name__ == "__main__":

    def generate_rdkit_conf(mol):
        test_mol = copy.deepcopy(mol)
        try:
            test_mol.RemoveConformer(0)
            cids = AllChem.EmbedMultipleConfs(test_mol, numConfs=1, numThreads=8, pruneRmsThresh=0.1, maxAttempts=5, useRandomCoords=False)
            

            if len(cids) < 1:
                # rdkit_failed_cnt += 1
                print('rdkit generate fail')
            else:
                AllChem.MMFFOptimizeMoleculeConfs(test_mol, numThreads=8)
        except Exception as e:
            print(f'exeption captured {e}')
        return test_mol


    pool = multiprocessing.Pool(64)
    # multi process 

    def lines():
        for idx, mol in enumerate(MOL_LST2):
            yield mol



    # for res in tqdm(pool.imap(generate_rdkit_conf, lines(), chunksize=64), total=mol_len):
    #     rdkit_conf_mol_lst.append(res)
    for mol in tqdm(MOL_LST2):
        test_mol = copy.deepcopy(mol)
        # cids = AllChem.EmbedMultipleConfs(test_mol, numConfs=1, numThreads=0)
        try:
            test_mol.RemoveConformer(0)
            cids = AllChem.EmbedMultipleConfs(test_mol, numConfs=1, numThreads=8, pruneRmsThresh=0.1, maxAttempts=5, useRandomCoords=False)
            

            if len(cids) < 1:
                rdkit_failed_cnt += 1
                print('rdkit generate fail')
            else:
                AllChem.MMFFOptimizeMoleculeConfs(test_mol, numThreads=8)
        except Exception as e:
            print(f'exeption captured {e}')
        rdkit_conf_mol_lst.append(test_mol)

        cnt += 1

    np.save('rdkit_mols_conf_lst.npy', rdkit_conf_mol_lst)