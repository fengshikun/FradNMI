import numpy as np
import lmdb
import pickle

root = '/data/protein/SKData/DenoisingData/pcq'
MOL_LST = lmdb.open(f'{root}/MOL_LMDB', readonly=True, subdir=True, lock=False)

# get the MOL_LST length
with MOL_LST.begin() as txn:
    _keys = list(txn.cursor().iternext(values=False))
    mol_len = len(_keys)



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
    # for mol in tqdm(MOL_LST2):
    generate_rdkit_num_lst = []
    
    # keep _keys 10w
    keep_keys = _keys[:100000]
    
    
    for ky in tqdm(keep_keys):
        serialized_data = MOL_LST.begin().get(ky)
        mol = pickle.loads(serialized_data)
        test_mol = copy.deepcopy(mol)
        # cids = AllChem.EmbedMultipleConfs(test_mol, numConfs=1, numThreads=0)
        try:
            test_mol.RemoveConformer(0)
            cids = AllChem.EmbedMultipleConfs(test_mol, numConfs=16, numThreads=16, pruneRmsThresh=0.1, maxAttempts=5, useRandomCoords=False)
            

            if len(cids) < 1:
                rdkit_failed_cnt += 1
                print('rdkit generate fail')
            else:
                AllChem.MMFFOptimizeMoleculeConfs(test_mol, numThreads=16)
        except Exception as e:
            print(f'exeption captured {e}')
        rdkit_conf_mol_lst.append(test_mol)
        generate_rdkit_num_lst.append(test_mol.GetNumConformers())
        cnt += 1

    np.save('rdkit_mols_conf_lst.npy', rdkit_conf_mol_lst)
    # save generate_rdkit_num_lst
    np.save('generate_rdkit_num_lst.npy', generate_rdkit_num_lst)