import numpy as np
from rdkit import Chem
import copy
from tqdm import tqdm
from torchmdnet.datasets import MD17





import multiprocessing
from rdkit.Geometry import Point3D





if __name__ == "__main__":

    md17_mols_type = ['benzene2017', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil']

    for md17_type in md17_mols_type:
        path = '/share/project/sharefs-skfeng/MD17/raw/'
        data_npz = np.load(f'{path}/{md17_type}_dft.npz')
        if md17_type == 'benzene2017':
            dataset_arg = 'benzene'
        else:
            dataset_arg = md17_type
        md17_data = MD17('/share/project/sharefs-skfeng/MD17', dataset_arg=dataset_arg)

        npy_file = f'/home/fengshikun/MD17_data/{md17_type}.npy'
        mols_lst = np.load(npy_file,allow_pickle=True)

        

        # fix md17 mol lst
        s_mol = mols_lst[0]
        no_h_mol = Chem.RemoveHs(s_mol)

        def process_item():
            for idx, mol in enumerate(md17_data):
                yield md17_data[idx]['pos'].numpy()

        def set_coordinate(org_pos):
            new_mol = copy.copy(s_mol)
            conf = new_mol.GetConformer()
            for j in range(new_mol.GetNumAtoms()):
                x, y , z = org_pos[j][0], org_pos[j][1], org_pos[j][2]
                conf.SetAtomPosition(j, Point3D(float(x), float(y), float(z)))
            return new_mol

        new_mol_lst = []
        data_length = data_npz["R"].shape[0]
        pool = multiprocessing.Pool(64)
        for new_mol in tqdm(pool.imap(set_coordinate, process_item(), chunksize=5), total=data_length):
            new_mol_lst.append(new_mol)

        pool.close()
        np.save(f'/home/fengshikun/MD17_data/processed/{md17_type}.npy', new_mol_lst)