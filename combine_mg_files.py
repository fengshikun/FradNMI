import os
import numpy as np


check_idx = [1, 2, 3, 4, 5, 6, 7, 8]

MG_E_All = []
MG_MOL_All = []
for idx in check_idx:
    print(f'process idx {idx}...')
    MG_E = np.load(f'MG_E_part_{idx}.npy', allow_pickle=True)
    MG_MOL = np.load(f'MG_MOL_LST_part_{idx}.npy', allow_pickle=True)
    MG_E_All.extend(MG_E)
    MG_MOL_All.extend(MG_MOL)

np.save('MG_All.npy', MG_E_All)
np.save('MG_MOL_All.npy', MG_MOL_All)
