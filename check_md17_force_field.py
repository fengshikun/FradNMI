from torchmdnet.datasets import MD17
import numpy as np
from tqdm import tqdm
from rdkit import Chem

from scipy.stats import pearsonr

npy_file = '/share/project/sharefs-skfeng/xyz2mol/aspirin.npy'
asp_mols = np.load(npy_file,allow_pickle=True)
equi_mol = asp_mols[0]
# org_pos = equi_mol.GetConformer().GetPositions()
md17_data = MD17('/share/project/sharefs-skfeng/MD17', dataset_arg='aspirin')

from sgdml.predict import GDMLPredict
model = np.load('/home/fengshikun/Backup/Denoising/data/md17/aspirin/raw/aspirin-aims.PBE.TS.light.tier.1-train200-sym6.npz')
gdml = GDMLPredict(model)


from torsion_utils import get_torsions, GetDihedral, apply_changes
import torch
def transform_noise(data, position_noise_scale):
    noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
    data_noise = data + noise.numpy()
    return data_noise

sample_num = 1000
base_pos = md17_data[0].pos
tau = 0.04
engery_lst = []
force_lst = []

# l_force_lst = []
# for i in tqdm(range(sample_num)):
#     # coord noise
#     noise_pos = transform_noise(base_pos, tau).numpy()
#     e, f = gdml.predict(noise_pos.reshape(1, -1))
#     # predict force field
#     engery_lst.append(e)
#     force_lst.append(f)
#     # estimated force field
#     l_force_lst.append( (base_pos-noise_pos).numpy() / (tau**2))


# pear_lst = []
# for i in range(sample_num):
#     res = pearsonr(l_force_lst[i].flatten(), force_lst[i].flatten())
#     pear_lst.append(res[0])
# prear_value = np.mean(pear_lst)
# print(f'coord pearson is {prear_value}')


def get_SIGMA(C, sigma, tau, N, m):
    C1 = C[:m, :] # m x m
    C2 = C[m:, :] # (3N - m) x m
    # print(C2.shape)

    SIGMA_lt =  (tau ** 2) * np.identity(m) + (sigma ** 2) * np.dot(C1, C1.T) # left top
    SIGMA_lb = (sigma ** 2) * np.dot(C2, C1.T) # left bottom
    SIGMA_rt = (sigma ** 2) * np.dot(C1, C2.T) # right top
    SIGMA_rb = (tau ** 2) * np.identity(3*N - m) + (sigma ** 2) * np.dot(C2, C2.T) # right bottom

    SIGMA = np.zeros((3*N, 3*N), dtype=np.float32) # 3N x 3N
    SIGMA[:m, :m] = SIGMA_lt # m x m
    SIGMA[m:, :m] = SIGMA_lb # (3N - m) x m
    SIGMA[:m, m:] = SIGMA_rt # m x (3N - m)
    SIGMA[m:, m:] = SIGMA_rb # (3N - m) * (3N -m)

    return SIGMA




no_h_mol = Chem.RemoveHs(equi_mol)
rotable_bonds = get_torsions([no_h_mol])
print(rotable_bonds)

org_angle = []
for rot_bond in rotable_bonds:
    org_angle.append(GetDihedral(equi_mol.GetConformer(), rot_bond))

sample_num = 1000
base_pos = md17_data[0].pos
tau = 0.04
sigma = 2
N = base_pos.shape[0]
m = len(org_angle)
engery_lst = []
force_lst = []
a_force_lst = []


delta_angle_lst = []
delta_pos_lst = []
new_angle_pos_lst = []
l_force_lst = []
for i in tqdm(range(sample_num)):
    # angle noise
    noise_angle = transform_noise(org_angle, sigma)
    new_mol = apply_changes(equi_mol, noise_angle, rotable_bonds)

    coord_conf = new_mol.GetConformer()
    new_pos = np.zeros((N, 3), dtype=np.float32)
    # pos_noise_coords = new_mol.GetConformer().GetPositions()
    for idx in range(N):
        c_pos = coord_conf.GetAtomPosition(idx)
        new_pos[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]


    # new_pos = new_mol.GetConformer().GetPositions()
    delta_pos = new_pos - base_pos.numpy()
    delta_angle = noise_angle - org_angle

    delta_angle_lst.append(delta_angle.reshape(1, -1))
    delta_pos_lst.append(delta_pos.reshape(1, -1))
    new_angle_pos_lst.append(new_pos)

# estimate C

print('estimate C')
delta_angle = np.concatenate(delta_angle_lst)
delta_pos = np.concatenate(delta_pos_lst)
C_res = np.linalg.lstsq(delta_angle, delta_pos,  rcond = -1)
residual_c = (np.linalg.norm(delta_angle.dot(C_res[0]) - delta_pos, axis=1) / np.linalg.norm(delta_pos, axis=1)).mean()
C = C_res[0].T
SIGMA = get_SIGMA(C, sigma, tau, N, m)
SIGMA_reverse = np.linalg.inv(SIGMA)
# exit(0)

for i in tqdm(range(sample_num)):
    # coord noise
    noise_pos = transform_noise(new_angle_pos_lst[i], tau)
    e, f = gdml.predict(noise_pos.reshape(1, -1))
    # predict force 
    engery_lst.append(e)
    force_lst.append(f)
    
    # estimate force
    a_force = SIGMA_reverse.dot((base_pos - noise_pos).reshape(-1, 1).numpy()).flatten()
    a_force_lst.append(a_force)

    l_force_lst.append( (base_pos-noise_pos).numpy() / (tau**2))
    

pear_lst = []
for i in range(sample_num):
    res = pearsonr(a_force_lst[i].flatten(), force_lst[i].flatten())
    pear_lst.append(res[0])
prear_value = np.mean(pear_lst)
print(f'angle coord pearson is {prear_value}')

# pear_lst = []
# for i in range(sample_num):
#     res = pearsonr(l_force_lst[i].flatten(), force_lst[i].flatten())
#     pear_lst.append(res[0])
# np.mean(pear_lst)

# prear_value = np.mean(pear_lst)
# print(f'angle coord2 pearson is {prear_value}')