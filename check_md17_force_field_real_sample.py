from torchmdnet.datasets import MD17
import numpy as np
from tqdm import tqdm
from rdkit import Chem

from scipy.stats import pearsonr
import argparse


# org_pos = equi_mol.GetConformer().GetPositions()

from sgdml.predict import GDMLPredict



from torsion_utils import get_torsions, GetDihedral, apply_changes
import torch
def transform_noise(data, position_noise_scale):
    noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
    data_noise = data + noise.numpy()
    return data_noise



def get_estimate_coord_ff(noise_pos_lst, base_pos, tau=0.04):
    estimate_force_lst = []
    print('estimate coord ff')
    for noise_pos in tqdm(noise_pos_lst):
        c_force = ((base_pos-noise_pos).numpy() / (tau**2)).flatten()
        estimate_force_lst.append(c_force)
    
    return estimate_force_lst



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

def estimate_SIGMA_reverse(equi_mol, base_pos, sample_num=1000, tau=0.04, sigma=2):
    no_h_mol = Chem.RemoveHs(equi_mol)
    rotable_bonds = get_torsions([no_h_mol])
    # print(rotable_bonds)

    org_angle = []
    for rot_bond in rotable_bonds:
        org_angle.append(GetDihedral(equi_mol.GetConformer(), rot_bond))

    N = base_pos.shape[0]
    m = len(org_angle)


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
    return SIGMA_reverse, residual_c


def get_estimate_angle_ff(noise_pos_lst, equi_mol, base_pos, sample_num=1000, tau=0.04, sigma=2):
    estimate_force_lst = []
    SIGMA_reverse, residual_c = estimate_SIGMA_reverse(equi_mol, base_pos, sample_num, tau, sigma)
    print('estimate angle ff')
    for noise_pos in tqdm(noise_pos_lst):
        a_force = SIGMA_reverse.dot((base_pos - noise_pos).reshape(-1, 1).numpy()).flatten()
        estimate_force_lst.append(a_force)
    
    return estimate_force_lst, residual_c


def gaussian_sample(base_pos, sample_num=100, tau=0.04):
    noise_pos_lst = []
    for i in tqdm(range(sample_num)):
        noise_pos = transform_noise(base_pos, tau).numpy()
        noise_pos_lst.append(noise_pos)
    return noise_pos_lst

def angle_coord_gaussian_sample(base_pos, equi_mol, sample_num=100, tau=0.04, sigma=2):
    no_h_mol = Chem.RemoveHs(equi_mol)
    rotable_bonds = get_torsions([no_h_mol])
    print(rotable_bonds)

    org_angle = []
    for rot_bond in rotable_bonds:
        org_angle.append(GetDihedral(equi_mol.GetConformer(), rot_bond))

    noise_pos_lst = []
    N = base_pos.shape[0]
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

        # coord noise
        noise_pos = transform_noise(new_pos, tau)
        noise_pos_lst.append(noise_pos)
    return noise_pos_lst
    
def get_mean_pearson(predict_lst, estimate_lst):
    sample_num = len(predict_lst)
    pear_lst = []
    for i in range(sample_num):
        res = pearsonr(estimate_lst[i].flatten(), predict_lst[i].flatten())
        pear_lst.append(res[0])
    prear_value = np.mean(pear_lst)
    return prear_value


def predict_noise_pos(equi_mol, base_pos, sample_num, noise_pos_lst, sigma_lst, tau, force_label_lst=None):
    predict_force_lst = []
    predict_energy_lst = []
    if force_label_lst is not None:
        predict_force_lst = force_label_lst
    else:
        print('predicting use gdml....')
        for noise_pos in tqdm(noise_pos_lst):
            e, f = gdml.predict(noise_pos.reshape(1, -1))

            predict_energy_lst.append(e)
            predict_force_lst.append(f)
            # coord
    c_force_lst = get_estimate_coord_ff(noise_pos_lst, base_pos, 0.04)
    
    
    print(f'inside tau {0.04}, coord pearson is {get_mean_pearson(predict_force_lst, c_force_lst)}')


    c_force_lst = get_estimate_coord_ff(noise_pos_lst, base_pos, 0.4)
    
    
    print(f'inside tau {0.4}, coord pearson is {get_mean_pearson(predict_force_lst, c_force_lst)}')



    # angle + coord
    for sigma in sigma_lst:
        a_force_lst, c_error = get_estimate_angle_ff(noise_pos_lst, equi_mol, base_pos, sample_num, tau, sigma)
        print(f'inside sigma is {sigma}, inside tau is {tau}, angle + coord pearson is {get_mean_pearson(predict_force_lst, a_force_lst)}, C error is {c_error}')
        # print(a_force_lst[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument("--job_prefix", type=str, default="job_prefix")
    args = parser.parse_args()
    job_prefix = args.job_prefix

    npy_file = '/share/project/sharefs-skfeng/xyz2mol/aspirin.npy'
    asp_mols = np.load(npy_file,allow_pickle=True)
    equi_mol = asp_mols[0]
    md17_data = MD17('/share/project/sharefs-skfeng/MD17', dataset_arg='aspirin')
    model = np.load('/home/fengshikun/Backup/Denoising/data/md17/aspirin/raw/aspirin-aims.PBE.TS.light.tier.1-train200-sym6.npz')
    gdml = GDMLPredict(model)
    sample_num = 10000
    base_pos = md17_data[0].pos

    sigma_lst = [20, 2, 1, 0.5, 0.1, 0.01, 0.001]
    tau = 0.04

    from tqdm import tqdm

    equi_threshold = 0.0319127058416426
    non_equi_threshold = 0.617926264702376
    pos_diff_lst = []
    for md_ele in tqdm(md17_data):
        pos = md_ele.pos.numpy()
        pos_diff = (np.abs(pos - base_pos)).mean()
        pos_diff_lst.append(pos_diff)
    pos_diff_lst_numpy = np.array(pos_diff_lst)
    # pos_diff_lst_numpy = np.load('pos_diff_lst_numpy.npy')

    top_sample_idx = (pos_diff_lst_numpy <  equi_threshold).nonzero()[0][1:]

    # energy_y_lst = []
    # for md_ele in tqdm(md17_data):
    #     energy_y_lst.append(md_ele.y[0][0].item())
    # energy_y_lst = np.array(energy_y_lst)
    # q10, q90 = np.percentile(energy_y_lst, [10, 90])

    
    # top_sample_idx = (energy_y_lst < q10).nonzero()[0]

    # top_sample_idx = top_sample_idx[1:] # erase

    top_sample_pos = []; top_sample_force = []

    for idx in top_sample_idx:
        top_sample_pos.append(md17_data[idx].pos.numpy())
        top_sample_force.append(md17_data[idx].dy.numpy().flatten())
    
    

    predict_noise_pos(equi_mol, base_pos, sample_num, top_sample_pos, sigma_lst, tau, force_label_lst=top_sample_force)

    # print('+++++++++++++++++++left++++++++++++++++++++')
    # left_sample_idx = (energy_y_lst > q90).nonzero()[0]

    # # top_sample_idx = top_sample_idx[1:] # erase

    left_sample_idx = ((pos_diff_lst_numpy < non_equi_threshold) & (pos_diff_lst_numpy > equi_threshold)).nonzero()[0]
    left_sample_pos = []; left_sample_force = []

    for idx in left_sample_idx:
        left_sample_pos.append(md17_data[idx].pos.numpy())
        left_sample_force.append(md17_data[idx].dy.numpy().flatten())
    
    predict_noise_pos(equi_mol, base_pos, sample_num, left_sample_pos, sigma_lst, tau, force_label_lst=left_sample_force)



    # gaussian noise, tau=0.04
    # noise_pos_lst = gaussian_sample(base_pos, sample_num, tau)
    # print(f'+++++++++++++++guassian noise {tau}+++++++++++++++')
    # predict_noise_pos(equi_mol, base_pos, sample_num, noise_pos_lst, sigma_lst, tau)
    # sigma = 20
    # print(f'+++++++++++++++angel + coord guassian noise {sigma}, {tau}+++++++++++++++')
    # noise_pos_lst = angle_coord_gaussian_sample(base_pos, equi_mol, sample_num, tau, sigma)
    # predict_noise_pos(equi_mol, base_pos, sample_num, noise_pos_lst, sigma_lst, tau)
    # sigma = 1
    # print(f'+++++++++++++++angel + coord guassian noise {sigma}, {tau}+++++++++++++++')
    # noise_pos_lst = angle_coord_gaussian_sample(base_pos, equi_mol, sample_num, tau, sigma)
    # predict_noise_pos(equi_mol, base_pos, sample_num, noise_pos_lst, sigma_lst, tau)