from rdkit import Chem
import numpy as np
from torchmdnet.datasets import PCQM4MV2 
import torch
from rdkit.Geometry import Point3D
import multiprocessing
import copy
from tqdm import tqdm
import pickle

pcqm_root_dir = '/sharefs/sharefs-skfeng/Denoising/data/pcq'
position_noise_scale = 0.06 # from examples/ET-PCQM4MV2.yaml
sample_num = 100

pcq_datset = PCQM4MV2(root=pcqm_root_dir, transform=None)

suppl = Chem.SDMolSupplier('pcqm4m-v2-train.sdf')
debug = False

# noise:
def transform(data, position_noise_scale):
    noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
    data_noise = data + noise.numpy()
    return data_noise


# code from equibind:

def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True

def get_geometry_graph_ring(lig, only_atom_ring=False, return_coords=False):
    coords = lig.GetConformer().GetPositions()
    rings = lig.GetRingInfo().AtomRings()
    bond_rings = lig.GetRingInfo().BondRings()
    edges_src = []
    edges_dst = []
    for i, atom in enumerate(lig.GetAtoms()):
        src_idx = atom.GetIdx()
        assert src_idx == i
        if not only_atom_ring:
            one_hop_dsts = [neighbor for neighbor in list(atom.GetNeighbors())]
            two_and_one_hop_idx = [neighbor.GetIdx() for neighbor in one_hop_dsts]
            for one_hop_dst in one_hop_dsts:
                for two_hop_dst in one_hop_dst.GetNeighbors():
                    two_and_one_hop_idx.append(two_hop_dst.GetIdx())
            all_dst_idx = list(set(two_and_one_hop_idx))
        else:
            all_dst_idx = []
        for ring_idx, ring in enumerate(rings):
            if src_idx in ring and isRingAromatic(lig,bond_rings[ring_idx]):
                all_dst_idx.extend(list(ring))
        all_dst_idx = list(set(all_dst_idx))
        if len(all_dst_idx) == 0: continue
        all_dst_idx.remove(src_idx)
        all_src_idx = [src_idx] *len(all_dst_idx)
        edges_src.extend(all_src_idx)
        edges_dst.extend(all_dst_idx)
    
    # graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=lig.GetNumAtoms(), idtype=torch.long)
    feat = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    if return_coords:
        return edges_src, edges_dst, coords
    else:
        return edges_src, edges_dst, feat
    # return {'edges_src': edges_src, 'edges_dst': edges_dst, 'feat': feat}
    # return graph


# return min loss, max loss, diff
def statistic_loss(mol, para=False):
    edges_src, edges_dst, feat = get_geometry_graph_ring(mol)
    edges_src2, edges_dst2, feat2 = get_geometry_graph_ring(mol, only_atom_ring=True)


    coords = mol.GetConformer().GetPositions()
    
    # noise
    loss_lst = []
    
    loss_lst_only_ring = []

    noise_coords_lst = []

    sample_num = 1000
    if para:
        repeat_coords = coords[np.newaxis, :].repeat(sample_num, axis=0)
        noise_coords = transform(repeat_coords, position_noise_scale)
        noise_feat = torch.from_numpy(np.linalg.norm(noise_coords[:,edges_src] - noise_coords[:,edges_dst], axis=2).astype(np.float32))
        loss_lst = torch.mean((noise_feat**2 - feat ** 2)**2, dim=1)
        
        pass

    for i in range(sample_num):
        noise_coords = transform(coords, position_noise_scale)
        noise_coords_lst.append(noise_coords)
        noise_feat = torch.from_numpy(np.linalg.norm(noise_coords[edges_src] - noise_coords[edges_dst], axis=1).astype(np.float32))
        Loss = torch.mean((noise_feat**2 - feat ** 2)**2)
        loss_lst.append(Loss.item())

        noise_feat2 = torch.from_numpy(np.linalg.norm(noise_coords[edges_src2] - noise_coords[edges_dst2], axis=1).astype(np.float32))
        Loss2 = torch.mean((noise_feat2**2 - feat2 ** 2)**2)
        loss_lst_only_ring.append(Loss2.item())
    
    
        
    
    loss_lst_numpy = np.array(loss_lst)
    sort_idx = np.argsort(loss_lst_numpy)
    min_noise, max_noise = loss_lst_numpy[sort_idx[0]], loss_lst_numpy[sort_idx[-1]]
    diff = max_noise - min_noise
    
    loss_lst_ring_numpy = np.array(loss_lst_only_ring)
    ring_sort_idx = np.argsort(loss_lst_ring_numpy)
    ring_min_noise, ring_max_noise = loss_lst_only_ring[ring_sort_idx[0]], loss_lst_only_ring[ring_sort_idx[-1]]
    ring_diff = ring_max_noise - ring_min_noise
    
    
    return max_noise, min_noise, diff, ring_max_noise, ring_min_noise, ring_diff


    if debug:
        # show mol coordinate
        mol_cpy = copy.copy(mol)
        conf = mol_cpy.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x,y,z = noise_coords_lst[min_noise][i]
            conf.SetAtomPosition(i,Point3D(x,y,z))
        

        writer = Chem.SDWriter('org.sdf')
        writer.write(mol)
        writer.close()

        # supplier = Chem.SDMolSupplier('v3000.sdf')
        writer = Chem.SDWriter('min_noise.sdf')
        writer.write(mol_cpy)
        writer.close()
        # show mol coordinate
        mol_cpy = copy.copy(mol)
        conf = mol_cpy.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x,y,z = noise_coords_lst[max_noise][i]
            conf.SetAtomPosition(i,Point3D(x,y,z))
        
        writer = Chem.SDWriter('max_noise.sdf')
        writer.write(mol_cpy)
        writer.close()

        

        smiles = Chem.MolToSmiles(mol)
        atom_lst = []
        atom_num_lst = []
        for atom in mol.GetAtoms():
            atom_lst.append(atom.GetSymbol())
            atom_num_lst.append(atom.GetAtomicNum())
        coord_data = pcq_datset[idx]
        print(coord_data.z)
        print(atom_num_lst)


        # check the smiles and graph atom, if they align, should obey the order: smiles--> generate graph

        # mol_wo_h = Chem.MolFromSmiles(smiles)
        # atom_lst2 = []
        # for atom in mol_wo_h.GetAtoms():
        #     atom_lst2.append(atom.GetSymbol())
    pass


import matplotlib.pyplot as plt
def draw_dist(dist_numpy, save_prefix='', delima=0.01):
    plt.cla()
    bins = np.arange(dist_numpy.min(), dist_numpy.max(), delima)
    plt.hist(dist_numpy, bins=bins, label='pos', color='steelblue', alpha=0.7)
    plt.title(f'{save_prefix}')
    plt.xlabel('value')
    plt.ylabel('number')
    plt.savefig(f'{save_prefix}.png')


# coordinate 
# parallel statistics

# iterate the smiles and 3d
cnt = 0
total_num = len(suppl)
pool = multiprocessing.Pool(64)

def lines():
    for idx, mol in enumerate(suppl):
        yield mol


# debug
save_pkl = []
save_pkl_2 = []
for mol in tqdm(suppl):
    try:
        save_pkl.append(get_geometry_graph_ring(mol, return_coords=True))
    except:
        print('error happens save_pkl')
        save_pkl.append([])
    try:
        save_pkl_2.append(get_geometry_graph_ring(mol, only_atom_ring=True, return_coords=True))
    except:
        print('error happens save_pkl_2')
        save_pkl_2.append([])

save_file = f'mol_iter_all.pickle'
save_file2 = f'mol_iter_all_only_ring.pickle'
with open(save_file, 'wb') as handle:
    pickle.dump(save_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_file2, 'wb') as handle:
    pickle.dump(save_pkl_2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(save_file, 'rb') as handle:
#     mol_lst = np.array(pickle.load(handle))

# np.save('mol_iter_all.npy', mol_lst)

exit(0)


# noise_info_lst = np.zeros((total_num, 6), dtype=np.float32)
# cnt = 0
# for res in tqdm(pool.imap(statistic_loss, lines(), chunksize=100), total=total_num):
#     noise_info_lst[cnt] = res
#     cnt += 1


save_file = f'var_{position_noise_scale}_iter_{sample_num}_noise_st.pickle'
# with open(save_file, 'wb') as handle:
#     pickle.dump(noise_info_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_file, 'rb') as handle:
    noise_info_lst = pickle.load(handle)

valid_idx = ~np.isnan(noise_info_lst[:,0])


draw_dist(noise_info_lst[:,0][valid_idx], 'max_noise')
draw_dist(noise_info_lst[:,1][valid_idx], 'min_noise')
draw_dist(noise_info_lst[:,2][valid_idx], 'diff_noise')


valid_idx = ~np.isnan(noise_info_lst[:,3])
draw_dist(noise_info_lst[:,3][valid_idx], 'ring_max_noise')
draw_dist(noise_info_lst[:,4][valid_idx], 'ring_min_noise')
draw_dist(noise_info_lst[:,5][valid_idx], 'ring_diff_noise')

pass
# max_noise, min_noise, diff, ring_max_noise, ring_min_noise, ring_diff
