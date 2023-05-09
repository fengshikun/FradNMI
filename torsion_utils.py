from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import copy
from collections import defaultdict
import collections
import random
import numpy as np
import math

def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol



def GetBondLength(conf, atom_idx):
    return rdMolTransforms.GetBondLength(conf, atom_idx[0], atom_idx[1])

def SetBondLength(conf, atom_idx, new_vale):
    return rdMolTransforms.SetBondLength(conf, atom_idx[0], atom_idx[1], new_vale)

def GetAngle(conf, atom_idx):
    return rdMolTransforms.GetAngleDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2])

def SetAngle(conf, atom_idx, new_vale):
    return rdMolTransforms.SetAngleDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], new_vale)


def apply_changes_bond_length(mol, values, bond_idx):
    opt_mol = copy.deepcopy(mol)
    [SetBondLength(opt_mol.GetConformer(), bond_idx[r], values[r]) for r in range(len(bond_idx))]
    return opt_mol

def apply_changes_angle(mol, values, bond_idx):
    opt_mol = copy.deepcopy(mol)
    [SetAngle(opt_mol.GetConformer(), bond_idx[r], values[r]) for r in range(len(bond_idx))]

    return opt_mol

# input: mol (rdkit) object; rotate_bonds list
# return: rotate_bonds order(list, each rotate bond maybe reverse), and depth
def get_rotate_order_info(mol, rotate_bonds):
    cut_bonds_set = []
    for rb in rotate_bonds:
        cut_bonds_set.append([rb[1], rb[2]])
    left_bond_set = []
    
    bond_set = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_set.append([a1, a2])
    for ele in bond_set:
        if [ele[0], ele[1]] not in cut_bonds_set and \
            [ele[1], ele[0]] not in cut_bonds_set:
                left_bond_set.append(ele)
    graph = defaultdict(list)

    for x, y in left_bond_set:
        graph[x].append(y)
        graph[y].append(x)
    visited = set()
    mol_num = mol.GetNumAtoms()
    labels = [-1 for _ in range(mol_num)]

    rg_nodes = defaultdict(set) # key is the node idx of rigid-node graph, value is the indices of corresponding node in original mol graph.

    def dfs(i, lb=-1):
        visited.add(i)
        labels[i] = lb
        rg_nodes[lb].add(i)
        for j in graph[i]:
            if j not in visited:
                dfs(j, lb)

    lb = 0
    for i in range(mol_num):
        if i not in visited:
            dfs(i, lb)
            lb += 1
    # construct rigid-nodes graph 
    rg_graph = defaultdict(list)

    edge_rotate_bond_dict = {} # key is edge of rigid-nodes graph, value is the idx of rotate bonds

    for idx, rb in enumerate(cut_bonds_set):
        rg_edge = []
        # print('rb is {}'.format(rb))
        for key in rg_nodes:
            if rb[0] in rg_nodes[key]:
                rg_edge.append(key)
            if rb[1] in rg_nodes[key]:
                rg_edge.append(key)
            if len(rg_edge) == 2:
                edge_rotate_bond_dict[str(rg_edge)] = idx
                edge_rotate_bond_dict[str([rg_edge[1], rg_edge[0]])] = idx
                break
        # print(rg_edge)
        rg_graph[rg_edge[0]].append(rg_edge[1])
        rg_graph[rg_edge[1]].append(rg_edge[0])


    # add the node number into rg_graph:
    rg_graph_w_s = {}
    for rg in rg_graph:
        rg_graph_w_s[rg] = [rg_graph[rg], len(rg_nodes[rg])]
    
    def bfs(graph, root, edge_rotate_bond_dict, rotable_bonds, rg_nodes):
        seen, queue = {root}, collections.deque([(root, 0, -1)]) # vertex, level, rotate_bond_idx
        visit_order = []
        levels = []
        rotate_idx_lst = []
        rotable_bonds_order_lst = []
        while queue:
            vertex, level, rotate_bond_idx = queue.popleft()
            visit_order.append(vertex)
            levels.append(level)
            rotate_idx_lst.append(rotate_bond_idx)
            for node in graph.get(vertex, []):
                if node not in seen:
                    seen.add(node)

                    # [vetex, node] as edge
                    rotate_bond_idx = edge_rotate_bond_dict[str([vertex, node])]
                    rotate_bond = list(rotable_bonds[rotate_bond_idx])

                
                    #vetext--> node , check the rotate_bond reverse or not
                    if rotate_bond[0] not in rg_nodes[vertex] and rotate_bond[1] not in rg_nodes[vertex]: # reverse
                        rotate_bond.reverse()
                    
                    rotable_bonds_order_lst.append(rotate_bond)
                    queue.append((node, level + 1, rotate_bond_idx))
        return rotable_bonds_order_lst, levels[1:]
    
    root_idx = sorted(rg_graph_w_s.items(), key=lambda kv: (len(kv[1][0]), kv[1][1]))[0][0]
    return bfs(rg_graph, root_idx, edge_rotate_bond_dict, rotate_bonds, rg_nodes)


def add_equi_noise(opt_mol, bond_var=0.04, angle_var=0.04, torsion_var=2):
    # bond noise, find all bond add noise
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()
    bond_label_lst = [] # [i, j, delta_len]
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
            continue

        org_bond_len = GetBondLength(conf, [i_idx, j_idx])
        # add gaussian noise:
        noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_var)
        # set bond_length
        SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
        bond_label_lst.append([i_idx, j_idx, noise_bond_len - org_bond_len])

    # angle noise
    angle_label_lst = [] # [i, j, k, delta_angle]
    for atom in mol.GetAtoms(): # j
        j_idx = atom.GetIdx()
        # atom_symbol = atom.GetSymbol()
        atom_degree = atom.GetDegree()
        if atom_degree >= 2:
            # get neighbors
            neighbors = atom.GetNeighbors()
            neb_lst = []
            for neb in neighbors:
                neb_lst.append(neb.GetIdx())

            if mol.GetAtomWithIdx(j_idx).IsInRing(): # if j in ring, must pick one neb in ring as i
                for n_idx in neb_lst:
                    if mol.GetAtomWithIdx(n_idx).IsInRing():
                        i_idx = n_idx
                        break
            else:
                # j not in ring, random pick one as i
                i_idx = random.choice(neb_lst)
            
            neb_lst.remove(i_idx)
            # iterate k
            for k_idx in neb_lst:
                # judge (i, j) and (j, k) in ring:
                if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
                    continue
                # add angle noise to (i, j, k)
                org_angle = GetAngle(conf, [i_idx, j_idx, k_idx])
                if math.isnan(org_angle): # may be nan
                    continue
                # add noise
                noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                SetAngle(conf, [i_idx, j_idx, k_idx], noise_angle)
                angle_label_lst.append([i_idx, j_idx, k_idx, noise_angle - org_angle])
    
    # dihedral angle(rotatable or not) [i, j, k, l]
    # get the all the rotatable angel idx
    rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
    
    rotable_sets = set([])
    for rb in rotable_bonds:
        rotable_sets.add(f'{rb[1]}_{rb[2]}')
        rotable_sets.add(f'{rb[2]}_{rb[1]}')

    dihedral_label_lst = [] # [i, j, k, l, delta_angle]
    rotate_dihedral_label_lst = []
    for bond in mol.GetBonds():

        is_rotate = False

        j_idx = bond.GetBeginAtomIdx()
        k_idx = bond.GetEndAtomIdx()
        # check (j_idx, k_idx) in ring or not
        if mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
            continue
        
        j_atom = mol.GetAtomWithIdx(j_idx)
        j_atom_degree = j_atom.GetDegree()
        k_atom = mol.GetAtomWithIdx(k_idx)
        k_atom_degree = k_atom.GetDegree()

        if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
            continue

        # get neibors
        j_neighbors = j_atom.GetNeighbors()
        j_neb_lst = []
        for neb in j_neighbors:
            j_neb_lst.append(neb.GetIdx())
        j_neb_lst.remove(k_idx)

        k_neighbors = k_atom.GetNeighbors()
        k_neb_lst = []
        for neb in k_neighbors:
            k_neb_lst.append(neb.GetIdx())
        k_neb_lst.remove(j_idx)

        # random pick one neighbor from j and k, taken as i, l
        i_idx = random.choice(j_neb_lst)
        l_idx = random.choice(k_neb_lst)

        if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
            deh_var = torsion_var
            is_rotate = True
        else:
            deh_var = angle_var
            is_rotate = False
        
        # add noise
        org_deh_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
        if math.isnan(org_deh_angle): # may be nan
            continue
        noise_deh_angle = np.random.normal(loc=org_deh_angle, scale=deh_var)
        SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_deh_angle)

        if is_rotate:
            rotate_dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        else:
            dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
    
    return mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst


if __name__ == "__main__":
    from tqdm import tqdm
    mol = Chem.SDMolSupplier('org_2.sdf')[0]
    rotate_bonds = get_torsions([mol])
    print(rotate_bonds)
    print(get_rotate_order_info(mol, rotate_bonds))
    dm = np.load('dihmol.npy', allow_pickle=True)[0]
    noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst = add_equi_noise(dm)
    # MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
    # for mol in tqdm(MOL_LST):
    #     noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst = add_equi_noise(mol)
    #     dihedral_label = np.array(dihedral_label_lst)
    #     if np.isnan(dihedral_label).sum():
    #         import pdb; pdb.set_trace()
    #         print('nan happens!')
