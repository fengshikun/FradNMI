	
#small_mol_md.py
# import yaml
# import sys
# import os
# import time
# import matplotlib.pyplot as plt
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
# from openforcefield.utils.toolkits import RDKitToolkitWrapper
# from openforcefield.utils.toolkits import AmberToolsToolkitWrapper
# from simtk import openmm
# from simtk import unit
from rdkit import Chem
from torsion_utils import get_torsions, GetDihedral
import numpy as np
from openmm.unit import kilojoules, mole, nanometer
# from torchmdnet.datasets import MD17
import copy 
import pickle
import random
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdmolops import GetDistanceMatrix
from pyscf import gto, dft
from scipy.stats import pearsonr  
import yaml

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def GetBondLength(conf, atom_idx):
    return rdMolTransforms.GetBondLength(conf, atom_idx[0], atom_idx[1])

def SetBondLength(conf, atom_idx, new_vale):
    return rdMolTransforms.SetBondLength(conf, atom_idx[0], atom_idx[1], new_vale)

def GetAngle(conf, atom_idx):
    return rdMolTransforms.GetAngleDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2])

def SetAngle(conf, atom_idx, new_vale):
    return rdMolTransforms.SetAngleDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], new_vale)


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol

def apply_changes_bond_length(mol, values, bond_idx):
    opt_mol = copy.deepcopy(mol)
    [SetBondLength(opt_mol.GetConformer(), bond_idx[r], values[r]) for r in range(len(bond_idx))]
    return opt_mol

def apply_changes_angle(mol, values, bond_idx):
    opt_mol = copy.deepcopy(mol)
    [SetAngle(opt_mol.GetConformer(), bond_idx[r], values[r]) for r in range(len(bond_idx))]

    return opt_mol

def get_ortho_vector(bond_vec_dict,i_idx, j_idx, k_idx):
    #vec1=vector(j->i) vec2=vector(j->k)
    if i_idx < j_idx:
        vec1 = bond_vec_dict[str((i_idx, j_idx))] 
    else:
        vec1 = -bond_vec_dict[str((j_idx, i_idx))] 
    if k_idx < j_idx:
        vec2 = bond_vec_dict[str((k_idx, j_idx))] 
    else:
        vec2 = -bond_vec_dict[str((j_idx, k_idx))] 
    #orthogornalization
    vec = vec2-vec1*np.dot(vec1,vec2) 
    vec_i_tan = vec/np.linalg.norm(vec) # tangent vector at atom i
    vec = vec1-vec2*np.dot(vec1,vec2) 
    vec_k_tan = vec/np.linalg.norm(vec) # tangent vector at atom k
    #the output vectors point to the inner direction between i and k
    # if i_idx > j_idx:
    #     vec_k_tan = -vec_k_tan
    # if k_idx > j_idx:
    #     vec_i_tan = -vec_i_tan
    return vec_i_tan, vec_k_tan 

def get_normal_vector(bond_vec_dict,i_idx, j_idx, k_idx):
    #get normal vector of the plane i,j,k: vec(k,j) cross vec(j,i)
    #vec1=vector(j->i) vec2=vector(j->k)
    vec1 = bond_vec_dict[str((i_idx, j_idx))]     
    vec2 = bond_vec_dict[str((k_idx, j_idx))] 
    #normal vector
    vec = np.cross(vec1,vec2)
    return vec 

def get_vector(conf,i_idx, j_idx):
    #return vec(j->i)
    vector = conf.GetAtomPosition(i_idx)-conf.GetAtomPosition(j_idx) #vec(j->i), i_idx < j_idx
    vector.Normalize()
    return np.array([vector.x,vector.y,vector.z]) 

def get_side_atoms(DMat,Natoms,the_other_side_idx,one_side_idx): 
    atom_lst = []
    for idx in range(Natoms):
        if (idx == the_other_side_idx) or (idx == one_side_idx):
            continue
        if DMat[the_other_side_idx][idx] > DMat[one_side_idx][idx]:
            atom_lst.append(idx) 
    return atom_lst
def save_conf(mol,conf,fname='aspirin'): 
    dict_a={'coord':[],'atoms':[],'bond':[]}   
    fpath= '/home/fengshikun/Pretraining-Denoising/' 
    f_save = open(fpath+fname + '.pkl', 'wb') 
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        dict_a['coord'].append(np.array(conf.GetAtomPosition(idx)))
        dict_a['atoms'].append(atom_symbol)
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        bondtype = bond.GetBondTypeAsDouble()
        dict_a['bond'].append([i_idx,j_idx,bondtype])
    pickle.dump(dict_a, f_save)     
    f_save.close()  

# add bond, angle, torsion angle(rotateble or not) noise

def add_equi_noise_and_calculate_force(opt_mol, para, bond_var=0.1, angle_var=2, dih_var_rigid=2, torsion_var=20):
    # add noise, calculate directioinal vector of noisy conf, calculate force 
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()    
    DMat = GetDistanceMatrix(mol)
    Natoms = mol.GetNumAtoms()
    bond_label_lst = [] # [i, j, delta_len]
    atomic_force_dict_1 = {} #{'atom index': atomic force vector}
    atomic_force_dict_side = {}
    atomic_force_dict_side2 = {}
    atomic_force_dict_2 = {}
    atomic_force_dict_n3 = {}
    bond_vec_dict = {} #{'(BeginAtomIdx,EndAtomIdx)':unit direction vector of the bond}
    
# add noise
    # find all bond add noise
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        # print(i_idx,j_idx)    

        if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
            continue

        org_bond_len = GetBondLength(conf, [i_idx, j_idx])
        # optimal_bond_len = para[0]["Bonds"][(i_idx, j_idx)].length._value
        # print('org_bond_len', org_bond_len, 'optimal_bond_len', optimal_bond_len)

    # add gaussian noise:
        noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_var)
        if noise_bond_len<=0.01:
            noise_bond_len=0.01
    #toy rand
        # noise_bond_len = org_bond_len + np.random.choice([-1,1],p=[0.5,0.5])*0.3
        # print(noise_bond_len)
    
    # set bond_length
        SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
        bond_label_lst.append([i_idx, j_idx, noise_bond_len - org_bond_len])
    # print(bond_label_lst)


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
            # random pick one as i
            i_idx = random.choice(neb_lst)
            neb_lst.remove(i_idx)
            # iterate k
            for k_idx in neb_lst:
                # judge (i, j) and (j, k) in ring:
                # print(i_idx,j_idx,k_idx)
                if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
                    continue
                # get original angle (i, j, k)
                org_angle = GetAngle(conf, [i_idx, j_idx, k_idx])
                # optimal_angle = para[0]["Angles"][(i_idx, j_idx, k_idx)].angle._value
                # print('org_angle', org_angle, 'optimal_angle', optimal_angle)
            # add noise
                noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                         
            #set angle
                SetAngle(conf, [i_idx, j_idx, k_idx], noise_angle)#fix ij, move k
                angle_label_lst.append([i_idx, j_idx, k_idx, noise_angle - org_angle])
    # print(angle_label_lst)

    # add noise on dihedral angle(rotatable or not) [i, j, k, l]
    # get the all the rotatable angel idx
    rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]    
    rotable_sets = set([])
    for rb in rotable_bonds:
        rotable_sets.add(f'{rb[1]}_{rb[2]}')
        rotable_sets.add(f'{rb[2]}_{rb[1]}')
    dihedral_label_lst = [] # [i, j, k, l, delta_angle]
    for bond in mol.GetBonds():
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
            dih_var = torsion_var
        else:
            dih_var = dih_var_rigid
        # print((i_idx, j_idx, k_idx, l_idx))
    # get original dihedral angle (i, j, k, l)
        org_dih_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
        # if (i_idx, j_idx, k_idx, l_idx) in para[0]["ProperTorsions"]:        
        #     optimal_dih_angle = para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].phase[0]._value
        # else:
        #     optimal_dih_angle = para[0]["ProperTorsions"][(l_idx, k_idx, j_idx, i_idx)].phase[0]._value
        # print('org_dih_angle', org_dih_angle, 'optima_dih_angle', optimal_dih_angle)
        
    # add noise and determine sign
        noise_dih_angle = np.random.normal(loc=org_dih_angle, scale=dih_var)        
        
    #set dihedral angle
        if i_idx<l_idx:
            SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_dih_angle)
            dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_dih_angle - org_dih_angle,noise_dih_angle , org_dih_angle])
        else:
            noise_dih_angle = - noise_dih_angle
            SetDihedral(conf, [l_idx, k_idx, j_idx, i_idx],  noise_dih_angle)
            dihedral_label_lst.append([l_idx, k_idx, j_idx, i_idx, noise_dih_angle - org_dih_angle,noise_dih_angle , org_dih_angle])
    # save_conf(mol,conf,str(bond_var)+'_'+str(angle_var)+'_'+ str(dih_var_rigid)+'_'+ str(torsion_var)+'_aspirin')  #NOTE

    opt_conf = opt_mol.GetConformer()    #opt_conf is of equilibrium， conf is of the noisy one

    for idx in range(Natoms):
        atomic_force_dict_side[str(idx)]= np.array([0,0,0]) #initialize atomic force
        atomic_force_dict_1[str(idx)]= np.array([0,0,0])
        atomic_force_dict_2[str(idx)]= np.array([0,0,0])
        atomic_force_dict_n3[str(idx)]= np.array([0,0,0])
        atomic_force_dict_side2[str(idx)]= np.array([0,0,0])
#calculate directional vector for all bonds of noisy conf, including that in rings
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()        
        bond_vec_dict[str((i_idx, j_idx))] = get_vector(conf,i_idx, j_idx)
        bond_vec_dict[str((j_idx, i_idx))] = - bond_vec_dict[str((i_idx, j_idx))]
    # calculate bond force
    for i in range(len(bond_label_lst)):
        if bond_var==0 :
            break  
        i_idx = bond_label_lst[i][0]
        j_idx = bond_label_lst[i][1]
        # get j side atoms
        j_side_lst = get_side_atoms(DMat,Natoms,i_idx,j_idx)
        i_side_lst = get_side_atoms(DMat,Natoms,j_idx,i_idx)
        bond_para = para[0]["Bonds"][(i_idx, j_idx)].k._value
        bond_force = bond_para * (bond_label_lst[i][2])      
        vector = bond_vec_dict[str((i_idx, j_idx))] 
         
        # oneside
        for j_side in j_side_lst:                    
            atomic_force_dict_side[str(j_side)] = atomic_force_dict_side[str(j_side)] + vector * bond_force    #   The interaction forces are in opposite directions  
            atomic_force_dict_side2[str(j_side)] = atomic_force_dict_side2[str(j_side)] + vector * bond_force
        atomic_force_dict_2[str(i_idx)] = atomic_force_dict_2[str(i_idx)] - vector * bond_force    #   force on the two atoms point to each other if noise len > org len   
        atomic_force_dict_2[str(j_idx)] = atomic_force_dict_2[str(j_idx)] + vector * bond_force
        atomic_force_dict_1[str(j_idx)] = atomic_force_dict_1[str(j_idx)] + vector * bond_force 
        # two sides
        for i_side in i_side_lst:                    
            atomic_force_dict_side2[str(i_side)] = atomic_force_dict_side[str(i_side)] - vector * bond_force 
    # calculate angle force
    for i in range(len(angle_label_lst)):
        if angle_var==0 :
            break  
        i_idx = angle_label_lst[i][0]
        j_idx = angle_label_lst[i][1]
        k_idx = angle_label_lst[i][2] 
        # get k side atoms
        k_side_lst = get_side_atoms(DMat,Natoms,j_idx,k_idx)      
        angle_para = para[0]["Angles"][(i_idx, j_idx, k_idx)].k._value
        angle_force = angle_para * (angle_label_lst[i][3])
        vector1, vector2 = get_ortho_vector(bond_vec_dict, i_idx, j_idx, k_idx)
        vector=get_vector(conf,i_idx, k_idx)
        #TODO oneside
        for k_side in k_side_lst:                    
            atomic_force_dict_side[str(k_side)] = atomic_force_dict_side[str(k_side)] - vector2  * angle_force   #   The interaction forces are in opposite directions  
        atomic_force_dict_n3[str(i_idx)] = atomic_force_dict_n3[str(i_idx)] - vector  * angle_force   #   The interaction forces are in opposite directions  
        atomic_force_dict_n3[str(k_idx)] = atomic_force_dict_n3[str(k_idx)] + vector  * angle_force   #   The interaction forces are in opposite directions  
        atomic_force_dict_1[str(k_idx)] = atomic_force_dict_1[str(k_idx)] + vector  * angle_force

        atomic_force_dict_2[str(i_idx)] =  atomic_force_dict_2[str(i_idx)] - vector1 * angle_force 
        atomic_force_dict_2[str(k_idx)] =  atomic_force_dict_2[str(k_idx)] - vector2  * angle_force    # the force that points to the outter direction has a positive sign
    
    # calculate dihedral angle force 
    for i in range(len(dihedral_label_lst)):
        if dih_var_rigid==0 and torsion_var==0 :
            break       
        i_idx = dihedral_label_lst[i][0] 
        j_idx = dihedral_label_lst[i][1] 
        k_idx = dihedral_label_lst[i][2] 
        l_idx = dihedral_label_lst[i][3]  
        # get k side atoms
        k_side_lst = get_side_atoms(DMat,Natoms,j_idx,k_idx)    
        # get k neighbor atoms
        k_neb_lst = []
        k_atom = mol.GetAtomWithIdx(k_idx)
        k_neighbors = k_atom.GetNeighbors()
        for neb in k_neighbors:
            k_neb_lst.append(neb.GetIdx())
        k_neb_lst.remove(j_idx)
    # get sign
        vector_l = opt_conf.GetAtomPosition(l_idx)-conf.GetAtomPosition(l_idx) # point to the opt conf
        vector_l_n = np.cross(bond_vec_dict[str((j_idx, k_idx))],bond_vec_dict[str((l_idx, k_idx))] ) # vec(k->j)cross vec(k->l)
        vector_sign_l = int(np.dot(vector_l,vector_l_n)>0)*2-1        

    # calculate torsion angle force
        if (i_idx, j_idx, k_idx, l_idx) in para[0]["ProperTorsions"]:        
            dih_angle_para = para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._value
        else:
            dih_angle_para = para[0]["ProperTorsions"][(l_idx, k_idx, j_idx, i_idx)].k[0]._value       
        dih_angle_force = dih_angle_para * (dihedral_label_lst[i][4])
        #NOTE oneside
        for k_side in k_side_lst:
            vector = np.cross(bond_vec_dict[str((j_idx, k_idx))],get_vector(conf,k_side, k_idx) ) # vec(k->j)cross vec(k->k_side)
            atomic_force_dict_side[str(k_side)] =  atomic_force_dict_side[str(k_side)] + vector_sign_l * vector * dih_angle_force
        for k_neb in k_neb_lst:
            vector = np.cross(bond_vec_dict[str((j_idx, k_idx))], get_vector(conf,k_neb, k_idx) ) # vec(k->j)cross vec(k->k_side)
            atomic_force_dict_1[str(k_neb)] =  atomic_force_dict_1[str(k_neb)] + vector_sign_l * vector * dih_angle_force
        
        
        # vector = np.cross(bond_vec_dict[str((j_idx, k_idx))],get_vector(conf,l_idx, k_idx) ) # vec(k->j)cross vec(k->k_side)
        # atomic_force_dict[str(l_idx)] =  atomic_force_dict[str(l_idx)] + vector_sign_l * vector * dih_angle_force
    #    
    return mol, atomic_force_dict_1,atomic_force_dict_2,atomic_force_dict_n3,atomic_force_dict_side,atomic_force_dict_side2

def calculate_dft_force(mol):
    # get dft input: atom type and coordinate
    conf = mol.GetConformer()
    noisy_atom_coord=''
    for atom in mol.GetAtoms():  
        idx = atom.GetIdx()     
        pos = conf.GetAtomPosition(idx)    
        noisy_atom_coord = noisy_atom_coord+' '+atom.GetSymbol()+' '+str(pos.x)+' '+str(pos.y)+' '+str(pos.z)+';'
    #dft calculate force 
    mol_hf = gto.M(atom = noisy_atom_coord, basis = '6-31g', symmetry = True)
    mf_hf = dft.RKS(mol_hf)
    mf_hf.xc = 'b3lyp'
    energy = mf_hf.kernel()
    g_2 = mf_hf.nuc_grad_method() 
    force = g_2.kernel()
    return force

def calculate_mean_correlation(array_1,array_2):    
    res = pearsonr(array_1.flatten(), array_2.flatten())
    return res

def calculate_equivariant_distance(array_1,array_2):
    #center at zero
    ff_1= array_1 - np.mean(array_1,0)
    print(np.mean(array_1,0))
    ff_2= array_2 - np.mean(array_2,0)
    print(np.mean(array_2,0))
    #rotation invariant distance
    # return calculate_mean_correlation(np.dot(np.transpose(ff_1),ff_1), np.dot(np.transpose(ff_2),ff_2))
    return calculate_mean_correlation(ff_1.flatten(), ff_2.flatten())

def add_coord_noise_get_Frad_force(opt_mol, coord_var=0.04, dihedral_var=2):
    # add coordinate noise, calculate force corresponding to Frad
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()    
    Natoms = mol.GetNumAtoms()
    #add dihedral noise
    rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]    
    rotable_sets = set([])
    for rb in rotable_bonds:
        rotable_sets.add(f'{rb[1]}_{rb[2]}')
        rotable_sets.add(f'{rb[2]}_{rb[1]}')
    dihedral_label_lst = [] # [i, j, k, l, delta_angle]
    for bond in mol.GetBonds():
        j_idx = bond.GetBeginAtomIdx()
        k_idx = bond.GetEndAtomIdx()
        if f'{j_idx}_{k_idx}' not in rotable_sets: # rotatable
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
    # get original dihedral angle (i, j, k, l)
        org_dih_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
    # add noise and determine sign
        noise_dih_angle = np.random.normal(loc=org_dih_angle, scale=dihedral_var)
    #set dihedral angle
        if i_idx<l_idx:
            SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_dih_angle)
            dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_dih_angle - org_dih_angle,noise_dih_angle , org_dih_angle])
        else:
            noise_dih_angle = - noise_dih_angle
            SetDihedral(conf, [l_idx, k_idx, j_idx, i_idx],  noise_dih_angle)
            dihedral_label_lst.append([l_idx, k_idx, j_idx, i_idx, noise_dih_angle - org_dih_angle,noise_dih_angle , org_dih_angle])    
    
    # add coord noise
    atomic_force_dict = {}
    for idx in range(Natoms):
        atomic_force_dict[str(idx)]= np.array([0,0,0]) #initialize atomic force
    for idx in range(0,Natoms,2):#TAG 
        pos=conf.GetAtomPosition(idx)
        pos_new = np.random.normal(loc=pos, scale=coord_var) 
        atomic_force_dict[str(idx)] = pos - pos_new 
        conf.SetAtomPosition(idx,pos_new)
    # save_conf(mol,conf,str(coord_var)+ str(dihedral_var)+'_aspirin')  #NOTE
    return mol, atomic_force_dict
    

def experiment(rd_mol, ff_applied_parameters,bond_var, angle_var, dih_var_rigid, torsion_var):
    rho=[]
    equiv_rho=[]
    equiv_pvalue=[]
    pvalue=[]
    for t in range(1):
        # new_mol, atomic_force_dict = add_equi_noise_and_calculate_force(rd_mol, ff_applied_parameters, config['bond_var'], config['angle_var'], config['dih_var_rigid'], config['torsion_var'])#1:20:200
        new_mol, atomic_force_dict_1,atomic_force_dict_2,atomic_force_dict_n3,atomic_force_dict_side,atomic_force_dict_side2 = add_equi_noise_and_calculate_force(rd_mol, ff_applied_parameters, bond_var, angle_var, dih_var_rigid, torsion_var)#1:20:200
        # print(*atomic_force_lst, sep="\n") 
        dft_force = calculate_dft_force(new_mol)
        atomic_force_dict = atomic_force_dict_1       #NOTE CHANGE
        atomic_force_lst = list(atomic_force_dict.values())   
        rho.append(calculate_mean_correlation(dft_force,np.asarray(atomic_force_lst))[0]) 
        pvalue.append(calculate_mean_correlation(dft_force,np.asarray(atomic_force_lst))[1])
        equiv_rho.append(calculate_equivariant_distance(dft_force,np.asarray(atomic_force_lst))[0])
        equiv_pvalue.append(calculate_equivariant_distance(dft_force,np.asarray(atomic_force_lst))[1])
    print('rho:',np.mean(rho),'pvalue:',np.mean(pvalue))
    print('equiv_rho:',np.mean(equiv_rho),'pvalue:',np.mean(equiv_pvalue))
    print('--------------')
    print('rho:',rho,'pvalue:',pvalue)
    print('equiv_rho:',equiv_rho,'pvalue:',equiv_pvalue)
    return None

def experiment_coord(rd_mol, coord_var=0.04, dihedral_var= 2):
    rho=[]
    equiv_rho=[]
    equiv_pvalue=[]
    pvalue=[]
    for t in range(1):
        new_mol, atomic_force_dict = add_coord_noise_get_Frad_force(rd_mol, coord_var, dihedral_var)
        atomic_force_lst = list(atomic_force_dict.values())   
        print(*atomic_force_lst, sep="\n") 
        dft_force = calculate_dft_force(new_mol)
        rho.append(calculate_mean_correlation(dft_force,np.asarray(atomic_force_lst))[0]) 
        pvalue.append(calculate_mean_correlation(dft_force,np.asarray(atomic_force_lst))[1])
        equiv_rho.append(calculate_equivariant_distance(dft_force,np.asarray(atomic_force_lst))[0])
        equiv_pvalue.append(calculate_equivariant_distance(dft_force,np.asarray(atomic_force_lst))[1])
    print('rho:',np.mean(rho),'pvalue:',np.mean(pvalue))
    print('equiv_rho:',np.mean(equiv_rho),'pvalue:',np.mean(equiv_pvalue))
    print('--------------')
    print('rho:',rho,'pvalue:',pvalue)
    print('equiv_rho:',equiv_rho,'pvalue:',equiv_pvalue)
    return None

if __name__=="__main__":
    forcefield = ForceField("openff-1.0.0.offxml")
    # mol=Chem.AddHs(Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O'))
    # aspirin = Molecule.from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    # topology = Topology.from_molecules(molecules=[aspirin])
    # system = forcefield.create_openmm_system(topology)
    # ff_applied_parameters = forcefield.label_molecules(topology)
    # config = yaml.load(open("nyymd.yml", "r"), yaml.Loader)
    # print(config)
    npy_file = '/share/project/sharefs-skfeng/xyz2mol/aspirin_4w.npy'
    asp_4w = np.load(npy_file,allow_pickle=True)
    rd_mol = asp_4w[0]
    print(rd_mol)
    equi_mol = Molecule.from_rdkit(rd_mol)
    topology = Topology.from_molecules(molecules=[equi_mol])
    system = forcefield.create_openmm_system(topology)
    ff_applied_parameters = forcefield.label_molecules(topology)

    # experiment_coord(rd_mol, coord_var=0.04, dihedral_var= 2)
    # experiment(rd_mol, ff_applied_parameters,bond_var=0.04, angle_var=0, dih_var_rigid=0, torsion_var=0) #1:0.04~ 100:2
    # experiment(rd_mol, ff_applied_parameters,bond_var=0, angle_var=2, dih_var_rigid=0, torsion_var=0)
    experiment(rd_mol, ff_applied_parameters,bond_var=0, angle_var=0, dih_var_rigid=2, torsion_var=20)
    # experiment(rd_mol, ff_applied_parameters,bond_var=0.04, angle_var=0.04, dih_var_rigid=0.04, torsion_var=2)
    # experiment(rd_mol, ff_applied_parameters,bond_var=0.1, angle_var=2, dih_var_rigid=0, torsion_var=0)
    # experiment(rd_mol, ff_applied_parameters,bond_var=0.04, angle_var=0.8, dih_var_rigid=0.8, torsion_var=2)


