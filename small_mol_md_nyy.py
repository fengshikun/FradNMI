	
#small_mol_md.py
# import yaml
# import sys
# import os
# import time
import matplotlib.pyplot as plt
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
import random
from rdkit.Chem import rdMolTransforms
from pyscf import gto, dft

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
    if i_idx < j_idx:
        vec1 = bond_vec_dict[str((i_idx, j_idx))] 
    else:
        vec1 = -bond_vec_dict[str((j_idx, i_idx))] 
    if k_idx < j_idx:
        vec2 = bond_vec_dict[str((k_idx, j_idx))] 
    else:
        vec2 = -bond_vec_dict[str((j_idx, k_idx))] 
    #normal vector
    vec = -np.cross(vec1,vec2)
    return vec 

class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(kilojoules/mole/nanometer) # /mole/nanometer
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))


# add bond, angle, torsion angle(rotateble or not) noise



def add_equi_noise_and_calculate_force(opt_mol, para, bond_var=0.04, angle_var=0.04, torsion_var=2):
    # add noise, calculate directioinal vector of noisy conf, calculate force 
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()    
    bond_label_lst = [] # [i, j, delta_len]
    atomic_force_dict = {} #{'atom index': atomic force vector}
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
            # dih_var = angle_var
            dih_var = 0 #TODO
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
        SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_dih_angle)
        dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_dih_angle - org_dih_angle])
    # print(dihedral_label_lst)    
    opt_conf = opt_mol.GetConformer()    #opt_conf is of equilibrium， conf is of the noisy one

    for idx in range(mol.GetNumAtoms()):
        atomic_force_dict[str(idx)]= np.array([0,0,0]) #initialize atomic force
#calculate directional vector for all bonds of noisy conf, including that in rings
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        vector = conf.GetAtomPosition(i_idx)-conf.GetAtomPosition(j_idx) #vec(j->i), i_idx < j_idx
        vector.Normalize()
        bond_vec_dict[str((i_idx, j_idx))] = np.array([vector.x,vector.y,vector.z])    
    # calculate bond force
    for i in range(len(bond_label_lst)):
        i_idx = bond_label_lst[i][0]
        j_idx = bond_label_lst[i][1]
        bond_para = para[0]["Bonds"][(i_idx, j_idx)].k._value
        bond_force = bond_para * (bond_label_lst[i][2])      
        vector = bond_vec_dict[str((i_idx, j_idx))]  
        atomic_force_dict[str(i_idx)] = atomic_force_dict[str(i_idx)] - vector * bond_force    #   force on the two atoms point to each other if noise len > org len   
        atomic_force_dict[str(j_idx)] = atomic_force_dict[str(j_idx)] + vector * bond_force    #   The interaction forces are in opposite directions  
    
    # calculate angle force
    for i in range(len(angle_label_lst)):
        i_idx = angle_label_lst[i][0]
        j_idx = angle_label_lst[i][1]
        k_idx = angle_label_lst[i][2]        
        angle_para = para[0]["Angles"][(i_idx, j_idx, k_idx)].k._value
        angle_force = angle_para * (angle_label_lst[i][3])
        vector1, vector2 = get_ortho_vector(bond_vec_dict, i_idx, j_idx, k_idx)
        # atomic_force_dict[str(i_idx)] =  atomic_force_dict[str(i_idx)] - vector1 * angle_force 
        atomic_force_dict[str(k_idx)] =  atomic_force_dict[str(k_idx)] - vector2  * angle_force    # the force that points to the outter direction has a positive sign
    
    # calculate dihedral angle force 
    for i in range(len(dihedral_label_lst)):
        i_idx = dihedral_label_lst[i][0] 
        j_idx = dihedral_label_lst[i][1] 
        k_idx = dihedral_label_lst[i][2] 
        l_idx = dihedral_label_lst[i][3]  
        # get neibors
        j_atom = mol.GetAtomWithIdx(j_idx)
        j_neighbors = j_atom.GetNeighbors()
        j_neb_lst = []
        for neb in j_neighbors:
            j_neb_lst.append(neb.GetIdx())
        j_neb_lst.remove(k_idx)
        k_atom = mol.GetAtomWithIdx(k_idx)
        k_neighbors = k_atom.GetNeighbors()
        k_neb_lst = []
        for neb in k_neighbors:
            k_neb_lst.append(neb.GetIdx())
        k_neb_lst.remove(j_idx)         
    
        vector_i = opt_conf.GetAtomPosition(i_idx)-conf.GetAtomPosition(i_idx) # force point to the opt conf
        vector_l = opt_conf.GetAtomPosition(l_idx)-conf.GetAtomPosition(l_idx)
        vector_i_n = get_normal_vector(bond_vec_dict, i_idx, j_idx, k_idx) #normal vector of the plane i,j,k:vec(k,j)cross vec(j,i)
        vector_l_n = get_normal_vector(bond_vec_dict, j_idx, k_idx, l_idx)
        print(np.dot(vector_i,vector_i_n), np.dot(vector_l,vector_l_n))
        vector_sign_i = int(np.dot(vector_i,vector_i_n)>0)*2-1
        vector_sign_l = int(np.dot(vector_l,vector_l_n)>0)*2-1        

    # calculate torsion angle force
        if (i_idx, j_idx, k_idx, l_idx) in para[0]["ProperTorsions"]:        
            dih_angle_para = para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._value
        else:
            dih_angle_para = para[0]["ProperTorsions"][(l_idx, k_idx, j_idx, i_idx)].k[0]._value       
        dih_angle_force = dih_angle_para * (dihedral_label_lst[i][4])
        for j_neb in j_neb_lst:
            vector = get_normal_vector(bond_vec_dict, j_neb, j_idx, k_idx) #normal vector of the plane i,j,k:vec(k,j)cross vec(j,i)
            atomic_force_dict[str(j_neb)] =  atomic_force_dict[str(j_neb)] + vector_sign_i * vector * dih_angle_force # the force that Rotate the bond(j,k) counterclockwise (look from j to k) has a positive sign
        for k_neb in k_neb_lst:
            vector = get_normal_vector(bond_vec_dict, j_idx, k_idx, k_neb)#normal vector of the plane j,k,l:vec(l,k)cross vec(k,j)
            atomic_force_dict[str(k_neb)] =  atomic_force_dict[str(k_neb)] + vector_sign_l * vector * dih_angle_force 
    return mol, atomic_force_dict

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

def calculate_mean_correlation(force_array_1,force_lst_2):
    from scipy.stats import pearsonr  
    res = pearsonr(force_array_1.flatten(), force_lst_2.flatten())
    return res

if __name__=="__main__":
    forcefield = ForceField("openff-1.0.0.offxml")
    # mol=Chem.AddHs(Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O'))
    # aspirin = Molecule.from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    # topology = Topology.from_molecules(molecules=[aspirin])
    # system = forcefield.create_openmm_system(topology)
    # ff_applied_parameters = forcefield.label_molecules(topology)
    
    npy_file = '/share/project/sharefs-skfeng/xyz2mol/aspirin_4w.npy'
    asp_4w = np.load(npy_file,allow_pickle=True)
    rd_mol = asp_4w[0]
    equi_mol = Molecule.from_rdkit(rd_mol)
    topology = Topology.from_molecules(molecules=[equi_mol])
    system = forcefield.create_openmm_system(topology)
    ff_applied_parameters = forcefield.label_molecules(topology)
    new_mol, atomic_force_dict = add_equi_noise_and_calculate_force(rd_mol, ff_applied_parameters, bond_var=0, angle_var=2, torsion_var=0)#1:20:200
    atomic_force_lst = list(atomic_force_dict.values())   
    print(*atomic_force_lst, sep="\n") 
    dft_force = calculate_dft_force(new_mol)
    rho = calculate_mean_correlation(dft_force,np.asarray(atomic_force_lst))
    print(rho)