	
#small_mol_md.py
# python small_mol_md.py --add-noise
# python small_mol_md.py --rdkit-gen true --rdkit-opt true
# python small_mol_md.py --rdkit-gen true --rdkit-opt false
# python small_mol_md.py
import yaml
import sys
import copy
import os
import time
import matplotlib.pyplot as plt
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils.toolkits import RDKitToolkitWrapper
from openforcefield.utils.toolkits import AmberToolsToolkitWrapper
from simtk import openmm
from simtk import unit
from rdkit import Chem
import numpy as np
from openmm.unit import kilojoules, mole, nanometer
import argparse
from rdkit.Chem import AllChem
from torsion_utils import add_equi_noise
import os
os.environ['CUDA_PATH'] = '/usr/local/cuda'

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
 
# todo rdkit generated conformations.

def run_md(molecule, confId=0, save_prefix=0):
    off_topology = molecule.to_topology()
    omm_topology = off_topology.to_openmm()
    system = forcefield.create_openmm_system(off_topology)
 
    time_step = config["time_step"] * unit.femtoseconds # more fine-grained
    temperature = config["temperature"] * unit.kelvin
    friction = 1 / unit.picosecond

    # platform = openmm.Platform.getPlatformByName('CUDA')

    # TODO fix params
    integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
     
    conf = molecule.conformers[confId]
    simulation = openmm.app.Simulation(omm_topology,
                                       system,
                                       integrator) # platform
    simulation.context.setPositions(conf)
    if not os.path.isdir('./log'):
        os.mkdir('./log')
    pdb_reporter = openmm.app.PDBReporter(f'./log/trj_{save_prefix}.pdb', config["trj_freq"])
    state_data_reporter = openmm.app.StateDataReporter(f"./log/data_{save_prefix}.csv",
                                                       config["data_freq"],
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True,
                                                       density=True)
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(state_data_reporter)

    simulation.reporters.append(ForceReporter(f'./log/forces_{save_prefix}.txt', 10))
    start = time.process_time()
    simulation.step(config["num_steps"])
    end = time.process_time()
    print(f"Elapsed time {end-start:.2f} sec")
    print("Done")




def run_md_spice(molecule, confId=0, save_prefix=0):
    off_topology = molecule.to_topology()
    omm_topology = off_topology.to_openmm()
    system = forcefield.create_openmm_system(off_topology)
 
    time_step = config["time_step"] * unit.femtoseconds # more fine-grained
    temperature = config["temperature"] * unit.kelvin
    friction = 1 / unit.picosecond

    # platform = openmm.Platform.getPlatformByName('CUDA')

    # TODO fix params
    # integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
    integrator = openmm.LangevinMiddleIntegrator(100*openmm.unit.kelvin, 1/openmm.unit.picosecond, 0.001*openmm.unit.picosecond) # TODO or 500 *openmm.unit.kelvin

    conf = molecule.conformers[confId]
    simulation = openmm.app.Simulation(omm_topology,
                                       system,
                                       integrator) # platform
    
    simulation.context.setPositions(conf)
    simulation.context.setVelocitiesToTemperature(100*openmm.unit.kelvin)
    if not os.path.isdir('./log_s'):
        os.mkdir('./log_s')
    pdb_reporter = openmm.app.PDBReporter(f'./log_s/trj_{save_prefix}.pdb', config["trj_freq"])
    state_data_reporter = openmm.app.StateDataReporter(f"./log_s/data_{save_prefix}.csv",
                                                       config["data_freq"],
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True,
                                                       density=True)
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(state_data_reporter)

    simulation.reporters.append(ForceReporter(f'./log/forces_{save_prefix}.txt', 10))
    start = time.process_time()
    simulation.step(config["num_steps"])
    end = time.process_time()
    print(f"Elapsed time {end-start:.2f} sec")
    print("Done")

def rdkit_gen_mol(mol, rdkit_opt=False):
    test_mol = copy.copy(mol)
    test_mol.RemoveConformer(0)
    cids = AllChem.EmbedMultipleConfs(test_mol, numConfs=1, numThreads=8, pruneRmsThresh=0.1, maxAttempts=5, useRandomCoords=False)
    if len(cids) < 1:
        print('Rdkit generate fail')
        return mol        
    else: # success
        print('Rdkit generate success')
        if rdkit_opt:
            print('Optimising...')
            AllChem.MMFFOptimizeMoleculeConfs(test_mol, numThreads=8)
        return test_mol





#TODO Consider calling AddHs()
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MOLECULE MD')
    parser.add_argument('--rdkit-gen', default=False, type=bool, help='if true, use rdkit generate conformation')
    parser.add_argument('--rdkit-opt', default=False, type=bool, help='if true, use rdkit force field optimization, only valid when')
    parser.add_argument('--add-noise', default=False, type=bool, help='if true, add noise')
    parser.add_argument('--spice', default=False, type=bool, help='change md env')
    args = parser.parse_args()
    rdkit_gen = args.rdkit_gen
    rdkit_opt = args.rdkit_opt

    add_noise = args.add_noise
    spice = args.spice


    if spice:
        forcefield = ForceField('openff_unconstrained-2.0.0-rc.2.offxml')
    else:
        forcefield = ForceField("openff-1.0.0.offxml")
    config = yaml.load(open("mdconf.yml", "r"), yaml.Loader)
    # molecule = Molecule.from_smiles(sys.argv[1])
    
    


    MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
    
    log_num = 10


    for i in range(log_num):
        rd_mol = MOL_LST[i]
        if rdkit_gen:
            rd_mol = rdkit_gen_mol(rd_mol, rdkit_opt=rdkit_opt)
        
        if add_noise:
            rd_mol, bond_label_lst, angle_label_lst, dihedral_label_lst = add_equi_noise(rd_mol, bond_var=0.04, angle_var=0.04, torsion_var=20)
        molecule = Molecule.from_rdkit(rd_mol)
        # molecule.generate_conformers()
        if rdkit_gen:
            save_prefix = f"rdkit_opt_{rdkit_opt}_{i}"
        else:
            save_prefix = f"{i}"
        if add_noise:
            save_prefix += f'_add_noise'
        if spice:
            run_md_spice(molecule, save_prefix=save_prefix)
        else:
            run_md(molecule, save_prefix=save_prefix)