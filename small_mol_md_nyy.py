	
#small_mol_md.py
import yaml
import sys
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
    # system = forcefield.create_openmm_system(off_topology)
 
    # time_step = config["time_step"] * unit.femtoseconds # more fine-grained
    # temperature = config["temperature"] * unit.kelvin
    # friction = 1 / unit.picosecond
    ff_applied_parameters = forcefield.label_molecules(off_topology)
    dict(ff_applied_parameters[0]["Bonds"])




    # TODO fix params
    integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
     
    conf = molecule.conformers[confId]
    simulation = openmm.app.Simulation(omm_topology,
                                       system,
                                       integrator)
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

    simulation.reporters.append(ForceReporter(f'forces_{save_prefix}.txt', 10))
    start = time.process_time()
    simulation.step(config["num_steps"])
    end = time.process_time()
    print(f"Elapsed time {end-start:.2f} sec")
    print("Done")
 
if __name__=="__main__":
    forcefield = ForceField("openff-1.0.0.offxml")

    # forcefield.get_parameter_handler('Electrostatics').method = 'PME'
    # forcefield.get_parameter_handler('Bonds').method = 'PM'

    config = yaml.load(open("mdconf.yml", "r"), yaml.Loader)
    # molecule = Molecule.from_smiles(sys.argv[1])
    
    MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
    
    log_num = 10


    for i in range(log_num):
        rd_mol = MOL_LST[i]
        molecule = Molecule.from_rdkit(rd_mol)
        # molecule.generate_conformers()
        run_md(molecule, save_prefix=i)