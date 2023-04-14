	
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
 
def run_md(molecule, confId=0):
    off_topology = molecule.to_topology()
    omm_topology = off_topology.to_openmm()
    system = forcefield.create_openmm_system(off_topology)
 
    time_step = config["time_step"] * unit.femtoseconds
    temperature = config["temperature"] * unit.kelvin
    friction = 1 / unit.picosecond
    integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
     
    conf = molecule.conformers[confId]
    simulation = openmm.app.Simulation(omm_topology,
                                       system,
                                       integrator)
    simulation.context.setPositions(conf)
    if not os.path.isdir('./log'):
        os.mkdir('./log')
    pdb_reporter = openmm.app.PDBReporter('./log/trj.pdb', config["trj_freq"])
    state_data_reporter = openmm.app.StateDataReporter("./log/data.csv",
                                                       config["data_freq"],
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True,
                                                       density=True)
    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(state_data_reporter)
    start = time.process_time()
    simulation.step(config["num_steps"])
    end = time.process_time()
    print(f"Elapsed time {end-start:.2f} sec")
    print("Done")
 
if __name__=="__main__":
    forcefield = ForceField("openff-1.0.0.offxml")
    config = yaml.load(open("mdconf.yml", "r"), yaml.Loader)
    molecule = Molecule.from_smiles(sys.argv[1])
    molecule.generate_conformers()
    run_md(molecule)