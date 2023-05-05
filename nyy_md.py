import numpy as np
import sys
sys.path.insert(0, "/home/fengshikun/Pretraining-Denoising/mindsponge/")
from mindspore import context
from mindsponge import Sponge
from mindsponge import Molecule
from mindsponge import ForceFieldBase
from mindsponge import UpdaterMD

from mindsponge.potential import BondEnergy, AngleEnergy
from mindsponge.callback import WriteH5MD, RunInfo
from mindsponge.function import VelocityGenerator
from mindsponge.control import LeapFrog, BerendsenThermostat
from mindsponge.partition import NeighbourList

import Xponge
import Xponge.forcefield.amber.gaff
from mindsponge import ForceFieldBase
from mindsponge import RunOneStepCell
from mindsponge import WithEnergyCell
from mindspore import Tensor
mindspore-gpu  
import pickle

# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

with open('/home/fengshikun/Pretraining-Denoising/0.042_aspirin.pkl', 'rb') as f:
    dict_a = pickle.load(f)


assign = Xponge.Assign()
for i in range(len(dict_a['atoms'])):
    assign.Add_Atom(element = dict_a['atoms'][i], x = dict_a['coord'][1][0], y = dict_a['coord'][1][1], z = dict_a['coord'][1][2])
for i in range(len(dict_a['bond'])):
    assign.Add_Bond(dict_a['bond'][i][0],dict_a['bond'][i][1],dict_a['bond'][i][2])
assign.determine_atom_type("gaff")
mol = assign.to_residuetype("mol")
molecule, energy = Xponge.get_mindsponge_system_energy(mol, use_pbc=False)
# molecule.pbc_box = None
molecule.pbc_box = Tensor([[99.9,99.9,99.9]])
potential = ForceFieldBase([energy.energies[i] for i in range(2,5)]) #

vgen = VelocityGenerator(300)
velocity = vgen(molecule.coordinate.shape, molecule.atom_mass)
opt = UpdaterMD(molecule,
                integrator=LeapFrog(molecule),
                thermostat=BerendsenThermostat(molecule, 300),
                time_step=1e-3
                )

sim = WithEnergyCell(molecule,potential)
run_one_step = RunOneStepCell(energy=sim, optimizer=opt,neighbour_list=NeighbourList(molecule, num_neighbours=20))
run_one_step()

