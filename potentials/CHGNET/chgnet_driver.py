"""
Copyright (c) 2023, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

from ase import Atoms
from ase.calculators.mixing import SumCalculator

from dftd3.ase import DFTD3
#from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from chgnet.model import CHGNet, CHGNetCalculator

import torch

def chgnet_initialize(model_name = None, as_path = False, dftd3 = False, gpu = True):
    """
    Initialize GNNP of CHGNet.
    Args:
        model_name (str): name of model for GNNP.
        as_path (bool): if true, model_name is path of model file.
        dftd3 (bool): to add correction of DFT-D3.
        gpu (bool): using GPU, if possible.
    Returns:
        cutoff: cutoff radius.
    """

    # Check gpu
    gpu_ = (gpu and torch.cuda.is_available())

    # Create CHGNetCalculator, that is pre-trained
    global myCalculator

    if model_name is None:
        myCHGNet = CHGNet.load()
    elif not as_path:
        myCHGNet = CHGNet.load(model_name)
    else:
        myCHGNet = CHGNet.from_file(model_name)

    myCalculator = CHGNetCalculator(
        model      = myCHGNet,
        use_device = ("cuda" if gpu_ else "cpu")
    )

    # Add DFT-D3 to calculator without three-body term
    global chgnetCalculator
    global dftd3Calculator

    chgnetCalculator = myCalculator
    dftd3Calculator  = None

    if dftd3:
        dftd3Calculator = DFTD3(
            method  = "PBE",
            damping = "d3zero",
            s9      = 0.0
        )
        #dftd3Calculator = TorchDFTD3Calculator(
        #    xc      = "pbe",
        #    damping = "zero",
        #    abc     = False
        #)

        myCalculator = SumCalculator([chgnetCalculator, dftd3Calculator])

    # Atoms object of ASE, that is empty here
    global myAtoms

    myAtoms = None

    ratom = float(myCHGNet.graph_converter.atom_graph_cutoff)
    rbond = float(myCHGNet.graph_converter.bond_graph_cutoff)
    return max(ratom, rbond)

def chgnet_get_energy_forces_stress(cell, atomic_numbers, positions):
    """
    Predict total energy, atomic forces and stress w/ pre-trained GNNP of CHGNet.
    Args:
        cell: lattice vectors in angstroms.
        atomic_numbers: atomic numbers for all atoms.
        positions: xyz coordinates for all atoms in angstroms.
    Returns:
        energy:  total energy.
        forcces: atomic forces.
        stress:  stress tensor (Voigt order).
    """

    # Initialize Atoms
    global myAtoms
    global myCalculator

    if myAtoms is None:
        myAtoms = Atoms(
            numbers   = atomic_numbers,
            positions = positions,
            cell      = cell,
            pbc       = [True, True, True]
        )

        myAtoms.calc = myCalculator

    else:
        myAtoms.set_cell(cell)
        myAtoms.set_atomic_numbers(atomic_numbers)
        myAtoms.set_positions(positions)

    # Predicting energy, forces and stress
    energy = myAtoms.get_potential_energy().item()
    forces = myAtoms.get_forces().tolist()

    global chgnetCalculator
    global dftd3Calculator

    if dftd3Calculator is None:
        stress = myAtoms.get_stress().tolist()
    else:
        # to avoid the bug of SumCalculator
        myAtoms.calc = chgnetCalculator
        stress1 = myAtoms.get_stress()

        myAtoms.calc = dftd3Calculator
        stress2 = myAtoms.get_stress()

        stress = stress1 + stress2
        stress = stress.tolist()

        myAtoms.calc = myCalculator

    return energy, forces, stress

