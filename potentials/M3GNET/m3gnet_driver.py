"""
Copyright (c) 2022, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

from ase import Atoms
from ase.calculators.mixing import SumCalculator

from m3gnet.models import M3GNet, M3GNetCalculator, Potential

def m3gnet_initialize(model_name = None, dftd3 = False):
    """
    Initialize GNNP of M3GNet.
    Args:
        model_name (str): name of model for GNNP.
        dftd3 (bool): to add correction of DFT-D3.
    Returns:
        cutoff: cutoff radius.
    """

    # Create M3GNetCalculator, that is pre-trained
    global myCalculator

    if model_name is not None:
        myM3GNet = M3GNet.load(model_name)
    else:
        myM3GNet = M3GNet.load()

    myPotential  = Potential(myM3GNet)

    myCalculator = M3GNetCalculator(
        potential      = myPotential,
        compute_stress = True,
        stress_weight  = 1.0
    )

    # Add DFT-D3 to calculator without three-body term
    global m3gnetCalculator
    global dftd3Calculator

    m3gnetCalculator = myCalculator
    dftd3Calculator  = None

    if dftd3:
        from dftd3.ase import DFTD3
        #from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

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

        myCalculator = SumCalculator([m3gnetCalculator, dftd3Calculator])

    # Atoms object of ASE, that is empty here
    global myAtoms

    myAtoms = None

    return myM3GNet.get_config().get("cutoff", 5.0)

def m3gnet_get_energy_forces_stress(cell, atomic_numbers, positions):
    """
    Predict total energy, atomic forces and stress w/ pre-trained GNNP of M3GNet.
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

    if myAtoms is not None and len(myAtoms.numbers) != len(atomic_numbers):
        myAtoms = None

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

    global m3gnetCalculator
    global dftd3Calculator

    if dftd3Calculator is None:
        stress = myAtoms.get_stress().tolist()
    else:
        # to avoid the bug of SumCalculator
        myAtoms.calc = m3gnetCalculator
        stress1 = myAtoms.get_stress()

        myAtoms.calc = dftd3Calculator
        stress2 = myAtoms.get_stress()

        stress = stress1 + stress2
        stress = stress.tolist()

        myAtoms.calc = myCalculator

    return energy, forces, stress

