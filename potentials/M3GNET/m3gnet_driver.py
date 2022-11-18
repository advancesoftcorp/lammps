"""
Copyright (c) 2022, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

import os
import pprint
import torch
import yaml

from ase import Atoms

from m3gnet.models import M3GNet, M3GNetCalculator, Potential

def m3gnet_initialize(model_name = None):
    """
    Initialize GNNP of M3GNet.
    Args:
        model_name (str): name of model for GNNP.
    Returns:
        cutoff: cutoff radius.
    """

    # Create M3GNetCalculator, that is pre-trained
    global myCalculator

    if model_name is not None:
        myM3GNet = M3GNet.load(model_name)
    elif
        myM3GNet = M3GNet.load()

    myPotential  = Potential(myM3GNet)

    myCalculator = M3GNetCalculator(
        potential      = myPotential,
        compute_stress = True,
        stress_weight  = 1.0
    )

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
    energy = myAtoms.get_potential_energy()
    forces = myAtoms.get_forces()
    stress = myAtoms.get_stress()

    return energy, forces, stress

