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

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    setup_imports,
    setup_logging,
)

from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs

def oc20_initialize(model_name, gpu = True):
    """
    Initialize GNNP of OC20 (i.e. S2EF).
    Args:
        model_name (str): name of model for GNNP. One can use the followings,
            - "DimeNet++"
            - "GemNet-dT"
            - "CGCNN"
            - "SchNet"
            - "SpinConv"
        gpu (bool): using GPU, if possible.
    Returns:
        cutoff: cutoff radius.
    """

    setup_imports()
    setup_logging()

    # Check model_name
    log_file = open("log.oc20", "w")
    log_file.write("\n");
    log_file.write("model_name = " + model_name + "\n");

    if model_name is not None:
        model_name = model_name.lower()

    if model_name == "DimeNet++".lower():
        config_yml = "oc20_configs/dimenetpp.yml"
        checkpoint = "oc20_checkpt/dimenetpp_all.pt"

    elif model_name == "GemNet-dT".lower():
        config_yml = "oc20_configs/gemnet.yml"
        checkpoint = "oc20_checkpt/gemnet_t_direct_h512_all.pt"

    elif model_name == "CGCNN".lower():
        config_yml = "oc20_configs/cgcnn.yml"
        checkpoint = "oc20_checkpt/cgcnn_all.pt"

    elif model_name == "SchNet".lower():
        config_yml = "oc20_configs/schnet.yml"
        checkpoint = "oc20_checkpt/schnet_all_large.pt"

    elif model_name == "SpinConv".lower():
        config_yml = "oc20_configs/spinconv.yml"
        checkpoint = "oc20_checkpt/spinconv_force_centric_all.pt"

    else:
        raise Exception("incorrect model_name.")

    basePath   = os.path.dirname(os.path.abspath(__file__))
    config_yml = os.path.normpath(os.path.join(basePath, config_yml))
    checkpoint = os.path.normpath(os.path.join(basePath, checkpoint))

    log_file.write("config_yml = " + config_yml + "\n");
    log_file.write("checkpoint = " + checkpoint + "\n");

    # Check gpu
    gpu_ = (gpu and torch.cuda.is_available())

    log_file.write("gpu (in)   = " + str(gpu)   + "\n");
    log_file.write("gpu (eff)  = " + str(gpu_)  + "\n");

    # Load configuration
    config = yaml.safe_load(open(config_yml, "r"))

    # Check max_neigh and cutoff
    max_neigh = config["model"].get("max_neighbors", 50)
    cutoff    = config["model"].get("cutoff", 6.0)

    log_file.write("max_neigh  = " + str(max_neigh) + "\n");
    log_file.write("cutoff     = " + str(cutoff) + "\n");

    assert max_neigh > 0
    assert cutoff    > 0.0

    # To calculate the edge indices on-the-fly
    config["model"]["otf_graph"] = True

    log_file.write("\nconfig:\n");
    log_file.write(pprint.pformat(config) + "\n");
    log_file.write("\n");
    log_file.close()

    # Create trainer, that is pre-trained
    global trainer

    trainer = registry.get_trainer_class(
        config.get("trainer", "forces")
    )(
        task       = config["task"],
        model      = config["model"],
        dataset    = None,
        normalizer = config["normalizer"],
        optimizer  = config["optim"],
        identifier = "",
        slurm      = config.get("slurm", {}),
        local_rank = config.get("local_rank", 0),
        is_debug   = config.get("is_debug", False),
        cpu        = not gpu_
    )

    # Load checkpoint
    trainer.load_checkpoint(checkpoint)

    # Atoms object of ASE, that is empty here
    global atoms

    atoms = None

    # Converter: Atoms -> Graphs (the edges on-the-fly)
    a2g = AtomsToGraphs(
        max_neigh   = max_neigh,
        radius      = cutoff,
        r_energy    = False,
        r_forces    = False,
        r_distances = False,
        r_edges     = False,
        r_fixed     = False
    )

    return cutoff

def oc20_get_energy_and_forces(cell, atomic_numbers, positions):
    """
    Predict total energy and atomic forces w/ pre-trained GNNP of OC20 (i.e. S2EF).
    Args:
        cell: lattice vectors in angstroms.
        atomic_numbers: atomic numbers for all atoms.
        positions: xyz coordinates for all atoms in angstroms.
    Returns:
        energy:  total energy.
        forcces: atomic forces.
    """

    # Initialize Atoms
    global atoms

    if atoms is None:
        atoms = Atoms(
            numbers   = atomic_numbers,
            positions = positions,
            cell      = cell,
            pbc       = [True, True, True]
        )

    else:
        atoms.set_cell(cell)
        atoms.set_atomic_numbers(atomic_numbers)
        atoms.set_positions(positions)

    # Preprossing atomic positions (the edges on-the-fly)
    data  = a2g.convert(atoms)
    batch = data_list_collater([data], otf_graph = True)

    # Predicting energy and forces
    global trainer

    predictions = trainer.predict(
        batch,
        per_image    = False,
        disable_tqdm = True
    )

    energy = predictions["energy"].item()
    forces = predictions["forces"].cpu().numpy().tolist()

    return energy, forces

