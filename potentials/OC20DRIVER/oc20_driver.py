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
        config_yml = "dimenetpp.yml"
        checkpoint = "dimenetpp_all.pt"

    elif model_name == "GemNet-dT".lower():
        config_yml = "gemnet.yml"
        checkpoint = "gemnet_t_direct_h512_all.pt"

    elif model_name == "CGCNN".lower():
        config_yml = "cgcnn.yml"
        checkpoint = "cgcnn_all.pt"

    elif model_name == "SchNet".lower():
        config_yml = "schnet.yml"
        checkpoint = "schnet_all_large.pt"

    elif model_name == "SpinConv".lower():
        config_yml = "spinconv.yml"
        checkpoint = "spinconv_force_centric_all.pt"

    else:
        raise Exception("incorrect model_name.")

    basePath   = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.normpath(os.path.join(basePath, "oc20_configs"))
    chekpt_dir = os.path.normpath(os.path.join(basePath, "oc20_checkpt"))
    config_yml = os.path.normpath(os.path.join(config_dir, config_yml))
    checkpoint = os.path.normpath(os.path.join(chekpt_dir, checkpoint))

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

    # Modify path of scale_file for GemNet-dT
    scale_file = config["model"].get("scale_file", None)

    if scale_file is not None:
        scale_file = os.path.normpath(os.path.join(config_dir, scale_file))
        config["model"]["scale_file"] = scale_file

    log_file.write("\nconfig:\n");
    log_file.write(pprint.pformat(config) + "\n");
    log_file.write("\n");
    log_file.close()

    # Create trainer, that is pre-trained
    global myTrainer

    myTrainer = registry.get_trainer_class(
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
    myTrainer.load_checkpoint(checkpoint)

    # Atoms object of ASE, that is empty here
    global myAtoms

    myAtoms = None

    # Converter: Atoms -> Graphs (the edges on-the-fly)
    global myA2G

    myA2G = AtomsToGraphs(
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
    global myAtoms

    if myAtoms is None:
        myAtoms = Atoms(
            numbers   = atomic_numbers,
            positions = positions,
            cell      = cell,
            pbc       = [True, True, True]
        )

    else:
        myAtoms.set_cell(cell)
        myAtoms.set_atomic_numbers(atomic_numbers)
        myAtoms.set_positions(positions)

    # Preprossing atomic positions (the edges on-the-fly)
    global myA2G

    data  = myA2G.convert(myAtoms)
    batch = data_list_collater([data], otf_graph = True)

    # Predicting energy and forces
    global myTrainer

    predictions = myTrainer.predict(
        batch,
        per_image    = False,
        disable_tqdm = True
    )

    energy = predictions["energy"].item()
    forces = predictions["forces"].cpu().numpy().tolist()

    return energy, forces

