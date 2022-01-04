/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_oc20_gpu.h"

using namespace LAMMPS_NS;

PairOC20GPU::PairOC20GPU(LAMMPS *lmp) : PairOC20(lmp)
{
    if (copymode)
    {
        return;
    }

    // NOP
}

PairOC20GPU::~PairOC20GPU()
{
    // NOP
}

int PairOC20GPU::withGPU()
{
    return 1;
}

