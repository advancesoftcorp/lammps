/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_chgnet_gpu.h"

using namespace LAMMPS_NS;

PairCHGNetGPU::PairCHGNetGPU(LAMMPS *lmp) : PairCHGNet(lmp)
{
    if (copymode)
    {
        return;
    }

    // NOP
}

PairCHGNetGPU::~PairCHGNetGPU()
{
    // NOP
}

int PairCHGNetGPU::withDFTD3()
{
    return 0;
}

int PairCHGNetGPU::withGPU()
{
    return 1;
}

