/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_chgnet_d3_gpu.h"

using namespace LAMMPS_NS;

PairCHGNetD3GPU::PairCHGNetD3GPU(LAMMPS *lmp) : PairCHGNet(lmp)
{
    if (copymode)
    {
        return;
    }

    // NOP
}

PairCHGNetD3GPU::~PairCHGNetD3GPU()
{
    // NOP
}

int PairCHGNetD3GPU::withDFTD3()
{
    return 1;
}

int PairCHGNetD3GPU::withGPU()
{
    return 1;
}

