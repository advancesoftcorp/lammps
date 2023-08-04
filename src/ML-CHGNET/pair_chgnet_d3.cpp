/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_chgnet_d3.h"

using namespace LAMMPS_NS;

PairCHGNetD3::PairCHGNetD3(LAMMPS *lmp) : PairCHGNet(lmp)
{
    if (copymode)
    {
        return;
    }

    // NOP
}

PairCHGNetD3::~PairCHGNetD3()
{
    // NOP
}

int PairCHGNetD3::withDFTD3()
{
    return 1;
}

int PairCHGNetD3::withGPU()
{
    return 0;
}

