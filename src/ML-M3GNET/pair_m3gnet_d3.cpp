/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_m3gnet_d3.h"

using namespace LAMMPS_NS;

PairM3GNetD3::PairM3GNetD3(LAMMPS *lmp) : PairM3GNet(lmp)
{
    if (copymode)
    {
        return;
    }

    // NOP
}

PairM3GNetD3::~PairM3GNetD3()
{
    // NOP
}

int PairM3GNetD3::withDFTD3()
{
    return 1;
}

