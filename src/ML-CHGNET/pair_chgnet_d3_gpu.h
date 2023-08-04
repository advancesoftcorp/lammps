/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef PAIR_CLASS

PairStyle(chgnet/d3/gpu, PairCHGNetD3GPU)

#else

#ifndef LMP_PAIR_CHGNET_D3_GPU_H_
#define LMP_PAIR_CHGNET_D3_GPU_H_

#include "pair_chgnet.h"

namespace LAMMPS_NS
{

class PairCHGNetD3GPU: public PairCHGNet
{
public:
    PairCHGNetD3GPU(class LAMMPS*);

    virtual ~PairCHGNetD3GPU() override;

protected:
    int withDFTD3() override;

    int withGPU() override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_CHGNET_D3_GPU_H_ */
#endif
