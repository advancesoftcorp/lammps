/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef PAIR_CLASS

PairStyle(chgnet/gpu, PairCHGNetGPU)

#else

#ifndef LMP_PAIR_CHGNET_GPU_H_
#define LMP_PAIR_CHGNET_GPU_H_

#include "pair_chgnet.h"

namespace LAMMPS_NS
{

class PairCHGNetGPU: public PairCHGNet
{
public:
    PairCHGNetGPU(class LAMMPS*);

    virtual ~PairCHGNetGPU() override;

protected:
    int withDFTD3() override;

    int withGPU() override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_CHGNET_GPU_H_ */
#endif
