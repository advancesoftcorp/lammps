/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef PAIR_CLASS

PairStyle(oc20/gpu, PairOC20GPU)

#else

#ifndef LMP_PAIR_OC20_GPU_H_
#define LMP_PAIR_OC20_GPU_H_

#include "pair_oc20.h"

namespace LAMMPS_NS
{

class PairOC20GPU: public PairOC20
{
public:
    PairOC20GPU(class LAMMPS*);

    virtual ~PairOC20GPU() override;

protected:
    int withGPU() override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_OC20_GPU_H_ */
#endif
