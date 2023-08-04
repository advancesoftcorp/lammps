/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef PAIR_CLASS

PairStyle(chgnet/d3, PairCHGNetD3)

#else

#ifndef LMP_PAIR_CHGNET_D3_H_
#define LMP_PAIR_CHGNET_D3_H_

#include "pair_chgnet.h"

namespace LAMMPS_NS
{

class PairCHGNetD3: public PairCHGNet
{
public:
    PairCHGNetD3(class LAMMPS*);

    virtual ~PairCHGNetD3() override;

protected:
    int withDFTD3() override;

    int withGPU() override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_CHGNET_D3_H_ */
#endif
