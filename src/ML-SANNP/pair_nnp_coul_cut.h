/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifdef PAIR_CLASS

PairStyle(nnp/coul/cut, PairNNPCoulCut)

#else

#ifndef LMP_PAIR_NNP_COUL_CUT_H_
#define LMP_PAIR_NNP_COUL_CUT_H_

#include "pair_nnp_charge.h"

namespace LAMMPS_NS
{

class PairNNPCoulCut : public PairNNPCharge
{
public:
    PairNNPCoulCut(class LAMMPS*);

    virtual ~PairNNPCoulCut() override;

    void compute(int, int) override;

    void settings(int, char **) override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_NNP_COUL_CUT_H_ */
#endif
