/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifdef PAIR_CLASS

PairStyle(nnp/coul/long, PairNNPCoulLong)

#else

#ifndef LMP_PAIR_NNP_COUL_LONG_H_
#define LMP_PAIR_NNP_COUL_LONG_H_

#include "pair_nnp_charge.h"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

namespace LAMMPS_NS
{

class PairNNPCoulLong : public PairNNPCharge
{
public:
    PairNNPCoulLong(class LAMMPS*);

    virtual ~PairNNPCoulLong();

    void compute(int, int);

    void init_style();

protected:
    double g_ewald;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_NNP_COUL_LONG_H_ */
#endif
