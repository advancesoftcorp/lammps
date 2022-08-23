/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifdef PAIR_CLASS

PairStyle(nnp/charge, PairNNPCharge)

#else

#ifndef LMP_PAIR_NNP_CHARGE_H_
#define LMP_PAIR_NNP_CHARGE_H_

#include "pair_nnp.h"
#include "kspace.h"

namespace LAMMPS_NS
{

class PairNNPCharge : public PairNNP
{
public:
    PairNNPCharge(class LAMMPS*);

    virtual ~PairNNPCharge() override;

    virtual void settings(int, char **) override;

    void coeff(int, char **) override;

    virtual void init_style() override;

    void *extract(const char *, int &) override;

    int pack_forward_comm(int, int *, double *, int, int *) override;

    void unpack_forward_comm(int, int, double *) override;

    int pack_reverse_comm(int, int, double *) override;

    void unpack_reverse_comm(int, int *, double *) override;

protected:
    double cutcoul;
    nnpreal* charges;

    void allocate() override;

    void prepareNN(bool* hasGrown) override;

    void performNN(int) override;

    double get_cutoff() override;
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_NNP_CHARGE_H_ */
#endif
