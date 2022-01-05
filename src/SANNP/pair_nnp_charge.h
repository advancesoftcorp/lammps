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

    virtual ~PairNNPCharge();

    virtual void settings(int, char **);

    void coeff(int, char **);

    virtual void init_style();

    void *extract(const char *, int &);

    int pack_forward_comm(int, int *, double *, int, int *);

    void unpack_forward_comm(int, int, double *);

    int pack_reverse_comm(int, int, double *);

    void unpack_reverse_comm(int, int *, double *);

protected:
    double cutcoul;
    nnpreal* charges;

    void allocate();

    bool prepareNN();

    void performNN(int);

    double get_cutoff();
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_NNP_CHARGE_H_ */
#endif
