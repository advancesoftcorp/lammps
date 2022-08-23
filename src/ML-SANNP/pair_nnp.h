/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifdef PAIR_CLASS

PairStyle(nnp, PairNNP)

#else

#ifndef LMP_PAIR_NNP_H_
#define LMP_PAIR_NNP_H_

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "pair.h"
#include "force.h"
#include "update.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neural_network_potential.h"

namespace LAMMPS_NS
{

class PairNNP: public Pair
{
public:
    PairNNP(class LAMMPS*);

    virtual ~PairNNP() override;

    virtual void compute(int, int) override;

    virtual void settings(int, char **) override;

    virtual void coeff(int, char **) override;

    double init_one(int, int) override;

    virtual void init_style() override;

protected:
    int*      typeMap;
    int       zeroEatom;
    Property* property;
    NNArch*   arch;

    int*       elements;
    nnpreal*   energies;
    nnpreal*** forces;

    int maxinum;
    int maxnneigh;
    int maxnneighAll;

    int*       numNeighbor;
    int**      idxNeighbor;
    int**      elemNeighbor;
    nnpreal*** posNeighbor;
    nnpreal*** posNeighborAll;

    virtual void allocate();

    virtual bool prepareNN();

    virtual void performNN(int);

    void computeLJLike(int);

    virtual double get_cutoff();

private:
    int dimensionPosNeighbor();
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_NNP_H_ */
#endif
