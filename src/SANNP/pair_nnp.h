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

    virtual ~PairNNP();

    virtual void compute(int, int);

    virtual void settings(int, char **);

    virtual void coeff(int, char **);

    double init_one(int, int);

    virtual void init_style();

protected:
    int* typeMap;
    Property* property;
    NNArch* arch;

    real* energies;
    real*** forces;

    int maxinum;
    int maxnneigh;
    int maxnneighNN;

    real*** posNeighbor;

    real*** posNeighborNN;
    int* numNeighborNN;
    int** elemNeighborNN;
    int** indexNeighborNN;

    virtual void allocate();

    virtual bool prepareNN();

    virtual void performNN(int);

    void clearNN();

    virtual double get_cutoff();
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_NNP_H_ */
#endif
