/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_REAX_POT_H_
#define NNP_REAX_POT_H_

#include "nnp_common.h"
#include "nnp_reax_param.h"
#include "memory.h"
#include "pair.h"
#include "atom.h"

class ReaxPot
{
public:
    ReaxPot(nnpreal rcut, nnpreal mixingRate, LAMMPS_NS::Memory* memory, FILE* fp, int rank, MPI_Comm world);
    virtual ~ReaxPot();

    void initElements(int ntypes, int* atomNums);

    void initGeometry(int locAtoms, int numAtoms, int* type,
                      int* ilist, int* numNeighbor, int** idxNeighbor, nnpreal*** posNeighbor);

    void calculatePotential(int eflag, LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom);

    nnpreal getRcutBond() const
    {
        return this->param->rcut_bond;
    }

    nnpreal getRcutVDW() const
    {
        return this->param->rcut_vdw;
    }

private:
    ReaxParam* param;

    nnpreal mixingRate;

    LAMMPS_NS::Memory* memory;

    int  locAtoms;
    int  locAtomsAll;
    int  numAtoms;
    int  numAtomsAll;
    int  maxAtoms;
    int  maxAtomsAll;
    int* mapForwardAtoms;   // map w/ ilist of LAMMPS
    int* mapBackwardAtoms1; // map w/ ilist of LAMMPS
    int* mapBackwardAtoms2; // map w/o ilist of LAMMPS

    int*       typeMap;
    int*       typeAll;
    int*       numNeighsAll;
    int**      idxNeighsAll;
    nnpreal*** posNeighsAll;

    int   maxNeighs;
    int   maxBonds;
    int   maxBondsAll;
    int*  numBonds;
    int** idxBonds;

    nnpreal*** BOs_raw;  // 0:sigma, 1:pi, 2:pipi
    nnpreal*** BOs_corr; // 0:total, 1:pi, 2:pipi, where sigma = total - pi - pipi
    nnpreal*   Deltas_raw;
    nnpreal*   Deltas_corr;
    nnpreal*   Deltas_e;
    nnpreal*   exp1Deltas;
    nnpreal*   exp2Deltas;
    nnpreal*   n0lps;
    nnpreal*   nlps;
    nnpreal*   Slps;
    nnpreal*   Tlps;

    nnpreal*** dBOdrs_raw;
    nnpreal*** dBOdBOs;
    nnpreal*** dBOdDeltas;
    nnpreal*** dEdBOs_raw;
    nnpreal*** dEdBOs_corr;
    nnpreal*   dEdDeltas_raw;
    nnpreal*   dEdDeltas_corr;
    nnpreal*   dEdSlps;
    nnpreal*   dn0lpdDeltas;
    nnpreal*   dnlpdDeltas;
    nnpreal*   dTlpdDeltas;
    nnpreal*   dDeltadSlps;
    nnpreal*   dDeltadDeltas;
    nnpreal*   coeff1Eovers;
    nnpreal*   coeff2Eovers;

    void calculateBondOrder();
    void calculateBondOrderRaw();
    void calculateBondOrderCorr();

    void calculateBondOrderForce(LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom);
    void calculateBondOrderForce0();
    void calculateBondOrderForce1();
    void calculateBondOrderForce2();
    void calculateBondOrderForce3();
    void calculateBondOrderForce4(LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom);

    void calculateBondEnergy(int eflag, LAMMPS_NS::Pair* pair);

    void calculateLonePairNumber();
    void calculateLonePairNumberN();
    void calculateLonePairNumberS();

    void calculateLonePairEnergy(int eflag, LAMMPS_NS::Pair* pair);

    void calculateOverCoordEnergy(int eflag, LAMMPS_NS::Pair* pair);

    void calculateVanDerWaalsEnergy(int eflag, LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom);

    int indexOfLAMMPS(int iatom) const
    {
        return this->mapBackwardAtoms1[iatom]; // w/ ilist
    }

    int getElement(int iatom) const
    {
        int Iatom = this->mapBackwardAtoms1[iatom]; // w/ ilist

        int itype = this->typeAll[Iatom];     // atomic type of LAMMPS
        int ielem = this->typeMap[itype - 1]; // atomic type of ReaxFF
        return ielem;
    }

    int numNeighbors(int iatom) const
    {
        int Iatom = this->mapBackwardAtoms1[iatom]; // w/ ilist
        return this->numNeighsAll[Iatom];
    }

    int* getNeighbors(int iatom) const
    {
        int Iatom = this->mapBackwardAtoms1[iatom]; // w/ ilist
        return this->idxNeighsAll[Iatom];
    }

    int getNeighbor(int* idxNeigh, int ineigh) const
    {
        int Jatom = idxNeigh[ineigh];
        Jatom &= NEIGHMASK;

        int jatom = this->mapForwardAtoms[Jatom]; // w/ ilist
        return jatom;
    }

    nnpreal** getPositions(int iatom) const
    {
        int Iatom = this->mapBackwardAtoms2[iatom]; // w/o ilist
        return this->posNeighsAll[Iatom];
    }
};

#endif /* NNP_REAX_POT_H_ */
