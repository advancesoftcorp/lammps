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

class ReaxPot
{
public:
    ReaxPot(nnpreal rcut, nnpreal mixingRate, const char* fileName);
    virtual ~ReaxPot();

    nnpreal getRcutBond() const
    {
        return this->param->rcut_bond;
    }

    nnpreal getRcutVDW() const
    {
        return this->param->rcut_vdw;
    }

    void removePotentialFrom(Geometry* geometry);

    nnpreal totalEnergyOfPotential(const Geometry* geometry);

    nnpreal energyAndForceOfPotential(const Geometry* geometry, nnpreal* energy, nnpreal* force);

private:
    ReaxParam* param;

    Geometry* geometry;
    int*      geometryMap;

    real mixingRate;

    int**      elemNeighs;
    nnpreal*** rxyzNeighs;

    int*  numBonds;
    int** idxBonds;

    nnpreal*** BOs_raw;  // 0:sigma, 1:pi, 2:pipi
    nnpreal*** BOs_corr; // 0:total, 1:pi, 2:pipi, where sigma = total - pi - pipi
    nnpreal*   Deltas_raw;
    nnpreal*   Deltas_corr;
    nnpreal*   Deltas_e;
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

    void clearAtomData();

    void clearPairData();

    void clearGeometry();

    void setGeometry(const Geometry* geometry);

    void calculatePotential(bool withForce);

    void createNeighbors(nnpreal rcut);

    void calculateBondOrder();
    void calculateBondOrderRaw();
    void calculateBondOrderCorr();

    void calculateBondOrderForce();
    void calculateBondOrderForce0();
    void calculateBondOrderForce1();
    void calculateBondOrderForce2();
    void calculateBondOrderForce3();
    void calculateBondOrderForce4();

    void calculateBondEnergy();

    void calculateLonePairNumber();
    void calculateLonePairNumberN();
    void calculateLonePairNumberS();

    void calculateLonePairEnergy();

    void calculateOverCoordEnergy();

    void calculateVanDerWaalsEnergy(bool withForce);
};

#endif /* NNP_REAX_POT_H_ */
