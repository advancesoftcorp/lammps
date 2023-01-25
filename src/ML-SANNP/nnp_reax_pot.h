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

    void removePotentialFrom(Geometry* geometry);

    nnpreal totalEnergyOfPotential(const Geometry* geometry);

    nnpreal energyAndForceOfPotential(const Geometry* geometry, real* energy, real* force);

private:
    ReaxParam* param;

    Geometry* geometry;
    int*      geometryMap;

    real mixingRate;

    int**   elemNeighs;
    real*** rxyzNeighs;

    int*  numBonds;
    int** idxBonds;

    real*** BOs_raw;  // 0:sigma, 1:pi, 2:pipi
    real*** BOs_corr; // 0:total, 1:pi, 2:pipi, where sigma = total - pi - pipi
    real*   Deltas_raw;
    real*   Deltas_corr;
    real*   Deltas_e;
    real*   n0lps;
    real*   nlps;
    real*   Slps;
    real*   Tlps;

    real*** dBOdrs_raw;
    real*** dBOdBOs;
    real*** dBOdDeltas;
    real*** dEdBOs_raw;
    real*** dEdBOs_corr;
    real*   dEdDeltas_raw;
    real*   dEdDeltas_corr;
    real*   dEdSlps;
    real*   dn0lpdDeltas;
    real*   dnlpdDeltas;
    real*   dTlpdDeltas;

    void clearAtomData();

    void clearPairData();

    void clearGeometry();

    void setGeometry(const Geometry* geometry);

    void calculatePotential(bool withForce);

    void createNeighbors(real rcut);

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
