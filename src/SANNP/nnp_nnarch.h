/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_NNARCH_H_
#define NNP_NNARCH_H_

#include "nnp_common.h"
#include "nnp_property.h"
#include "nnp_symm_func.h"
#include "nnp_symm_func_manybody.h"
#include "nnp_symm_func_behler.h"
#include "nnp_symm_func_chebyshev.h"
#include "nnp_nnlayer.h"

class NNArch
{
public:
    NNArch(int mode, int numElems, const Property* property);
    virtual ~NNArch();

    int getAtomNum(int ielem) const
    {
        return this->atomNum[ielem];
    }

    void restoreNN(const FILE* fp, int numElems, const char** elemNames, int rank, MPI_Comm world);

    void initGeometry(int numAtoms, const int* elements,
                      const int* numNeighbor, const int** elemNeighbor, const real*** posNeighbor)

    void clearGeometry();

    void calculateSymmFuncs();

    void renormalizeSymmFuncs();

    void initLayers();

    void goForwardOnEnergy();

    void goBackwardOnForce();

    void goForwardOnCharge();

    void obtainEnergies(real* energies) const;

    void obtainForces(real*** forces) const;

    void obtainCharges(real* charges) const;

private:
    int mode;

    bool elemWeight;
    int  numElems;
    int  numAtoms;

    const Property* property;

    int* atomNum;

    const int*    elements;
    const int*    numNeighbor;
    const int**   elemNeighbor;
    const real*** posNeighbor;

    int   mbatch;
    int*  nbatch;
    int*  ibatch;

    real** energyData;
    real** energyGrad;

    real*** forceData;

    real** chargeData;

    real**    symmData;
    real**    symmDiff;
    real*     symmAve;
    real*     symmDev;
    SymmFunc* symmFunc;

    NNLayer*** interLayersEnergy;
    NNLayer**  lastLayersEnergy;

    NNLayer*** interLayersCharge;
    NNLayer**  lastLayersCharge;

    bool isEnergyMode() const
    {
        return this->mode == NNARCH_MODE_BOTH || this->mode == NNARCH_MODE_ENERGY;
    }

    bool isChargeMode() const
    {
        return this->mode == NNARCH_MODE_BOTH || this->mode == NNARCH_MODE_CHARGE;
    }

    SymmFunc* getSymmFunc();
};

#endif /* NNP_NNARCH_H_ */
