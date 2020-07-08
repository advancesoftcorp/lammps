/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_PROPERTY_H_
#define NNP_PROPERTY_H_

#include "nnp_common.h"

class Property
{
public:
    Property();
    virtual ~Property();

    void peekProperty(FILE* fp, int rank, MPI_Comm world);

    int getSymmFunc() const
    {
        return this->symmFunc;
    }

    int getM2() const
    {
        return this->m2;
    }

    int getM3() const
    {
        return this->m3;
    }

    real getRinner() const
    {
        return this->rinner;
    }

    real getRouter() const
    {
        return this->router;
    }

    int getNumRadius() const
    {
        return this->numRadius;
    }

    int getNumAngle() const
    {
        return this->numAngle;
    }

    const real* getBehlerEta1() const
    {
        return this->behlerEta1;
    }

    const real* getBehlerEta2() const
    {
        return this->behlerEta2;
    }

    const real* getBehlerRs() const
    {
        return this->behlerRs;
    }

    const real* getBehlerZeta() const
    {
        return this->behlerZeta;
    }

    int getLayersEnergy() const
    {
        return this->layersEnergy;
    }

    int getNodesEnergy() const
    {
        return this->nodesEnergy;
    }

    int getActivEnergy() const
    {
        return this->activEnergy;
    }

    int getLayersCharge() const
    {
        return this->layersCharge;
    }

    int getNodesCharge() const
    {
        return this->nodesCharge;
    }

    int getActivCharge() const
    {
        return this->activCharge;
    }

    int getWithCharge() const
    {
        return this->withCharge;
    }

private:
    // about symmetry functions
    int symmFunc;

    int m2;
    int m3;
    real rinner;
    real router;

    int   numRadius;
    int   numAngle;
    real* behlerEta1;
    real* behlerEta2;
    real* behlerRs;
    real* behlerZeta;

    // about neural networks
    int  layersEnergy;
    int  nodesEnergy;
    int  activEnergy;

    int  layersCharge;
    int  nodesCharge;
    int  activCharge;

    int  withCharge;
};

#endif /* NNP_PROPERTY_H_ */
