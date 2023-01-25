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

    void readProperty(FILE* fp, int rank, int nproc, MPI_Comm world);

    int getSymmFunc() const
    {
        return this->symmFunc;
    }

    int getElemWeight() const
    {
        return this->elemWeight;
    }

    int getTanhCutoff() const
    {
        return this->tanhCutoff;
    }

    nnpreal getRcutoff() const
    {
        if (this->symmFunc == SYMM_FUNC_MANYBODY)
        {
            return this->router;
        }

        if (this->numRadius > 0 && this->numAngle > 0)
        {
            return max(this->rcutRadius, this->rcutAngle);
        }
        else if (this->numRadius > 0)
        {
            return this->rcutRadius;
        }
        else if (this->numAngle > 0)
        {
            return this->rcutAngle;
        }

        return 0.0;
    }

    int getM2() const
    {
        return this->m2;
    }

    int getM3() const
    {
        return this->m3;
    }

    nnpreal getRinner() const
    {
        return this->rinner;
    }

    nnpreal getRouter() const
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

    nnpreal getRcutRadius() const
    {
        return this->rcutRadius;
    }

    nnpreal getRcutAngle() const
    {
        return this->rcutAngle;
    }

    int getBehlerG4() const
    {
        return this->behlerG4;
    }

    const nnpreal* getBehlerEta1() const
    {
        return this->behlerEta1;
    }

    const nnpreal* getBehlerEta2() const
    {
        return this->behlerEta2;
    }

    const nnpreal* getBehlerRs1() const
    {
        return this->behlerRs1;
    }

    const nnpreal* getBehlerRs2() const
    {
        return this->behlerRs2;
    }

    const nnpreal* getBehlerZeta() const
    {
        return this->behlerZeta;
    }

    int getCutoffMode() const
    {
        return this->cutoffMode;
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

    int getWithClassical() const
    {
        return this->withClassical;
    }

    int getWithReaxFF() const
    {
        return this->withReaxFF;
    }

    nnpreal getRcutReaxFF() const
    {
        return this->rcutReaxFF;
    }

    nnpreal getRateReaxFF() const
    {
        return this->rateReaxFF;
    }

#ifdef _NNP_GPU
    int getGpuThreads() const
    {
        return this->gpuThreads;
    }

    int getGpuAtomBlock() const
    {
        return this->gpuAtomBlock;
    }
#endif

private:
    // about symmetry functions
    int symmFunc;
    int elemWeight;
    int tanhCutoff;

    int m2;
    int m3;
    nnpreal rinner;
    nnpreal router;

    int     numRadius;
    int     numAngle;
    nnpreal rcutRadius;
    nnpreal rcutAngle;

    int      behlerG4;
    nnpreal* behlerEta1;
    nnpreal* behlerEta2;
    nnpreal* behlerRs1;
    nnpreal* behlerRs2;
    nnpreal* behlerZeta;

    int cutoffMode;

    // about neural networks
    int layersEnergy;
    int nodesEnergy;
    int activEnergy;

    int layersCharge;
    int nodesCharge;
    int activCharge;

    int     withCharge;
    int     withClassical;
    int     withReaxFF;
    nnpreal rcutReaxFF;
    nnpreal rateReaxFF;

    void printProperty();

    void activToString(char* str, int activ);

#ifdef _NNP_GPU
    // for GPU
    int gpuThreads;
    int gpuAtomBlock;

    void readGpuProperty(int rank, int nproc, MPI_Comm world);
#endif
};

#endif /* NNP_PROPERTY_H_ */
