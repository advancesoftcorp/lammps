/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_NNLAYER_H_
#define NNP_NNLAYER_H_

#include "nnp_common.h"

class NNLayer
{
public:
    NNLayer(int numInpNodes, int numOutNodes, int activation);
    virtual ~NNLayer();

    void setSizeOfBatch(int sizeBatch);

    void scanWeight(const FILE* fp, int rank, MPI_Comm world);

    void projectWeightFrom(NNLayer* src, int* mapInpNodes);

    void goForward(real* outData) const;

    void goBackward(const real* outData, real* outGrad, bool toInpGrad);

    int getNumInpNodes() const
    {
        return this->numInpNodes;
    }

    int getNumOutNodes() const
    {
        return this->numOutNodes;
    }

    real* getData()
    {
        return this->inpData;
    }

    real* getGrad()
    {
        return this->inpGrad;
    }

private:
    int numInpNodes;
    int numOutNodes;
    int sizeBatch;

    int activation;

    real* inpData;
    real* inpGrad;

    real* weight;
    real* bias;

    void operateActivation(real* outData) const;

    void deriveActivation(const real* outData, real* outGrad) const;
};

#endif /* NNP_NNLAYER_H_ */
