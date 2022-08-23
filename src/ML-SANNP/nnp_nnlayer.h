/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_NNLAYER_H_
#define NNP_NNLAYER_H_

#include "nnp_common.h"
#include "memory.h"

class NNLayer
{
public:
    NNLayer(int numInpNodes, int numOutNodes, int activation, int imemory, LAMMPS_NS::Memory* memory);
    NNLayer(int numInpNodes, int numOutNodes, int activation);
    virtual ~NNLayer();

    void setSizeOfBatch(int sizeBatch);

    void scanWeight(FILE* fp, bool zeroBias, int rank, MPI_Comm world);

    void scanWeight(FILE* fp, int rank, MPI_Comm world)
    {
        this->scanWeight(fp, false, rank, world);
    }

    void projectWeightFrom(NNLayer* src, int* mapInpNodes);

    void goForward(nnpreal* outData) const;

    void goBackward(nnpreal* outGrad, bool toInpGrad);

    int getNumInpNodes() const
    {
        return this->numInpNodes;
    }

    int getNumOutNodes() const
    {
        return this->numOutNodes;
    }

    nnpreal* getData()
    {
        return this->inpData;
    }

    nnpreal* getGrad()
    {
        return this->inpGrad;
    }

private:
    int numInpNodes;
    int numOutNodes;
    int sizeBatch;
    int sizeBatchMax;

    int activation;

    int imemory;
    LAMMPS_NS::Memory* memory;

    nnpreal* inpData;
    nnpreal* inpGrad;
    nnpreal* outDrv1;

    nnpreal* weight;
    nnpreal* bias;

    void operateActivation(nnpreal* outData) const;
};

#endif /* NNP_NNLAYER_H_ */
