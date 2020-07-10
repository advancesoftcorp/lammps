/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_nnlayer.h"

NNLayer::NNLayer(int numInpNodes, int numOutNodes, int activation)
{
    if (numInpNodes < 1)
    {
        stop_by_error("input size of neural network is not positive.");
    }

    if (numOutNodes < 1)
    {
        stop_by_error("output size of neural network is not positive.");
    }

    this->numInpNodes = numInpNodes;
    this->numOutNodes = numOutNodes;
    this->sizeBatch = 0;

    this->activation = activation;

    this->inpData = NULL;
    this->inpGrad = NULL;

    this->weight = new real[this->numInpNodes * this->numOutNodes];
    this->bias = new real[this->numOutNodes];
}

NNLayer::~NNLayer()
{
    if (this->inpData != NULL)
    {
        delete[] this->inpData;
    }
    if (this->inpGrad != NULL)
    {
        delete[] this->inpGrad;
    }

    delete[] this->weight;
    delete[] this->bias;
}

void NNLayer::setSizeOfBatch(int sizeBatch)
{
    if (sizeBatch < 1)
    {
        stop_by_error("size of batch is not positive.");
    }

    if (this->sizeBatch == sizeBatch)
    {
        return;
    }

    this->sizeBatch = sizeBatch;

    if (this->inpData != NULL)
    {
        delete[] this->inpData;
    }

    if (this->inpGrad != NULL)
    {
        delete[] this->inpGrad;
    }

    this->inpData = new real[this->numInpNodes * this->sizeBatch];
    this->inpGrad = new real[this->numInpNodes * this->sizeBatch];
}

void NNLayer::scanWeight(FILE* fp, int rank, MPI_Comm world)
{
    int iweight;
    int nweight = this->numInpNodes * this->numOutNodes;

    int ibias;
    int nbias = this->numOutNodes;

    int ierr;

    ierr = 0;
    if (rank == 0)
    {
        for (iweight = 0; iweight < nweight; ++iweight)
        {
            if (fscanf(fp, IFORM_F1, &(this->weight[iweight])) != 1)
            {
                ierr = 1;
                break;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot scan neural network @weight");

    ierr = 0;
    if (rank == 0)
    {
        for (ibias = 0; ibias < nbias; ++ibias)
        {
            if (fscanf(fp, IFORM_F1, &(this->bias[ibias])) != 1)
            {
                ierr = 1;
                break;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot scan neural network @bias");

    MPI_Bcast(&(this->weight[0]), nweight, MPI_REAL0, 0, world);
    MPI_Bcast(&(this->bias[0]),   nbias,   MPI_REAL0, 0, world);
}

void NNLayer::projectWeightFrom(NNLayer* src, int* mapInpNodes)
{
    if (src == NULL)
    {
        stop_by_error("source layer is null.");
    }

    if (mapInpNodes == NULL)
    {
        stop_by_error("map of input nodes is null.");
    }

    int numInpNodes1 = src ->numInpNodes;
    int numInpNodes2 = this->numInpNodes;
    int numOutNodes1 = src ->numOutNodes;
    int numOutNodes2 = this->numOutNodes;
    int activation1  = src ->activation;
    int activation2  = this->activation;

    if (numOutNodes1 != numOutNodes2)
    {
        stop_by_error("cannot projet weight, for incorrect #out-nodes.");
    }

    if (activation1 != activation2)
    {
        stop_by_error("cannot projet weight, for incorrect activation.");
    }

    int ioutNodes;
    int iinpNodes1;
    int iinpNodes2;

    #pragma omp parallel for private (ioutNodes, iinpNodes1, iinpNodes2)
    for (ioutNodes = 0; ioutNodes < numOutNodes1; ++ioutNodes)
    {
        for (iinpNodes1 = 0; iinpNodes1 < numInpNodes1; ++iinpNodes1)
        {
            iinpNodes2 = mapInpNodes[iinpNodes1];

            if (0 <= iinpNodes2 && iinpNodes2 < numInpNodes2)
            {
                this->weight[iinpNodes2 + ioutNodes * numInpNodes2] =
                src ->weight[iinpNodes1 + ioutNodes * numInpNodes1];
            }
        }
    }
}

void NNLayer::goForward(real* outData) const
{
    if (outData == NULL)
    {
        stop_by_error("outData is null.");
    }

    if (this->inpData == NULL)
    {
        stop_by_error("inpData is null.");
    }

    if (this->sizeBatch < 1)
    {
        stop_by_error("size of batch is not positive.");
    }

    // inpData -> outData, through neural network
    real a0 = ZERO;
    real a1 = ONE;

    xgemm_("T", "N", &(this->numOutNodes), &(this->sizeBatch), &(this->numInpNodes),
           &a1, this->weight, &(this->numInpNodes), this->inpData, &(this->numInpNodes),
           &a0, outData, &(this->numOutNodes));

    int ibatch;
    int ioutNode;

    #pragma omp parallel for private (ibatch, ioutNode)
    for (ibatch = 0; ibatch < this->sizeBatch; ++ibatch)
    {
        for (ioutNode = 0; ioutNode < this->numOutNodes; ++ioutNode)
        {
            outData[ioutNode + ibatch * this->numOutNodes] += this->bias[ioutNode];
        }
    }

    // operate activation function
    this->operateActivation(outData);
}

void NNLayer::goBackward(const real* outData, real* outGrad, bool toInpGrad)
{
    if (outData == NULL)
    {
        stop_by_error("outData is null.");
    }

    if (outGrad == NULL)
    {
        stop_by_error("outGrad is null.");
    }

    if (this->sizeBatch < 1)
    {
        stop_by_error("size of batch is not positive.");
    }

    // derive activation function
    this->deriveActivation(outData, outGrad);

    real a0 = ZERO;
    real a1 = ONE;

    // outGrad -> inpGrad, through neural network
    if (toInpGrad)
    {
        if (this->inpGrad == NULL)
        {
            stop_by_error("inpGrad is null.");
        }

        xgemm_("N", "N", &(this->numInpNodes), &(this->sizeBatch), &(this->numOutNodes),
               &a1, this->weight, &(this->numInpNodes), outGrad, &(this->numOutNodes),
               &a0, this->inpGrad, &(this->numInpNodes));
    }
}

void NNLayer::operateActivation(real* outData) const
{
    real x;
    int idata;
    int ndata = this->sizeBatch * this->numOutNodes;

    if (this->activation == ACTIVATION_ASIS)
    {
        // NOP
    }

    else if (this->activation == ACTIVATION_SIGMOID)
    {
        #pragma omp parallel for private (idata, x)
        for (idata = 0; idata < ndata; ++idata)
        {
            x = outData[idata];
            outData[idata] = ONE / (ONE + exp(-x));
        }
    }

    else if (this->activation == ACTIVATION_TANH)
    {
        #pragma omp parallel for private (idata, x)
        for (idata = 0; idata < ndata; ++idata)
        {
            x = outData[idata];
            outData[idata] = tanh(x);
        }
    }

    else if (this->activation == ACTIVATION_ELU)
    {
        #pragma omp parallel for private (idata, x)
        for (idata = 0; idata < ndata; ++idata)
        {
            x = outData[idata];
            outData[idata] = (x >= ZERO) ? x : (exp(x) - ONE);
        }
    }
}

void NNLayer::deriveActivation(const real* outData, real* outGrad) const
{
    real y;
    int idata;
    int ndata = this->sizeBatch * this->numOutNodes;

    if (this->activation == ACTIVATION_ASIS)
    {
        // NOP
    }

    else if (this->activation == ACTIVATION_SIGMOID)
    {
        #pragma omp parallel for private (idata, y)
        for (idata = 0; idata < ndata; ++idata)
        {
            y = outData[idata];
            outGrad[idata] *= y * (ONE - y);
        }
    }

    else if (this->activation == ACTIVATION_TANH)
    {
        #pragma omp parallel for private (idata, y)
        for (idata = 0; idata < ndata; ++idata)
        {
            y = outData[idata];
            outGrad[idata] *= ONE - y * y;
        }
    }

    else if (this->activation == ACTIVATION_ELU)
    {
        #pragma omp parallel for private (idata, y)
        for (idata = 0; idata < ndata; ++idata)
        {
            y = outData[idata];
            outGrad[idata] *= (y >= ZERO) ? ONE : (y + ONE);
        }
    }
}
