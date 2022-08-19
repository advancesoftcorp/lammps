/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func_gpu.h"

SymmFuncGPU::SymmFuncGPU(int numElems, bool tanhCutFunc, bool elemWeight, int sizeRad, int sizeAng,
                         nnpreal rcutRad, nnpreal rcutAng, int cutoffMode) : SymmFunc(numElems, tanhCutFunc, elemWeight)
{
    if (sizeRad < 1)
    {
        stop_by_error("size of radius basis is not positive.");
    }

    if (sizeAng < 0)
    {
        stop_by_error("size of angle basis is negative.");
    }

    if (rcutRad <= ZERO)
    {
        stop_by_error("cutoff for radius is not positive.");
    }

    if (sizeAng > 0 && rcutAng <= ZERO)
    {
        stop_by_error("cutoff for angle is not positive.");
    }

    this->maxThreadsPerBlock = 1;

    this->sizeRad = sizeRad;
    this->sizeAng = sizeAng;

    if (this->elemWeight)
    {
        this->numRadBasis = this->sizeRad;
        this->numAngBasis = this->sizeAng;
    }
    else
    {
        this->numRadBasis = this->sizeRad * this->numElems;
        this->numAngBasis = this->sizeAng * (this->numElems * (this->numElems + 1) / 2);
    }

    this->numBasis = this->numRadBasis + this->numAngBasis;

    this->rcutRad = rcutRad;
    this->rcutAng = rcutAng;

    this->sizeLenAtoms = 0;
    this->sizeTotNeigh = 0;

    if (cutoffMode == CUTOFF_MODE_SINGLE)
    {
        this->sizePosNeighbor = 6;
    }
    else if (cutoffMode == CUTOFF_MODE_DOUBLE || cutoffMode == CUTOFF_MODE_IPSO)
    {
        this->sizePosNeighbor = 8;
    }
    else
    {
        this->sizePosNeighbor = 4;
    }

    this->numNeighs        = nullptr;
    this->numNeighs_d      = nullptr;
    this->idxNeighs        = nullptr;
    this->idxNeighs_d      = nullptr;
    this->elementAll       = nullptr;
    this->elementAll_d     = nullptr;
    this->posNeighborAll   = nullptr;
    this->posNeighborAll_d = nullptr;
    this->symmDataAll      = nullptr;
    this->symmDataAll_d    = nullptr;
    this->symmDiffAll      = nullptr;
    this->symmDiffAll_d    = nullptr;

#ifdef _NNP_SINGLE
    cudaError_t error = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
    cudaError_t error = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif

    if (error != cudaSuccess)
    {
        char message[512];
        sprintf(message, "error of cudaDeviceSetSharedMemConfig: %s\n", cudaGetErrorString(error));
        stop_by_error(message);
    }
}

SymmFuncGPU::~SymmFuncGPU()
{
    if (this->numNeighs        != nullptr) cudaFreeHost(this->numNeighs);
    if (this->numNeighs_d      != nullptr) cudaFree    (this->numNeighs_d);
    if (this->idxNeighs        != nullptr) cudaFreeHost(this->idxNeighs);
    if (this->idxNeighs_d      != nullptr) cudaFree    (this->idxNeighs_d);
    if (this->elementAll       != nullptr) cudaFreeHost(this->elementAll);
    if (this->elementAll_d     != nullptr) cudaFree    (this->elementAll_d);
    if (this->posNeighborAll   != nullptr) cudaFreeHost(this->posNeighborAll);
    if (this->posNeighborAll_d != nullptr) cudaFree    (this->posNeighborAll_d);
    if (this->symmDataAll      != nullptr) cudaFreeHost(this->symmDataAll);
    if (this->symmDataAll_d    != nullptr) cudaFree    (this->symmDataAll_d);
    if (this->symmDiffAll      != nullptr) cudaFreeHost(this->symmDiffAll);
    if (this->symmDiffAll_d    != nullptr) cudaFree    (this->symmDiffAll_d);
}

void SymmFuncGPU::calculate(int lenAtoms, int* numNeighbor, int** elemNeighbor, nnpreal*** posNeighbor,
                            nnpreal** symmData, nnpreal** symmDiff);
{
    if (numAtoms < 0)
    {
        stop_by_error("#atoms is not positive.");
    }

    if (numNeighbor == nullptr || elemNeighbor == nullptr || posNeighbor == nullptr)
    {
        stop_by_error("neighbor is null.");
    }

    if (symmData == nullptr)
    {
        stop_by_error("symmData is null.");
    }

    if (symmDiff == nullptr)
    {
        stop_by_error("symmDiff is null.");
    }

    // define varialbes
    int iatom;
    int ineigh, jneigh;
    int ifree;
    int ibase;
    int ipos;

    int numNeigh;
    int idxNeigh;
    int maxNeigh;
    int totNeigh;

    int numPos;
    int idxPos;

    int idxBase;
    int idxDiff;

    int numModeBatchs;
    int modesPerBatch;
    int dimBasis;

    dim3 grid;
    dim3 block;

    size_t sizeShared;

    // allocate memory about lenAtoms
    if (this->sizeLenAtoms < lenAtoms)
    {
        if (this->numNeighs   != nullptr) cudaFreeHost(this->numNeighs);
        if (this->numNeighs_d != nullptr) cudaFree    (this->numNeighs_d);
        if (this->idxNeighs   != nullptr) cudaFreeHost(this->idxNeighs);
        if (this->idxNeighs_d != nullptr) cudaFree    (this->idxNeighs_d);

        cudaMallocHost(&(this->numNeighs),   sizeof(int) * lenAtoms);
        cudaMalloc    (&(this->numNeighs_d), sizeof(int) * lenAtoms);
        cudaMallocHost(&(this->idxNeighs),   sizeof(int) * lenAtoms);
        cudaMalloc    (&(this->idxNeighs_d), sizeof(int) * lenAtoms);

        this->sizeLenAtoms = lenAtoms;
    }

    // count neighbors
    maxNeigh = 0;
    totNeigh = 0;

    for (iatom = 0; iatom < lenAtoms; ++iatom)
    {
        numNeigh = numNeighbor[iatom];
        this->numNeighs[iatom] = numNeigh;
        this->idxNeighs[iatom] = totNeigh;

        maxNeigh  = max(maxNeigh, numNeigh);
        totNeigh += numNeigh;
    }

    if (maxNeigh > this->maxThreadsPerBlock)
    {
        stop_by_error("too less #threads a block for GPU.");
    }

    if (maxNeigh < 1 || totNeigh < 1)
    {
        #pragma omp parallel for private (iatom, ibase, ineigh, numNeigh, ifree)
        for (iatom = 0; iatom < lenAtoms; ++iatom)
        {
            for (ibase = 0; ibase < this->numBasis; ++ibase)
            {
                symmData[iatom][ibase] = ZERO;
            }

            for (ibase = 0; ibase < this->numBasis; ++ibase)
            {
                symmDiff[iatom][ibase + 0 * this->numBasis] = ZERO;
                symmDiff[iatom][ibase + 1 * this->numBasis] = ZERO;
                symmDiff[iatom][ibase + 2 * this->numBasis] = ZERO;
            }

            numNeigh = this->numNeighs[iatom];

            if (numNeigh < 1)
            {
                continue;
            }

            for (ineigh = 0; ineigh < numNeigh; ++ineigh)
            {
                ifree   = 3 * (ineigh + 1);

                for (ibase = 0; ibase < this->numBasis; ++ibase)
                {
                    symmDiff[iatom][ibase + (ifree + 0) * this->numBasis] = ZERO;
                    symmDiff[iatom][ibase + (ifree + 1) * this->numBasis] = ZERO;
                    symmDiff[iatom][ibase + (ifree + 2) * this->numBasis] = ZERO;
                }
            }
        }

        return;
    }

    // allocate memory about totNeigh
    if (this->sizeTotNeigh < totNeigh)
    {
        if (this->elementAll       != nullptr) cudaFreeHost(this->elementAll);
        if (this->elementAll_d     != nullptr) cudaFree    (this->elementAll_d);
        if (this->posNeighborAll   != nullptr) cudaFreeHost(this->posNeighborAll);
        if (this->posNeighborAll_d != nullptr) cudaFree    (this->posNeighborAll_d);
        if (this->symmDataAll      != nullptr) cudaFreeHost(this->symmDataAll);
        if (this->symmDataAll_d    != nullptr) cudaFree    (this->symmDataAll_d);
        if (this->symmDiffAll      != nullptr) cudaFreeHost(this->symmDiffAll);
        if (this->symmDiffAll_d    != nullptr) cudaFree    (this->symmDiffAll_d);

        cudaMallocHost(&(this->elementAll),       sizeof(gint)    * totNeigh);
        cudaMalloc    (&(this->elementAll_d),     sizeof(gint)    * totNeigh);
        cudaMallocHost(&(this->posNeighborAll),   sizeof(nnpreal) * this->sizePosNeighbor * totNeigh);
        cudaMalloc    (&(this->posNeighborAll_d), sizeof(nnpreal) * this->sizePosNeighbor * totNeigh);
        cudaMallocHost(&(this->symmDataAll),      sizeof(nnpreal) * this->numBasis * totNeigh);
        cudaMalloc    (&(this->symmDataAll_d),    sizeof(nnpreal) * this->numBasis * totNeigh);
        cudaMallocHost(&(this->symmDiffAll),      sizeof(nnpreal) * this->numBasis * 3 * totNeigh);
        cudaMalloc    (&(this->symmDiffAll_d),    sizeof(nnpreal) * this->numBasis * 3 * totNeigh);

        this->sizeTotNeigh = totNeigh;
    }

    // serialize all data of neighbors
    #pragma omp parallel for private (iatom, ineigh, jneigh, numNeigh, idxNeigh, ipos, numPos, idxPos)
    for (iatom = 0; iatom < lenAtoms; ++iatom)
    {
        numNeigh = this->numNeighs[iatom];
        idxNeigh = this->idxNeighs[iatom];

        if (numNeigh < 1 || idxNeigh < 0)
        {
            continue;
        }

        // element / atomnum of neighbor atoms
        for (ineigh = 0; ineigh < numNeigh; ++ineigh)
        {
            this->elementAll[ineigh + idxNeigh] = elemNeighbor[iatom][ineigh]
        }

        // positions of neighbor atoms
        for (ineigh = 0; ineigh < numNeigh; ++ineigh)
        {
            jneigh = ineigh + idxNeigh;
            numPos = this->sizePosNeighbor;
            idxPos = jneigh * numPos;

            for (ipos = 0; ipos < numPos; ++ipos)
            {
                this->posNeighborAll[ipos + idxPos] = posNeighbor[iatom][ineigh][ipos];
            }
        }
    }

    // copy memory host -> gpu
    cudaMemcpy(this->numNeighs_d,      this->numNeighs,      sizeof(int)     * lenAtoms,                         cudaMemcpyHostToDevice);
    cudaMemcpy(this->idxNeighs_d,      this->idxNeighs,      sizeof(int)     * lenAtoms,                         cudaMemcpyHostToDevice);
    cudaMemcpy(this->elementAll_d,     this->elementAll,     sizeof(gint)    * totNeigh,                         cudaMemcpyHostToDevice);
    cudaMemcpy(this->posNeighborAll_d, this->posNeighborAll, sizeof(nnpreal) * totNeigh * this->sizePosNeighbor, cudaMemcpyHostToDevice);

    /*
     * radial part
     *   imode   -> threadIdx.x + blockIdx.y * blockDim.x
     *   ineigh1 -> threadIdx.y
     *   iatom   -> blockIdx.x
     *   (num of mode batch) -> blockDim.x
     *   (idx of mode batch) -> blockIdx.y
     */
    if (this->sizeRad > 0)
    {
        this->getSizeOfModeBatchs(&numModeBatchs, &modesPerBatch, this->sizeRad, maxNeigh);

        block = dim3(modesPerBatch, maxNeigh, 1);
        grid  = dim3(lenAtoms, numModeBatchs, 1);

        this->calculateRadial(grid, block);
    }

    /*
     * angular  part
     *   imode   -> threadIdx.x + blockIdx.y * blockDim.x
     *   ineigh1 -> threadIdx.y
     *   iatom   -> blockIdx.x
     *   (num of mode batch) -> blockDim.x
     *   (idx of mode batch) -> blockIdx.y
     */
    if (this->sizeAng > 0)
    {
        this->getSizeOfModeBatchs(&numModeBatchs, &modesPerBatch, this->sizeAng, maxNeigh);

        block = dim3(modesPerBatch, maxNeigh, 1);
        grid  = dim3(lenAtoms, numModeBatchs, 1);

        sizeShared = sizeof(gint)    * maxNeigh
                   + sizeof(nnpreal) * maxNeigh * 5;

        if (this->elemWeight)
        {
            this->calculateAnglarElemWeight(grid, block, sizeShared);
        }
        else
        {
            dimBasis = this->numElems * (this->numElems + 1) / 2;

            if (dimBasis > MAX_ELEMENT_PAIRS)
            {
                stop_by_error("too much elements for symmetry functions on GPU, please use CPU.");
            }

            this->calculateAnglarNotElemWeight(grid, block, sizeShared, dimBasis);
        }
    }

    // copy memory gpu -> host
    cudaMemcpy(this->symmDataAll, this->symmDataAll_d, sizeof(nnpreal) * this->numBasis *     totNeigh, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->symmDiffAll, this->symmDiffAll_d, sizeof(nnpreal) * this->numBasis * 3 * totNeigh, cudaMemcpyDeviceToHost);

    #pragma omp parallel for private (iatom, ineigh, jneigh, numNeigh, idxNeigh, ifree, ibase, idxBase, idxDiff)
    for (iatom = 0; iatom < lenAtoms; ++iatom)
    {
        for (ibase = 0; ibase < this->numBasis; ++ibase)
        {
            symmData[iatom][ibase] = ZERO;
        }

        for (ibase = 0; ibase < this->numBasis; ++ibase)
        {
            symmDiff[iatom][ibase + 0 * this->numBasis] = ZERO;
            symmDiff[iatom][ibase + 1 * this->numBasis] = ZERO;
            symmDiff[iatom][ibase + 2 * this->numBasis] = ZERO;
        }

        numNeigh = this->numNeighs[iatom];
        idxNeigh = this->idxNeighs[iatom];

        if (numNeigh < 1 || idxNeigh < 0)
        {
            continue;
        }

        for (ineigh = 0; ineigh < numNeigh; ++ineigh)
        {
            ifree   = 3 * (ineigh + 1);
            jneigh  = ineigh + idxNeigh;
            idxBase = jneigh * this->numBasis;

            for (ibase = 0; ibase < this->numBasis; ++ibase)
            {
                symmData[iatom][ibase] += this->symmDataAll[ibase + idxBase];
            }

            for (ibase = 0; ibase < this->numBasis; ++ibase)
            {
                idxDiff = (ibase + idxBase) * 3;
                symmDiff[iatom][ibase + 0 * this->numBasis] -= this->symmDiffAll[idxDiff + 0];
                symmDiff[iatom][ibase + 1 * this->numBasis] -= this->symmDiffAll[idxDiff + 1];
                symmDiff[iatom][ibase + 2 * this->numBasis] -= this->symmDiffAll[idxDiff + 2];
                symmDiff[iatom][ibase + (ifree + 0) * this->numBasis] = this->symmDiffAll[idxDiff + 0];
                symmDiff[iatom][ibase + (ifree + 1) * this->numBasis] = this->symmDiffAll[idxDiff + 1];
                symmDiff[iatom][ibase + (ifree + 2) * this->numBasis] = this->symmDiffAll[idxDiff + 2];
            }
        }
    }

    // check error of cuda
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        char message[512];
        sprintf(message, "cudaError: %s\n", cudaGetErrorString(error));
        stop_by_error(message);
    }
}
