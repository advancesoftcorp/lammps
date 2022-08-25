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

    this->transDiff  = true;

#ifdef SYMMDIFF_HIDDEN
    this->hiddenDiff = true;
#else
    this->hiddenDiff = false;
#endif

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

    this->sizeLenAtoms  = 0;
    this->sizeMaxAtoms  = 0;
    this->sizeTotNeigh1 = 0;
    this->sizeTotNeigh2 = 0;
    this->sizeFullNeigh = 0;

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
    this->symmDataSum      = nullptr;
    this->symmDataSum_d    = nullptr;
    this->symmDataAll_d    = nullptr;
    this->symmDiffAll      = nullptr;
    this->symmDiffAll_d    = nullptr;
    this->symmDiffFull_d   = nullptr;
    this->symmGrad         = nullptr;
    this->symmGrad_d       = nullptr;
    this->forceData        = nullptr;
    this->forceData_d      = nullptr;

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
    if (this->symmDataSum      != nullptr) cudaFreeHost(this->symmDataSum);
    if (this->symmDataSum_d    != nullptr) cudaFree    (this->symmDataSum_d);
    if (this->symmDataAll_d    != nullptr) cudaFree    (this->symmDataAll_d);
    if (this->symmDiffAll      != nullptr) cudaFreeHost(this->symmDiffAll);
    if (this->symmDiffAll_d    != nullptr) cudaFree    (this->symmDiffAll_d);
    if (this->symmDiffFull_d   != nullptr) cudaFree    (this->symmDiffFull_d);
    if (this->symmGrad         != nullptr) cudaFreeHost(this->symmGrad);
    if (this->symmGrad_d       != nullptr) cudaFree    (this->symmGrad);
    if (this->forceData        != nullptr) cudaFreeHost(this->forceData);
    if (this->forceData_d      != nullptr) cudaFree    (this->forceData_d);
}

__global__ void sumupSymmData(int* numNeighs, int* idxNeighs, nnpreal* symmData, nnpreal* symmDataSum)
{
    const int iatom    = blockIdx.x;
    const int ibase    = threadIdx.x;
    const int numBasis = blockDim.x;
    const int numNeigh = numNeighs[iatom];
    const int idxNeigh = idxNeighs[iatom];
    const int idxData  = ibase * numNeigh + numBasis * idxNeigh;

    int ineigh;

    nnpreal symmData0 = ZERO;

    for (ineigh = 0; ineigh < numNeigh; ++ineigh)
    {
        symmData0 += symmData[ineigh + idxData];
    }

    symmDataSum[ibase + iatom * numBasis] = symmData0;
}

__global__ void multSymmDiff(int* numNeighs, int* idxNeighs,
                             nnpreal* symmDiff, nnpreal* symmGrad, nnpreal** forceData, int numBasis)
{
    const int iatom    = blockIdx.x;
    const int ineigh   = threadIdx.x;
    const int numNeigh = numNeighs[iatom];
    const int idxNeigh = idxNeighs[iatom];
    const int idxData0 = ineigh + numBasis * idxNeigh;
    const int idxBasis = iatom * numBasis;
    const int idxForce = (ineigh + idxNeigh) * 3;

    if (ineigh >= numNeigh)
    {
        return;
    }

    int ibase;
    int idxData;
    int idxDiff;

    nnpreal symmGrad0;

    nnpreal forceX = ZERO;
    nnpreal forceY = ZERO;
    nnpreal forceZ = ZERO;

    for (ibase = 0; ibase < numBasis; ++ibase)
    {
        idxData = ibase * numNeigh + idxData0;
        idxDiff = idxData * 3;

        symmGrad0 = symmGrad[ibase + idxBasis];

        forceX += symmDiff[idxDiff + 0] * symmGrad0;
        forceY += symmDiff[idxDiff + 1] * symmGrad0;
        forceZ += symmDiff[idxDiff + 2] * symmGrad0;
    }

    forceData[idxForce + 0] = forceX;
    forceData[idxForce + 1] = forceY;
    forceData[idxForce + 2] = forceZ;
}

void SymmFuncGPU::calculate(int lenAtoms, int* numNeighbor, int* idxNeighbor, int** elemNeighbor, nnpreal*** posNeighbor,
                            nnpreal* symmData, nnpreal* symmDiff)
{
    if (lenAtoms < 0)
    {
        stop_by_error("#atoms is not positive.");
    }

    if (numNeighbor == nullptr || idxNeighbor == nullptr || elemNeighbor == nullptr || posNeighbor == nullptr)
    {
        stop_by_error("neighbor is null.");
    }

    if (symmData == nullptr)
    {
        stop_by_error("symmData is null.");
    }

#ifndef SYMMDIFF_HIDDEN
    if (symmDiff == nullptr)
    {
        stop_by_error("symmDiff is null.");
    }
#endif

    if (this->numBasis > this->maxThreadsPerBlock)
    {
        stop_by_error("too less #threads a block for GPU (#threads < numBasis).");
    }

    // define varialbes
    int iatom;
    int ineigh, jneigh;
    int idata;
    int ipos;

    int numNeigh;
    int idxNeigh;
    int maxNeigh;
    int totNeigh;

    int numData;

    int numPos;
    int idxPos;

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
    totNeigh = numNeighbor[lenAtoms - 1] + idxNeighbor[lenAtoms - 1] - idxNeighbor[0];

    #pragma omp parallel for private(iatom, numNeigh, idxNeigh) reduction(max:maxNeigh)
    for (iatom = 0; iatom < lenAtoms; ++iatom)
    {
        numNeigh = numNeighbor[iatom];
        idxNeigh = idxNeighbor[iatom] - idxNeighbor[0];
        this->numNeighs[iatom] = numNeigh;
        this->idxNeighs[iatom] = idxNeigh;

        maxNeigh  = max(maxNeigh, numNeigh);
    }

    if (maxNeigh > this->maxThreadsPerBlock)
    {
        stop_by_error("too less #threads a block for GPU (#threads < maxNeigh).");
    }

    if (maxNeigh < 1 || totNeigh < 1)
    {
        // because this->numNeighs[iatom] is always 0,
        // there is no need to do symmDiff = ZERO.

        numData = lenAtoms * this->numBasis;

        #pragma omp parallel for private (idata)
        for (idata = 0; idata < numData; ++idata)
        {
            symmData[idata] = ZERO;
        }

        return;
    }

    // allocate memory about totNeigh
    if (this->sizeTotNeigh1 < totNeigh)
    {
        if (this->elementAll       != nullptr) cudaFreeHost(this->elementAll);
        if (this->elementAll_d     != nullptr) cudaFree    (this->elementAll_d);
        if (this->posNeighborAll   != nullptr) cudaFreeHost(this->posNeighborAll);
        if (this->posNeighborAll_d != nullptr) cudaFree    (this->posNeighborAll_d);
        if (this->symmDataSum      != nullptr) cudaFreeHost(this->symmDataSum);
        if (this->symmDiffAll      != nullptr) cudaFreeHost(this->symmDiffAll);
        if (this->symmDataSum_d    != nullptr) cudaFree    (this->symmDataSum_d);
        if (this->symmDataAll_d    != nullptr) cudaFree    (this->symmDataAll_d);
        if (this->symmDiffAll_d    != nullptr) cudaFree    (this->symmDiffAll_d);

        cudaMallocHost(&(this->elementAll),       sizeof(gint)    * totNeigh);
        cudaMalloc    (&(this->elementAll_d),     sizeof(gint)    * totNeigh);
        cudaMallocHost(&(this->posNeighborAll),   sizeof(nnpreal) * this->sizePosNeighbor * totNeigh);
        cudaMalloc    (&(this->posNeighborAll_d), sizeof(nnpreal) * this->sizePosNeighbor * totNeigh);
#ifndef SYMMFUNC_DIRECT_COPY
        cudaMallocHost(&(this->symmDataSum),      sizeof(nnpreal)     * lenAtoms * this->numBasis);
#ifndef SYMMDIFF_HIDDEN
        cudaMallocHost(&(this->symmDiffAll),      sizeof(nnpreal) * 3 * totNeigh * this->numBasis);
#endif
#endif
        cudaMalloc    (&(this->symmDataSum_d),    sizeof(nnpreal)     * lenAtoms * this->numBasis);
        cudaMalloc    (&(this->symmDataAll_d),    sizeof(nnpreal)     * totNeigh * this->numBasis);
#ifndef SYMMDIFF_HIDDEN
        cudaMalloc    (&(this->symmDiffAll_d),    sizeof(nnpreal) * 3 * totNeigh * this->numBasis);
#endif

        this->sizeTotNeigh1 = totNeigh;
    }

    // serialize all data of neighbors
    #pragma omp parallel for private (iatom, ineigh, jneigh, numNeigh, idxNeigh, ipos, numPos, idxPos)
    for (iatom = 0; iatom < lenAtoms; ++iatom)
    {
        numNeigh = this->numNeighs[iatom];
        idxNeigh = this->idxNeighs[iatom];

        // element / atomnum of neighbor atoms
        for (ineigh = 0; ineigh < numNeigh; ++ineigh)
        {
            this->elementAll[ineigh + idxNeigh] = elemNeighbor[iatom][ineigh];
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

        this->calculateRadial(grid, block, idxNeighbor[0]);
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
            this->calculateAnglarElemWeight(grid, block, sizeShared, idxNeighbor[0]);
        }
        else
        {
            dimBasis = this->numElems * (this->numElems + 1) / 2;

            if (dimBasis > MAX_ELEMENT_PAIRS)
            {
                stop_by_error("too much elements for symmetry functions on GPU, please use CPU.");
            }

            this->calculateAnglarNotElemWeight(grid, block, sizeShared, idxNeighbor[0], dimBasis);
        }
    }

    // sum up symmData
    block = dim3(this->numBasis, 1, 1);
    grid  = dim3(lenAtoms, 1, 1);

    sumupSymmData<<<grid, block>>>(
                 this->numNeighs_d, this->idxNeighs_d, this->symmDataAll_d, this->symmDataSum_d);

    // copy memory gpu -> host
#ifdef SYMMFUNC_DIRECT_COPY
    cudaMemcpy(symmData, this->symmDataSum_d, sizeof(nnpreal)     * lenAtoms * this->numBasis, cudaMemcpyDeviceToHost);
#ifndef SYMMDIFF_HIDDEN
    cudaMemcpy(symmDiff, this->symmDiffAll_d, sizeof(nnpreal) * 3 * totNeigh * this->numBasis, cudaMemcpyDeviceToHost);
#endif
#else
    cudaMemcpy(this->symmDataSum, this->symmDataSum_d, sizeof(nnpreal)     * lenAtoms * this->numBasis, cudaMemcpyDeviceToHost);
    memcpy    (      symmData,    this->symmDataSum,   sizeof(nnpreal)     * lenAtoms * this->numBasis);
#ifndef SYMMDIFF_HIDDEN
    cudaMemcpy(this->symmDiffAll, this->symmDiffAll_d, sizeof(nnpreal) * 3 * totNeigh * this->numBasis, cudaMemcpyDeviceToHost);
    memcpy    (      symmDiff,    this->symmDiffAll,   sizeof(nnpreal) * 3 * totNeigh * this->numBasis);
#endif
#endif

    // check error of cuda
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        char message[512];
        sprintf(message, "cudaError: %s\n", cudaGetErrorString(error));
        stop_by_error(message);
    }
}

void SymmFuncGPU::allocHiddenDiff(int maxAtoms, int fullNeigh)
{
#ifdef SYMMDIFF_HIDDEN
    if (maxAtoms < 0)
    {
        stop_by_error("max of atoms is not positive.");
    }

    // allocate memory about maxAtoms
    if (this->sizeMaxAtoms < maxAtoms)
    {
        if (this->symmGrad   != nullptr) cudaFreeHost(this->symmGrad);
        if (this->symmGrad_d != nullptr) cudaFree    (this->symmGrad_d);

        cudaMallocHost(&(this->symmGrad),   sizeof(int) * maxAtoms * this->numBasis);
        cudaMalloc    (&(this->symmGrad_d), sizeof(int) * maxAtoms * this->numBasis);

        this->sizeMaxAtoms = maxAtoms;
    }

    if (fullNeigh < 1)
    {
        return;
    }

    // allocate memory about fullNeigh
    if (this->sizeFullNeigh < fullNeigh)
    {
        if (this->symmDiffFull_d != nullptr) cudaFree(this->symmDiffFull_d);

        cudaMalloc(&(this->symmDiffFull_d), sizeof(nnpreal) * 3 * fullNeigh * this->numBasis);

        this->sizeFullNeigh = fullNeigh;
    }
#endif
}

void SymmFuncGPU::driveHiddenDiff(int lenAtoms, int* numNeighbor, int* idxNeighbor, nnpreal* forceData)
{
#ifdef SYMMDIFF_HIDDEN
    if (lenAtoms < 0)
    {
        stop_by_error("#atoms is not positive.");
    }

    if (numNeighbor == nullptr || idxNeighbor == nullptr)
    {
        stop_by_error("neighbor is null.");
    }

    if (forceData == nullptr)
    {
        stop_by_error("forceData is null.");
    }

    // define varialbes
    int iatom;

    int numNeigh;
    int idxNeigh;
    int maxNeigh;
    int totNeigh;

    dim3 grid;
    dim3 block;

    // check sizeLenAtoms
    if (this->sizeLenAtoms < lenAtoms)
    {
        stop_by_error("size of atoms is not correct.");
    }

    // count neighbors
    maxNeigh = 0;
    totNeigh = numNeighbor[lenAtoms - 1] + idxNeighbor[lenAtoms - 1] - idxNeighbor[0];

    #pragma omp parallel for private(iatom, numNeigh, idxNeigh) reduction(max:maxNeigh)
    for (iatom = 0; iatom < lenAtoms; ++iatom)
    {
        numNeigh = numNeighbor[iatom];
        idxNeigh = idxNeighbor[iatom] - idxNeighbor[0];
        this->numNeighs[iatom] = numNeigh;
        this->idxNeighs[iatom] = idxNeigh;

        maxNeigh  = max(maxNeigh, numNeigh);
    }

    if (maxNeigh > this->maxThreadsPerBlock)
    {
        stop_by_error("too less #threads a block for GPU (#threads < maxNeigh).");
    }

    if (maxNeigh < 1 || totNeigh < 1)
    {
        // because this->numNeighs[iatom] is always 0,
        // there is no need to do forceData = ZERO.
        return;
    }

    // allocate memory about totNeigh
    if (this->sizeTotNeigh2 < totNeigh)
    {
        if (this->forceData   != nullptr) cudaFreeHost(this->forceData);
        if (this->forceData_d != nullptr) cudaFree    (this->forceData_d);

#ifndef SYMMFUNC_DIRECT_COPY
        cudaMallocHost(&(this->forceData),   sizeof(nnpreal) * 3 * totNeigh);
#endif
        cudaMalloc    (&(this->forceData_d), sizeof(nnpreal) * 3 * totNeigh);

        this->sizeTotNeigh2 = totNeigh;
    }

    // copy memory host -> gpu
    cudaMemcpy(this->numNeighs_d, this->numNeighs, sizeof(int)     * lenAtoms,                  cudaMemcpyHostToDevice);
    cudaMemcpy(this->idxNeighs_d, this->idxNeighs, sizeof(int)     * lenAtoms,                  cudaMemcpyHostToDevice);
    cudaMemcpy(this->symmGrad_d,  this->symmGrad,  sizeof(nnpreal) * lenAtoms * this->numBasis, cudaMemcpyHostToDevice);

    // forceData = symmDiff * symmGrad
    idxNeigh = idxNeighbor[0];
    idxDiff  = 3 * idxNeigh * this->numBasis;

    block = dim3(maxNeigh, 1, 1);
    grid  = dim3(lenAtoms, 1, 1);

    multSymmDiff<<<grid, block>>>(this->numNeighs_d, this->idxNeighs_d,
                                  &(this->symmDiffFull_d[idxDiff]), this->forceData_d, this->numBasis);

    // copy memory gpu -> host
#ifdef SYMMFUNC_DIRECT_COPY
    cudaMemcpy(      forceData, this->forceData_d, sizeof(nnpreal) * 3 * totNeigh, cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(this->forceData, this->forceData_d, sizeof(nnpreal) * 3 * totNeigh, cudaMemcpyDeviceToHost);
    memcpy    (      forceData, this->forceData,   sizeof(nnpreal) * 3 * totNeigh);
#endif

    // check error of cuda
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        char message[512];
        sprintf(message, "cudaError: %s\n", cudaGetErrorString(error));
        stop_by_error(message);
    }
#endif
}

