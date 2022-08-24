/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func_gpu_chebyshev.h"

#define SMALL_SIN  NNPREAL(0.001)
#define SMALL_ANG  NNPREAL(0.00001)

SymmFuncGPUChebyshev::SymmFuncGPUChebyshev(int numElems, bool tanhCutFunc, bool elemWeight, int sizeRad, int sizeAng,
                                           nnpreal rcutRad, nnpreal rcutAng, int cutoffMode) :
                      SymmFuncGPU(numElems, tanhCutFunc, elemWeight, sizeRad, sizeAng, rcutRad, rcutAng, cutoffMode)
{
    // NOP
}

SymmFuncGPUChebyshev::~SymmFuncGPUChebyshev()
{
    // NOP
}

__device__ inline void chebyshevTrigonometric(nnpreal* t, nnpreal* dt, nnpreal r, int n)
{
    nnpreal k = (nnpreal) n;
    t [0]     = cos(k * r);
    dt[0]     = r < SMALL_ANG ? (k * k * (ONE - (k * k - ONE) / NNPREAL(6.0) * r * r))
              : (k * sin(k * r) / sin(r));
}

__global__ void calculateChebyshevRad(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeRad, nnpreal rcutRad, int numBasis, bool elemWeight)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh = numNeighs[iatom];

    if (imode >= sizeRad || numNeigh < 1 || ineigh1 >= numNeigh)
    {
        return;
    }

    const int idxNeigh0 = idxNeighs[iatom];
    const int idxNeigh  = ineigh1 + idxNeigh0;
    const int idxPos    = idxNeigh * sizePosNeighbor;

    const int jbase   = elemWeight ? 0 : (((int) element[idxNeigh]) * sizeRad);
    const int ibase   = imode + jbase;
    const int idxData = ineigh1 + ibase * numNeigh + numBasis * idxNeigh0;
    const int idxDiff = idxData * 3;

    const nnpreal r1 = posNeighbor[0 + idxPos];

    if (r1 >= rcutRad)
    {
        symmData[idxData]     = ZERO;
        symmDiff[idxDiff + 0] = ZERO;
        symmDiff[idxDiff + 1] = ZERO;
        symmDiff[idxDiff + 2] = ZERO;
        return;
    }

    const nnpreal x1      = posNeighbor[1 + idxPos];
    const nnpreal y1      = posNeighbor[2 + idxPos];
    const nnpreal z1      = posNeighbor[3 + idxPos];
    const nnpreal fc1     = posNeighbor[4 + idxPos];
    const nnpreal dfc1dr1 = posNeighbor[5 + idxPos];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;

    const nnpreal zscale = elemWeight ? ((nnpreal) element[idxNeigh]) : ONE;

    const nnpreal scheby = acos(NNPREAL(2.0) * r1 / rcutRad - ONE);

    nnpreal phi;
    nnpreal dphi;

    chebyshevTrigonometric(&phi, &dphi, scheby, imode);

    dphi *= NNPREAL(2.0) / rcutRad / r1;
    const nnpreal dphidx1 = x1 * dphi;
    const nnpreal dphidy1 = y1 * dphi;
    const nnpreal dphidz1 = z1 * dphi;

    const nnpreal g     = zscale * phi * fc1;
    const nnpreal dgdx1 = zscale * (dphidx1 * fc1 + phi * dfc1dx1);
    const nnpreal dgdy1 = zscale * (dphidy1 * fc1 + phi * dfc1dy1);
    const nnpreal dgdz1 = zscale * (dphidz1 * fc1 + phi * dfc1dz1);

    symmData[idxData]     = g;
    symmDiff[idxDiff + 0] = dgdx1;
    symmDiff[idxDiff + 1] = dgdy1;
    symmDiff[idxDiff + 2] = dgdz1;
}

__global__ void calculateChebyshevAngEW(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeAng, nnpreal rcutAng, int numBasis, int offsetBasis)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh  = numNeighs[iatom];
    const int idxNeigh0 = idxNeighs[iatom];
    const int idxNeigh  = ineigh1 + idxNeigh0;
    const int idxPos    = idxNeigh * sizePosNeighbor;
    const int idxPos1   = ineigh1  * 5;

    if (numNeigh < 1)
    {
        return;
    }

    // set shared memory
    extern __shared__ gint totalShared[];

    gint* element0 = totalShared;

    nnpreal* posNeighbor0 = (nnpreal*) (&element0[numNeigh]);

    if (jmode == 0 && ineigh1 < numNeigh)
    {
        element0[ineigh1] = element[idxNeigh];

        posNeighbor0[0 + idxPos1] = posNeighbor[0 + idxPos];
        posNeighbor0[1 + idxPos1] = posNeighbor[1 + idxPos];
        posNeighbor0[2 + idxPos1] = posNeighbor[2 + idxPos];
        posNeighbor0[3 + idxPos1] = posNeighbor[3 + idxPos];
        posNeighbor0[4 + idxPos1] = posNeighbor[6 + idxPos];
    }

    __syncthreads();

    if (imode >= sizeAng || ineigh1 >= numNeigh)
    {
        return;
    }

    const int ibase   = imode + offsetBasis;
    const int idxData = ineigh1 + ibase * numNeigh + numBasis * idxNeigh0;
    const int idxDiff = idxData * 3;

    const nnpreal r1 = posNeighbor0[0 + idxPos1];

    if (r1 >= rcutAng)
    {
        symmData[idxData]     = ZERO;
        symmDiff[idxDiff + 0] = ZERO;
        symmDiff[idxDiff + 1] = ZERO;
        symmDiff[idxDiff + 2] = ZERO;
        return;
    }

    const nnpreal x1      = posNeighbor0[1 + idxPos1];
    const nnpreal y1      = posNeighbor0[2 + idxPos1];
    const nnpreal z1      = posNeighbor0[3 + idxPos1];
    const nnpreal fc1     = posNeighbor0[4 + idxPos1];
    const nnpreal dfc1dr1 = posNeighbor [7 + idxPos ];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;
    const nnpreal zanum1  = (nnpreal) element0[ineigh1];

    int ineigh2;
    int idxPos2;

    nnpreal r2;
    nnpreal x2;
    nnpreal y2;
    nnpreal z2;
    nnpreal fc2;
    nnpreal zanum2;
    nnpreal zscale;

    nnpreal cos0;
    nnpreal sin0;
    nnpreal coef0;
    nnpreal coef1;
    nnpreal coef2;
    nnpreal tht;
    nnpreal dthtdx1;
    nnpreal dthtdy1;
    nnpreal dthtdz1;

    nnpreal scheby;
    nnpreal phi;
    nnpreal dphidth;
    nnpreal dphidx1;
    nnpreal dphidy1;
    nnpreal dphidz1;

    nnpreal g;
    nnpreal dgdx1;
    nnpreal dgdy1;
    nnpreal dgdz1;

    nnpreal symmData0 = ZERO;
    nnpreal symmDiffX = ZERO;
    nnpreal symmDiffY = ZERO;
    nnpreal symmDiffZ = ZERO;

    for (ineigh2 = 0; ineigh2 < numNeigh; ++ineigh2)
    {
        idxPos2 = ineigh2 * 5;
        r2 = posNeighbor0[0 + idxPos2];

        if (ineigh1 != ineigh2 && r2 < rcutAng)
        {
            x2     = posNeighbor0[1 + idxPos2];
            y2     = posNeighbor0[2 + idxPos2];
            z2     = posNeighbor0[3 + idxPos2];
            fc2    = posNeighbor0[4 + idxPos2];
            zanum2 = (nnpreal) element0[ineigh2];
            zscale = sqrt(zanum1 * zanum2);

            cos0 = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            cos0 = cos0 >  ONE ?  ONE : cos0;
            cos0 = cos0 < -ONE ? -ONE : cos0;
            sin0 = sqrt(ONE - cos0 * cos0);
            sin0 = sin0 < SMALL_SIN ? SMALL_SIN : sin0;

            coef0   =  ONE / r1 / r2;
            coef1   = cos0 / r1 / r1;
            coef2   = -ONE / sin0;
            tht     = acos(cos0);
            dthtdx1 = (coef0 * x2 - coef1 * x1) * coef2;
            dthtdy1 = (coef0 * y2 - coef1 * y1) * coef2;
            dthtdz1 = (coef0 * z2 - coef1 * z1) * coef2;

            scheby  = NNPREAL(2.0) * tht / PI - ONE;
            scheby  = acos(scheby);

            chebyshevTrigonometric(&phi, &dphidth, scheby, imode);

            dphidth *= NNPREAL(2.0) / PI;
            dphidx1 = dphidth * dthtdx1;
            dphidy1 = dphidth * dthtdy1;
            dphidz1 = dphidth * dthtdz1;

            g     = zscale * phi * fc1 * fc2;
            dgdx1 = zscale * (dphidx1 * fc1 + phi * dfc1dx1) * fc2;
            dgdy1 = zscale * (dphidy1 * fc1 + phi * dfc1dy1) * fc2;
            dgdz1 = zscale * (dphidz1 * fc1 + phi * dfc1dz1) * fc2;

            symmData0 += g;
            symmDiffX += dgdx1;
            symmDiffY += dgdy1;
            symmDiffZ += dgdz1;
        }
    }

    symmData[idxData]     = symmData0 * NNPREAL(0.5);
    symmDiff[idxDiff + 0] = symmDiffX;
    symmDiff[idxDiff + 1] = symmDiffY;
    symmDiff[idxDiff + 2] = symmDiffZ;
}

__global__ void calculateChebyshevAngNotEW(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeAng, nnpreal rcutAng, int numBasis, int offsetBasis, int dimBasis)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh  = numNeighs[iatom];
    const int idxNeigh0 = idxNeighs[iatom];
    const int idxNeigh  = ineigh1 + idxNeigh0;
    const int idxPos    = idxNeigh * sizePosNeighbor;
    const int idxPos1   = ineigh1  * 5;
    const int idxData0  = ineigh1 + numBasis * idxNeigh0;

    if (numNeigh < 1)
    {
        return;
    }

    // set shared memory
    extern __shared__ gint totalShared[];

    gint* element0 = totalShared;

    nnpreal* posNeighbor0 = (nnpreal*) (&element0[numNeigh]);

    if (jmode == 0 && ineigh1 < numNeigh)
    {
        element0[ineigh1] = element[idxNeigh];

        posNeighbor0[0 + idxPos1] = posNeighbor[0 + idxPos];
        posNeighbor0[1 + idxPos1] = posNeighbor[1 + idxPos];
        posNeighbor0[2 + idxPos1] = posNeighbor[2 + idxPos];
        posNeighbor0[3 + idxPos1] = posNeighbor[3 + idxPos];
        posNeighbor0[4 + idxPos1] = posNeighbor[6 + idxPos];
    }

    __syncthreads();

    if (imode >= sizeAng || ineigh1 >= numNeigh)
    {
        return;
    }

    int ibase;
    int jbase;
    int kbase;
    int idxData;
    int idxDiff;

    const nnpreal r1 = posNeighbor0[0 + idxPos1];

    if (r1 >= rcutAng)
    {
        for (kbase = 0; kbase < dimBasis; ++kbase)
        {
            jbase   = kbase * sizeAng;
            ibase   = imode + jbase + offsetBasis;
            idxData = ibase * numNeigh + idxData0;
            idxDiff = idxData * 3;

            symmData[idxData]     = ZERO;
            symmDiff[idxDiff + 0] = ZERO;
            symmDiff[idxDiff + 1] = ZERO;
            symmDiff[idxDiff + 2] = ZERO;
        }

        return;
    }

    const nnpreal x1      = posNeighbor0[1 + idxPos1];
    const nnpreal y1      = posNeighbor0[2 + idxPos1];
    const nnpreal z1      = posNeighbor0[3 + idxPos1];
    const nnpreal fc1     = posNeighbor0[4 + idxPos1];
    const nnpreal dfc1dr1 = posNeighbor [7 + idxPos ];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;
    const int     ielem1  = (int) element0[ineigh1];

    int  ineigh2;
    int  idxPos2;

    nnpreal r2;
    nnpreal x2;
    nnpreal y2;
    nnpreal z2;
    nnpreal fc2;
    int     ielem2;
    int     jelem1;
    int     jelem2;

    nnpreal cos0;
    nnpreal sin0;
    nnpreal coef0;
    nnpreal coef1;
    nnpreal coef2;
    nnpreal tht;
    nnpreal dthtdx1;
    nnpreal dthtdy1;
    nnpreal dthtdz1;

    nnpreal scheby;
    nnpreal phi;
    nnpreal dphidth;
    nnpreal dphidx1;
    nnpreal dphidy1;
    nnpreal dphidz1;

    nnpreal g;
    nnpreal dgdx1;
    nnpreal dgdy1;
    nnpreal dgdz1;

    nnpreal symmData0[MAX_ELEMENT_PAIRS];
    nnpreal symmDiff0[MAX_ELEMENT_PAIRS3];

    for (kbase = 0; kbase < dimBasis; ++kbase)
    {
        symmData0[kbase]         = ZERO;
        symmDiff0[kbase * 3 + 0] = ZERO;
        symmDiff0[kbase * 3 + 1] = ZERO;
        symmDiff0[kbase * 3 + 2] = ZERO;
    }

    for (ineigh2 = 0; ineigh2 < numNeigh; ++ineigh2)
    {
        idxPos2 = ineigh2 * 5;
        r2 = posNeighbor0[0 + idxPos2];

        if (ineigh1 != ineigh2 && r2 < rcutAng)
        {
            x2     = posNeighbor0[1 + idxPos2];
            y2     = posNeighbor0[2 + idxPos2];
            z2     = posNeighbor0[3 + idxPos2];
            fc2    = posNeighbor0[4 + idxPos2];
            ielem2 = (int) element0[ineigh2];

            cos0 = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            cos0 = cos0 >  ONE ?  ONE : cos0;
            cos0 = cos0 < -ONE ? -ONE : cos0;
            sin0 = sqrt(ONE - cos0 * cos0);
            sin0 = sin0 < SMALL_SIN ? SMALL_SIN : sin0;

            coef0   =  ONE / r1 / r2;
            coef1   = cos0 / r1 / r1;
            coef2   = -ONE / sin0;
            tht     = acos(cos0);
            dthtdx1 = (coef0 * x2 - coef1 * x1) * coef2;
            dthtdy1 = (coef0 * y2 - coef1 * y1) * coef2;
            dthtdz1 = (coef0 * z2 - coef1 * z1) * coef2;

            scheby  = NNPREAL(2.0) * tht / PI - ONE;
            scheby  = acos(scheby);

            chebyshevTrigonometric(&phi, &dphidth, scheby, imode);

            dphidth *= NNPREAL(2.0) / PI;
            dphidx1 = dphidth * dthtdx1;
            dphidy1 = dphidth * dthtdy1;
            dphidz1 = dphidth * dthtdz1;

            g     = phi * fc1 * fc2;
            dgdx1 = (dphidx1 * fc1 + phi * dfc1dx1) * fc2;
            dgdy1 = (dphidy1 * fc1 + phi * dfc1dy1) * fc2;
            dgdz1 = (dphidz1 * fc1 + phi * dfc1dz1) * fc2;

            jelem1  = min(ielem1, ielem2);
            jelem2  = max(ielem1, ielem2);
            kbase   = (jelem1 + jelem2 * (jelem2 + 1) / 2);

            symmData0[kbase]         += g;
            symmDiff0[kbase * 3 + 0] += dgdx1;
            symmDiff0[kbase * 3 + 1] += dgdy1;
            symmDiff0[kbase * 3 + 2] += dgdz1;
        }
    }

    for (kbase = 0; kbase < dimBasis; ++kbase)
    {
        jbase   = kbase * sizeAng;
        ibase   = imode + jbase + offsetBasis;
        idxData = ibase * numNeigh + idxData0;
        idxDiff = idxData * 3;

        symmData[idxData]     = symmData0[kbase] * NNPREAL(0.5);
        symmDiff[idxDiff + 0] = symmDiff0[kbase * 3 + 0];
        symmDiff[idxDiff + 1] = symmDiff0[kbase * 3 + 1];
        symmDiff[idxDiff + 2] = symmDiff0[kbase * 3 + 2];
    }
}

void SymmFuncGPUChebyshev::calculateRadial(dim3 grid, dim3 block)
{
    calculateChebyshevRad<<<grid, block>>>(
    this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
    this->symmDataAll_d, this->symmDiffAll_d, this->sizeRad, this->rcutRad, this->numBasis, this->elemWeight);
}

void SymmFuncGPUChebyshev::calculateAnglarElemWeight(dim3 grid, dim3 block, size_t sizeShared)
{
    calculateChebyshevAngEW<<<grid, block, sizeShared>>>(
    this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
    this->symmDataAll_d, this->symmDiffAll_d, this->sizeAng, this->rcutAng, this->numBasis, this->numRadBasis);
}

void SymmFuncGPUChebyshev::calculateAnglarNotElemWeight(dim3 grid, dim3 block, size_t sizeShared, int dimBasis)
{
    calculateChebyshevAngNotEW<<<grid, block, sizeShared>>>(
    this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
    this->symmDataAll_d, this->symmDiffAll_d, this->sizeAng, this->rcutAng, this->numBasis, this->numRadBasis, dimBasis);
}

