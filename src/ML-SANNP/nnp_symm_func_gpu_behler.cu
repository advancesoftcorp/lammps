/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func_gpu_behler.h"

#define CHI0_THR  NNPREAL(1.0e-6)

SymmFuncGPUBehler::SymmFuncGPUBehler(int numElems, bool tanhCutFunc, bool elemWeight, int sizeRad, int sizeAng,
                                     nnpreal rcutRad, nnpreal rcutAng, int cutoffMode) :
                   SymmFuncGPU(numElems, tanhCutFunc, elemWeight, sizeRad, sizeAng, rcutRad, rcutAng, cutoffMode)
{
    if ((this->sizeAng % 2) != 0)
    {
        stop_by_error("sizeAng of SymmFuncGPUBehler is not even.");
    }

    this->radiusEta   = nullptr;
    this->radiusShift = nullptr;

    this->angleMod    = false;
    this->angleEta    = nullptr;
    this->angleZeta   = nullptr;
    this->angleShift  = nullptr;
}

SymmFuncGPUBehler::~SymmFuncGPUBehler()
{
    if (this->radiusEta   != nullptr) cudaFree(this->radiusEta);
    if (this->radiusShift != nullptr) cudaFree(this->radiusShift);

    if (this->angleEta    != nullptr) cudaFree(this->angleEta);
    if (this->angleZeta   != nullptr) cudaFree(this->angleZeta);
    if (this->angleShift  != nullptr) cudaFree(this->angleShift);
}

__device__ inline void cutoffFunction(nnpreal* fc, nnpreal* dfcdr, nnpreal r, nnpreal rc, bool tanhCutFunc)
{
    if (tanhCutFunc)
    {
        nnpreal tanh1 = tanh(ONE - r / rc);
        nnpreal tanh2 = tanh1 * tanh1;
        fc[0]         = tanh1 * tanh2;
        dfcdr[0]      = -NNPREAL(3.0) * tanh2 * (ONE - tanh2) / rc;
    }

    else
    {
        nnpreal fac = PI / rc;
        fc[0]       =  NNPREAL(0.5) * (cos(fac * r) + ONE);
        dfcdr[0]    = -NNPREAL(0.5) * fac * sin(fac * r);
    }
}

__global__ void calculateBehlerG2(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeRad, nnpreal rcutRad,
                nnpreal* radiusEta, nnpreal* radiusShift,
                int numBasis, bool elemWeight)
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

    const int idxNeigh = ineigh1 + idxNeighs[iatom];
    const int idxPos   = idxNeigh * sizePosNeighbor;

    const nnpreal r1 = posNeighbor[0 + idxPos];

    if (r1 >= rcutRad)
    {
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

    const nnpreal zscale  = elemWeight ? ((nnpreal) element[idxNeigh]) : ONE;
    const int     jbase   = elemWeight ? 0 : (((int) element[idxNeigh]) * sizeRad);
    const int     ibase   = imode + jbase;
    const int     idxBase = ibase + idxNeigh * numBasis;
    const int     idxDiff = idxBase * 3;

    const nnpreal eta = radiusEta  [imode];
    const nnpreal rs  = radiusShift[imode];

    const nnpreal dr = r1 - rs;
    const nnpreal rr = dr * dr;

    const nnpreal gau     = exp(-eta * rr);
    const nnpreal coef    = -NNPREAL(2.0) * eta * dr / r1 * gau;
    const nnpreal dgaudx1 = x1 * coef;
    const nnpreal dgaudy1 = y1 * coef;
    const nnpreal dgaudz1 = z1 * coef;

    const nnpreal g     = zscale * gau * fc1;
    const nnpreal dgdx1 = zscale * (dgaudx1 * fc1 + gau * dfc1dx1);
    const nnpreal dgdy1 = zscale * (dgaudy1 * fc1 + gau * dfc1dy1);
    const nnpreal dgdz1 = zscale * (dgaudz1 * fc1 + gau * dfc1dz1);

    symmData[idxBase]     = g;
    symmDiff[idxDiff + 0] = dgdx1;
    symmDiff[idxDiff + 1] = dgdy1;
    symmDiff[idxDiff + 2] = dgdz1;
}

__global__ void calculateBehlerG3EW(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeAng, nnpreal rcutAng,
                nnpreal* angleEta, nnpreal* angleZeta, nnpreal* angleShift,
                bool tanhCutFunc, int numBasis, int offsetBasis)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh = numNeighs[iatom];
    const int idxNeigh = ineigh1 + idxNeighs[iatom];
    const int idxPos   = idxNeigh * sizePosNeighbor;
    const int idxPos1  = ineigh1  * 5;

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

    const nnpreal r1 = posNeighbor0[0 + idxPos1];

    if (r1 >= rcutAng)
    {
        return;
    }

    const int  sizeAng0 = sizeAng / 2;
    const int  imode0   = imode % sizeAng0;
    const int  ilambda  = imode / sizeAng0;

    const nnpreal eta     = angleEta  [imode0];
    const nnpreal rs      = angleShift[imode0];
    const nnpreal zeta    = angleZeta [imode0];
    const nnpreal zeta0   = pow(NNPREAL(2.0), ONE - zeta);
    const nnpreal lambda  = (nnpreal) (1 - 2 * ilambda);

    const nnpreal x1      = posNeighbor0[1 + idxPos1];
    const nnpreal y1      = posNeighbor0[2 + idxPos1];
    const nnpreal z1      = posNeighbor0[3 + idxPos1];
    const nnpreal fc1     = posNeighbor0[4 + idxPos1];
    const nnpreal dfc1dr1 = posNeighbor [7 + idxPos ];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;
    const nnpreal zanum1  = (nnpreal) element0[ineigh1];
    const nnpreal rr1     = (r1 - rs) * (r1 - rs);

    const int  ibase    = imode + offsetBasis;
    const int  idxBase  = ibase + idxNeigh * numBasis;
    const int  idxDiff  = idxBase * 3;

    int ineigh2;
    int idxPos2;

    nnpreal r2, r3;
    nnpreal x2, x3;
    nnpreal y2, y3;
    nnpreal z2, z3;
    nnpreal zanum2;
    nnpreal zscale;
    nnpreal rr3, rr;

    nnpreal fc2, fc3, fc0;
    nnpreal dfc3dr3;
    nnpreal dfc3dx3;
    nnpreal dfc3dy3;
    nnpreal dfc3dz3;
    nnpreal dfc0dx1, dfc0dx3;
    nnpreal dfc0dy1, dfc0dy3;
    nnpreal dfc0dz1, dfc0dz3;

    nnpreal gau;
    nnpreal dgaudx1, dgaudx3;
    nnpreal dgaudy1, dgaudy3;
    nnpreal dgaudz1, dgaudz3;

    nnpreal psi;
    nnpreal dpsidx1;
    nnpreal dpsidy1;
    nnpreal dpsidz1;

    nnpreal chi;
    nnpreal chi0;
    nnpreal dchidpsi;

    nnpreal g;
    nnpreal dgdx1, dgdx3;
    nnpreal dgdy1, dgdy3;
    nnpreal dgdz1, dgdz3;

    nnpreal coef0;
    nnpreal coef1;
    nnpreal coef3;

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
            x2 = posNeighbor0[1 + idxPos2];
            y2 = posNeighbor0[2 + idxPos2];
            z2 = posNeighbor0[3 + idxPos2];

            x3  = x1 - x2;
            y3  = y1 - y2;
            z3  = z1 - z2;
            rr3 = x3 * x3 + y3 * y3 + z3 * z3;

            psi  = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            chi0 = ONE + lambda * psi;

            if (rr3 < rcutAng * rcutAng && chi0 >= CHI0_THR)
            {
                fc2    = posNeighbor0[4 + idxPos2];
                zanum2 = (nnpreal) element0[ineigh2];
                zscale = sqrt(zanum1 * zanum2);

                r3 = sqrt(rr3);

                cutoffFunction(&fc3, &dfc3dr3, r3, rcutAng, tanhCutFunc);
                dfc3dx3 = x3 / r3 * dfc3dr3;
                dfc3dy3 = y3 / r3 * dfc3dr3;
                dfc3dz3 = z3 / r3 * dfc3dr3;

                fc0 = fc1 * fc2 * fc3;
                dfc0dx1 = dfc1dx1 * fc2 * fc3;
                dfc0dy1 = dfc1dy1 * fc2 * fc3;
                dfc0dz1 = dfc1dz1 * fc2 * fc3;
                dfc0dx3 = fc1 * fc2 * dfc3dx3;
                dfc0dy3 = fc1 * fc2 * dfc3dy3;
                dfc0dz3 = fc1 * fc2 * dfc3dz3;

                coef0   = ONE / r1 / r2;
                coef1   = psi / r1 / r1;
                dpsidx1 = coef0 * x2 - coef1 * x1;
                dpsidy1 = coef0 * y2 - coef1 * y1;
                dpsidz1 = coef0 * z2 - coef1 * z1;

                chi      = zeta0 * pow(chi0, zeta);
                dchidpsi = zeta * lambda * chi / chi0;

                rr = rr1 + (r2 - rs) * (r2 - rs) + (r3 - rs) * (r3 - rs);

                gau     = exp(-eta * rr);
                coef0   = -NNPREAL(2.0) * eta * gau;
                coef1   = coef0 * (r1 - rs) / r1;
                coef3   = coef0 * (r3 - rs) / r3;
                dgaudx1 = coef1 * x1;
                dgaudy1 = coef1 * y1;
                dgaudz1 = coef1 * z1;
                dgaudx3 = coef3 * x3;
                dgaudy3 = coef3 * y3;
                dgaudz3 = coef3 * z3;

                g     = zscale * chi * gau * fc0;
                dgdx1 = zscale * (dchidpsi * dpsidx1 * gau * fc0 + chi * dgaudx1 * fc0 + chi * gau * dfc0dx1);
                dgdy1 = zscale * (dchidpsi * dpsidy1 * gau * fc0 + chi * dgaudy1 * fc0 + chi * gau * dfc0dy1);
                dgdz1 = zscale * (dchidpsi * dpsidz1 * gau * fc0 + chi * dgaudz1 * fc0 + chi * gau * dfc0dz1);
                dgdx3 = zscale * (chi * dgaudx3 * fc0 + chi * gau * dfc0dx3);
                dgdy3 = zscale * (chi * dgaudy3 * fc0 + chi * gau * dfc0dy3);
                dgdz3 = zscale * (chi * dgaudz3 * fc0 + chi * gau * dfc0dz3);

                symmData0 += g;
                symmDiffX += dgdx1 + dgdx3;
                symmDiffY += dgdy1 + dgdy3;
                symmDiffZ += dgdz1 + dgdz3;
            }
        }
    }

    symmData[idxBase]     = symmData0 * NNPREAL(0.5);
    symmDiff[idxDiff + 0] = symmDiffX;
    symmDiff[idxDiff + 1] = symmDiffY;
    symmDiff[idxDiff + 2] = symmDiffZ;
}

__global__ void calculateBehlerG3NotEW(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeAng, nnpreal rcutAng,
                nnpreal* angleEta, nnpreal* angleZeta, nnpreal* angleShift,
                bool tanhCutFunc, int numBasis, int offsetBasis, int dimBasis)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh = numNeighs[iatom];
    const int idxNeigh = ineigh1 + idxNeighs[iatom];
    const int idxPos   = idxNeigh * sizePosNeighbor;
    const int idxPos1  = ineigh1  * 5;

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

    const nnpreal r1 = posNeighbor0[0 + idxPos1];

    if (r1 >= rcutAng)
    {
        return;
    }

    const int  sizeAng0 = sizeAng / 2;
    const int  imode0   = imode % sizeAng0;
    const int  ilambda  = imode / sizeAng0;

    const nnpreal eta     = angleEta  [imode0];
    const nnpreal rs      = angleShift[imode0];
    const nnpreal zeta    = angleZeta [imode0];
    const nnpreal zeta0   = pow(NNPREAL(2.0), ONE - zeta);
    const nnpreal lambda  = (nnpreal) (1 - 2 * ilambda);

    const nnpreal x1      = posNeighbor0[1 + idxPos1];
    const nnpreal y1      = posNeighbor0[2 + idxPos1];
    const nnpreal z1      = posNeighbor0[3 + idxPos1];
    const nnpreal fc1     = posNeighbor0[4 + idxPos1];
    const nnpreal dfc1dr1 = posNeighbor [7 + idxPos ];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;
    const nnpreal rr1     = (r1 - rs) * (r1 - rs);
    const int     ielem1  = (int) element0[ineigh1];

    int ibase;
    int jbase;
    int kbase;
    int idxBase;
    int idxDiff;

    int ineigh2;
    int idxPos2;

    nnpreal r2, r3;
    nnpreal x2, x3;
    nnpreal y2, y3;
    nnpreal z2, z3;
    nnpreal rr3, rr;
    int     ielem2;
    int     jelem1;
    int     jelem2;

    nnpreal fc2, fc3, fc0;
    nnpreal dfc3dr3;
    nnpreal dfc3dx3;
    nnpreal dfc3dy3;
    nnpreal dfc3dz3;
    nnpreal dfc0dx1, dfc0dx3;
    nnpreal dfc0dy1, dfc0dy3;
    nnpreal dfc0dz1, dfc0dz3;

    nnpreal gau;
    nnpreal dgaudx1, dgaudx3;
    nnpreal dgaudy1, dgaudy3;
    nnpreal dgaudz1, dgaudz3;

    nnpreal psi;
    nnpreal dpsidx1;
    nnpreal dpsidy1;
    nnpreal dpsidz1;

    nnpreal chi;
    nnpreal chi0;
    nnpreal dchidpsi;

    nnpreal g;
    nnpreal dgdx1, dgdx3;
    nnpreal dgdy1, dgdy3;
    nnpreal dgdz1, dgdz3;

    nnpreal coef0;
    nnpreal coef1;
    nnpreal coef3;

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
            x2 = posNeighbor0[1 + idxPos2];
            y2 = posNeighbor0[2 + idxPos2];
            z2 = posNeighbor0[3 + idxPos2];

            x3  = x1 - x2;
            y3  = y1 - y2;
            z3  = z1 - z2;
            rr3 = x3 * x3 + y3 * y3 + z3 * z3;

            psi  = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            chi0 = ONE + lambda * psi;

            if (rr3 < rcutAng * rcutAng && chi0 >= CHI0_THR)
            {
                fc2    = posNeighbor0[4 + idxPos2];
                ielem2 = (int) element0[ineigh2];

                r3 = sqrt(rr3);

                cutoffFunction(&fc3, &dfc3dr3, r3, rcutAng, tanhCutFunc);
                dfc3dx3 = x3 / r3 * dfc3dr3;
                dfc3dy3 = y3 / r3 * dfc3dr3;
                dfc3dz3 = z3 / r3 * dfc3dr3;

                fc0 = fc1 * fc2 * fc3;
                dfc0dx1 = dfc1dx1 * fc2 * fc3;
                dfc0dy1 = dfc1dy1 * fc2 * fc3;
                dfc0dz1 = dfc1dz1 * fc2 * fc3;
                dfc0dx3 = fc1 * fc2 * dfc3dx3;
                dfc0dy3 = fc1 * fc2 * dfc3dy3;
                dfc0dz3 = fc1 * fc2 * dfc3dz3;

                coef0   = ONE / r1 / r2;
                coef1   = psi / r1 / r1;
                dpsidx1 = coef0 * x2 - coef1 * x1;
                dpsidy1 = coef0 * y2 - coef1 * y1;
                dpsidz1 = coef0 * z2 - coef1 * z1;

                chi      = zeta0 * pow(chi0, zeta);
                dchidpsi = zeta * lambda * chi / chi0;

                rr = rr1 + (r2 - rs) * (r2 - rs) + (r3 - rs) * (r3 - rs);

                gau     = exp(-eta * rr);
                coef0   = -NNPREAL(2.0) * eta * gau;
                coef1   = coef0 * (r1 - rs) / r1;
                coef3   = coef0 * (r3 - rs) / r3;
                dgaudx1 = coef1 * x1;
                dgaudy1 = coef1 * y1;
                dgaudz1 = coef1 * z1;
                dgaudx3 = coef3 * x3;
                dgaudy3 = coef3 * y3;
                dgaudz3 = coef3 * z3;

                g     = chi * gau * fc0;
                dgdx1 = dchidpsi * dpsidx1 * gau * fc0 + chi * dgaudx1 * fc0 + chi * gau * dfc0dx1;
                dgdy1 = dchidpsi * dpsidy1 * gau * fc0 + chi * dgaudy1 * fc0 + chi * gau * dfc0dy1;
                dgdz1 = dchidpsi * dpsidz1 * gau * fc0 + chi * dgaudz1 * fc0 + chi * gau * dfc0dz1;
                dgdx3 = chi * dgaudx3 * fc0 + chi * gau * dfc0dx3;
                dgdy3 = chi * dgaudy3 * fc0 + chi * gau * dfc0dy3;
                dgdz3 = chi * dgaudz3 * fc0 + chi * gau * dfc0dz3;

                jelem1  = min(ielem1, ielem2);
                jelem2  = max(ielem1, ielem2);
                kbase   = (jelem1 + jelem2 * (jelem2 + 1) / 2);

                symmData0[kbase]         += g;
                symmDiff0[kbase * 3 + 0] += dgdx1 + dgdx3;
                symmDiff0[kbase * 3 + 1] += dgdy1 + dgdy3;
                symmDiff0[kbase * 3 + 2] += dgdz1 + dgdz3;
            }
        }
    }

    for (kbase = 0; kbase < dimBasis; ++kbase)
    {
        jbase   = kbase * sizeAng;
        ibase   = imode + jbase + offsetBasis;
        idxBase = ibase + idxNeigh * numBasis;
        idxDiff = idxBase * 3;

        symmData[idxBase]     = symmData0[kbase] * NNPREAL(0.5);
        symmDiff[idxDiff + 0] = symmDiff0[kbase * 3 + 0];
        symmDiff[idxDiff + 1] = symmDiff0[kbase * 3 + 1];
        symmDiff[idxDiff + 2] = symmDiff0[kbase * 3 + 2];
    }
}

__global__ void calculateBehlerG4EW(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeAng, nnpreal rcutAng,
                nnpreal* angleEta, nnpreal* angleZeta, nnpreal* angleShift,
                int numBasis, int offsetBasis)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh = numNeighs[iatom];
    const int idxNeigh = ineigh1 + idxNeighs[iatom];
    const int idxPos   = idxNeigh * sizePosNeighbor;
    const int idxPos1  = ineigh1  * 5;

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

    const nnpreal r1 = posNeighbor0[0 + idxPos1];

    if (r1 >= rcutAng)
    {
        return;
    }

    const int  sizeAng0 = sizeAng / 2;
    const int  imode0   = imode % sizeAng0;
    const int  ilambda  = imode / sizeAng0;

    const nnpreal eta     = angleEta  [imode0];
    const nnpreal rs      = angleShift[imode0];
    const nnpreal zeta    = angleZeta [imode0];
    const nnpreal zeta0   = pow(NNPREAL(2.0), ONE - zeta);
    const nnpreal lambda  = (nnpreal) (1 - 2 * ilambda);

    const nnpreal x1      = posNeighbor0[1 + idxPos1];
    const nnpreal y1      = posNeighbor0[2 + idxPos1];
    const nnpreal z1      = posNeighbor0[3 + idxPos1];
    const nnpreal fc1     = posNeighbor0[4 + idxPos1];
    const nnpreal dfc1dr1 = posNeighbor [7 + idxPos ];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;
    const nnpreal zanum1  = (nnpreal) element0[ineigh1];
    const nnpreal rr1     = (r1 - rs) * (r1 - rs);

    const int  ibase    = imode + offsetBasis;
    const int  idxBase  = ibase + idxNeigh * numBasis;
    const int  idxDiff  = idxBase * 3;

    int ineigh2;
    int idxPos2;

    nnpreal r2;
    nnpreal x2;
    nnpreal y2;
    nnpreal z2;
    nnpreal fc2;
    nnpreal zanum2;
    nnpreal zscale;
    nnpreal rr;

    nnpreal gau;
    nnpreal dgaudx1;
    nnpreal dgaudy1;
    nnpreal dgaudz1;

    nnpreal psi;
    nnpreal dpsidx1;
    nnpreal dpsidy1;
    nnpreal dpsidz1;

    nnpreal chi;
    nnpreal chi0;
    nnpreal dchidpsi;

    nnpreal g;
    nnpreal dgdx1;
    nnpreal dgdy1;
    nnpreal dgdz1;

    nnpreal coef0;
    nnpreal coef1;

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
            x2 = posNeighbor0[1 + idxPos2];
            y2 = posNeighbor0[2 + idxPos2];
            z2 = posNeighbor0[3 + idxPos2];

            psi  = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            chi0 = ONE + lambda * psi;

            if (chi0 >= CHI0_THR)
            {
                fc2    = posNeighbor0[4 + idxPos2];
                zanum2 = (nnpreal) element0[ineigh2];
                zscale = sqrt(zanum1 * zanum2);

                coef0   = ONE / r1 / r2;
                coef1   = psi / r1 / r1;
                dpsidx1 = coef0 * x2 - coef1 * x1;
                dpsidy1 = coef0 * y2 - coef1 * y1;
                dpsidz1 = coef0 * z2 - coef1 * z1;

                chi      = zeta0 * pow(chi0, zeta);
                dchidpsi = zeta * lambda * chi / chi0;

                rr = rr1 + (r2 - rs) * (r2 - rs);

                gau     = exp(-eta * rr);
                coef0   = -NNPREAL(2.0) * eta * gau;
                coef1   = coef0 * (r1 - rs) / r1;
                dgaudx1 = coef1 * x1;
                dgaudy1 = coef1 * y1;
                dgaudz1 = coef1 * z1;

                g     = zscale * chi * gau * fc1 * fc2;
                dgdx1 = zscale * (dchidpsi * dpsidx1 * gau * fc1 + chi * dgaudx1 * fc1 + chi * gau * dfc1dx1) * fc2;
                dgdy1 = zscale * (dchidpsi * dpsidy1 * gau * fc1 + chi * dgaudy1 * fc1 + chi * gau * dfc1dy1) * fc2;
                dgdz1 = zscale * (dchidpsi * dpsidz1 * gau * fc1 + chi * dgaudz1 * fc1 + chi * gau * dfc1dz1) * fc2;

                symmData0 += g;
                symmDiffX += dgdx1;
                symmDiffY += dgdy1;
                symmDiffZ += dgdz1;
            }
        }
    }

    symmData[idxBase]     = symmData0 * NNPREAL(0.5);
    symmDiff[idxDiff + 0] = symmDiffX;
    symmDiff[idxDiff + 1] = symmDiffY;
    symmDiff[idxDiff + 2] = symmDiffZ;
}

__global__ void calculateBehlerG4NotEW(
                int* numNeighs, int* idxNeighs, gint* element, nnpreal* posNeighbor, int sizePosNeighbor,
                nnpreal* symmData, nnpreal* symmDiff, int sizeAng, nnpreal rcutAng,
                nnpreal* angleEta, nnpreal* angleZeta, nnpreal* angleShift,
                int numBasis, int offsetBasis, int dimBasis)
{
    const int iatom         = blockIdx.x;
    const int ineigh1       = threadIdx.y;
    const int modesPerBatch = blockDim.x;
    const int jmode         = threadIdx.x;
    const int imode         = jmode + blockIdx.y * modesPerBatch;

    const int numNeigh = numNeighs[iatom];
    const int idxNeigh = ineigh1 + idxNeighs[iatom];
    const int idxPos   = idxNeigh * sizePosNeighbor;
    const int idxPos1  = ineigh1  * 5;

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

    const nnpreal r1 = posNeighbor0[0 + idxPos1];

    if (r1 >= rcutAng)
    {
        return;
    }

    const int  sizeAng0 = sizeAng / 2;
    const int  imode0   = imode % sizeAng0;
    const int  ilambda  = imode / sizeAng0;

    const nnpreal eta     = angleEta  [imode0];
    const nnpreal rs      = angleShift[imode0];
    const nnpreal zeta    = angleZeta [imode0];
    const nnpreal zeta0   = pow(NNPREAL(2.0), ONE - zeta);
    const nnpreal lambda  = (nnpreal) (1 - 2 * ilambda);

    const nnpreal x1      = posNeighbor0[1 + idxPos1];
    const nnpreal y1      = posNeighbor0[2 + idxPos1];
    const nnpreal z1      = posNeighbor0[3 + idxPos1];
    const nnpreal fc1     = posNeighbor0[4 + idxPos1];
    const nnpreal dfc1dr1 = posNeighbor [7 + idxPos ];
    const nnpreal dfc1dx1 = x1 / r1 * dfc1dr1;
    const nnpreal dfc1dy1 = y1 / r1 * dfc1dr1;
    const nnpreal dfc1dz1 = z1 / r1 * dfc1dr1;
    const nnpreal rr1     = (r1 - rs) * (r1 - rs);
    const int     ielem1  = (int) element0[ineigh1];

    int ibase;
    int jbase;
    int kbase;
    int idxBase;
    int idxDiff;

    int ineigh2;
    int idxPos2;

    nnpreal r2;
    nnpreal x2;
    nnpreal y2;
    nnpreal z2;
    nnpreal fc2;
    nnpreal rr;
    int     ielem2;
    int     jelem1;
    int     jelem2;

    nnpreal gau;
    nnpreal dgaudx1;
    nnpreal dgaudy1;
    nnpreal dgaudz1;

    nnpreal psi;
    nnpreal dpsidx1;
    nnpreal dpsidy1;
    nnpreal dpsidz1;

    nnpreal chi;
    nnpreal chi0;
    nnpreal dchidpsi;

    nnpreal g;
    nnpreal dgdx1;
    nnpreal dgdy1;
    nnpreal dgdz1;

    nnpreal coef0;
    nnpreal coef1;

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
            x2 = posNeighbor0[1 + idxPos2];
            y2 = posNeighbor0[2 + idxPos2];
            z2 = posNeighbor0[3 + idxPos2];

            psi  = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            chi0 = ONE + lambda * psi;

            if (chi0 >= CHI0_THR)
            {
                fc2    = posNeighbor0[4 + idxPos2];
                ielem2 = (int) element0[ineigh2];

                coef0   = ONE / r1 / r2;
                coef1   = psi / r1 / r1;
                dpsidx1 = coef0 * x2 - coef1 * x1;
                dpsidy1 = coef0 * y2 - coef1 * y1;
                dpsidz1 = coef0 * z2 - coef1 * z1;

                chi      = zeta0 * pow(chi0, zeta);
                dchidpsi = zeta * lambda * chi / chi0;

                rr = rr1 + (r2 - rs) * (r2 - rs);

                gau     = exp(-eta * rr);
                coef0   = -NNPREAL(2.0) * eta * gau;
                coef1   = coef0 * (r1 - rs) / r1;
                dgaudx1 = coef1 * x1;
                dgaudy1 = coef1 * y1;
                dgaudz1 = coef1 * z1;

                g     = chi * gau * fc1 * fc2;
                dgdx1 = (dchidpsi * dpsidx1 * gau * fc1 + chi * dgaudx1 * fc1 + chi * gau * dfc1dx1) * fc2;
                dgdy1 = (dchidpsi * dpsidy1 * gau * fc1 + chi * dgaudy1 * fc1 + chi * gau * dfc1dy1) * fc2;
                dgdz1 = (dchidpsi * dpsidz1 * gau * fc1 + chi * dgaudz1 * fc1 + chi * gau * dfc1dz1) * fc2;

                jelem1  = min(ielem1, ielem2);
                jelem2  = max(ielem1, ielem2);
                kbase   = (jelem1 + jelem2 * (jelem2 + 1) / 2);

                symmData0[kbase]         += g;
                symmDiff0[kbase * 3 + 0] += dgdx1;
                symmDiff0[kbase * 3 + 1] += dgdy1;
                symmDiff0[kbase * 3 + 2] += dgdz1;
            }
        }
    }

    for (kbase = 0; kbase < dimBasis; ++kbase)
    {
        jbase   = kbase * sizeAng;
        ibase   = imode + jbase + offsetBasis;
        idxBase = ibase + idxNeigh * numBasis;
        idxDiff = idxBase * 3;

        symmData[idxBase]     = symmData0[kbase] * NNPREAL(0.5);
        symmDiff[idxDiff + 0] = symmDiff0[kbase * 3 + 0];
        symmDiff[idxDiff + 1] = symmDiff0[kbase * 3 + 1];
        symmDiff[idxDiff + 2] = symmDiff0[kbase * 3 + 2];
    }
}

void SymmFuncGPUBehler::calculateRadial(dim3 grid, dim3 block)
{
    calculateBehlerG2<<<grid, block>>>(
    this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
    this->symmDataAll_d, this->symmDiffAll_d, this->sizeRad, this->rcutRad,
    this->radiusEta, this->radiusShift,
    this->numBasis, this->elemWeight);
}

void SymmFuncGPUBehler::calculateAnglarElemWeight(dim3 grid, dim3 block, size_t sizeShared)
{
    if (this->angleMod)
    {
        calculateBehlerG4EW<<<grid, block, sizeShared>>>(
        this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
        this->symmDataAll_d, this->symmDiffAll_d, this->sizeAng, this->rcutAng,
        this->angleEta, this->angleZeta, this->angleShift,
        this->numBasis, this->numRadBasis);
    }
    else
    {
        calculateBehlerG3EW<<<grid, block, sizeShared>>>(
        this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
        this->symmDataAll_d, this->symmDiffAll_d, this->sizeAng, this->rcutAng,
        this->angleEta, this->angleZeta, this->angleShift,
        this->tanhCutFunc, this->numBasis, this->numRadBasis);
    }
}

void SymmFuncGPUBehler::calculateAnglarNotElemWeight(dim3 grid, dim3 block, size_t sizeShared, int dimBasis)
{
    if (this->angleMod)
    {
        calculateBehlerG4NotEW<<<grid, block, sizeShared>>>(
        this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
        this->symmDataAll_d, this->symmDiffAll_d, this->sizeAng, this->rcutAng,
        this->angleEta, this->angleZeta, this->angleShift,
        this->numBasis, this->numRadBasis, dimBasis);
    }
    else
    {
        calculateBehlerG3NotEW<<<grid, block, sizeShared>>>(
        this->numNeighs_d, this->idxNeighs_d, this->elementAll_d, this->posNeighborAll_d, this->sizePosNeighbor,
        this->symmDataAll_d, this->symmDiffAll_d, this->sizeAng, this->rcutAng,
        this->angleEta, this->angleZeta, this->angleShift,
        this->tanhCutFunc, this->numBasis, this->numRadBasis, dimBasis);
    }
}

