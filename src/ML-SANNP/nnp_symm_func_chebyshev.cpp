/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func_chebyshev.h"

#define SMALL_SIN  NNPREAL(0.001)

SymmFuncChebyshev::SymmFuncChebyshev(int numElems, bool tanhCutFunc, bool elemWeight, int sizeRad, int sizeAng,
                                     nnpreal rcutRad, nnpreal rcutAng) : SymmFunc(numElems, tanhCutFunc, elemWeight)
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
}

SymmFuncChebyshev::~SymmFuncChebyshev()
{
    // NOP
}

void SymmFuncChebyshev::calculate(int numNeighbor, int* elemNeighbor, nnpreal** posNeighbor,
                                  nnpreal* symmData, nnpreal* symmDiff)
{
    if (elemNeighbor == nullptr || posNeighbor == nullptr)
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
    const int numFree = 3 * numNeighbor;

    int ineigh1, ineigh2;
    int mneigh;

    int jelem1, jelem2;
    int ifree1, ifree2;

    int jbase;

    nnpreal x1, x2;
    nnpreal y1, y2;
    nnpreal z1, z2;
    nnpreal r1, r2, rr;

    nnpreal fc1, fc2;
    nnpreal dfc1dr1, dfc2dr2;
    nnpreal dfc1dx1, dfc2dx2;
    nnpreal dfc1dy1, dfc2dy2;
    nnpreal dfc1dz1, dfc2dz2;

    nnpreal fc0;
    nnpreal dfc0dx1, dfc0dx2;
    nnpreal dfc0dy1, dfc0dy2;
    nnpreal dfc0dz1, dfc0dz2;

    nnpreal tht;
    nnpreal dthtdx1, dthtdx2;
    nnpreal dthtdy1, dthtdy2;
    nnpreal dthtdz1, dthtdz2;

    nnpreal zanum1, zanum2;
    nnpreal zscale;

    nnpreal cos0, sin0;
    nnpreal coef0, coef1, coef2, coef3;

    nnpreal scheby;

#ifndef CHEBYSHEV_TRIGONO
    const int ncheby = max(2, max(this->sizeRad, this->sizeAng));
    nnpreal tcheby[ncheby];
    nnpreal dcheby[ncheby];
#endif

    // initialize symmetry functions
    for (int ibase = 0; ibase < this->numBasis; ++ibase)
    {
        symmData[ibase] = ZERO;
    }

    for (ifree1 = 0; ifree1 < numFree; ++ifree1)
    {
        for (int ibase = 0; ibase < this->numBasis; ++ibase)
        {
            symmDiff[ibase + ifree1 * this->numBasis] = ZERO;
        }
    }

    if (numNeighbor < 1)
    {
        return;
    }

    // radial part
    for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
    {
        r1 = posNeighbor[ineigh1][0];
        if (r1 >= this->rcutRad)
        {
            continue;
        }

        x1 = posNeighbor[ineigh1][1];
        y1 = posNeighbor[ineigh1][2];
        z1 = posNeighbor[ineigh1][3];

        ifree1 = 3 * ineigh1;

        if (this->elemWeight)
        {
            jelem1 = 0;
            zanum1 = (nnpreal) elemNeighbor[ineigh1];
        }
        else
        {
            jelem1 = elemNeighbor[ineigh1];
            zanum1 = ONE;
        }

        jbase  = jelem1 * this->sizeRad;
        zscale = zanum1;

        fc1     = posNeighbor[ineigh1][4];
        dfc1dr1 = posNeighbor[ineigh1][5];
        dfc1dx1 = x1 / r1 * dfc1dr1;
        dfc1dy1 = y1 / r1 * dfc1dr1;
        dfc1dz1 = z1 / r1 * dfc1dr1;

        scheby = NNPREAL(2.0) * r1 / this->rcutRad - ONE;
        coef0  = NNPREAL(2.0) / this->rcutRad / r1;

#ifdef CHEBYSHEV_TRIGONO
        scheby = acos(scheby);
#else
        this->chebyshevFunction(tcheby, dcheby, scheby, this->sizeRad);
#endif

        #pragma omp simd
        for (int imode = 0; imode < this->sizeRad; ++imode)
        {
#ifdef CHEBYSHEV_TRIGONO
            nnpreal phi;
            nnpreal dphi;
            this->chebyshevTrigonometric(&phi, &dphi, scheby, imode);
            dphi *= coef0;
#else
            const nnpreal phi  = tcheby[imode];
            const nnpreal dphi = dcheby[imode] * coef0;
#endif
            const nnpreal dphidx1 = x1 * dphi;
            const nnpreal dphidy1 = y1 * dphi;
            const nnpreal dphidz1 = z1 * dphi;

            const nnpreal g     = zscale * phi * fc1;
            const nnpreal dgdx1 = zscale * (dphidx1 * fc1 + phi * dfc1dx1);
            const nnpreal dgdy1 = zscale * (dphidy1 * fc1 + phi * dfc1dy1);
            const nnpreal dgdz1 = zscale * (dphidz1 * fc1 + phi * dfc1dz1);

            const int ibase = imode + jbase;

            symmData[ibase] += g;

            symmDiff[ibase + (ifree1 + 0) * this->numBasis] += dgdx1;
            symmDiff[ibase + (ifree1 + 1) * this->numBasis] += dgdy1;
            symmDiff[ibase + (ifree1 + 2) * this->numBasis] += dgdz1;
        }
    }

    if (numNeighbor < 2 || this->sizeAng < 1)
    {
        return;
    }

    // angular part
    for (ineigh2 = 0; ineigh2 < numNeighbor; ++ineigh2)
    {
        r2 = posNeighbor[ineigh2][0];
        if (r2 >= this->rcutAng)
        {
            continue;
        }

        x2 = posNeighbor[ineigh2][1];
        y2 = posNeighbor[ineigh2][2];
        z2 = posNeighbor[ineigh2][3];

        ifree2 = 3 * ineigh2;

        if (this->elemWeight)
        {
            jelem2 = 0;
            zanum2 = (nnpreal) elemNeighbor[ineigh2];
            mneigh = ineigh2;
        }
        else
        {
            jelem2 = elemNeighbor[ineigh2];
            zanum2 = ONE;
            mneigh = numNeighbor;
        }

        fc2     = posNeighbor[ineigh2][6];
        dfc2dr2 = posNeighbor[ineigh2][7];
        dfc2dx2 = x2 / r2 * dfc2dr2;
        dfc2dy2 = y2 / r2 * dfc2dr2;
        dfc2dz2 = z2 / r2 * dfc2dr2;

        for (ineigh1 = 0; ineigh1 < mneigh; ++ineigh1)
        {
            r1 = posNeighbor[ineigh1][0];
            if (r1 >= this->rcutAng)
            {
                continue;
            }

            x1 = posNeighbor[ineigh1][1];
            y1 = posNeighbor[ineigh1][2];
            z1 = posNeighbor[ineigh1][3];

            ifree1 = 3 * ineigh1;

            if (this->elemWeight)
            {
                jelem1 = 0;
                zanum1 = (nnpreal) elemNeighbor[ineigh1];
                zscale = sqrt(zanum1 * zanum2);
            }
            else
            {
                jelem1 = elemNeighbor[ineigh1];
                zanum1 = ONE;
                zscale = ONE;

                if (jelem1 > jelem2 || (jelem1 == jelem2 && ineigh1 >= ineigh2))
                {
                    continue;
                }
            }

            jbase = (jelem1 + jelem2 * (jelem2 + 1) / 2) * this->sizeAng;

            fc1     = posNeighbor[ineigh1][6];
            dfc1dr1 = posNeighbor[ineigh1][7];
            dfc1dx1 = x1 / r1 * dfc1dr1;
            dfc1dy1 = y1 / r1 * dfc1dr1;
            dfc1dz1 = z1 / r1 * dfc1dr1;

            fc0 = fc1 * fc2;
            dfc0dx1 = dfc1dx1 * fc2;
            dfc0dy1 = dfc1dy1 * fc2;
            dfc0dz1 = dfc1dz1 * fc2;
            dfc0dx2 = fc1 * dfc2dx2;
            dfc0dy2 = fc1 * dfc2dy2;
            dfc0dz2 = fc1 * dfc2dz2;

            cos0 = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            cos0 = cos0 >  ONE ?  ONE : cos0;
            cos0 = cos0 < -ONE ? -ONE : cos0;
            sin0 = sqrt(ONE - cos0 * cos0);
            sin0 = sin0 < SMALL_SIN ? SMALL_SIN : sin0;

            coef0   =  ONE / r1 / r2;
            coef1   = cos0 / r1 / r1;
            coef2   = cos0 / r2 / r2;
            coef3   = -ONE / sin0;
            tht     = acos(cos0);
            dthtdx1 = (coef0 * x2 - coef1 * x1) * coef3;
            dthtdy1 = (coef0 * y2 - coef1 * y1) * coef3;
            dthtdz1 = (coef0 * z2 - coef1 * z1) * coef3;
            dthtdx2 = (coef0 * x1 - coef2 * x2) * coef3;
            dthtdy2 = (coef0 * y1 - coef2 * y2) * coef3;
            dthtdz2 = (coef0 * z1 - coef2 * z2) * coef3;

            scheby = NNPREAL(2.0) * tht / PI - ONE;
            coef0  = NNPREAL(2.0) / PI;

#ifdef CHEBYSHEV_TRIGONO
            scheby = acos(scheby);
#else
            this->chebyshevFunction(tcheby, dcheby, scheby, this->sizeAng);
#endif

            #pragma omp simd
            for (int imode = 0; imode < this->sizeAng; ++imode)
            {
#ifdef CHEBYSHEV_TRIGONO
                nnpreal phi;
                nnpreal dphidth;
                this->chebyshevTrigonometric(&phi, &dphidth, scheby, imode);
                dphidth *= coef0;
#else
                const nnpreal phi     = tcheby[imode];
                const nnpreal dphidth = dcheby[imode] * coef0;
#endif
                const nnpreal dphidx1 = dphidth * dthtdx1;
                const nnpreal dphidy1 = dphidth * dthtdy1;
                const nnpreal dphidz1 = dphidth * dthtdz1;
                const nnpreal dphidx2 = dphidth * dthtdx2;
                const nnpreal dphidy2 = dphidth * dthtdy2;
                const nnpreal dphidz2 = dphidth * dthtdz2;

                const nnpreal g     = zscale * phi * fc0;
                const nnpreal dgdx1 = zscale * (dphidx1 * fc0 + phi * dfc0dx1);
                const nnpreal dgdy1 = zscale * (dphidy1 * fc0 + phi * dfc0dy1);
                const nnpreal dgdz1 = zscale * (dphidz1 * fc0 + phi * dfc0dz1);
                const nnpreal dgdx2 = zscale * (dphidx2 * fc0 + phi * dfc0dx2);
                const nnpreal dgdy2 = zscale * (dphidy2 * fc0 + phi * dfc0dy2);
                const nnpreal dgdz2 = zscale * (dphidz2 * fc0 + phi * dfc0dz2);

                const int ibase = this->numRadBasis + imode + jbase;

                symmData[ibase] += g;

                symmDiff[ibase + (ifree1 + 0) * this->numBasis] += dgdx1;
                symmDiff[ibase + (ifree1 + 1) * this->numBasis] += dgdy1;
                symmDiff[ibase + (ifree1 + 2) * this->numBasis] += dgdz1;

                symmDiff[ibase + (ifree2 + 0) * this->numBasis] += dgdx2;
                symmDiff[ibase + (ifree2 + 1) * this->numBasis] += dgdy2;
                symmDiff[ibase + (ifree2 + 2) * this->numBasis] += dgdz2;
            }
        }
    }
}
