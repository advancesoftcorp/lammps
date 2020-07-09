/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func_behler.h"

SymmFuncBehler::SymmFuncBehler(int numElems, bool tanhCutFunc, bool elemWeight, int sizeRad, int sizeAng,
                               real rcutRad, real rcutAng) : SymmFunc(numElems, tanhCutFunc, elemWeight)
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
        this->numAngBasis = this->sizeAng * 2;
    }
    else
    {
        this->numRadBasis = this->sizeRad * this->numElems;
        this->numAngBasis = this->sizeAng * 2 * (this->numElems * (this->numElems + 1) / 2);
    }

    this->numBasis = this->numRadBasis + this->numAngBasis;

    this->rcutRad = rcutRad;
    this->rcutAng = rcutAng;

    this->radiusEta   = NULL;
    this->radiusShift = NULL;

    this->angleMod   = false;
    this->angleEta   = NULL;
    this->angleZeta  = NULL;
    this->angleShift = NULL;
}

SymmFuncBehler::~SymmFuncBehler()
{
    // NOP
}

void SymmFuncBehler::calculate(int numNeighbor, real** posNeighbor, int* elemNeighbor,
                               real* symmData, real* symmDiff) const
{
    if (posNeighbor == NULL || elemNeighbor == NULL)
    {
        stop_by_error("neighbor is null.");
    }

    if (symmData == NULL)
    {
        stop_by_error("symmData is null.");
    }

    if (symmDiff == NULL)
    {
        stop_by_error("symmDiff is null.");
    }

    // define varialbes
    const int numFree = 3 * (1 + numNeighbor);

    int ineigh1, ineigh2;
    int mneigh;

    int jelem1, jelem2;
    int ifree1, ifree2;

    int imode;
    int ibase, jbase, kbase;

    real x1, x2, x3;
    real y1, y2, y3;
    real z1, z2, z3;
    real r1, r2, r3, dr, rr;

    real  rs, eta;
    real  zeta, zeta0;
    real* zeta1;

    int  ilambda;
    real lambda;

    real fc1, fc2, fc3;
    real dfc1dr1, dfc2dr2, dfc3dr3;
    real dfc1dx1, dfc2dx2, dfc3dx3;
    real dfc1dy1, dfc2dy2, dfc3dy3;
    real dfc1dz1, dfc2dz2, dfc3dz3;

    real fc0;
    real dfc0dx1, dfc0dx2, dfc0dx3;
    real dfc0dy1, dfc0dy2, dfc0dy3;
    real dfc0dz1, dfc0dz2, dfc0dz3;

    real gau;
    real dgaudx1, dgaudx2, dgaudx3;
    real dgaudy1, dgaudy2, dgaudy3;
    real dgaudz1, dgaudz2, dgaudz3;

    real psi;
    real dpsidx1, dpsidx2;
    real dpsidy1, dpsidy2;
    real dpsidz1, dpsidz2;

    real chi;
    real chi0;
    real dchidpsi;
    const real chi0_thr = REAL(1.0e-6);

    real g;
    real dgdx1, dgdx2, dgdx3;
    real dgdy1, dgdy2, dgdy3;
    real dgdz1, dgdz2, dgdz3;

    real zanum1, zanum2;
    real zscale;

    real coef0, coef1, coef2, coef3;

    // initialize symmetry functions
    for (ibase = 0; ibase < this->numBasis; ++ibase)
    {
        symmData[ibase] = ZERO;
    }

    for (ifree1 = 0; ifree1 < numFree; ++ifree1)
    {
        for (ibase = 0; ibase < this->numBasis; ++ibase)
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

        ifree1 = 3 * (ineigh1 + 1);

        if (this->elemWeight)
        {
            jelem1 = 0;
            zanum1 = (real) elemNeighbor[ineigh1];
        }
        else
        {
            jelem1 = elemNeighbor[ineigh1];
            zanum1 = ONE;
        }

        jbase  = jelem1 * this->sizeRad;
        zscale = zanum1;

        this->cutoffFunction(&fc1, &dfc1dr1, r1, this->rcutRad);
        dfc1dx1 = x1 / r1 * dfc1dr1;
        dfc1dy1 = y1 / r1 * dfc1dr1;
        dfc1dz1 = z1 / r1 * dfc1dr1;

        for (imode = 0; imode < this->sizeRad; ++imode)
        {
            eta = this->radiusEta[imode];
            rs  = this->radiusShift[imode];

            dr  = r1 - rs;
            rr  = dr * dr;

            gau     = exp(-eta * rr);
            coef0   = -REAL(2.0) * eta * dr / r1 * gau;
            dgaudx1 = x1 * coef0;
            dgaudy1 = y1 * coef0;
            dgaudz1 = z1 * coef0;

            g     = zscale * gau * fc1;
            dgdx1 = zscale * (dgaudx1 * fc1 + gau * dfc1dx1);
            dgdy1 = zscale * (dgaudy1 * fc1 + gau * dfc1dy1);
            dgdz1 = zscale * (dgaudz1 * fc1 + gau * dfc1dz1);

            ibase = imode + jbase;

            symmData[ibase] += g;

            symmDiff[ibase + 0 * this->numBasis] -= dgdx1;
            symmDiff[ibase + 1 * this->numBasis] -= dgdy1;
            symmDiff[ibase + 2 * this->numBasis] -= dgdz1;

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
    zeta1 = new real[this->sizeAng];
    for (imode = 0; imode < this->sizeAng; ++imode)
    {
        zeta = this->angleZeta[imode];
        zeta1[imode] = pow(REAL(2.0), ONE - zeta);
    }

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

        ifree2 = 3 * (ineigh2 + 1);

        if (this->elemWeight)
        {
            jelem2 = 0;
            zanum2 = (real) elemNeighbor[ineigh2];
            mneigh = ineigh2;
        }
        else
        {
            jelem2 = elemNeighbor[ineigh2];
            zanum2 = ONE;
            mneigh = numNeigh;
        }

        this->cutoffFunction(&fc2, &dfc2dr2, r2, this->rcutAng);
        dfc2dx2 = x2 / r2 * dfc2dr2;
        dfc2dy2 = y2 / r2 * dfc2dr2;
        dfc2dz2 = z2 / r2 * dfc2dr2;

        for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
        {
            r1 = posNeighbor[ineigh1][0];
            if (r1 >= this->rcutAng)
            {
                continue;
            }

            x1 = posNeighbor[ineigh1][1];
            y1 = posNeighbor[ineigh1][2];
            z1 = posNeighbor[ineigh1][3];

            ifree1 = 3 * (ineigh1 + 1);

            if (this->elemWeight)
            {
                jelem1 = 0;
                zanum1 = (real) elemNeighbor[ineigh1];
            }
            else
            {
                jelem1 = elemNeighbor[ineigh1];
                zanum1 = ONE;

                if (jelem1 > jelem2 || (jelem1 == jelem2 && ineigh1 >= ineigh2))
                {
                    continue;
                }
            }

            kbase  = (jelem1 + jelem2 * (jelem2 + 1) / 2) * 2 * this->sizeAng;
            zscale = zanum1 * zanum2;

            this->cutoffFunction(&fc1, &dfc1dr1, r1, this->rcutAng);
            dfc1dx1 = x1 / r1 * dfc1dr1;
            dfc1dy1 = y1 / r1 * dfc1dr1;
            dfc1dz1 = z1 / r1 * dfc1dr1;

            if (this->angleMod)
            {
                fc0 = fc1 * fc2;
                dfc0dx1 = dfc1dx1 * fc2;
                dfc0dy1 = dfc1dy1 * fc2;
                dfc0dz1 = dfc1dz1 * fc2;
                dfc0dx2 = fc1 * dfc2dx2;
                dfc0dy2 = fc1 * dfc2dy2;
                dfc0dz2 = fc1 * dfc2dz2;
            }

            else
            {
                x3 = x1 - x2;
                y3 = y1 - y2;
                z3 = z1 - z2;
                rr = x3 * x3 + y3 * y3 + z3 * z3;

                if (rr >= this->rcutAng * this->rcutAng)
                {
                    continue;
                }

                r3 = sqrt(rr);

                this->cutoffFunction(&fc3, &dfc3dr3, r3, this->rcutAng);
                dfc3dx3 = x3 / r3 * dfc3dr3;
                dfc3dy3 = y3 / r3 * dfc3dr3;
                dfc3dz3 = z3 / r3 * dfc3dr3;

                fc0 = fc1 * fc2 * fc3;
                dfc0dx1 = dfc1dx1 * fc2 * fc3;
                dfc0dy1 = dfc1dy1 * fc2 * fc3;
                dfc0dz1 = dfc1dz1 * fc2 * fc3;
                dfc0dx2 = fc1 * dfc2dx2 * fc3;
                dfc0dy2 = fc1 * dfc2dy2 * fc3;
                dfc0dz2 = fc1 * dfc2dz2 * fc3;
                dfc0dx3 = fc1 * fc2 * dfc3dx3;
                dfc0dy3 = fc1 * fc2 * dfc3dy3;
                dfc0dz3 = fc1 * fc2 * dfc3dz3;
            }

            psi     = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            coef0   = ONE / r1 / r2;
            coef1   = psi / r1 / r1;
            coef2   = psi / r2 / r2;
            dpsidx1 = coef0 * x2 - coef1 * x1;
            dpsidy1 = coef0 * y2 - coef1 * y1;
            dpsidz1 = coef0 * z2 - coef1 * z1;
            dpsidx2 = coef0 * x1 - coef2 * x2;
            dpsidy2 = coef0 * y1 - coef2 * y2;
            dpsidz2 = coef0 * z1 - coef2 * z2;

            for (ilambda = 0; ilambda < 2; ++ilambda)
            {
                lambda = (ilambda == 0) ? ONE : (-ONE);

                chi0 = ONE + lambda * psi;
                if (chi0 < chi0_thr)
                {
                    continue;
                }

                jbase = ilambda * this->sizeAng;

                if (this->angleMod)
                {
                    for (imode = 0; imode < this->sizeAng; ++imode)
                    {
                        eta   = this->angleEta[imode];
                        rs    = this->angleShift[imode];
                        zeta  = this->angleZeta[imode];
                        zeta0 = zeta1[imode];

                        chi      = zeta0 * pow(chi0, zeta);
                        dchidpsi = zeta * lambda * chi / chi0;

                        rr = (r1 - rs) * (r1 - rs) + (r2 - rs) * (r2 - rs);

                        gau     = exp(-eta * rr);
                        coef0   = -REAL(2.0) * eta * gau;
                        coef1   = coef0 * (r1 - rs) / r1;
                        coef2   = coef0 * (r2 - rs) / r2;
                        dgaudx1 = coef1 * x1;
                        dgaudy1 = coef1 * y1;
                        dgaudz1 = coef1 * z1;
                        dgaudx2 = coef2 * x2;
                        dgaudy2 = coef2 * y2;
                        dgaudz2 = coef2 * z2;

                        g     = zscale * chi * gau * fc0;
                        dgdx1 = zscale * (dchidpsi * dpsidx1 * gau * fc0 + chi * dgaudx1 * fc0 + chi * gau * dfc0dx1);
                        dgdy1 = zscale * (dchidpsi * dpsidy1 * gau * fc0 + chi * dgaudy1 * fc0 + chi * gau * dfc0dy1);
                        dgdz1 = zscale * (dchidpsi * dpsidz1 * gau * fc0 + chi * dgaudz1 * fc0 + chi * gau * dfc0dz1);
                        dgdx2 = zscale * (dchidpsi * dpsidx2 * gau * fc0 + chi * dgaudx2 * fc0 + chi * gau * dfc0dx2);
                        dgdy2 = zscale * (dchidpsi * dpsidy2 * gau * fc0 + chi * dgaudy2 * fc0 + chi * gau * dfc0dy2);
                        dgdz2 = zscale * (dchidpsi * dpsidz2 * gau * fc0 + chi * dgaudz2 * fc0 + chi * gau * dfc0dz2);

                        ibase = this->numRadBasis + imode + jbase + kbase;

                        symmData[ibase] += g;

                        symmDiff[ibase + 0 * this->numBasis] -= dgdx1 + dgdx2;
                        symmDiff[ibase + 1 * this->numBasis] -= dgdy1 + dgdy2;
                        symmDiff[ibase + 2 * this->numBasis] -= dgdz1 + dgdz2;

                        symmDiff[ibase + (ifree1 + 0) * this->numBasis] += dgdx1;
                        symmDiff[ibase + (ifree1 + 1) * this->numBasis] += dgdy1;
                        symmDiff[ibase + (ifree1 + 2) * this->numBasis] += dgdz1;

                        symmDiff[ibase + (ifree2 + 0) * this->numBasis] += dgdx2;
                        symmDiff[ibase + (ifree2 + 1) * this->numBasis] += dgdy2;
                        symmDiff[ibase + (ifree2 + 2) * this->numBasis] += dgdz2;
                    }
                }

                else
                {
                    for (imode = 0; imode < this->sizeAng; ++imode)
                    {
                        eta   = this->angleEta[imode];
                        rs    = this->angleShift[imode];
                        zeta  = this->angleZeta[imode];
                        zeta0 = zeta1[imode];

                        chi      = zeta0 * pow(chi0, zeta);
                        dchidpsi = zeta * lambda * chi / chi0;

                        rr = (r1 - rs) * (r1 - rs) + (r2 - rs) * (r2 - rs) + (r3 - rs) * (r3 - rs);

                        gau     = exp(-eta * rr);
                        coef0   = -REAL(2.0) * eta * gau;
                        coef1   = coef0 * (r1 - rs) / r1;
                        coef2   = coef0 * (r2 - rs) / r2;
                        coef3   = coef0 * (r3 - rs) / r3;
                        dgaudx1 = coef1 * x1;
                        dgaudy1 = coef1 * y1;
                        dgaudz1 = coef1 * z1;
                        dgaudx2 = coef2 * x2;
                        dgaudy2 = coef2 * y2;
                        dgaudz2 = coef2 * z2;
                        dgaudx3 = coef3 * x3;
                        dgaudy3 = coef3 * y3;
                        dgaudz3 = coef3 * z3;

                        g     = zscale * (chi * gau * fc0);
                        dgdx1 = zscale * (dchidpsi * dpsidx1 * gau * fc0 + chi * dgaudx1 * fc0 + chi * gau * dfc0dx1);
                        dgdy1 = zscale * (dchidpsi * dpsidy1 * gau * fc0 + chi * dgaudy1 * fc0 + chi * gau * dfc0dy1);
                        dgdz1 = zscale * (dchidpsi * dpsidz1 * gau * fc0 + chi * dgaudz1 * fc0 + chi * gau * dfc0dz1);
                        dgdx2 = zscale * (dchidpsi * dpsidx2 * gau * fc0 + chi * dgaudx2 * fc0 + chi * gau * dfc0dx2);
                        dgdy2 = zscale * (dchidpsi * dpsidy2 * gau * fc0 + chi * dgaudy2 * fc0 + chi * gau * dfc0dy2);
                        dgdz2 = zscale * (dchidpsi * dpsidz2 * gau * fc0 + chi * dgaudz2 * fc0 + chi * gau * dfc0dz2);
                        dgdx3 = zscale * (chi * dgaudx3 * fc0 + chi * gau * dfc0dx3);
                        dgdy3 = zscale * (chi * dgaudy3 * fc0 + chi * gau * dfc0dy3);
                        dgdz3 = zscale * (chi * dgaudz3 * fc0 + chi * gau * dfc0dz3);

                        ibase = this->numRadBasis + imode + jbase + kbase;

                        symmData[ibase] += g;

                        symmDiff[ibase + 0 * this->numBasis] -= dgdx1 + dgdx2;
                        symmDiff[ibase + 1 * this->numBasis] -= dgdy1 + dgdy2;
                        symmDiff[ibase + 2 * this->numBasis] -= dgdz1 + dgdz2;

                        symmDiff[ibase + (ifree1 + 0) * this->numBasis] += dgdx1 + dgdx3;
                        symmDiff[ibase + (ifree1 + 1) * this->numBasis] += dgdy1 + dgdy3;
                        symmDiff[ibase + (ifree1 + 2) * this->numBasis] += dgdz1 + dgdz3;

                        symmDiff[ibase + (ifree2 + 0) * this->numBasis] += dgdx2 - dgdx3;
                        symmDiff[ibase + (ifree2 + 1) * this->numBasis] += dgdy2 - dgdy3;
                        symmDiff[ibase + (ifree2 + 2) * this->numBasis] += dgdz2 - dgdz3;
                    }
                }
            }
        }
    }

    delete[] zeta1;
}
