/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"
#include <cmath>
using namespace std;

void ReaxPot::calculateVanDerWaalsEnergy(bool withForce)
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int ineigh;
    int nneigh;

    int ielem;
    int jelem;

    int*   elemNeigh;
    real** rxyzNeigh;

    real r, r2, r3, r4, r5, r6, r7;

    real D;
    real alpha;
    real gamma;
    real rvdw;
    real pvdw = this->param->p_vdw;
    real pc1, pc2, pc3;

    real rp, gp;
    real f13, g13, h13, i13;
    real df13dr;

    real Tap;
    real dTapdr;

    real Tap0 = this->param->Tap_vdw[0];
    real Tap1 = this->param->Tap_vdw[1];
    real Tap2 = this->param->Tap_vdw[2];
    real Tap3 = this->param->Tap_vdw[3];
    real Tap4 = this->param->Tap_vdw[4];
    real Tap5 = this->param->Tap_vdw[5];
    real Tap6 = this->param->Tap_vdw[6];
    real Tap7 = this->param->Tap_vdw[7];

    real Evdw;
    real Evdw0;
    real Ecore;
    real dEvdw0dr;
    real dEcoredr;

    real dEdr;
    real dx, dy, dz;
    real Fx, Fy, Fz;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh    = this->geometry->getNumNeighbors(iatom);
        elemNeigh = this->elemNeighs[iatom];
        rxyzNeigh = this->rxyzNeighs[iatom];

        ielem = elemNeigh[0];

        Evdw = ZERO;
        Fx   = ZERO;
        Fy   = ZERO;
        Fz   = ZERO;

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            jelem = elemNeigh[ineigh + 1];
            r     = rxyzNeigh[ineigh][0];
            dx    = rxyzNeigh[ineigh][1];
            dy    = rxyzNeigh[ineigh][2];
            dz    = rxyzNeigh[ineigh][3];

            D     = this->param->D_vdw    [ielem][jelem];
            alpha = this->param->alpha_vdw[ielem][jelem];
            gamma = this->param->gammaw   [ielem][jelem];
            rvdw  = this->param->r_vdw    [ielem][jelem];
            pc1   = this->param->p_core1  [ielem][jelem];
            pc2   = this->param->p_core2  [ielem][jelem];
            pc3   = this->param->p_core3  [ielem][jelem];

            if (this->param->shielding)
            {
                rp     = pow(r, pvdw);
                gp     = pow(ONE / gamma, pvdw);
                f13    = pow(rp + gp, ONE / pvdw);
                df13dr = rp / r * f13 / (rp + gp);
            }
            else
            {
                f13    = r;
                df13dr = ONE;
            }

            g13 = alpha * (ONE - f13 / rvdw);
            h13 = exp(g13);
            i13 = exp(REAL(0.5) * g13);

            Evdw0    = D * (h13 - REAL(2.0) * i13);
            dEvdw0dr = D * alpha / rvdw * (i13 - h13) * df13dr;

            if (this->param->innerWall)
            {
                Ecore    = pc2 * exp(pc3 * (ONE - r / pc1));
                dEcoredr = -pc3 / pc1 * Ecore;
            }
            else
            {
                Ecore    = ZERO;
                dEcoredr = ZERO;
            }

            r2 = r * r;
            r3 = r * r2;
            r4 = r * r3;
            r5 = r * r4;
            r6 = r * r5;
            r7 = r * r6;

            Tap  = Tap0;
            Tap += Tap1 * r;
            Tap += Tap2 * r2;
            Tap += Tap3 * r3;
            Tap += Tap4 * r4;
            Tap += Tap5 * r5;
            Tap += Tap6 * r6;
            Tap += Tap7 * r7;

            Evdw += Tap * (Evdw0 + Ecore);

            if (withForce)
            {
                dTapdr  = Tap1;
                dTapdr += Tap2 * r  * REAL(2.0);
                dTapdr += Tap3 * r2 * REAL(3.0);
                dTapdr += Tap4 * r3 * REAL(4.0);
                dTapdr += Tap5 * r4 * REAL(5.0);
                dTapdr += Tap6 * r5 * REAL(6.0);
                dTapdr += Tap7 * r6 * REAL(7.0);

                dEdr = dTapdr * (Evdw0 + Ecore) + Tap * (dEvdw0dr + dEcoredr);

                Fx += dEdr * dx / r;
                Fy += dEdr * dy / r;
                Fz += dEdr * dz / r;
            }
        }

        Evdw *= REAL(0.5);

        this->geometry->addEnergy(iatom, (double) Evdw);
        this->geometry->addForce (iatom, Fx, Fy, Fz);
    }
}

