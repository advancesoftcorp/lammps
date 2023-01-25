/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"
#include <cmath>
using namespace std;

void ReaxPot::calculateLonePairNumber()
{
    this->calculateLonePairNumberN();

    this->calculateLonePairNumberS();
}

void ReaxPot::calculateLonePairNumberN()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int ielem;

    real mass;

    real r1      = this->param->r1_lp;
    real r2      = this->param->r2_lp;
    real l1      = -r2;
    real l2      = -r1;
    real lambda  = this->param->lambda_lp;
    real lambda2 = lambda * lambda;
    real lamCosD;

    real xr  = r2 - r1;
    real xr2 = xr  * xr;
    real xr3 = xr2 * xr;
    real xr4 = xr3 * xr;
    real xr5 = xr4 * xr;
    real xr6 = xr5 * xr;
    real xr7 = xr6 * xr;

    real T4 = -REAL(35.0) / xr4;
    real T5 =  REAL(84.0) / xr5;
    real T6 = -REAL(70.0) / xr6;
    real T7 =  REAL(20.0) / xr7;

    real Tap;
    real dTapdDelta;

    real f;
    real dfdDelta;

    real Delta_e;
    real nlp;
    real dnlpdDelta;

    this->n0lps        = new real[natom];
    this->nlps         = new real[natom];
    this->dn0lpdDeltas = new real[natom];
    this->dnlpdDeltas  = new real[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->elemNeighs[iatom][0];

        Delta_e = this->Deltas_e[iatom];

        if (l2 <= Delta_e && Delta_e < r1)
        {
            nlp        = ZERO;
            dnlpdDelta = ZERO;
        }
        else
        {
            lamCosD  = lambda * cos(PI * Delta_e);
            f        = lambda * sin(-PI * Delta_e) / (lamCosD - ONE);
            f        = -REAL(0.5) * Delta_e - REAL(0.5) - atan(f) / PI;
            dfdDelta = REAL(2.0) * lambda2 - REAL(4.0) * lamCosD + REAL(2.0);
            dfdDelta = (lambda2 - ONE) / dfdDelta;

            if (Delta_e < l2)
            {
                nlp        = f;
                dnlpdDelta = dfdDelta;

                if (Delta_e >= l1)
                {
                    xr  = Delta_e - l1;
                    xr2 = xr  * xr;
                    xr3 = xr2 * xr;
                    xr4 = xr3 * xr;
                    xr5 = xr4 * xr;
                    xr6 = xr5 * xr;
                    xr7 = xr6 * xr;

                    Tap = ONE;
                    Tap += T4 * xr4;
                    Tap += T5 * xr5;
                    Tap += T6 * xr6;
                    Tap += T7 * xr7;

                    dTapdDelta =  REAL(4.0) * T4 * xr3;
                    dTapdDelta += REAL(5.0) * T5 * xr4;
                    dTapdDelta += REAL(6.0) * T6 * xr5;
                    dTapdDelta += REAL(7.0) * T7 * xr6;

                    nlp        *= Tap;
                    dnlpdDelta *= Tap;
                    dnlpdDelta += f * dTapdDelta;
                }
            }
            else //if(Delta_e >= r1)
            {
                nlp        = f + ONE;
                dnlpdDelta = dfdDelta;

                if (Delta_e < r2)
                {
                    xr  = r2 - Delta_e;
                    xr2 = xr  * xr;
                    xr3 = xr2 * xr;
                    xr4 = xr3 * xr;
                    xr5 = xr4 * xr;
                    xr6 = xr5 * xr;
                    xr7 = xr6 * xr;

                    Tap = ONE;
                    Tap += T4 * xr4;
                    Tap += T5 * xr5;
                    Tap += T6 * xr6;
                    Tap += T7 * xr7;

                    dTapdDelta =  REAL(4.0) * T4 * xr3;
                    dTapdDelta += REAL(5.0) * T5 * xr4;
                    dTapdDelta += REAL(6.0) * T6 * xr5;
                    dTapdDelta += REAL(7.0) * T7 * xr6;

                    nlp        *= Tap;
                    dnlpdDelta *= Tap;
                    dnlpdDelta -= f * dTapdDelta;
                }
            }
        }

        this->n0lps       [iatom] = nlp;
        this->dn0lpdDeltas[iatom] = dnlpdDelta;

        mass = this->param->mass[ielem];

        if (mass > REAL(21.0))
        {
            this->nlps       [iatom] = this->param->n_lp_opt[ielem];
            this->dnlpdDeltas[iatom] = ZERO;
        }
        else
        {
            this->nlps       [iatom] = this->n0lps       [iatom];
            this->dnlpdDeltas[iatom] = this->dn0lpdDeltas[iatom];
        }
    }
}

void ReaxPot::calculateLonePairNumberS()
{
    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;
    const int** neighbor;

    int ielem;

    real** BO_corr;
    real   BO_pi;
    real   BO_pipi;

    real Delta;
    real nlpopt;
    real nlp;
    real dndDelta;
    real Slp;
    real Tlp;
    real dTdDelta;

    this->Slps        = new real[natom];
    this->Tlps        = new real[natom];
    this->dTlpdDeltas = new real[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->elemNeighs[iatom][0];

        Delta    = this->Deltas_corr    [iatom];
        nlpopt   = this->param->n_lp_opt[ielem];
        nlp      = this->nlps           [iatom];
        dndDelta = this->dnlpdDeltas[iatom];

        Tlp      = Delta - nlpopt + nlp;
        dTdDelta = ONE + dndDelta;

        this->Tlps       [iatom] = Tlp;
        this->dTlpdDeltas[iatom] = dTdDelta;
    }

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond     = this->numBonds[iatom];
        idxBond   = this->idxBonds[iatom];
        neighbor  = this->geometry->getNeighbors(iatom);

        BO_corr = this->BOs_corr[iatom];

        Slp = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = neighbor[ineigh][0];

            Tlp     = this->Tlps[jatom];
            BO_pi   = BO_corr[ibond][1];
            BO_pipi = BO_corr[ibond][2];

            Slp += Tlp * (BO_pi + BO_pipi);
        }

        this->Slps[iatom] = Slp;
    }
}

