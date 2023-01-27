/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

void ReaxPot::calculateLonePairNumber()
{
    this->calculateLonePairNumberN();

    this->calculateLonePairNumberS();
}

void ReaxPot::calculateLonePairNumberN()
{
    int iatom;
    int natom = this->numAtoms;

    int ielem;

    nnpreal mass;

    nnpreal r1      = this->param->r1_lp;
    nnpreal r2      = this->param->r2_lp;
    nnpreal l1      = -r2;
    nnpreal l2      = -r1;
    nnpreal lambda  = this->param->lambda_lp;
    nnpreal lambda2 = lambda * lambda;
    nnpreal lamCosD;

    nnpreal xr  = r2 - r1;
    nnpreal xr2 = xr  * xr;
    nnpreal xr3 = xr2 * xr;
    nnpreal xr4 = xr3 * xr;
    nnpreal xr5 = xr4 * xr;
    nnpreal xr6 = xr5 * xr;
    nnpreal xr7 = xr6 * xr;

    nnpreal T4 = -NNPREAL(35.0) / xr4;
    nnpreal T5 =  NNPREAL(84.0) / xr5;
    nnpreal T6 = -NNPREAL(70.0) / xr6;
    nnpreal T7 =  NNPREAL(20.0) / xr7;

    nnpreal Tap;
    nnpreal dTapdDelta;

    nnpreal f;
    nnpreal dfdDelta;

    nnpreal Delta_e;
    nnpreal nlp;
    nnpreal dnlpdDelta;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->getElement(iatom);

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
            f        = -NNPREAL(0.5) * Delta_e - NNPREAL(0.5) - atan(f) / PI;
            dfdDelta = NNPREAL(2.0) * lambda2 - NNPREAL(4.0) * lamCosD + NNPREAL(2.0);
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

                    dTapdDelta =  NNPREAL(4.0) * T4 * xr3;
                    dTapdDelta += NNPREAL(5.0) * T5 * xr4;
                    dTapdDelta += NNPREAL(6.0) * T6 * xr5;
                    dTapdDelta += NNPREAL(7.0) * T7 * xr6;

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

                    dTapdDelta =  NNPREAL(4.0) * T4 * xr3;
                    dTapdDelta += NNPREAL(5.0) * T5 * xr4;
                    dTapdDelta += NNPREAL(6.0) * T6 * xr5;
                    dTapdDelta += NNPREAL(7.0) * T7 * xr6;

                    nlp        *= Tap;
                    dnlpdDelta *= Tap;
                    dnlpdDelta -= f * dTapdDelta;
                }
            }
        }

        this->n0lps       [iatom] = nlp;
        this->dn0lpdDeltas[iatom] = dnlpdDelta;

        mass = this->param->mass[ielem];

        if (mass > NNPREAL(21.0))
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
    int natom = this->numAtoms;
    int latom = this->locAtoms;

    int  ineigh;
    int* idxNeigh;

    int  ibond;
    int  nbond;
    int* idxBond;

    int ielem;

    nnpreal** BO_corr;
    nnpreal   BO_pi;
    nnpreal   BO_pipi;

    nnpreal Delta;
    nnpreal nlpopt;
    nnpreal nlp;
    nnpreal dndDelta;
    nnpreal Slp;
    nnpreal Tlp;
    nnpreal dTdDelta;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->getElement(iatom);

        Delta    = this->Deltas_corr    [iatom];
        nlpopt   = this->param->n_lp_opt[ielem];
        nlp      = this->nlps           [iatom];
        dndDelta = this->dnlpdDeltas[iatom];

        Tlp      = Delta - nlpopt + nlp;
        dTdDelta = ONE + dndDelta;

        this->Tlps       [iatom] = Tlp;
        this->dTlpdDeltas[iatom] = dTdDelta;
    }

    for (iatom = 0; iatom < latom; ++iatom)
    {
        idxNeigh = this->getNeighbors(iatom);

        nbond     = this->numBonds[iatom];
        idxBond   = this->idxBonds[iatom];

        BO_corr = this->BOs_corr[iatom];

        Slp = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);

            Tlp     = this->Tlps[jatom];
            BO_pi   = BO_corr[ibond][1];
            BO_pipi = BO_corr[ibond][2];

            Slp += Tlp * (BO_pi + BO_pipi);
        }

        this->Slps[iatom] = Slp;
    }
}

