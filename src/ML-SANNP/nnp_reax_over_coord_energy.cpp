/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"
#include <cmath>
using namespace std;

#define SMALL_VAL REAL(1.0e-8)

void ReaxPot::calculateOverCoordEnergy()
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
    int jelem;

    int* elemNeigh;

    real pover1;
    real pover2;
    real pover3 = this->param->p_over3;
    real pover4 = this->param->p_over4;

    real** BO_corr;
    real   BO;
    real   De_sigma;
    real   DeBO;

    real  Val;
    real  Delta;
    real  Delta_lp;
    real  dDeltadnlp;
    real* dDeltadSlp   = new real[natom];
    real* dDeltadDelta = new real[natom];

    real nlpopt;
    real nlp;
    real dnlp;
    real dnlpdDelta;
    real Slp;

    real expS;
    real expDelta;
    real DeltaVal;

    real  coeff0;
    real  coeff1i;
    real  coeff1j;
    real* coeff1 = new real[natom];
    real* coeff2 = new real[natom];

    real   Eover;
    real** dEdBO_corr;
    real   dEdDelta_lp;

    this->dEdSlps = new real[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->elemNeighs[iatom][0];

        pover2     = this->param->p_over2 [ielem];
        Val        = this->param->Val     [ielem];
        Delta      = this->Deltas_corr    [iatom];
        nlpopt     = this->param->n_lp_opt[ielem];
        nlp        = this->nlps           [iatom];
        dnlpdDelta = this->dnlpdDeltas    [iatom];
        Slp        = this->Slps           [iatom];

        expS     = pover3 * exp(pover4 * Slp);
        dnlp     = nlpopt - nlp;
        Delta_lp = Delta - dnlp / (ONE + expS);

        dDeltadnlp          = ONE / (ONE + expS);
        dDeltadSlp  [iatom] = dnlp * pover4 * expS * dDeltadnlp * dDeltadnlp;
        dDeltadDelta[iatom] = ONE + dDeltadnlp * dnlpdDelta;

        expDelta      = exp(pover2 * Delta_lp);
        DeltaVal      = Delta_lp + Val + SMALL_VAL;
        coeff0        = ONE / DeltaVal / (ONE + expDelta);
        coeff1i       = Delta_lp * coeff0;
        coeff1[iatom] = coeff1i;
        coeff2[iatom] = (Val * coeff0 / DeltaVal - pover2 * coeff1i * expDelta / (ONE + expDelta));
    }

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond     = this->numBonds[iatom];
        idxBond   = this->idxBonds[iatom];
        elemNeigh = this->elemNeighs[iatom];
        neighbor  = this->geometry->getNeighbors(iatom);

        ielem = elemNeigh[0];

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];

        coeff1i = coeff1[iatom];

        DeBO = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jelem  = elemNeigh[ineigh + 1];
            jatom  = neighbor[ineigh][0];

            pover1   = this->param->p_over1 [ielem][jelem];
            De_sigma = this->param->De_sigma[ielem][jelem];

            BO = BO_corr[ibond][0];

            coeff1j = coeff1[jatom];

            DeBO += pover1 * De_sigma * BO;

            dEdBO_corr[ibond][0] += pover1 * De_sigma * (coeff1i + coeff1j);
        }

        Eover = DeBO * coeff1i;

        dEdDelta_lp = DeBO * coeff2[iatom];

        this->dEdSlps[iatom] = dEdDelta_lp * dDeltadSlp[iatom];

        this->dEdDeltas_corr[iatom] += dEdDelta_lp * dDeltadDelta[iatom];

        this->geometry->addEnergy(iatom, (double) Eover);
    }

    delete[] dDeltadSlp;
    delete[] dDeltadDelta;

    delete[] coeff1;
    delete[] coeff2;
}

