/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

#define SMALL_VAL NNPREAL(1.0e-8)

void ReaxPot::calculateOverCoordEnergy(int eflag, LAMMPS_NS::Pair* pair)
{
    int iatom, Iatom;
    int jatom;
    int latom = this->locAtoms;
    int natom = this->numAtoms;

    int  ineigh;
    int* idxNeigh;

    int  ibond;
    int  nbond;
    int* idxBond;

    int ielem;
    int jelem;

    nnpreal pover1;
    nnpreal pover2;
    nnpreal pover3 = this->param->p_over3;
    nnpreal pover4 = this->param->p_over4;

    nnpreal** BO_corr;
    nnpreal   BO;
    nnpreal   De_sigma;
    nnpreal   DeBO;

    nnpreal Val;
    nnpreal Delta;
    nnpreal Delta_lp;
    nnpreal dDeltadnlp;
    nnpreal dDeltadSlp;
    nnpreal dDeltadDelta;

    nnpreal nlpopt;
    nnpreal nlp;
    nnpreal dnlp;
    nnpreal dnlpdDelta;
    nnpreal Slp;

    nnpreal expS;
    nnpreal expDelta;
    nnpreal DeltaVal;

    nnpreal coeff0;
    nnpreal coeff1i;
    nnpreal coeff1j;
    nnpreal coeff2;

    nnpreal   Eover;
    nnpreal** dEdBO_corr;
    nnpreal   dEdDelta_lp;

    double escale = (double) (this->mixingRate * KCAL2EV);

    for (iatom = 0; iatom < latom; ++iatom)
    {
        ielem = this->getElement(iatom);

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

        expDelta = exp(pover2 * Delta_lp);
        DeltaVal = Delta_lp + Val + SMALL_VAL;
        coeff0   = ONE / DeltaVal / (ONE + expDelta);
        coeff1i  = Delta_lp * coeff0;
        coeff2   = (Val * coeff0 / DeltaVal - pover2 * coeff1i * expDelta / (ONE + expDelta));

        dDeltadnlp   = ONE / (ONE + expS);
        dDeltadSlp   = dnlp * pover4 * expS * dDeltadnlp * dDeltadnlp;
        dDeltadDelta = ONE + dDeltadnlp * dnlpdDelta;

        this->dDeltadSlps  [iatom] = dDeltadSlp;
        this->dDeltadDeltas[iatom] = dDeltadDelta;
        this->coeff1Eovers [iatom] = coeff1i;
        this->coeff2Eovers [iatom] = coeff2;
    }

    for (iatom = 0; iatom < latom; ++iatom)
    {
        ielem    = this->getElement(iatom);
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];

        dDeltadSlp   = this->dDeltadSlps  [iatom];
        dDeltadDelta = this->dDeltadDeltas[iatom];
        coeff1i      = this->coeff1Eovers [iatom];
        coeff2       = this->coeff2Eovers [iatom];

        DeBO = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);
            jelem  = this->getElement(jatom);

            pover1   = this->param->p_over1 [ielem][jelem];
            De_sigma = this->param->De_sigma[ielem][jelem];

            BO = BO_corr[ibond][0];

            DeBO += pover1 * De_sigma * BO;

            if (jatom < latom)
            {
                coeff1j = this->coeff1Eovers[jatom];

                dEdBO_corr[ibond][0] += pover1 * De_sigma * (coeff1i + coeff1j);
            }
            else
            {
                dEdBO_corr[ibond][0] += pover1 * De_sigma * coeff1i;
            }
        }

        Eover = DeBO * coeff1i;

        dEdDelta_lp = DeBO * coeff2;

        this->dEdSlps[iatom] = dEdDelta_lp * dDeltadSlp;

        this->dEdDeltas_corr[iatom] += dEdDelta_lp * dDeltadDelta;

        if (eflag)
        {
            Iatom = this->indexOfLAMMPS(iatom);
            if (pair->eflag_global) pair->eng_vdwl     += escale * ((double) Eover);
            if (pair->eflag_atom)   pair->eatom[Iatom] += escale * ((double) Eover);
        }
    }

    for (iatom = latom; iatom < natom; ++iatom)
    {
        ielem    = this->getElement(iatom);
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        dEdBO_corr = this->dEdBOs_corr[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);

            if (jatom >= latom) continue;

            jelem = this->getElement(jatom);

            pover1   = this->param->p_over1 [ielem][jelem];
            De_sigma = this->param->De_sigma[ielem][jelem];

            coeff1j = this->coeff1Eovers[jatom];

            dEdBO_corr[ibond][0] += pover1 * De_sigma * coeff1j;
        }
    }
}

