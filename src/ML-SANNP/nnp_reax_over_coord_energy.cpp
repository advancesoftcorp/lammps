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
    nnpreal pover5;
    nnpreal pover6 = this->param->p_over6;
    nnpreal pover7 = this->param->p_over7;
    nnpreal pover8 = this->param->p_over8;

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

    nnpreal exp34S;
    nnpreal exp78S;
    nnpreal exp2Delta;
    nnpreal exp6Delta;
    nnpreal DeltaVal;

    nnpreal coeff;
    nnpreal Aoveri;
    nnpreal Aoverj;
    nnpreal Bover;

    nnpreal Aunder, dAunder;
    nnpreal Bunder, dBunder;
    nnpreal Eunder;
    nnpreal dEunderdSlp;
    nnpreal dEunderdDelta;

    nnpreal   Eover;
    nnpreal** dEdBO_corr;
    nnpreal   dEdDelta_lp;

    double escale = (double) (this->mixingRate * KCAL2EV);

    for (iatom = 0; iatom < latom; ++iatom)
    {
        ielem = this->getElement(iatom);

        pover2     = this->param->p_over2 [ielem];
        pover5     = this->param->p_over5 [ielem];
        Val        = this->param->Val     [ielem];
        Delta      = this->Deltas_corr    [iatom];
        nlpopt     = this->param->n_lp_opt[ielem];
        nlp        = this->nlps           [iatom];
        dnlpdDelta = this->dnlpdDeltas    [iatom];
        Slp        = this->Slps           [iatom];

        exp34S   = pover3 * exp(pover4 * Slp);
        exp78S   = pover7 * exp(pover8 * Slp);
        dnlp     = nlpopt - nlp;
        Delta_lp = Delta - dnlp / (ONE + exp34S);

        dDeltadnlp   = ONE / (ONE + exp34S);
        dDeltadSlp   = dnlp * pover4 * exp34S * dDeltadnlp * dDeltadnlp;
        dDeltadDelta = ONE + dDeltadnlp * dnlpdDelta;

        exp2Delta = exp(pover2 * Delta_lp);
        exp6Delta = exp(pover2 * Delta_lp);
        DeltaVal  = Delta_lp + Val + SMALL_VAL;

        coeff  = ONE / DeltaVal / (ONE + exp2Delta);
        Aoveri = Delta_lp * coeff;
        Bover  = (Val * coeff / DeltaVal - pover2 * Aoveri * exp2Delta / (ONE + exp2Delta));

        coeff   = ONE / (ONE + ONE / exp2Delta);
        Aunder  = (ONE - exp6Delta) * coeff;
        dAunder = pover2 / exp2Delta * (ONE - exp6Delta) * coeff * coeff - pover6 * exp6Delta * coeff;
        Bunder  = ONE / (ONE + exp78S);
        dBunder = -pover8 * exp78S * Bunder * Bunder;

        Eunder        = -pover5 * Aunder  * Bunder;
        dEunderdSlp   = -pover5 * Aunder  * dBunder;
        dEunderdDelta = -pover5 * dAunder * Bunder;

        this->dDeltadSlps   [iatom] = dDeltadSlp;
        this->dDeltadDeltas [iatom] = dDeltadDelta;
        this->Aovers        [iatom] = Aoveri;
        this->Bovers        [iatom] = Bover;
        this->Eunders       [iatom] = Eunder;
        this->dEunderdSlps  [iatom] = dEunderdSlp;
        this->dEunderdDeltas[iatom] = dEunderdDelta;
    }

    for (iatom = 0; iatom < latom; ++iatom)
    {
        ielem    = this->getElement(iatom);
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];

        dDeltadSlp    = this->dDeltadSlps   [iatom];
        dDeltadDelta  = this->dDeltadDeltas [iatom];
        Aoveri        = this->Aovers        [iatom];
        Bover         = this->Bovers        [iatom];
        Eunder        = this->Eunders       [iatom];
        dEunderdSlp   = this->dEunderdSlps  [iatom];
        dEunderdDelta = this->dEunderdDeltas[iatom];

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
                Aoverj = this->Aovers[jatom];

                dEdBO_corr[ibond][0] += pover1 * De_sigma * (Aoveri + Aoverj);
            }
            else
            {
                dEdBO_corr[ibond][0] += pover1 * De_sigma * Aoveri;
            }
        }

        Eover = DeBO * Aoveri + Eunder;

        dEdDelta_lp = DeBO * Bover + dEunderdDelta;

        this->dEdSlps[iatom] = dEdDelta_lp * dDeltadSlp + dEunderdSlp;

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

            Aoverj = this->Aovers[jatom];

            dEdBO_corr[ibond][0] += pover1 * De_sigma * Aoverj;
        }
    }
}

