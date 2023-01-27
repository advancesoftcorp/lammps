/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

void ReaxPot::calculateLonePairEnergy(int eflag, LAMMPS_NS::Pair* pair)
{
    int iatom, Iatom;
    int latom = this->locAtoms;
    int natom = this->numAtoms;

    int ielem;

    nnpreal plp2;
    nnpreal nlpopt;
    nnpreal nlp;
    nnpreal dnlp;
    nnpreal expNlp;
    nnpreal facNlp;

    nnpreal Elp;
    nnpreal dEdnlp;
    nnpreal dnlpdDelta;

    double escale = (double) (this->mixingRate * KCAL2EV);

    for (iatom = 0; iatom < latom; ++iatom)
    {
        ielem = this->getElement(iatom);

        plp2       = this->param->p_lp2   [ielem];
        nlpopt     = this->param->n_lp_opt[ielem];
        nlp        = this->n0lps          [iatom];
        dnlpdDelta = this->dn0lpdDeltas   [iatom];

        dnlp   = nlpopt - nlp;
        expNlp = exp(-NNPREAL(75.0) * dnlp);
        facNlp = ONE / (ONE + expNlp);

        Elp    =  plp2 * dnlp * facNlp;
        dEdnlp = -plp2 * (ONE + (ONE + NNPREAL(75.0) * dnlp) * expNlp) * facNlp * facNlp;

        this->dEdDeltas_corr[iatom] = dEdnlp * dnlpdDelta;

        if (eflag)
        {
            Iatom = this->indexOfLAMMPS(iatom);
            if (pair->eflag_global) pair->eng_vdwl     += escale * ((double) Elp);
            if (pair->eflag_atom)   pair->eatom[Iatom] += escale * ((double) Elp);
        }
    }

    for (iatom = latom; iatom < natom; ++iatom)
    {
        this->dEdDeltas_corr[iatom] = ZERO;
    }
}

