/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"
#include <cmath>
using namespace std;

void ReaxPot::calculateLonePairEnergy()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int ielem;

    real plp2;
    real nlpopt;
    real nlp;
    real dnlp;
    real expNlp;
    real facNlp;

    real Elp;
    real dEdnlp;
    real dnlpdDelta;

    this->dEdDeltas_corr = new real[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->elemNeighs[iatom][0];

        plp2       = this->param->p_lp2   [ielem];
        nlpopt     = this->param->n_lp_opt[ielem];
        nlp        = this->n0lps          [iatom];
        dnlpdDelta = this->dn0lpdDeltas   [iatom];

        dnlp   = nlpopt - nlp;
        expNlp = exp(-REAL(75.0) * dnlp);
        facNlp = ONE / (ONE + expNlp);

        Elp    =  plp2 * dnlp * facNlp;
        dEdnlp = -plp2 * (ONE + (ONE + REAL(75.0) * dnlp) * expNlp) * facNlp * facNlp;

        this->dEdDeltas_corr[iatom] = dEdnlp * dnlpdDelta;

        this->geometry->addEnergy(iatom, (double) Elp);
    }
}

