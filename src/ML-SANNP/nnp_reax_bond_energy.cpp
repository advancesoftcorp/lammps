/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"
#include <cmath>
using namespace std;

void ReaxPot::calculateBondEnergy()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;

    int ielem;
    int jelem;

    int* elemNeigh;

    real** BO_corr;
    real   BO_sigma;
    real   BO_pi;
    real   BO_pipi;

    real De_sigma;
    real De_pi;
    real De_pipi;
    real pbe1, pbe2;

    real powBO_sigma;
    real expBO_sigma;
    real Ebond_sigma;
    real Ebond_pi;
    real Ebond_pipi;
    real Ebond;

    real** dEdBO_corr;
    real   dEdBO_sigma;

    this->dEdBOs_corr= new real**[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond     = this->numBonds[iatom];
        idxBond   = this->idxBonds[iatom];
        elemNeigh = this->elemNeighs[iatom];

        ielem = elemNeigh[0];

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = new real*[natom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            dEdBO_corr[ibond] = new real[3];
        }

        Ebond = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jelem  = elemNeigh[ineigh + 1];

            BO_pi    = BO_corr[ibond][1];
            BO_pipi  = BO_corr[ibond][2];
            BO_sigma = BO_corr[ibond][0] - BO_pi - BO_pipi;

            De_sigma = this->param->De_sigma[ielem][jelem];
            De_pi    = this->param->De_pi   [ielem][jelem];
            De_pipi  = this->param->De_pipi [ielem][jelem];
            pbe1     = this->param->p_be1   [ielem][jelem];
            pbe2     = this->param->p_be2   [ielem][jelem];

            powBO_sigma = pow(BO_sigma, pbe2);
            expBO_sigma = exp(pbe1 * (ONE - powBO_sigma));
            Ebond_sigma = -De_sigma * BO_sigma * expBO_sigma;
            Ebond_pi    = -De_pi    * BO_pi;
            Ebond_pipi  = -De_pipi  * BO_pipi;

            Ebond += Ebond_sigma + Ebond_pi + Ebond_pipi;

            dEdBO_sigma = De_sigma * (pbe1 * pbe2 * powBO_sigma - ONE) * expBO_sigma;

            // [BO'(sigma), BO'(pi), BO'(pipi)] -> [BO', BO'(pi), BO'(pipi)]
            dEdBO_corr[ibond][0] = dEdBO_sigma;
            dEdBO_corr[ibond][1] = -De_pi   - dEdBO_sigma;
            dEdBO_corr[ibond][2] = -De_pipi - dEdBO_sigma;
        }

        Ebond *= REAL(0.5);

        this->dEdBOs_corr[iatom] = dEdBO_corr;

        this->geometry->addEnergy(iatom, (double) Ebond);
    }
}

