/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

void ReaxPot::calculateBondEnergy(int eflag, LAMMPS_NS::Pair* pair)
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

    nnpreal** BO_corr;
    nnpreal   BO_sigma;
    nnpreal   BO_pi;
    nnpreal   BO_pipi;

    nnpreal De_sigma;
    nnpreal De_pi;
    nnpreal De_pipi;
    nnpreal pbe1, pbe2;
    nnpreal coeff, coeffi, coeffj;

    nnpreal powBO_sigma;
    nnpreal expBO_sigma;
    nnpreal Ebond_sigma;
    nnpreal Ebond_pi;
    nnpreal Ebond_pipi;
    nnpreal Ebond;

    nnpreal** dEdBO_corr;
    nnpreal   dEdBO_sigma;

    double escale = (double) (this->mixingRate * KCAL2EV);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem    = this->getElement(iatom);
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];

        Ebond = ZERO;

        coeffi = (iatom < latom) ? NNPREAL(0.5) : ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);

            if (iatom >= latom && jatom >= latom)
            {
                dEdBO_corr[ibond][0] = ZERO;
                dEdBO_corr[ibond][1] = ZERO;
                dEdBO_corr[ibond][2] = ZERO;
                continue;
            }

            jelem = this->getElement(jatom);

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

            Ebond += coeffi * (Ebond_sigma + Ebond_pi + Ebond_pipi);

            coeffj = (jatom < latom) ? NNPREAL(0.5) : ZERO;
            coeff  = coeffi + coeffj;

            dEdBO_sigma = De_sigma * (pbe1 * pbe2 * powBO_sigma - ONE) * expBO_sigma;

            // [BO'(sigma), BO'(pi), BO'(pipi)] -> [BO', BO'(pi), BO'(pipi)]
            dEdBO_corr[ibond][0] = coeff * dEdBO_sigma;
            dEdBO_corr[ibond][1] = coeff * (-De_pi   - dEdBO_sigma);
            dEdBO_corr[ibond][2] = coeff * (-De_pipi - dEdBO_sigma);
        }

        if (eflag)
        {
            if (iatom < latom)
            {
                Iatom = this->indexOfLAMMPS(iatom);
                if (pair->eflag_global) pair->eng_vdwl     += escale * ((double) Ebond);
                if (pair->eflag_atom)   pair->eatom[Iatom] += escale * ((double) Ebond);
            }
        }
    }
}

