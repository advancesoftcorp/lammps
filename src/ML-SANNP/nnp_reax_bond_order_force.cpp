/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"

void ReaxPot::calculateBondOrderForce()
{
    this->calculateBondOrderForce0();

    this->calculateBondOrderForce1();

    this->calculateBondOrderForce2();

    this->calculateBondOrderForce3();

    this->calculateBondOrderForce4();
}

// dE/dBO    = dE/dSlp * dSlp/dBO
// dE/dDelta = dE/dSlp * dSlp/dDelta
void ReaxPot::calculateBondOrderForce0()
{
    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;
    const int** neighbor;

    real** BO_corr;
    real   BO_pi;
    real   BO_pipi;

    real** dEdBO_corr;
    real   dEdBO_pi;

    real dEdSlpi;
    real dEdSlpj;
    real dEdTlp;
    real dEdDelta;

    real Tlpi;
    real Tlpj;
    real dTdDelta;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond    = this->numBonds[iatom];
        idxBond  = this->idxBonds[iatom];
        neighbor = this->geometry->getNeighbors(iatom);

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];
        dEdSlpi    = this->dEdSlps[iatom];
        Tlpi       = this->Tlps[iatom];
        dTdDelta   = this->dTlpdDeltas[iatom];

        dEdTlp = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = neighbor[ineigh][0];

            dEdSlpj = this->dEdSlps[jatom];
            Tlpj    = this->Tlps[jatom];

            BO_pi   = BO_corr[ibond][1];
            BO_pipi = BO_corr[ibond][2];

            dEdBO_pi = dEdSlpi * Tlpj + dEdSlpj * Tlpi;

            dEdBO_corr[ibond][1] += dEdBO_pi;
            dEdBO_corr[ibond][2] += dEdBO_pi;

            dEdTlp += dEdSlpj * (BO_pi + BO_pipi);
        }

        dEdDelta = dEdTlp * dTdDelta;

        this->dEdDeltas_corr[iatom] += dEdDelta;
    }
}

// dE/dBO = dE/dDelta * dDelta/dBO
void ReaxPot::calculateBondOrderForce1()
{
    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;
    const int** neighbor;

    real** dEdBO_corr;

    real dEdDeltai;
    real dEdDeltaj;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond    = this->numBonds[iatom];
        idxBond  = this->idxBonds[iatom];
        neighbor = this->geometry->getNeighbors(iatom);

        dEdBO_corr = this->dEdBOs_corr[iatom];
        dEdDeltai  = this->dEdDeltas_corr[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = neighbor[ineigh][0];

            dEdDeltaj = this->dEdDeltas_corr[jatom];

            dEdBO_corr[ibond][0] += dEdDeltai + dEdDeltaj;
        }
    }
}

// dE/dBO'    = dE/dBO * dBO/dBO'
// dE/dDelta' = dE/dBO * dBO/dDelta'
void ReaxPot::calculateBondOrderForce2()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int ibond;
    int nbond;

    real** dBOdBO;
    real** dBOdDelta;

    real dBOdBO_tot;
    real dBOdBO_pi1;
    real dBOdBO_pipi1;
    real dBOdBO_pi2;
    real dBOdBO_pipi2;

    real dBOdDelta_tot;
    real dBOdDelta_pi;
    real dBOdDelta_pipi;

    real** dEdBO_raw;
    real** dEdBO_corr;
    real   dEdDelta;

    real dEdBOr_tot;
    real dEdBOr_pi;
    real dEdBOr_pipi;
    real dEdBOc_tot;
    real dEdBOc_pi;
    real dEdBOc_pipi;

    this->dEdBOs_raw    = new real**[natom];
    this->dEdDeltas_raw = new real  [natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond = this->numBonds[iatom];

        dBOdBO     = this->dBOdBOs[iatom];
        dBOdDelta  = this->dBOdDeltas[iatom];

        dEdBO_raw  = new real*[nbond];
        dEdBO_corr = this->dEdBOs_corr[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            dEdBO_raw[ibond] = new real[3];
        }

        dEdDelta = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            dBOdBO_tot   = dBOdBO[ibond][0];  // dBO/dBO'
            dBOdBO_pi1   = dBOdBO[ibond][1];  // dBO(pi)/dBO'(pi)
            dBOdBO_pipi1 = dBOdBO[ibond][2];  // dBO(pipi)/dBO'(pipi)
            dBOdBO_pi2   = dBOdBO[ibond][3];  // dBO(pi)/dBO'
            dBOdBO_pipi2 = dBOdBO[ibond][4];  // dBO(pipi)/dBO'

            dBOdDelta_tot  = dBOdDelta[ibond][0];  // dBO/dDelta'
            dBOdDelta_pi   = dBOdDelta[ibond][1];  // dBO(pi)/dDelta'
            dBOdDelta_pipi = dBOdDelta[ibond][2];  // dBO(pipi)/dDelta'

            dEdBOc_tot  = dEdBO_corr[ibond][0];  // dE/dBO
            dEdBOc_pi   = dEdBO_corr[ibond][1];  // dE/dBO(pi)
            dEdBOc_pipi = dEdBO_corr[ibond][2];  // dE/dBO(pipi)

            dEdBOr_tot   = dEdBOc_tot  * dBOdBO_tot;    // dE/dBO' = dE/dBO * dBO/dBO'
            dEdBOr_pi    = dEdBOc_pi   * dBOdBO_pi1     // dE/dBO'(pi) = dE/dBO(pi) * dBO(pi)/dBO'(pi)
                         + dEdBOc_tot  * dBOdBO_pi2;    //             + dE/dBO * dBO/dBO(pi)'
            dEdBOr_pipi  = dEdBOc_pipi * dBOdBO_pipi1   // dE/dBO'(pipi) = dE/dBO(pipi) * dBO(pipi)/dBO'(pipi)
                         + dEdBOc_tot  * dBOdBO_pipi2;  //               + dE/dBO * dBO/dBO(pipi)'

            // [BO', BO'(pi), BO'(pipi)] -> [BO'(sigma), BO'(pi), BO'(pipi)]
            dEdBO_raw[ibond][0] = dEdBOr_tot;
            dEdBO_raw[ibond][1] = dEdBOr_tot + dEdBOr_pi;
            dEdBO_raw[ibond][2] = dEdBOr_tot + dEdBOr_pipi;

            dEdDelta += dEdBOc_tot  * dBOdDelta_tot    // dE/dDelta' = dE/dBO * dBO/dDelta'
                     +  dEdBOc_pi   * dBOdDelta_pi     //            + dE/dBO(pi) * dBO(pi)/dDelta'
                     +  dEdBOc_pipi * dBOdDelta_pipi;  //            + dE/dBO(pipi) * dBO(pipi)/dDelta'
        }

        this->dEdBOs_raw   [iatom] = dEdBO_raw;
        this->dEdDeltas_raw[iatom] = dEdDelta;
    }
}

// dE/dBO' = dE/dDelta' * dDelta'/dBO'
void ReaxPot::calculateBondOrderForce3()
{
    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;
    const int** neighbor;

    real** dEdBO_raw;

    real dEdDeltai;
    real dEdDeltaj;
    real dEdDelta;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond    = this->numBonds[iatom];
        idxBond  = this->idxBonds[iatom];
        neighbor = this->geometry->getNeighbors(iatom);

        dEdBO_raw = this->dEdBOs_raw[iatom];
        dEdDeltai = this->dEdDeltas_raw[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = neighbor[ineigh][0];

            dEdDeltaj = this->dEdDeltas_raw[jatom];

            dEdDelta = dEdDeltai + dEdDeltaj;

            dEdBO_raw[ibond][0] += dEdDelta;
            dEdBO_raw[ibond][1] += dEdDelta;
            dEdBO_raw[ibond][2] += dEdDelta;
        }
    }
}

// dE/dr = dE/dBO' * dBO'/dr
void ReaxPot::calculateBondOrderForce4()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;

    real** rxyzNeigh;

    real** dBOdr_raw;
    real** dEdBO_raw;

    real dBOdr_sigma;
    real dBOdr_pi;
    real dBOdr_pipi;

    real dEdBO_sigma;
    real dEdBO_pi;
    real dEdBO_pipi;

    real r, dEdr;
    real dx, dy, dz;
    real Fx, Fy, Fz;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond     = this->numBonds[iatom];
        idxBond   = this->idxBonds[iatom];
        rxyzNeigh = this->rxyzNeighs[iatom];

        dBOdr_raw = this->dBOdrs_raw[iatom];
        dEdBO_raw = this->dEdBOs_raw[iatom];

        Fx = ZERO;
        Fy = ZERO;
        Fz = ZERO;

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];

            r  = rxyzNeigh[ineigh][0];
            dx = rxyzNeigh[ineigh][1];
            dy = rxyzNeigh[ineigh][2];
            dz = rxyzNeigh[ineigh][3];

            dBOdr_sigma = dBOdr_raw[ibond][0];
            dBOdr_pi    = dBOdr_raw[ibond][1];
            dBOdr_pipi  = dBOdr_raw[ibond][2];

            dEdBO_sigma = dEdBO_raw[ibond][0];
            dEdBO_pi    = dEdBO_raw[ibond][1];
            dEdBO_pipi  = dEdBO_raw[ibond][2];

            dEdr = dEdBO_sigma * dBOdr_sigma
                 + dEdBO_pi    * dBOdr_pi
                 + dEdBO_pipi  * dBOdr_pipi;

            Fx += dEdr * dx / r;
            Fy += dEdr * dy / r;
            Fz += dEdr * dz / r;
        }

        this->geometry->addForce(iatom, Fx, Fy, Fz);
    }
}

