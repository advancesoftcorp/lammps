/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

void ReaxPot::calculateBondOrderForce(LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom)
{
    this->calculateBondOrderForce0();

    this->calculateBondOrderForce1();

    this->calculateBondOrderForce2();

    this->calculateBondOrderForce3();

    this->calculateBondOrderForce4(pair, atom);
}

// dE/dBO    = dE/dSlp * dSlp/dBO
// dE/dDelta = dE/dSlp * dSlp/dDelta
void ReaxPot::calculateBondOrderForce0()
{
    int iatom;
    int jatom;
    int latom = this->locAtoms;
    int natom = this->numAtoms;

    int  ineigh;
    int* idxNeigh;

    int  ibond;
    int  nbond;
    int* idxBond;

    nnpreal** BO_corr;
    nnpreal   BO_pi;
    nnpreal   BO_pipi;

    nnpreal** dEdBO_corr;
    nnpreal   dEdBO_pi;

    nnpreal dEdSlpi;
    nnpreal dEdSlpj;
    nnpreal dEdTlp;
    nnpreal dEdDelta;

    nnpreal Tlpi;
    nnpreal Tlpj;
    nnpreal dTdDelta;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        BO_corr    = this->BOs_corr[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];
        dEdSlpi    = iatom < latom ? this->dEdSlps[iatom] : ZERO;
        Tlpi       = this->Tlps[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);

            if (iatom >= latom && jatom >= latom) continue;

            dEdSlpj  = jatom < latom ? this->dEdSlps[jatom] : ZERO;
            Tlpj     = this->Tlps[jatom];
            dTdDelta = this->dTlpdDeltas[jatom];

            BO_pi   = BO_corr[ibond][1];
            BO_pipi = BO_corr[ibond][2];

            dEdBO_pi = dEdSlpi * Tlpj + dEdSlpj * Tlpi;

            dEdBO_corr[ibond][1] += dEdBO_pi;
            dEdBO_corr[ibond][2] += dEdBO_pi;

            dEdTlp   = dEdSlpi * (BO_pi + BO_pipi);
            dEdDelta = dEdTlp * dTdDelta;

            this->dEdDeltas_corr[jatom] += dEdDelta;
        }
    }
}

// dE/dBO = dE/dDelta * dDelta/dBO
void ReaxPot::calculateBondOrderForce1()
{
    int iatom;
    int jatom;
    int natom = this->numAtoms;

    int  ineigh;
    int* idxNeigh;

    int  ibond;
    int  nbond;
    int* idxBond;

    nnpreal** dEdBO_corr;

    nnpreal dEdDeltai;
    nnpreal dEdDeltaj;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        dEdBO_corr = this->dEdBOs_corr[iatom];
        dEdDeltai  = this->dEdDeltas_corr[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);

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
    int natom = this->numAtoms;

    int ibond;
    int nbond;

    nnpreal** dBOdBO;
    nnpreal** dBOdDelta;

    nnpreal dBOdBO_tot;
    nnpreal dBOdBO_pi1;
    nnpreal dBOdBO_pipi1;
    nnpreal dBOdBO_pi2;
    nnpreal dBOdBO_pipi2;

    nnpreal dBOdDelta_tot;
    nnpreal dBOdDelta_pi;
    nnpreal dBOdDelta_pipi;

    nnpreal** dEdBO_raw;
    nnpreal** dEdBO_corr;
    nnpreal   dEdDelta;

    nnpreal dEdBOr_tot;
    nnpreal dEdBOr_pi;
    nnpreal dEdBOr_pipi;
    nnpreal dEdBOc_tot;
    nnpreal dEdBOc_pi;
    nnpreal dEdBOc_pipi;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond = this->numBonds[iatom];

        dBOdBO     = this->dBOdBOs[iatom];
        dBOdDelta  = this->dBOdDeltas[iatom];

        dEdBO_raw  = this->dEdBOs_raw[iatom];
        dEdBO_corr = this->dEdBOs_corr[iatom];

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

        this->dEdDeltas_raw[iatom] = dEdDelta;
    }
}

// dE/dBO' = dE/dDelta' * dDelta'/dBO'
void ReaxPot::calculateBondOrderForce3()
{
    int iatom;
    int jatom;
    int natom = this->numAtoms;

    int  ineigh;
    int* idxNeigh;

    int  ibond;
    int  nbond;
    int* idxBond;

    nnpreal** dEdBO_raw;

    nnpreal dEdDeltai;
    nnpreal dEdDeltaj;
    nnpreal dEdDelta;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        idxNeigh = this->getNeighbors(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        dEdBO_raw = this->dEdBOs_raw[iatom];
        dEdDeltai = this->dEdDeltas_raw[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);

            dEdDeltaj = this->dEdDeltas_raw[jatom];

            dEdDelta = dEdDeltai + dEdDeltaj;

            dEdBO_raw[ibond][0] += dEdDelta;
            dEdBO_raw[ibond][1] += dEdDelta;
            dEdBO_raw[ibond][2] += dEdDelta;
        }
    }
}

// dE/dr = dE/dBO' * dBO'/dr
void ReaxPot::calculateBondOrderForce4(LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom)
{
    int iatom, Iatom;
    int jatom, Jatom;
    int natom = this->numAtoms;

    int       ineigh;
    int*      idxNeigh;
    nnpreal** posNeigh;

    int  ibond;
    int  nbond;
    int* idxBond;

    nnpreal** dBOdr_raw;
    nnpreal** dEdBO_raw;

    nnpreal dBOdr_sigma;
    nnpreal dBOdr_pi;
    nnpreal dBOdr_pipi;

    nnpreal dEdBO_sigma;
    nnpreal dEdBO_pi;
    nnpreal dEdBO_pipi;
    nnpreal dEdr;

    double r,  dx, dy, dz;
    double Fr, Fx, Fy, Fz;

    double escale = (double) (this->mixingRate * KCAL2EV);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        idxNeigh = this->getNeighbors(iatom);
        posNeigh = this->getPositions(iatom);
        Iatom    = this->indexOfLAMMPS(iatom);

        nbond   = this->numBonds[iatom];
        idxBond = this->idxBonds[iatom];

        dBOdr_raw = this->dBOdrs_raw[iatom];
        dEdBO_raw = this->dEdBOs_raw[iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = this->getNeighbor(idxNeigh, ineigh);
            Jatom  = this->indexOfLAMMPS(jatom);

            if (Iatom >= Jatom) continue;

            r  = (double)   posNeigh[ineigh][0];
            dx = (double) (-posNeigh[ineigh][1]);
            dy = (double) (-posNeigh[ineigh][2]);
            dz = (double) (-posNeigh[ineigh][3]);

            dBOdr_sigma = dBOdr_raw[ibond][0];
            dBOdr_pi    = dBOdr_raw[ibond][1];
            dBOdr_pipi  = dBOdr_raw[ibond][2];

            dEdBO_sigma = dEdBO_raw[ibond][0];
            dEdBO_pi    = dEdBO_raw[ibond][1];
            dEdBO_pipi  = dEdBO_raw[ibond][2];

            dEdr = dEdBO_sigma * dBOdr_sigma
                 + dEdBO_pi    * dBOdr_pi
                 + dEdBO_pipi  * dBOdr_pipi;

            Fr = -escale * ((double) dEdr) / r;
            Fx = Fr * dx;
            Fy = Fr * dy;
            Fz = Fr * dz;

            atom->f[Iatom][0] += Fx;
            atom->f[Iatom][1] += Fy;
            atom->f[Iatom][2] += Fz;

            atom->f[Jatom][0] -= Fx;
            atom->f[Jatom][1] -= Fy;
            atom->f[Jatom][2] -= Fz;

            if (pair->evflag)
            {
                pair->ev_tally(Iatom, Jatom, atom->nlocal, 1, // newton_pair has to be ON
                               0.0, 0.0, Fr, dx, dy, dz);
            }
        }
    }
}

