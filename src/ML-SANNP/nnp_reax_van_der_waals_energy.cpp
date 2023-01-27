/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

void ReaxPot::calculateVanDerWaalsEnergy(int eflag, LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom)
{
    int iatom, Iatom;
    int jatom, Jatom;
    int latom = this->locAtoms;

    int       ineigh;
    int       nneigh;
    int*      idxNeigh;
    nnpreal** posNeigh;

    int ielem;
    int jelem;

    nnpreal r, r2, r3, r4, r5, r6, r7;
    nnpreal rcut = this->param->rcut_vdw;

    nnpreal D;
    nnpreal alpha;
    nnpreal gamma;
    nnpreal rvdw;
    nnpreal pvdw = this->param->p_vdw;
    nnpreal pc1, pc2, pc3;

    nnpreal rp, gp;
    nnpreal f13, g13, h13, i13;
    nnpreal df13dr;

    nnpreal Tap;
    nnpreal dTapdr;

    nnpreal Tap0 = this->param->Tap_vdw[0];
    nnpreal Tap1 = this->param->Tap_vdw[1];
    nnpreal Tap2 = this->param->Tap_vdw[2];
    nnpreal Tap3 = this->param->Tap_vdw[3];
    nnpreal Tap4 = this->param->Tap_vdw[4];
    nnpreal Tap5 = this->param->Tap_vdw[5];
    nnpreal Tap6 = this->param->Tap_vdw[6];
    nnpreal Tap7 = this->param->Tap_vdw[7];

    double  Evdw;
    nnpreal Evdw0;
    nnpreal Ecore;
    nnpreal dEvdw0dr;
    nnpreal dEcoredr;
    nnpreal dEdr;

    double dx, dy, dz;
    double Fr, Fx, Fy, Fz;

    LAMMPS_NS::tagint itag, jtag;
    double xtmp, ytmp, ztmp;

    double escale = (double) (this->mixingRate * KCAL2EV);

    for (iatom = 0; iatom < latom; ++iatom)
    {
        ielem    = this->getElement(iatom);
        nneigh   = this->numNeighbors(iatom);
        idxNeigh = this->getNeighbors(iatom);
        posNeigh = this->getPositions(iatom);
        Iatom    = this->indexOfLAMMPS(iatom);

        itag = atom->tag[Iatom];
        xtmp = atom->x[Iatom][0];
        ytmp = atom->x[Iatom][1];
        ztmp = atom->x[Iatom][2];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            r = posNeigh[ineigh][0];
            if (r <= ZERO || rcut <= r) continue;

            jatom = this->getNeighbor(idxNeigh, ineigh);
            jelem = this->getElement(jatom);
            if (jelem < 0) continue;

            Jatom = this->indexOfLAMMPS(jatom);

            // skip half of atoms
            jtag = atom->tag[Jatom];
            if (itag > jtag) {
                if ((itag + jtag) % 2 == 0) continue;
            } else if (itag < jtag) {
                if ((itag + jtag) % 2 == 1) continue;
            } else {
                if (atom->x[Jatom][2] < ztmp) continue;
                if (atom->x[Jatom][2] == ztmp && atom->x[Jatom][1] < ytmp) continue;
                if (atom->x[Jatom][2] == ztmp && atom->x[Jatom][1] == ytmp && atom->x[Jatom][0] < xtmp) continue;
            }

            dx = (double) (-posNeigh[ineigh][1]);
            dy = (double) (-posNeigh[ineigh][2]);
            dz = (double) (-posNeigh[ineigh][3]);

            D     = this->param->D_vdw    [ielem][jelem];
            alpha = this->param->alpha_vdw[ielem][jelem];
            gamma = this->param->gammaw   [ielem][jelem];
            rvdw  = this->param->r_vdw    [ielem][jelem];
            pc1   = this->param->p_core1  [ielem][jelem];
            pc2   = this->param->p_core2  [ielem][jelem];
            pc3   = this->param->p_core3  [ielem][jelem];

            if (this->param->shielding != 0)
            {
                rp     = pow(r, pvdw);
                gp     = pow(ONE / gamma, pvdw);
                f13    = pow(rp + gp, ONE / pvdw);
                df13dr = rp / r * f13 / (rp + gp);
            }
            else
            {
                f13    = r;
                df13dr = ONE;
            }

            g13 = alpha * (ONE - f13 / rvdw);
            h13 = exp(g13);
            i13 = exp(NNPREAL(0.5) * g13);

            Evdw0    = D * (h13 - NNPREAL(2.0) * i13);
            dEvdw0dr = D * alpha / rvdw * (i13 - h13) * df13dr;

            if (this->param->innerWall != 0)
            {
                Ecore    = pc2 * exp(pc3 * (ONE - r / pc1));
                dEcoredr = -pc3 / pc1 * Ecore;
            }
            else
            {
                Ecore    = ZERO;
                dEcoredr = ZERO;
            }

            r2 = r * r;
            r3 = r * r2;
            r4 = r * r3;
            r5 = r * r4;
            r6 = r * r5;
            r7 = r * r6;

            Tap  = Tap0;
            Tap += Tap1 * r;
            Tap += Tap2 * r2;
            Tap += Tap3 * r3;
            Tap += Tap4 * r4;
            Tap += Tap5 * r5;
            Tap += Tap6 * r6;
            Tap += Tap7 * r7;

            dTapdr  = Tap1;
            dTapdr += Tap2 * r  * NNPREAL(2.0);
            dTapdr += Tap3 * r2 * NNPREAL(3.0);
            dTapdr += Tap4 * r3 * NNPREAL(4.0);
            dTapdr += Tap5 * r4 * NNPREAL(5.0);
            dTapdr += Tap6 * r5 * NNPREAL(6.0);
            dTapdr += Tap7 * r6 * NNPREAL(7.0);

            Evdw = eflag ? ((double) (Tap * (Evdw0 + Ecore))) : 0.0;
            Evdw *= escale;

            dEdr = dTapdr * (Evdw0 + Ecore) + Tap * (dEvdw0dr + dEcoredr);
            dEdr *= escale;

            Fr = (double) (-dEdr / r);
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
                               Evdw, 0.0, Fr, dx, dy, dz);
            }
        }
    }
}

