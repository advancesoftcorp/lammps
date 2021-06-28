/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "pair_nnp_coul_cut.h"

using namespace LAMMPS_NS;

PairNNPCoulCut::PairNNPCoulCut(LAMMPS *lmp) : PairNNPCharge(lmp)
{
    // NOP
}

PairNNPCoulCut::~PairNNPCoulCut()
{
    // NOP
}

void PairNNPCoulCut::compute(int eflag, int vflag)
{
    double** x = atom->x;
    double** f = atom->f;
    double* q = atom->q;
    tagint *tag = atom->tag;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    int inum = list->inum;
    int* ilist = list->ilist;
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;

    double r;
    double rcut = this->property->getRcutoff();

    double delx, dely, delz;
    double fx, fy, fz;

    int i, j;
    int ii, jj, jnum;
    int *jlist;
    tagint itag, jtag;
    double xtmp, ytmp, ztmp, qtmp;
    double r2inv, rinv;
    double ecoul, fpair;
    double forcecoul, factor_coul, fc, dfcdr;

    double *special_coul = force->special_coul;
    double qqrd2e = force->qqrd2e;

    ecoul = 0.0;
    ev_init(eflag, vflag);

    prepareNN();

    performNN(eflag);

    clearNN();

    computeLJLike(eflag);

    for (ii = 0; ii < inum; ii++)
    {
        i = ilist[ii];
        itag = tag[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        qtmp = q[i];

        jlist = firstneigh[i];
        jnum = numneigh[i];

        for (jj = 0; jj < jnum; jj++)
        {
            j = jlist[jj];
            factor_coul = special_coul[sbmask(j)];
            j &= NEIGHMASK;

            // skip half of atoms
            jtag = tag[j];
            if (itag > jtag) {
                if ((itag + jtag) % 2 == 0) continue;
            } else if (itag < jtag) {
                if ((itag + jtag) % 2 == 1) continue;
            } else {
                if (x[j][2] < ztmp) continue;
                if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
                if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
            }

            r    =  this->posNeighbor[ii][jj][0];
            delx = -this->posNeighbor[ii][jj][1];
            dely = -this->posNeighbor[ii][jj][2];
            delz = -this->posNeighbor[ii][jj][3];

            if (r < this->cutcoul)
            {
                rinv = 1.0 / r;
                r2inv = rinv * rinv;
                forcecoul = qqrd2e * qtmp * q[j] * rinv;

                if (r < rcut)
                {
                    fc = 0.5 * (1.0 - cos(PI * r / rcut));
                    dfcdr = 0.5 * PId / rcut * sin(PId * r / rcut);
                    fpair = factor_coul * forcecoul * (rinv * fc - dfcdr) * rinv;
                }
                else
                {
                    fc = 1.0;
                    fpair = factor_coul * forcecoul * r2inv;
                }

                fx = delx * fpair;
                fy = dely * fpair;
                fz = delz * fpair;

                f[i][0] += fx;
                f[i][1] += fy;
                f[i][2] += fz;

                f[j][0] -= fx;
                f[j][1] -= fy;
                f[j][2] -= fz;

                if (eflag)
                {
                    ecoul = factor_coul * forcecoul * fc;
                }

                if (evflag)
                {
                    ev_tally(i, j, nlocal, newton_pair,
                             0.0, ecoul, fpair, delx, dely, delz);
                }
            }
        }
    }

    if (vflag_fdotr)
    {
        virial_fdotr_compute();
    }
}

void PairNNPCoulCut::settings(int narg, char **arg)
{
    if (narg != 1)
    {
        error->all(FLERR, "Illegal number of arguments for pair_style nnp/coul/cut command.");
    }

    this->cutcoul = force->numeric(FLERR, arg[0]);
}

