/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "pair_nnp.h"

using namespace LAMMPS_NS;

PairNNP::PairNNP(LAMMPS *lmp) : Pair(lmp)
{
    this->typeMap = NULL;
    this->property = NULL;
    this->arch = NULL;

    int max = 10;
    this->maxinum = max;
    this->maxnneigh = max;
    this->maxnneighNN = max;
}

PairNNP::~PairNNP()
{
    if (copymode)
    {
        return;
    }

    if (this->typeMap != NULL)
    {
        delete this->typeMap;
    }

    if (this->property != NULL)
    {
        delete this->property;
    }

    if (this->arch != NULL)
    {
        delete this->arch;
    }

    if (allocated)
    {
        memory->destroy(cutsq);
        memory->destroy(setflag);
        memory->destroy(energies);
        memory->destroy(forces);
        memory->destroy(posNeighbor);
        memory->destroy(posNeighborNN);
        memory->destroy(numNeighborNN);
        memory->destroy(elemNeighborNN);
        memory->destroy(indexNeighborNN);
    }
}

void PairNNP::allocate()
{
    allocated = 1;
    int ntypes = atom->ntypes;

    memory->create(cutsq, ntypes + 1, ntypes + 1, "pair:cutsq");
    memory->create(setflag, ntypes + 1, ntypes + 1, "pair:setflag");

    memory->create(energies, this->maxinum, "pair:energies");
    memory->create(forces, this->maxinum, this->maxnneighNN, 3, "pair:forces");

    memory->create(posNeighbor, this->maxinum, this->maxnneigh, 4, "pair:posNeighbor");
    memory->create(indexNeighborNN, this->maxinum, this->maxnneigh, "pair:indexNeighborNN");

    memory->create(posNeighborNN, this->maxinum, this->maxnneighNN, 4, "pair:posNeighborNN");
    memory->create(numNeighborNN, this->maxinum, "pair:numNeighborNN");
    memory->create(elemNeighborNN, this->maxinum, this->maxnneighNN, "pair:elemNeighborNN");
}

void PairNNP::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);

    prepareNN();

    performNN(eflag);

    clearNN();

    if (vflag_fdotr)
    {
        virial_fdotr_compute();
    }
}

bool PairNNP::prepareNN()
{
    int i, j;
    int iatom, jatom;
    int ineigh, nneigh;

    int* type = atom->type;
    double** x = atom->x;

    int inum = list->inum;
    int* ilist = list->ilist;
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;

    real x0, y0, z0, dx, dy, dz, r, rr;
    real rcut = this->property->getRouter();

    bool hasGrown = false;

    // allocate
    // grow inum, nneigh
    nneigh = 0;
    #pragma omp parallel for private(iatom) reduction(max:nneigh)
    for (int iatom = 0; iatom < inum; ++iatom)
    {
        nneigh = max(nneigh, numneigh[iatom]);
    }

    if (inum > this->maxinum)
    {
        hasGrown = true;

        this->maxinum = inum + this->maxinum / 2;

        memory->grow(energies, this->maxinum, "pair:energies");
        memory->grow(numNeighborNN, this->maxinum, "pair:numNeighborNN");

        if (nneigh > this->maxnneigh)
        {
            this->maxnneigh = nneigh + this->maxnneigh / 2;
        }

        memory->grow(indexNeighborNN, this->maxinum, this->maxnneigh, "pair:indexNeighborNN");
        memory->grow(posNeighbor, this->maxinum, this->maxnneigh, 4, "pair:posNighbor");
    }
    else if (nneigh > this->maxnneigh)
    {
        this->maxnneigh = nneigh + this->maxnneigh / 2;
        memory->grow(indexNeighborNN, this->maxinum, this->maxnneigh, "pair:indexNeighborNN");
        memory->grow(posNeighbor, this->maxinum, this->maxnneigh, 4, "pair:posNighbor");
    }

    // generate posNeighbor
    #pragma omp parallel for private(iatom, i, x0, y0, z0, nneigh, ineigh, j, dx, dy, dz, rr, r)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        x0 = x[i][0];
        y0 = x[i][1];
        z0 = x[i][2];

        nneigh = numneigh[i];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            j = firstneigh[i][ineigh];
            j &= NEIGHMASK;

            dx = x[j][0] - x0;
            dy = x[j][1] - y0;
            dz = x[j][2] - z0;

            rr = dx * dx + dy * dy + dz * dz;
            r = sqrt(rr);

            posNeighbor[iatom][ineigh][0] = r;
            posNeighbor[iatom][ineigh][1] = dx;
            posNeighbor[iatom][ineigh][2] = dy;
            posNeighbor[iatom][ineigh][3] = dz;
        }
    }

    // generate indexNeighborNN
    #pragma omp parallel for private(iatom, nneigh, ineigh)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        nneigh = numneigh[ilist[iatom]];
        numNeighborNN[iatom] = 0;

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            if (posNeighbor[iatom][ineigh][0] < rcut)
            {
                indexNeighborNN[iatom][numNeighborNN[iatom]] = ineigh;
                ++numNeighborNN[iatom];
            }
        }
    }

    // grow nneighNN
    nneigh = 0;
    #pragma omp parallel reduction(max:nneigh)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        nneigh = max(nneigh, numNeighborNN[iatom]);
    }

    if (nneigh > this->maxnneighNN)
    {
        this->maxnneighNN = nneigh + this->maxnneighNN / 2;

        memory->grow(forces, this->maxinum, this->maxnneighNN, 3, "pair:forces");
        memory->grow(posNeighborNN, this->maxinum, this->maxnneighNN, 4, "pair:posNeighborNN");
        memory->grow(elemNeighborNN, this->maxinum, this->maxnneighNN, "pair:elemNeighborNN");
    }
    else if (hasGrown)
    {
        memory->grow(forces, this->maxinum, this->maxnneighNN, 3, "pair:forces");
        memory->grow(posNeighborNN, this->maxinum, this->maxnneighNN, 4, "pair:posNeighborNN");
        memory->grow(elemNeighborNN, this->maxinum, this->maxnneighNN, "pair:elemNeighborNN");
    }

    // generate posNeighborNN
    #pragma omp parallel for private(iatom, i, nneigh, ineigh, jatom, j)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];
        nneigh = numNeighborNN[iatom];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            jatom = indexNeighborNN[iatom][ineigh];
            j = firstneigh[i][jatom];
            j &= NEIGHMASK;

            posNeighborNN[iatom][ineigh][0] = posNeighbor[iatom][jatom][0];
            posNeighborNN[iatom][ineigh][1] = posNeighbor[iatom][jatom][1];
            posNeighborNN[iatom][ineigh][2] = posNeighbor[iatom][jatom][2];
            posNeighborNN[iatom][ineigh][3] = posNeighbor[iatom][jatom][3];
            elemNeighborNN[iatom][ineigh] = this->typeMap[type[j]] - 1;
        }
    }

    this->arch->initGeometry(inum, ilist, type, this->typeMap, numNeighborNN);

    return hasGrown;
}

void PairNNP::performNN(int eflag)
{
    int i, j;
    int iatom;
    int ineigh, nneigh;

    double** f = atom->f;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    int inum = list->inum;
    int* ilist = list->ilist;
    int** firstneigh = list->firstneigh;

    double delx, dely, delz;
    double fx, fy, fz;

    double evdwl = 0.0;

    this->arch->calculateSymmFuncs(numNeighborNN, elemNeighborNN, posNeighborNN);

    this->arch->renormalizeSymmFuncs();

    this->arch->goForwardOnEnergy();
    this->arch->obtainEnergies(energies);

    this->arch->goBackwardOnForce();
    this->arch->obtainForces(forces);

    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        fx = forces[iatom][0][0];
        fy = forces[iatom][0][1];
        fz = forces[iatom][0][2];

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;

        if (eflag)
        {
            evdwl = energies[iatom];
            if (eflag_global) eng_vdwl += evdwl;
            if (eflag_atom)   eatom[i] += evdwl;
        }

        nneigh = numNeighborNN[iatom];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            j = firstneigh[i][indexNeighborNN[iatom][ineigh]];
            j &= NEIGHMASK;

            fx = forces[iatom][ineigh + 1][0];
            fy = forces[iatom][ineigh + 1][1];
            fz = forces[iatom][ineigh + 1][2];

            f[j][0] += fx;
            f[j][1] += fy;
            f[j][2] += fz;

            if (evflag)
            {
                delx = posNeighborNN[iatom][ineigh][1];
                dely = posNeighborNN[iatom][ineigh][2];
                delz = posNeighborNN[iatom][ineigh][3];

                ev_tally_xyz(i, j, nlocal, newton_pair,
                             0.0, 0.0, -fx, -fy, -fz, delx, dely, delz);
            }
        }
    }
}

void PairNNP::clearNN()
{
    this->arch->clearGeometry();
}

void PairNNP::settings(int narg, char **arg)
{
    if (narg != 0)
    {
        error->all(FLERR, "pair_style nnp command has unnecessary argument(s).");
    }
}

void PairNNP::coeff(int narg, char **arg)
{
    int i, j;
    int count;
    double r, rr;
    FILE* fp;

    int ntypes = atom->ntypes;
    int ntypesEff;
    char** typeNames;

    if (!allocated)
    {
        allocate();
    }

    if (narg != 3 + ntypes)
    {
        error->all(FLERR, "Incorrect number of arguments for pair_coeff.");
    }

    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    {
        error->all(FLERR, "Only wildcard asterisk is allowed in place of atom types for pair_coeff.");
    }

    if (this->typeMap != NULL)
    {
        delete this->typeMap;
    }

    this->typeMap = new int[ntypes + 1];

    ntypesEff = 0;
    typeNames = new char*[ntypes];

    for (i = 0; i < ntypes; ++i)
    {
        if (strcmp(arg[i + 3], "NULL") == 0)
        {
            this->typeMap[i + 1] = 0;
        }
        else
        {
            this->typeMap[i + 1] = ntypesEff + 1;
            typeNames[ntypesEff] = arg[i + 3];
        	ntypesEff++;
        }
    }

    if (ntypesEff < 1)
    {
        error->all(FLERR, "There are no elements for pair_coeff of NNP.");
    }

    if (comm->me == 0) {
        fp = fopen(arg[2], "r");

        if (fp == NULL)
        {
            error->all(FLERR, "cannot open ffield file.");
        }
    }

    if (this->property != NULL)
    {
        delete this->property;
    }

    this->property = new Property();
    this->property->peekProperty(fp, comm->me, world);

    if (this->arch != NULL)
    {
        delete this->arch;
    }

    bool withCharge = (this->property->getWithCharge() != 0);
    this->arch = new NNArch(withCharge ? NNARCH_MODE_BOTH : NNARCH_MODE_ENERGY, ntypesEff, this->property);

    this->arch->initLayers();
    this->arch->restoreNN(fp, ntypesEff, typeNames, comm->me, world);

    if (comm->me == 0) {
        fclose(fp);
    }

    delete[] typeNames;

    count = 0;
    r = get_cutoff();
    rr = r * r;

    for (i = 1; i <= ntypes; ++i)
    {
        for (j = i; j <= ntypes; ++j)
        {
            if (this->typeMap[i] > 0 && this->typeMap[j] > 0)
            {
                cutsq[i][j] = rr;
                setflag[i][j] = 1;
                count++;
            }
            else
            {
                setflag[i][j] = 0;
            }
        }
    }

    if (count == 0)
    {
        error->all(FLERR, "Incorrect args for pair coefficients");
    }
}

double PairNNP::init_one(int i, int j)
{
    if (setflag[i][j] == 0)
    {
        error->all(FLERR, "All pair coeffs are not set");
    }

    return get_cutoff();
}

void PairNNP::init_style()
{
    if (atom->tag_enable == 0)
    {
        error->all(FLERR,"Pair style NNP requires atom IDs");
    }

    if (force->newton_pair == 0)
    {
        error->all(FLERR, "Pair style NNP requires newton pair on");
    }

    int irequest = neighbor->request(this, instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
}

double PairNNP::get_cutoff()
{
    double router = this->property->getRouter();
    return router;
}

