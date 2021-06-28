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
    this->typeMap  = NULL;
    this->property = NULL;
    this->arch     = NULL;

    const int max   = 10;
    this->maxinum   = max;
    this->maxnneigh = max;
}

PairNNP::~PairNNP()
{
    if (copymode)
    {
        return;
    }

    if (this->typeMap != NULL)
    {
        delete[] this->typeMap;
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
        memory->destroy(this->elements);
        memory->destroy(this->energies);
        memory->destroy(this->forces);
        memory->destroy(this->numNeighbor);
        memory->destroy(this->elemNeighbor);
        memory->destroy(this->posNeighbor);
    }
}

void PairNNP::allocate()
{
    allocated = 1;

    const int ntypes = atom->ntypes;

    const int dim = this->dimensionPosNeighbor();

    memory->create(cutsq,   ntypes + 1, ntypes + 1, "pair:cutsq");
    memory->create(setflag, ntypes + 1, ntypes + 1, "pair:setflag");

    memory->create(this->elements, this->maxinum,                         "pair:elements");
    memory->create(this->energies, this->maxinum,                         "pair:energies");
    memory->create(this->forces,   this->maxinum, this->maxnneigh + 1, 3, "pair:forces");

    memory->create(this->numNeighbor,  this->maxinum,                       "pair:numNeighbor");
    memory->create(this->elemNeighbor, this->maxinum, this->maxnneigh,      "pair:elemNeighbor");
    memory->create(this->posNeighbor,  this->maxinum, this->maxnneigh, dim, "pair:posNeighbor");
}

void PairNNP::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);

    prepareNN();

    performNN(eflag);

    clearNN();

    computeLJLike(eflag);

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

    int itype, jtype;
    int* type = atom->type;
    double** x = atom->x;

    int inum = list->inum;
    int* ilist = list->ilist;
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;

    real x0, y0, z0, dx, dy, dz, r, rr, fc, dfcdr;

    const int elemWeight = this->property->getElemWeight();
    const int cutoffMode = this->property->getCutoffMode();
    const double rcutRad = this->property->getRcutRadius();
    const double rcutAng = this->property->getRcutAngle();

    SymmFunc* symmFunc = this->arch->getSymmFunc();

    bool hasGrown = false;

    // grow with inum and nneigh
    nneigh = 0;
    #pragma omp parallel for private(iatom) reduction(max:nneigh)
    for (int iatom = 0; iatom < inum; ++iatom)
    {
        nneigh = max(nneigh, numneigh[ilist[iatom]]);
    }

    if (inum > this->maxinum)
    {
        hasGrown = true;

        this->maxinum = inum + this->maxinum / 2;

        memory->grow(this->elements,    this->maxinum, "pair:elements");
        memory->grow(this->energies,    this->maxinum, "pair:energies");
        memory->grow(this->numNeighbor, this->maxinum, "pair:numNeighbor");
    }

    if (hasGrown || nneigh > this->maxnneigh)
    {
        if (nneigh > this->maxnneigh)
        {
            this->maxnneigh = nneigh + this->maxnneigh / 2;
        }

        const int dim = this->dimensionPosNeighbor();

        memory->grow(this->forces,       this->maxinum, this->maxnneigh + 1, 3, "pair:forces");
        memory->grow(this->elemNeighbor, this->maxinum, this->maxnneigh,        "pair:elemNighbor");
        memory->grow(this->posNeighbor,  this->maxinum, this->maxnneigh,   dim, "pair:posNighbor");
    }

    // generate numNeighbor, elemNeighbor, posNeighbor
    #pragma omp parallel for private(iatom, i, itype, x0, y0, z0, \
                                     nneigh, ineigh, j, jtype, dx, dy, dz, rr, r, fc, dfcdr)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        x0 = x[i][0];
        y0 = x[i][1];
        z0 = x[i][2];

        nneigh = numneigh[i];

        this->numNeighbor[iatom] = nneigh;

        itype = this->typeMap[type[i]];
        this->elements[iatom] = itype - 1;

        if (elemWeight == 0)
        {
            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                j = firstneigh[i][ineigh];
                j &= NEIGHMASK;

                jtype = this->typeMap[type[j]];
                this->elemNeighbor[iatom][ineigh] = jtype - 1;
            }
        }
        else
        {
            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                j = firstneigh[i][ineigh];
                j &= NEIGHMASK;

                jtype = this->typeMap[type[j]];
                this->elemNeighbor[iatom][ineigh] = this->arch->getAtomNum(jtype - 1);
            }
        }

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            j = firstneigh[i][ineigh];
            j &= NEIGHMASK;

            dx = x[j][0] - x0;
            dy = x[j][1] - y0;
            dz = x[j][2] - z0;

            rr = dx * dx + dy * dy + dz * dz;
            r = sqrt(rr);

            this->posNeighbor[iatom][ineigh][0] = r;
            this->posNeighbor[iatom][ineigh][1] = dx;
            this->posNeighbor[iatom][ineigh][2] = dy;
            this->posNeighbor[iatom][ineigh][3] = dz;
        }

        if (cutoffMode == CUTOFF_MODE_SINGLE)
        {
            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                r = this->posNeighbor[iatom][ineigh][0];

                if (r < rcutRad)
                {
                    symmFunc->cutoffFunction(&fc, &dfcdr, r, rcutRad);
                }
                else
                {
                    fc    = 0.0;
                    dfcdr = 0.0;
                }

                this->posNeighbor[iatom][ineigh][4] = fc;
                this->posNeighbor[iatom][ineigh][5] = dfcdr;
            }
        }

        else if (cutoffMode == CUTOFF_MODE_DOUBLE)
        {
            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                r = this->posNeighbor[iatom][ineigh][0];

                if (r < rcutRad)
                {
                    symmFunc->cutoffFunction(&fc, &dfcdr, r, rcutRad);
                }
                else
                {
                    fc    = 0.0;
                    dfcdr = 0.0;
                }

                this->posNeighbor[iatom][ineigh][4] = fc;
                this->posNeighbor[iatom][ineigh][5] = dfcdr;

                if (r < rcutAng)
                {
                    symmFunc->cutoffFunction(&fc, &dfcdr, r, rcutAng);
                }
                else
                {
                    fc    = 0.0;
                    dfcdr = 0.0;
                }

                this->posNeighbor[iatom][ineigh][6] = fc;
                this->posNeighbor[iatom][ineigh][7] = dfcdr;
            }
        }

        else if (cutoffMode == CUTOFF_MODE_IPSO)
        {
            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                r = this->posNeighbor[iatom][ineigh][0];

                if (r < rcutRad)
                {
                    symmFunc->cutoffFunction(&fc, &dfcdr, r, rcutRad);
                }
                else
                {
                    fc    = 0.0;
                    dfcdr = 0.0;
                }

                this->posNeighbor[iatom][ineigh][4] = fc;
                this->posNeighbor[iatom][ineigh][5] = dfcdr;
                this->posNeighbor[iatom][ineigh][6] = fc;
                this->posNeighbor[iatom][ineigh][7] = dfcdr;
            }
        }
    }

    if (inum > 0)
    {
        this->arch->initGeometry(inum, this->elements,
                                 this->numNeighbor, this->elemNeighbor, this->posNeighbor);
    }

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

    if (inum > 0)
    {
        this->arch->calculateSymmFuncs();

        this->arch->renormalizeSymmFuncs();

        this->arch->goForwardOnEnergy();
        this->arch->obtainEnergies(energies);

        this->arch->goBackwardOnForce();
        this->arch->obtainForces(forces);
    }

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

        nneigh = this->numNeighbor[iatom];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            j = firstneigh[i][ineigh];
            j &= NEIGHMASK;

            fx = forces[iatom][ineigh + 1][0];
            fy = forces[iatom][ineigh + 1][1];
            fz = forces[iatom][ineigh + 1][2];

            f[j][0] += fx;
            f[j][1] += fy;
            f[j][2] += fz;

            if (evflag)
            {
                delx = this->posNeighbor[iatom][ineigh][1];
                dely = this->posNeighbor[iatom][ineigh][2];
                delz = this->posNeighbor[iatom][ineigh][3];

                ev_tally_xyz(i, j, nlocal, newton_pair,
                             0.0, 0.0, -fx, -fy, -fz, delx, dely, delz);
            }
        }
    }
}

void PairNNP::clearNN()
{
    int inum = list->inum;

    if (inum > 0)
    {
        this->arch->clearGeometry();
    }
}

void PairNNP::computeLJLike(int eflag)
{
    if (this->property->getWithClassical() == 0)
    {
        return;
    }

    int* type = atom->type;
    double** x = atom->x;
    double** f = atom->f;
    tagint *tag = atom->tag;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    int inum = list->inum;
    int* ilist = list->ilist;
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;

    double r, r2, r6, r8, r10, r12;
    double rcut = this->property->getRcutoff();

    double delx, dely, delz;
    double fx, fy, fz;

    int i, j;
    int ii, jj, jnum;
    int ielem1, jelem1;
    int ielem2, jelem2;
    int kelem;
    int *jlist;
    tagint itag, jtag;
    double xtmp, ytmp, ztmp;
    double evdwl, fpair;

    double* ljlikeA1 = this->arch->getLJLikeA1();
    double* ljlikeA2 = this->arch->getLJLikeA2();
    double* ljlikeA3 = this->arch->getLJLikeA3();
    double* ljlikeA4 = this->arch->getLJLikeA4();
    double A1, A2, A3, A4;
    double B1, B2, B3, B4;

    evdwl = 0.0;

    for (ii = 0; ii < inum; ii++)
    {
        i = ilist[ii];
        itag = tag[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];

        ielem1 = this->typeMap[type[i]] - 1;

        jlist = firstneigh[i];
        jnum = numneigh[i];

        for (jj = 0; jj < jnum; jj++)
        {
            j = jlist[jj];
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

            r    = this->posNeighbor[ii][jj][0];
            delx = this->posNeighbor[ii][jj][1];
            dely = this->posNeighbor[ii][jj][2];
            delz = this->posNeighbor[ii][jj][3];

            jelem1 = this->typeMap[type[j]] - 1;

            if (r < rcut)
            {
                ielem2 = max(ielem1, jelem1);
                jelem2 = min(ielem1, jelem1);
                kelem  = jelem2 + ielem2 * (ielem2 + 1) / 2;

                A1 = ljlikeA1[kelem];
                A2 = ljlikeA2[kelem];
                A3 = ljlikeA3[kelem];
                A4 = ljlikeA4[kelem];

                r2  = r * r;
                r6  = r2 * r2 * r2;
                r8  = r2 * r6;
                r10 = r2 * r8;
                r12 = r2 * r10;

                B1 = A1 / r12;
                B2 = A2 / r10;
                B3 = A3 / r8;
                B4 = A4 / r6;

                fpair = 12.0 * B1 + 10.0 * B2 + 8.0 * B3 + 6.0 * B4;
                fpair /= r2;

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
                    evdwl = B1 + B2 + B3 + B4;
                }

                if (evflag)
                {
                    ev_tally(i, j, nlocal, newton_pair,
                             evdwl, 0.0, fpair, delx, dely, delz);
                }
            }
        }
    }
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

    if (narg != (3 + ntypes))
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
        this->typeMap[i + 1] = 0;

        if (strcmp(arg[i + 3], "NULL") == 0)
        {
            continue;
        }

        for (j = 0; j < i; ++j)
        {
            if (strcmp(arg[i + 3], arg[j + 3]) == 0)
            {
                this->typeMap[i + 1] = this->typeMap[j + 1];
                break;
            }
        }

        if (this->typeMap[i + 1] == 0)
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
    this->property->readProperty(fp, comm->me, world);

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

    if (!allocated)
    {
        allocate();
    }

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
    return this->property->getRcutoff();
}

int PairNNP::dimensionPosNeighbor()
{
    const int cutoffMode = this->property->getCutoffMode();

    int dim;

    if (cutoffMode == CUTOFF_MODE_SINGLE)
    {
        dim = 6;
    }
    else if (cutoffMode == CUTOFF_MODE_DOUBLE || cutoffMode == CUTOFF_MODE_IPSO)
    {
        dim = 8;
    }
    else
    {
        dim = 4;
    }

    return dim;
}

