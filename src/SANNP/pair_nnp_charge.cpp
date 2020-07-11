/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "pair_nnp_charge.h"

using namespace LAMMPS_NS;

PairNNPCharge::PairNNPCharge(LAMMPS *lmp) : PairNNP(lmp)
{
    comm_forward = comm_reverse = 1;
}

PairNNPCharge::~PairNNPCharge()
{
    if (copymode)
    {
        return;
    }

    if (allocated)
    {
        memory->destroy(charges);
    }
}

void PairNNPCharge::allocate() {
    PairNNP::allocate();
    memory->create(charges, this->maxinum, "pair:charges");
}

bool PairNNPCharge::prepareNN()
{
    bool hasGrown = PairNNP::prepareNN();

    if (hasGrown)
    {
        memory->grow(charges, this->maxinum, "pair:charges");
    }

    return hasGrown;
}

void PairNNPCharge::performNN(int eflag)
{
    PairNNP::performNN(eflag);

    int i;
    int iatom;

    double qsum;
    double qsumlocal;
    double qoffset;

    double* q = atom->q;
    int nlocal = atom->nlocal;
    bigint natoms = atom->natoms;

    int inum = list->inum;
    int* ilist = list->ilist;

    if (this->property->getWithCharge() == 0)
    {
        return;
    }

    if (inum > 0)
    {
        this->arch->goForwardOnCharge();
        this->arch->obtainCharges(charges);
    }

    #pragma omp parallel for private(iatom, i)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];
        q[i] = charges[iatom];
    }

    qsumlocal = 0.0;

    #pragma omp parallel for reduction(+:qsumlocal) private(i)
    for (int i = 0; i < nlocal; i++) {
        qsumlocal += q[i];
    }

    MPI_Allreduce(&qsumlocal, &qsum, 1, MPI_DOUBLE, MPI_SUM, world);

    qoffset = qsum / natoms;

    #pragma omp parallel for private(iatom, i)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];
        q[i] -= qoffset;
    }

    comm->forward_comm_pair(this);

    if (force->kspace)
    {
        force->kspace->qsum_qsq();
    }
}

void PairNNPCharge::settings(int narg, char **arg)
{
    if (narg == 1)
    {
        this->cutcoul = force->numeric(FLERR, arg[0]);
    }

    else if (narg == 0)
    {
        this->cutcoul = -1.0; // setting when coeff
    }

    else
    {
        error->all(FLERR, "Illegal number of arguments for Pair style NNP with charge.");
    }
}

void PairNNPCharge::coeff(int narg, char **arg)
{
    PairNNP::coeff(narg, arg);

    if (this->cutcoul <= 0.0)
    {
        this->cutcoul = this->property->getRcutoff();
    }
}

void PairNNPCharge::init_style()
{
    if (!atom->q_flag)
    {
        error->all(FLERR, "Pair style NNP with charge requires atom attribute q");
    }

    PairNNP::init_style();
}

double PairNNPCharge::get_cutoff()
{
    double rcut = this->property->getRcutoff();
    return max(this->cutcoul, rcut);
}

void *PairNNPCharge::extract(const char *str, int &dim)
{
    dim = 1;
    if (strcmp(str,"cut_coul") == 0) return (void *) &(this->cutcoul);
    return NULL;
}

int PairNNPCharge::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
    int m;
    for(m = 0; m < n; m++) buf[m] = atom->q[list[m]];
    return m;
}

void PairNNPCharge::unpack_forward_comm(int n, int first, double *buf)
{
    int i, m;
    for(m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
}

int PairNNPCharge::pack_reverse_comm(int n, int first, double *buf)
{
    int i, m;
    for(m = 0, i = first; m < n; m++, i++) buf[m] = atom->q[i];
    return m;
}

void PairNNPCharge::unpack_reverse_comm(int n, int *list, double *buf)
{
    int m;
    for(m = 0; m < n; m++) atom->q[list[m]] += buf[m];
}

