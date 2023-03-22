/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

ReaxPot::ReaxPot(nnpreal rcut, nnpreal mixingRate, LAMMPS_NS::Memory* memory, FILE* fp, int rank, MPI_Comm world)
{
    if (memory == nullptr)
    {
        stop_by_error("memory is null.");
    }

    this->param = new ReaxParam(rcut, fp, rank, world);

    this->mixingRate = mixingRate;

    this->memory = memory;

    const int imax = 10;

    this->locAtoms    = 0;
    this->locAtomsAll = 0;
    this->numAtoms    = 0;
    this->numAtomsAll = 0;
    this->maxAtoms    = imax;
    this->maxAtomsAll = imax;

    this->memory->create(this->mapForwardAtoms,   this->maxAtomsAll, "nnpReax:mapForwardAtoms");
    this->memory->create(this->mapBackwardAtoms1, this->maxAtomsAll, "nnpReax:mapBackwardAtoms1");
    this->memory->create(this->mapBackwardAtoms2, this->maxAtomsAll, "nnpReax:mapBackwardAtoms2");

    this->typeMap       = nullptr;
    this->typeAll       = nullptr;
    this->numNeighsAll  = nullptr;
    this->idxNeighsAll  = nullptr;
    this->posNeighsAll  = nullptr;

    this->maxNeighs   = imax;
    this->maxBonds    = imax;
    this->maxBondsAll = imax;
    this->memory->create(this->numBonds,       this->maxAtoms,                       "nnpReax:numBonds");
    this->memory->create(this->idxBonds,       this->maxAtoms, this->maxNeighs,      "nnpReax:idxBonds");

    this->memory->create(this->BOs_raw,        this->maxAtoms, this->maxBondsAll, 3, "nnpReax:BOs_raw");
    this->memory->create(this->BOs_corr,       this->maxAtoms, this->maxBonds,    3, "nnpReax:BOs_corr");
    this->memory->create(this->Deltas_raw,     this->maxAtoms,                       "nnpReax:Deltas_raw");
    this->memory->create(this->Deltas_corr,    this->maxAtoms,                       "nnpReax:Deltas_corr");
    this->memory->create(this->Deltas_e,       this->maxAtoms,                       "nnpReax:Deltas_e");
    this->memory->create(this->exp1Deltas,     this->maxAtoms,                       "nnpReax:exp1Deltas");
    this->memory->create(this->exp2Deltas,     this->maxAtoms,                       "nnpReax:exp2Deltas");
    this->memory->create(this->n0lps,          this->maxAtoms,                       "nnpReax:n0lps");
    this->memory->create(this->nlps,           this->maxAtoms,                       "nnpReax:nlps");
    this->memory->create(this->Slps,           this->maxAtoms,                       "nnpReax:Slps");
    this->memory->create(this->Tlps,           this->maxAtoms,                       "nnpReax:Tlps");

    this->memory->create(this->dBOdrs_raw,     this->maxAtoms, this->maxBondsAll, 3, "nnpReax:dBOdrs_raw");
    this->memory->create(this->dBOdBOs,        this->maxAtoms, this->maxBonds,    5, "nnpReax:dBOdBOs");
    this->memory->create(this->dBOdDeltas,     this->maxAtoms, this->maxBonds,    3, "nnpReax:dBOdDeltas");
    this->memory->create(this->dEdBOs_raw,     this->maxAtoms, this->maxBonds,    3, "nnpReax:dEdBOs_raw");
    this->memory->create(this->dEdBOs_corr,    this->maxAtoms, this->maxBonds,    3, "nnpReax:dEdBOs_corr");
    this->memory->create(this->dEdDeltas_raw,  this->maxAtoms,                       "nnpReax:dEdDeltas_raw");
    this->memory->create(this->dEdDeltas_corr, this->maxAtoms,                       "nnpReax:dEdDeltas_corr");
    this->memory->create(this->dEdSlps,        this->maxAtoms,                       "nnpReax:dEdSlps");
    this->memory->create(this->dn0lpdDeltas,   this->maxAtoms,                       "nnpReax:dn0lpdDeltas");
    this->memory->create(this->dnlpdDeltas,    this->maxAtoms,                       "nnpReax:dnlpdDeltas");
    this->memory->create(this->dTlpdDeltas,    this->maxAtoms,                       "nnpReax:dTlpdDeltas");
    this->memory->create(this->dDeltadSlps,    this->maxAtoms,                       "nnpReax:dDeltadSlps");
    this->memory->create(this->dDeltadDeltas,  this->maxAtoms,                       "nnpReax:dDeltadDeltas");
    this->memory->create(this->Aovers,         this->maxAtoms,                       "nnpReax:Aovers");
    this->memory->create(this->Bovers,         this->maxAtoms,                       "nnpReax:Bovers");
    this->memory->create(this->Eunders,        this->maxAtoms,                       "nnpReax:Eunders");
    this->memory->create(this->dEunderdSlps,   this->maxAtoms,                       "nnpReax:dEunderdSlps");
    this->memory->create(this->dEunderdDeltas, this->maxAtoms,                       "nnpReax:dEunderdDeltas");
}

ReaxPot::~ReaxPot()
{
    delete this->param;

    if (this->typeMap != nullptr)
    {
        delete[] this->typeMap;
    }

    this->memory->destroy(this->mapForwardAtoms);
    this->memory->destroy(this->mapBackwardAtoms1);
    this->memory->destroy(this->mapBackwardAtoms2);

    this->memory->destroy(this->numBonds);
    this->memory->destroy(this->idxBonds);

    this->memory->destroy(this->BOs_raw);
    this->memory->destroy(this->BOs_corr);
    this->memory->destroy(this->Deltas_raw);
    this->memory->destroy(this->Deltas_corr);
    this->memory->destroy(this->Deltas_e);
    this->memory->destroy(this->exp1Deltas);
    this->memory->destroy(this->exp2Deltas);
    this->memory->destroy(this->n0lps);
    this->memory->destroy(this->nlps);
    this->memory->destroy(this->Slps);
    this->memory->destroy(this->Tlps);

    this->memory->destroy(this->dBOdrs_raw);
    this->memory->destroy(this->dBOdBOs);
    this->memory->destroy(this->dBOdDeltas);
    this->memory->destroy(this->dEdBOs_raw);
    this->memory->destroy(this->dEdBOs_corr);
    this->memory->destroy(this->dEdDeltas_raw);
    this->memory->destroy(this->dEdDeltas_corr);
    this->memory->destroy(this->dEdSlps);
    this->memory->destroy(this->dn0lpdDeltas);
    this->memory->destroy(this->dnlpdDeltas);
    this->memory->destroy(this->dTlpdDeltas);
    this->memory->destroy(this->dDeltadSlps);
    this->memory->destroy(this->dDeltadDeltas);
    this->memory->destroy(this->Aovers);
    this->memory->destroy(this->Bovers);
    this->memory->destroy(this->Eunders);
    this->memory->destroy(this->dEunderdSlps);
    this->memory->destroy(this->dEunderdDeltas);
}

void ReaxPot::initElements(int ntypes, int* atomNums)
{
    if (ntypes < 1 || atomNums == nullptr)
    {
        stop_by_error("type's data is empty.");
    }

    if (this->typeMap != nullptr)
    {
        delete[] this->typeMap;
    }

    this->typeMap = new int[ntypes];

    int itype;
    int ielem;
    int nelem = this->param->numElems;
    int atomNum;

    for (itype = 0; itype < ntypes; ++itype)
    {
        // atomic type of LAMMPS -> atomic number
        atomNum = atomNums[itype];

        // atomic number -> atomic type of ReaxFF
        ielem = atomNum > 0 ? this->param->atomNumToElement(atomNum) : -1;

        if (0 <= ielem && ielem < nelem)
        {
            this->typeMap[itype] = ielem;
        }
        else
        {
            this->typeMap[itype] = -1;
        }
    }
}

void ReaxPot::initGeometry(int locAtoms, int numAtoms, int* type,
                           int* ilist, int* numNeighbor, int** idxNeighbor, nnpreal*** posNeighbor)
{
    if (type == nullptr)
    {
        stop_by_error("type's data is null.");
    }

    if (ilist == nullptr || numNeighbor == nullptr || idxNeighbor == nullptr || posNeighbor == nullptr)
    {
        stop_by_error("geometric data is null.");
    }

    this->locAtomsAll = locAtoms;
    this->numAtomsAll = numAtoms;

    this->typeAll      = type;
    this->numNeighsAll = numNeighbor;
    this->idxNeighsAll = idxNeighbor;
    this->posNeighsAll = posNeighbor;

    int Iatom;
    int itype;
    int ielem;
    int nelem = this->param->numElems;

    if (this->maxAtomsAll < this->numAtomsAll)
    {
        this->maxAtomsAll = good_memory_size(this->numAtomsAll);
        this->memory->grow(this->mapForwardAtoms,   this->maxAtomsAll, "nnpReax:mapForwardAtoms");
        this->memory->grow(this->mapBackwardAtoms1, this->maxAtomsAll, "nnpReax:mapBackwardAtoms1");
        this->memory->grow(this->mapBackwardAtoms2, this->maxAtomsAll, "nnpReax:mapBackwardAtoms2");
    }

    // count available atoms
    this->locAtoms = 0;

    for (Iatom = 0; Iatom < this->locAtomsAll; ++Iatom)
    {
        itype = this->typeAll[Iatom];
        ielem = this->typeMap[itype - 1];

        if (0 <= ielem && ielem < nelem)
        {
            this->mapForwardAtoms  [ilist[Iatom]]   = this->locAtoms;
            this->mapBackwardAtoms1[this->locAtoms] = ilist[Iatom];
            this->mapBackwardAtoms2[this->locAtoms] = Iatom;
            this->locAtoms++;
        }
        else
        {
            this->mapForwardAtoms[ilist[Iatom]] = -1;
        }
    }

    this->numAtoms = this->locAtoms;

    for (Iatom = this->locAtomsAll; Iatom < this->numAtomsAll; ++Iatom)
    {
        itype = this->typeAll[Iatom];
        ielem = this->typeMap[itype - 1];

        if (0 <= ielem && ielem < nelem)
        {
            this->mapForwardAtoms  [ilist[Iatom]]   = this->numAtoms;
            this->mapBackwardAtoms1[this->numAtoms] = ilist[Iatom];
            this->mapBackwardAtoms2[this->numAtoms] = Iatom;
            this->numAtoms++;
        }
        else
        {
            this->mapForwardAtoms[ilist[Iatom]] = -1;
        }
    }
}

void ReaxPot::calculatePotential(int eflag, LAMMPS_NS::Pair* pair, LAMMPS_NS::Atom* atom)
{
    if (pair == nullptr)
    {
        stop_by_error("pair is null.");
    }

    if (atom == nullptr)
    {
        stop_by_error("atom is null.");
    }

    if (this->locAtoms < 1)
    {
        return;
    }

    this->calculateBondOrder();
    this->calculateLonePairNumber();
    this->calculateBondEnergy(eflag, pair);
    this->calculateLonePairEnergy(eflag, pair);
    this->calculateOverCoordEnergy(eflag, pair);
    this->calculateBondOrderForce(pair, atom);
    this->calculateVanDerWaalsEnergy(eflag, pair, atom);
}

