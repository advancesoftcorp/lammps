/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_nnarch.h"

#define MIN_NEIGHBOR  10

NNArch::NNArch(int numElems, const Property* property, LAMMPS_NS::Memory* memory)
{
    if (numElems < 1)
    {
        stop_by_error("#elements is not positive.");
    }

    if (property == nullptr)
    {
        stop_by_error("property is null.");
    }

    if (memory == nullptr)
    {
        stop_by_error("memory is null.");
    }

    int ielem;

    this->mode     = (property->getWithCharge() != 0) ? NNARCH_MODE_BOTH : NNARCH_MODE_ENERGY;
    this->numElems = numElems;
    this->numAtoms = 0;
    this->property = property;
    this->memory   = memory;

    this->atomNum = new int[this->numElems];

    this->elements     = nullptr;
    this->numNeighbor  = nullptr;
    this->idxNeighbor  = nullptr;
    this->elemNeighbor = nullptr;
    this->posNeighbor  = nullptr;

    this->sizeNumAtom  = 0;
    this->sizeTotNeigh = 0;
    this->sizeNbatch   = new int[this->numElems];

    for (ielem = 0; ielem < this->numElems; ++ielem)
    {
        this->sizeNbatch[ielem] = 0;
    }

    this->nbatch = new int[this->numElems];
    this->ibatch = nullptr;

    if (this->isEnergyMode())
    {
        this->energyData = new nnpreal*[this->numElems];
        this->energyGrad = new nnpreal*[this->numElems];

        for (ielem = 0; ielem < this->numElems; ++ielem)
        {
            this->energyData[ielem] = nullptr;
            this->energyGrad[ielem] = nullptr;
        }
    }
    else
    {
        this->energyData = nullptr;
        this->energyGrad = nullptr;
    }

    this->forceData = nullptr;

    if (this->isChargeMode())
    {
        this->chargeData = new nnpreal*[this->numElems];

        for (ielem = 0; ielem < this->numElems; ++ielem)
        {
            this->chargeData[ielem] = nullptr;
        }
    }
    else
    {
        this->chargeData = nullptr;
    }

    this->symmData = nullptr;
    this->symmDiff = nullptr;
    this->symmAve  = new nnpreal[this->numElems];
    this->symmDev  = new nnpreal[this->numElems];
    this->symmFunc = nullptr;

    this->interLayersEnergy = nullptr;
    this->lastLayersEnergy  = nullptr;

    this->interLayersCharge = nullptr;
    this->lastLayersCharge  = nullptr;

    this->ljlikeA1 = nullptr;
    this->ljlikeA2 = nullptr;
    this->ljlikeA3 = nullptr;
    this->ljlikeA4 = nullptr;

    this->reaxPot = nullptr;
}

NNArch::~NNArch()
{
    int ielem;
    int nelem = this->numElems;

    int ilayer;
    int nlayerEnergy = this->property->getLayersEnergy();
    int nlayerCharge = this->property->getLayersCharge();

    delete[] this->atomNum;

    if (this->idxNeighbor != nullptr)
    {
        this->memory->destroy(this->idxNeighbor);
    }

    delete[] this->sizeNbatch;

    delete[] this->nbatch;

    if (this->ibatch != nullptr)
    {
        this->memory->destroy(this->ibatch);
    }

    if (this->energyData != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->energyData[ielem] != nullptr)
            {
                this->memory->destroy(this->energyData[ielem]);
            }
        }
        delete[] this->energyData;
    }

    if (this->energyGrad != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->energyGrad[ielem] != nullptr)
            {
                this->memory->destroy(this->energyGrad[ielem]);
            }
        }
        delete[] this->energyGrad;
    }

    if (this->forceData != nullptr)
    {
        this->memory->destroy(this->forceData);
    }

    if (this->chargeData != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->chargeData[ielem] != nullptr)
            {
                this->memory->destroy(this->chargeData[ielem]);
            }
        }
        delete[] this->chargeData;
    }

    if (this->symmData != nullptr)
    {
        this->memory->destroy(this->symmData);
    }

    if (this->symmDiff != nullptr)
    {
        this->memory->destroy(this->symmDiff);
    }

    delete[] this->symmAve;
    delete[] this->symmDev;

    if (this->symmFunc != nullptr)
    {
        delete this->symmFunc;
    }

    if (this->interLayersEnergy != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            for (ilayer = 0; ilayer < nlayerEnergy; ++ilayer)
            {
                delete this->interLayersEnergy[ielem][ilayer];
            }
            delete[] this->interLayersEnergy[ielem];
        }
        delete[] this->interLayersEnergy;
    }

    if (this->lastLayersEnergy != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            delete this->lastLayersEnergy[ielem];
        }
        delete[] this->lastLayersEnergy;
    }

    if (this->interLayersCharge != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            for (ilayer = 0; ilayer < nlayerCharge; ++ilayer)
            {
                delete this->interLayersCharge[ielem][ilayer];
            }
            delete[] this->interLayersCharge[ielem];
        }
        delete[] this->interLayersCharge;
    }

    if (this->lastLayersCharge != nullptr)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            delete this->lastLayersCharge[ielem];
        }
        delete[] this->lastLayersCharge;
    }

    if (this->ljlikeA1 != nullptr)
    {
        delete[] this->ljlikeA1;
    }
    if (this->ljlikeA2 != nullptr)
    {
        delete[] this->ljlikeA2;
    }
    if (this->ljlikeA3 != nullptr)
    {
        delete[] this->ljlikeA3;
    }
    if (this->ljlikeA4 != nullptr)
    {
        delete[] this->ljlikeA4;
    }

    if (this->reaxPot != nullptr)
    {
        delete this->reaxPot;
    }
}

void NNArch::restoreNN(FILE* fp, char** elemNames, bool zeroEatom, int rank, MPI_Comm world)
{
    int  symmFunc = this->property->getSymmFunc();
    int  m2 = this->property->getM2();
    int  m3 = this->property->getM3();
    int  mm2, mm3;
    int  i2;
    int  i3,  ii3;
    int  i3_, ii3_;
    int  j3,  jj3;
    int  j3_, jj3_;
    int  k3;
    int  nrad = this->property->getNumRadius();
    int  nang = this->property->getNumAngle();
    int  irad, iang;
    int  nbase, ibase, jbase;

    int ilayer;
    int nlayerEnergy = this->property->getLayersEnergy();
    int nlayerCharge = this->property->getLayersCharge();
    int nnodeEnergy  = this->property->getNodesEnergy();
    int nnodeCharge  = this->property->getNodesCharge();
    int activEnergy  = this->property->getActivEnergy();
    int activCharge  = this->property->getActivCharge();
    int withCharge   = this->property->getWithCharge();

    int ielem,  jelem;
    int kelem,  lelem;
    int kelem_, lelem_;
    int ielem1, ielem2;
    int jelem1, jelem2;
    int kelem1, kelem2;
    int nelemOld;
    int nelemNew = this->numElems;
    int melemOld;
    int melemNew;

    const int lenElemName = 32;
    char   elemName1[lenElemName];
    char   elemName2[lenElemName];
    char** elemNamesOld;
    char** elemNamesNew;

    int* mapElem;

    nnpreal* symmAveOld;
    nnpreal* symmDevOld;
    int*     atomNumOld;

    nnpreal A1, A2, A3, A4;

    nnpreal rcutReaxFF;
    nnpreal rateReaxFF;

    int* mapSymmFunc;

    NNLayer* oldLayer;

    int ierr;

    const int lenLine = 256;
    char line[lenLine];

    // read number of elements
    ierr = 0;
    if (rank == 0)
    {
        if (fgets(line, lenLine, fp) == nullptr)
        {
            ierr = 1;
        }

        if (ierr == 0)
        {
            if (sscanf(line, "%d", &nelemOld) != 1)
            {
                ierr = 1;
            }
            else if (nelemOld < 1)
            {
                ierr = 1;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot read ffield file, at nelem");

    MPI_Bcast(&nelemOld, 1, MPI_INT, 0, world);

    // read element's properties
    elemNamesOld = new char*[nelemOld];
    elemNamesNew = new char*[nelemNew];

    symmAveOld = new nnpreal[nelemOld];
    symmDevOld = new nnpreal[nelemOld];
    atomNumOld = new int    [nelemOld];

    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        elemNamesOld[ielem] = new char[lenElemName];
        symmAveOld  [ielem] = ZERO;
        symmDevOld  [ielem] = -ONE;
        atomNumOld  [ielem] = 0;

        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == nullptr)
            {
                ierr = 1;
            }

            if (ierr == 0)
            {
                if (sscanf(line, IFORM_S1_F2_D1, elemNamesOld[ielem], &symmAveOld[ielem],
                                                  &symmDevOld[ielem], &atomNumOld[ielem]) != 4)
                {
                    if (sscanf(line, IFORM_S1_F2, elemNamesOld[ielem], &symmAveOld[ielem],
                                                   &symmDevOld[ielem]) != 3)
                    {
                        ierr = 1;
                    }
                    else
                    {
                        atomNumOld[ielem] = 0;
                    }
                }
            }

            if (ierr == 0)
            {
                if (symmDevOld[ielem] <= ZERO)
                {
                    ierr = 1;
                }

                if (this->property->getElemWeight() != 0 || this->property->getWithReaxFF() != 0)
                {
                    if (atomNumOld[ielem] < 1)
                    {
                        ierr = 1;
                    }
                }
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot read ffield file, at element.");

        MPI_Bcast(elemNamesOld[ielem], lenElemName, MPI_CHAR, 0, world);
    }

    MPI_Bcast(&symmAveOld[0], nelemOld, MPI_NNPREAL, 0, world);
    MPI_Bcast(&symmDevOld[0], nelemOld, MPI_NNPREAL, 0, world);
    MPI_Bcast(&atomNumOld[0], nelemOld, MPI_INT,     0, world);

    for (ielem = 0; ielem < nelemNew; ++ielem)
    {
        elemNamesNew[ielem] = new char[lenElemName];
        strcpy(elemNamesNew[ielem], elemNames[ielem]);
    }

    // map of elements
    mapElem = new int[nelemOld];

    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        kelem = -1;
        for (jelem = 0; jelem < nelemNew; ++jelem)
        {
            if (strcmp(elemNamesOld[ielem], elemNamesNew[jelem]) == 0)
            {
                kelem = jelem;
                break;
            }
        }

        mapElem[ielem] = kelem;
    }

    // has all elements ?
    for (jelem = 0; jelem < nelemNew; ++jelem)
    {
        bool hasElem = false;

        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            if (jelem == mapElem[ielem])
            {
                hasElem = true;
                break;;
            }
        }

        if (!hasElem)
        {
            char message[256];
            sprintf(message, "ffield file does not have the element: %s", elemNamesNew[jelem]);
            stop_by_error(message);
        }
    }

    // set symmetry function's properties
    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        kelem = mapElem[ielem];
        if (kelem >= 0)
        {
            this->symmAve[kelem] = symmAveOld[ielem];
            this->symmDev[kelem] = symmDevOld[ielem];
            this->atomNum[kelem] = atomNumOld[ielem];
        }
    }

    // read parameters of LJ-like potential
    if (this->property->getWithClassical() != 0)
    {
        melemNew = nelemNew * (nelemNew + 1) / 2;

        this->ljlikeA1 = new nnpreal[melemNew];
        this->ljlikeA2 = new nnpreal[melemNew];
        this->ljlikeA3 = new nnpreal[melemNew];
        this->ljlikeA4 = new nnpreal[melemNew];

        for (kelem = 0; kelem < melemNew; ++kelem)
        {
            this->ljlikeA1[kelem] = ZERO;
            this->ljlikeA2[kelem] = ZERO;
            this->ljlikeA3[kelem] = ZERO;
            this->ljlikeA4[kelem] = ZERO;
        }

        ierr = 0;
        if (rank == 0)
        {
            melemOld = 0;
            if (fgets(line, lenLine, fp) == nullptr)
            {
                ierr = 1;
            }

            if (ierr == 0)
            {
                if (sscanf(line, "%d", &melemOld) != 1)
                {
                    ierr = 1;
                }
            }

            if (ierr != 0)
            {
                melemOld = 0;
            }

            for (lelem = 0; lelem < melemOld; ++lelem)
            {
                if (fgets(line, lenLine, fp) == nullptr)
                {
                    ierr = 1;
                    break;
                }

                if (sscanf(line, IFORM_S2_F4, elemName1, elemName2, &A1, &A2, &A3, &A4) != 6)
                {
                    ierr = 1;
                    break;
                }

                ielem1 = -1;
                ielem2 = -1;

                for (ielem = 0; ielem < nelemOld; ++ielem)
                {
                    if (ielem1 < 0 && strcmp(elemName1, elemNamesOld[ielem]) == 0)
                    {
                        ielem1 = ielem;
                    }

                    if (ielem2 < 0 &&strcmp(elemName2, elemNamesOld[ielem]) == 0)
                    {
                        ielem2 = ielem;
                    }

                    if (ielem1 > -1 && ielem2 > -1)
                    {
                        break;
                    }
                }

                if (ielem1 < 0 || ielem2 < 0)
                {
                    continue;
                }

                jelem1 = mapElem[ielem1];
                jelem2 = mapElem[ielem2];

                if (jelem1 < 0 || jelem2 < 0)
                {
                    continue;
                }

                kelem1 = max(jelem1, jelem2);
                kelem2 = min(jelem1, jelem2);
                kelem  = kelem2 + kelem1 * (kelem1 + 1) / 2;

                this->ljlikeA1[kelem] = A1;
                this->ljlikeA2[kelem] = A2;
                this->ljlikeA3[kelem] = A3;
                this->ljlikeA4[kelem] = A4;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot read ffield file, at LJ-like.");

        MPI_Bcast(&(this->ljlikeA1[0]), melemNew, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->ljlikeA2[0]), melemNew, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->ljlikeA3[0]), melemNew, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->ljlikeA4[0]), melemNew, MPI_NNPREAL, 0, world);
    }

    // release memory
    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        delete[] elemNamesOld[ielem];
    }

    for (ielem = 0; ielem < nelemNew; ++ielem)
    {
        delete[] elemNamesNew[ielem];
    }

    delete[] elemNamesOld;
    delete[] elemNamesNew;

    delete[] symmAveOld;
    delete[] symmDevOld;
    delete[] atomNumOld;

    // read parameters of ReaxFF
    if (this->property->getWithReaxFF() != 0)
    {
        rcutReaxFF = this->property->getRcutReaxFF();
        rateReaxFF = this->property->getRateReaxFF();

        this->reaxPot = new ReaxPot(rcutReaxFF, rateReaxFF, this->memory, fp, rank, world);
    }

    // map of symmetry functions
    if (this->property->getElemWeight() != 0)
    {
        int mang = symmFunc == SYMM_FUNC_BEHLER ? (nang * 2) : nang;
        nbase = nrad + mang;

        mapSymmFunc = nullptr;
    }

    else if (symmFunc == SYMM_FUNC_MANYBODY)
    {
        mm2 = nelemOld * m2;
        mm3 = nelemOld * m3;

        ibase = 0;
        nbase = mm2 + mm3 * (mm3 + 1) / 2 * m3;

        mapSymmFunc = new int[nbase];

        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            if (kelem < 0)
            {
                for (i2 = 0; i2 < m2; ++i2)
                {
                    mapSymmFunc[ibase] = -1;
                    ibase++;
                }
            }

            else
            {
                jbase = m2 * kelem;

                for (i2 = 0; i2 < m2; ++i2)
                {
                    mapSymmFunc[ibase] = i2 + jbase;
                    ibase++;
                }
            }
        }

        for (k3 = 0; k3 < m3; ++k3)
        {
            jbase = mm2 + k3 * mm3 * (mm3 + 1) / 2;

            for (jj3 = 0; jj3 < mm3; ++jj3)
            {
                j3    = jj3 % m3;
                jelem = jj3 / m3;
                lelem = mapElem[jelem];

                for (ii3 = 0; ii3 <= jj3; ++ii3)
                {
                    i3    = ii3 % m3;
                    ielem = ii3 / m3;
                    kelem = mapElem[ielem];

                    if (lelem < 0 || kelem < 0)
                    {
                        mapSymmFunc[ibase] = -1;
                    }

                    else
                    {
                        lelem_ = max(kelem, lelem);
                        kelem_ = min(kelem, lelem);

                        if (lelem_ == kelem_)
                        {
                            j3_ = max(i3, j3);
                            i3_ = min(i3, j3);
                        }
                        else
                        {
                            j3_ = j3;
                            i3_ = i3;
                        }

                        jj3_ = j3_ + m3 * lelem_;
                        ii3_ = i3_ + m3 * kelem_;

                        mapSymmFunc[ibase] = ii3_ + jj3_ * (jj3_ + 1) / 2 + jbase;
                    }

                    ibase++;
                }
            }
        }
    }

    else //if (symmFunc == SYMM_FUNC_BEHLER || symmFunc == SYMM_FUNC_CHEBYSHEV)
    {
        int mang = symmFunc == SYMM_FUNC_BEHLER ? (nang * 2) : nang;

        ibase = 0;
        nbase = nrad * nelemOld + mang * nelemOld * (nelemOld + 1) / 2;

        mapSymmFunc = new int[nbase];

        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            if (kelem < 0)
            {
                for (irad = 0; irad < nrad; ++irad)
                {
                    mapSymmFunc[ibase] = -1;
                    ibase++;
                }
            }

            else
            {
                jbase = nrad * kelem;

                for (irad = 0; irad < nrad; ++irad)
                {
                    mapSymmFunc[ibase] = irad + jbase;
                    ibase++;
                }
            }
        }

        for (jelem = 0; jelem < nelemOld; ++jelem)
        {
            lelem = mapElem[jelem];
            for (ielem = 0; ielem <= jelem; ++ielem)
            {
                kelem = mapElem[ielem];

                if (lelem < 0 || kelem < 0)
                {
                    for (iang = 0; iang < mang; ++iang)
                    {
                        mapSymmFunc[ibase] = -1;
                        ibase++;
                    }
                }

                else
                {
                    lelem_ = max(kelem, lelem);
                    kelem_ = min(kelem, lelem);

                    jbase = nrad * nelemNew + mang * (kelem_ + lelem_ * (lelem_ + 1) / 2);

                    for (iang = 0; iang < mang; ++iang)
                    {
                        mapSymmFunc[ibase] = iang + jbase;
                        ibase++;
                    }
                }
            }
        }
    }

    // read NN energy
    if (interLayersEnergy != nullptr && lastLayersEnergy != nullptr)
    {
        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            // the first layer
            if (mapSymmFunc != nullptr)
            {
                oldLayer = new NNLayer(nbase, nnodeEnergy, activEnergy);
                oldLayer->scanWeight(fp, rank, world);

                if (kelem >= 0)
                {
                    interLayersEnergy[kelem][0]->projectWeightFrom(oldLayer, mapSymmFunc);
                }

                delete oldLayer;
            }

            else
            {
                if (kelem < 0)
                {
                    oldLayer = new NNLayer(nbase, nnodeEnergy, activEnergy);
                    oldLayer->scanWeight(fp, rank, world);
                    delete oldLayer;
                }
                else
                {
                    interLayersEnergy[kelem][0]->scanWeight(fp, rank, world);
                }
            }


            // the 2nd ~ last layers (dummy)
            if (kelem < 0)
            {
                for (ilayer = 1; ilayer < nlayerEnergy; ++ilayer)
                {
                    oldLayer = new NNLayer(nnodeEnergy, nnodeEnergy, activEnergy);
                    oldLayer->scanWeight(fp, rank, world);
                    delete oldLayer;
                }

                oldLayer = new NNLayer(nnodeEnergy, 1, ACTIVATION_ASIS);
                oldLayer->scanWeight(fp, rank, world);
                delete oldLayer;
            }

            // the 2nd ~ last layers (real)
            else
            {
                for (ilayer = 1; ilayer < nlayerEnergy; ++ilayer)
                {
                    interLayersEnergy[kelem][ilayer]->scanWeight(fp, rank, world);
                }

                lastLayersEnergy[kelem]->scanWeight(fp, zeroEatom, rank, world);
            }
        }
    }

    // read NN charge
    if (withCharge != 0 && interLayersCharge != nullptr && lastLayersCharge != nullptr)
    {
        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            // the first layer
            if (mapSymmFunc != nullptr)
            {
                oldLayer = new NNLayer(nbase, nnodeCharge, activCharge);
                oldLayer->scanWeight(fp, rank, world);

                if (kelem >= 0)
                {
                    interLayersCharge[kelem][0]->projectWeightFrom(oldLayer, mapSymmFunc);
                }

                delete oldLayer;
            }

            else
            {
                if (kelem < 0)
                {
                    oldLayer = new NNLayer(nbase, nnodeCharge, activCharge);
                    oldLayer->scanWeight(fp, rank, world);
                    delete oldLayer;
                }
                else
                {
                    interLayersCharge[kelem][0]->scanWeight(fp, rank, world);
                }
            }

            // the 2nd ~ last layers (dummy)
            if (kelem < 0)
            {
                for (ilayer = 1; ilayer < nlayerCharge; ++ilayer)
                {
                    oldLayer = new NNLayer(nnodeCharge, nnodeCharge, activCharge);
                    oldLayer->scanWeight(fp, rank, world);
                    delete oldLayer;
                }

                oldLayer = new NNLayer(nnodeCharge, 1, ACTIVATION_ASIS);
                oldLayer->scanWeight(fp, rank, world);
                delete oldLayer;
            }

            // the 2nd ~ last layers (real)
            else
            {
                for (ilayer = 1; ilayer < nlayerCharge; ++ilayer)
                {
                    interLayersCharge[kelem][ilayer]->scanWeight(fp, rank, world);
                }

                lastLayersCharge[kelem]->scanWeight(fp, rank, world);
            }
        }
    }

    delete[] mapElem;

    if (mapSymmFunc != nullptr)
    {
        delete[] mapSymmFunc;
    }
}

void NNArch::initGeometry(int numAtoms, int* elements,
                          int* numNeighbor, int** elemNeighbor, nnpreal*** posNeighbor)
{
    this->numAtoms = numAtoms;

    if (this->numAtoms < 1)
    {
        stop_by_error("#atoms is not positive.");
    }

    if (elements == nullptr || numNeighbor == nullptr || elemNeighbor == nullptr || posNeighbor == nullptr)
    {
        stop_by_error("geometric data is null.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;
    int nelem = this->numElems;

    int totNeigh;

    int ilayer;
    int nlayer;

    int mbatch;
    int jbatch;
    int sizeJbatch;

    int nbase = this->getSymmFunc()->getNumBasis();

    this->elements     = elements;
    this->numNeighbor  = numNeighbor;
    this->elemNeighbor = elemNeighbor;
    this->posNeighbor  = posNeighbor;

    int natomNew    = -1;
    int totNeighNew = -1;
    int jbatchNew;
    int nbatchNew[nelem];

    for (ielem = 0; ielem < nelem; ++ielem)
    {
        nbatchNew[ielem] = -1;
    }

    if (this->sizeNumAtom < natom)
    {
        natomNew = good_memory_size(natom);
    }

    // count index of neighbor
    if (natomNew > 0)
    {
        if (this->idxNeighbor == nullptr)
        {
            this->memory->create(this->idxNeighbor, natomNew, "nnp:idxNeighbor");
        }
        else
        {
            this->memory->grow  (this->idxNeighbor, natomNew, "nnp:idxNeighbor");
        }
    }

    totNeigh = 0;
    for (iatom = 0; iatom < natom; ++iatom)
    {
        this->idxNeighbor[iatom] = totNeigh;
        totNeigh += this->numNeighbor[iatom];
    }

    if (this->sizeTotNeigh < 1)
    {
        totNeigh = max(totNeigh, natom * MIN_NEIGHBOR);
    }

    if (this->sizeTotNeigh < totNeigh)
    {
        totNeighNew = good_memory_size(totNeigh);
    }

    // count size of batch
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        this->nbatch[ielem] = 0;
    }

    if (natomNew > 0)
    {
        if (this->ibatch == nullptr)
        {
            this->memory->create(this->ibatch, natomNew, "nnp:ibatch");
        }
        else
        {
            this->memory->grow  (this->ibatch, natomNew, "nnp:ibatch");
        }
    }

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->elements[iatom];
        this->ibatch[iatom] = this->nbatch[ielem];
        this->nbatch[ielem]++;
    }

    mbatch = 0;
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        mbatch += this->nbatch[ielem];
    }

    if (mbatch < 1)
    {
        return;
    }

    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        jbatch     = this->nbatch    [ielem];
        sizeJbatch = this->sizeNbatch[ielem];

        if (sizeJbatch < jbatch)
        {
            nbatchNew[ielem] = good_memory_size(jbatch);
        }
    }

    if (this->isEnergyMode())
    {
        // (re)allocate memory of energies
        nlayer = this->property->getLayersEnergy();

        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->nbatch[ielem] < 1)
            {
                continue;
            }

            jbatch    = nbatch   [ielem];
            jbatchNew = nbatchNew[ielem];

            if (jbatchNew > 0)
            {
                char nameData[64];
                char nameGrad[64];
                sprintf(nameData, "nnp:energyData%d", ielem);
                sprintf(nameGrad, "nnp:energyGrad%d", ielem);

                if (this->energyData[ielem] == nullptr)
                {
                    this->memory->create(this->energyData[ielem], jbatchNew, nameData);
                }
                else
                {
                    this->memory->grow  (this->energyData[ielem], jbatchNew, nameData);
                }

                if (this->energyGrad[ielem] == nullptr)
                {
                    this->memory->create(this->energyGrad[ielem], jbatchNew, nameGrad);
                }
                else
                {
                    this->memory->grow  (this->energyGrad[ielem], jbatchNew, nameGrad);
                }
            }

            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersEnergy[ielem][ilayer]->setSizeOfBatch(jbatch);
            }

            this->lastLayersEnergy[ielem]->setSizeOfBatch(jbatch);
        }

        // (re)allocate memory of forces
        if (totNeighNew > 0)
        {
            if (this->forceData == nullptr)
            {
                this->memory->create(this->forceData, 3 * totNeighNew, "nnp:forceData");
            }
            else
            {
                this->memory->grow  (this->forceData, 3 * totNeighNew, "nnp:forceData");
            }
        }
    }

    if (this->isChargeMode())
    {
        // (re)allocate memory of charges
        nlayer = this->property->getLayersCharge();

        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->nbatch[ielem] < 1)
            {
                continue;
            }

            jbatch    = nbatch   [ielem];
            jbatchNew = nbatchNew[ielem];

            if (jbatchNew > 0)
            {
                char nameData[64];
                sprintf(nameData, "nnp:chargeData%d", ielem);

                if (this->chargeData[ielem] == nullptr)
                {
                    this->memory->create(this->chargeData[ielem], jbatchNew, nameData);
                }
                else
                {
                    this->memory->grow  (this->chargeData[ielem], jbatchNew, nameData);
                }
            }

            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersCharge[ielem][ilayer]->setSizeOfBatch(jbatch);
            }

            this->lastLayersCharge[ielem]->setSizeOfBatch(jbatch);
        }
    }

    // (re)allocate memory of symmetry functions
    if (natomNew > 0)
    {
        if (this->symmData == nullptr)
        {
            this->memory->create(this->symmData, natomNew * nbase, "nnp:symmData");
        }
        else
        {
            this->memory->grow  (this->symmData, natomNew * nbase, "nnp:symmData");
        }
    }

    if (totNeighNew > 0)
    {
        if (this->getSymmFunc()->isHiddenDiff())
        {
#ifdef _NNP_GPU
            this->getSymmFunc()->allocHiddenDiff(this->property->getGpuAtomBlock(), totNeighNew);
#else
            stop_by_error("hiddenDiff is only for GPU.");
#endif
        }
        else
        {
            if (this->symmDiff == nullptr)
            {
                this->memory->create(this->symmDiff, 3 * totNeighNew * nbase, "nnp:symmDiff");
            }
            else
            {
                this->memory->grow  (this->symmDiff, 3 * totNeighNew * nbase, "nnp:symmDiff");
            }
        }
    }

    // save size of memory
    if (natomNew > 0)
    {
        this->sizeNumAtom = natomNew;
    }

    if (totNeighNew > 0)
    {
        this->sizeTotNeigh = totNeighNew;
    }

    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        jbatchNew = nbatchNew[ielem];

        if (jbatchNew > 0)
        {
            this->sizeNbatch[ielem] = jbatchNew;
        }
    }
}

SymmFunc* NNArch::getSymmFunc()
{
    if (this->symmFunc == nullptr)
    {
        if (this->property->getSymmFunc() == SYMM_FUNC_MANYBODY)
        {
            int  m2 = this->property->getM2();
            int  m3 = this->property->getM3();

            nnpreal rinner = this->property->getRinner();
            nnpreal router = this->property->getRouter();

            this->symmFunc = new SymmFuncManyBody(this->numElems, false, m2, m3, rinner, router);
        }

        else if (this->property->getSymmFunc() == SYMM_FUNC_BEHLER)
        {
            int     nrad    = this->property->getNumRadius();
            int     nang    = this->property->getNumAngle();
            nnpreal rrad    = this->property->getRcutRadius();
            nnpreal rang    = this->property->getRcutAngle();
            bool    weight  = (this->property->getElemWeight() != 0);
            bool    tanhCut = (this->property->getTanhCutoff() != 0);

            bool useG4 = (this->property->getBehlerG4() != 0);

            const nnpreal* eta1 = this->property->getBehlerEta1();
            const nnpreal* eta2 = this->property->getBehlerEta2();
            const nnpreal* rs1  = this->property->getBehlerRs1();
            const nnpreal* rs2  = this->property->getBehlerRs2();
            const nnpreal* zeta = this->property->getBehlerZeta();

#ifdef _NNP_GPU
            // sizeAng = 2 * nang, for GPU
            SymmFuncGPUBehler* symmFuncBehler = nullptr;
            symmFuncBehler = new SymmFuncGPUBehler(
            this->numElems, tanhCut, weight, nrad, 2 * nang, rrad, rang, this->property->getCutoffMode());
            symmFuncBehler->setMaxThreadsPerBlock(property->getGpuThreads());
#else
            // sizeAng = nang, for CPU
            SymmFuncBehler* symmFuncBehler = nullptr;
            symmFuncBehler = new SymmFuncBehler(this->numElems, tanhCut, weight, nrad, nang, rrad, rang);
#endif

            symmFuncBehler->setRadiusData(eta1, rs1);
            symmFuncBehler->setAngleData(useG4, eta2, zeta, rs2);

            this->symmFunc = symmFuncBehler;
        }

        else if (this->property->getSymmFunc() == SYMM_FUNC_CHEBYSHEV)
        {
            int     nrad    = this->property->getNumRadius();
            int     nang    = this->property->getNumAngle();
            nnpreal rrad    = this->property->getRcutRadius();
            nnpreal rang    = this->property->getRcutAngle();
            bool    weight  = (this->property->getElemWeight() != 0);
            bool    tanhCut = (this->property->getTanhCutoff() != 0);

#ifdef _NNP_GPU
            SymmFuncGPUChebyshev* symmFuncChebyshev = nullptr;
            symmFuncChebyshev = new SymmFuncGPUChebyshev(
            this->numElems, tanhCut, weight, nrad, nang, rrad, rang, this->property->getCutoffMode());
            symmFuncChebyshev->setMaxThreadsPerBlock(property->getGpuThreads());
#else
            SymmFuncChebyshev* symmFuncChebyshev = nullptr;
            symmFuncChebyshev = new SymmFuncChebyshev(this->numElems, tanhCut, weight, nrad, nang, rrad, rang);
#endif

            this->symmFunc = symmFuncChebyshev;
        }

        if (this->symmFunc == nullptr)
        {
            stop_by_error("cannot create symmFunc.");
        }

        int nbase = this->symmFunc->getNumBasis();
        if (nbase < 1)
        {
            stop_by_error("there are no symmetry functions.");
        }
    }

    return this->symmFunc;
}

void NNArch::calculateSymmFuncs()
{
    int iatom;
    int natom = this->numAtoms;
#ifdef _NNP_GPU
    int lenAtoms;
    int iatomBlock = this->property->getGpuAtomBlock();
#endif

    int idata;
    int idiff;
    int nbase = this->getSymmFunc()->getNumBasis();

#ifdef _NNP_GPU
    nnpreal* symmDiff;

    for (iatom = 0; iatom < natom; iatom += iatomBlock)
    {
        idata = iatom * nbase;
        idiff = 3 * this->idxNeighbor[iatom] * nbase;

        if (this->getSymmFunc()->isHiddenDiff())
        {
            symmDiff = nullptr;
        }
        else
        {
            symmDiff = &(this->symmDiff[idiff]);
        }

        lenAtoms = min(iatom + iatomBlock, natom) - iatom;

        this->symmFunc->calculate(lenAtoms, &(this->numNeighbor[iatom]), &(this->idxNeighbor[iatom]),
                                  &(this->elemNeighbor[iatom]), &(this->posNeighbor[iatom]),
                                  &(this->symmData[idata]), symmDiff);
    }
#else
    #pragma omp parallel for private(iatom, idata, idiff)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        idata = iatom * nbase;
        idiff = 3 * this->idxNeighbor[iatom] * nbase;

        this->symmFunc->calculate(this->numNeighbor[iatom],
                                  this->elemNeighbor[iatom], this->posNeighbor[iatom],
                                  &(this->symmData[idata]), &(this->symmDiff[idiff]));
    }
#endif
}

void NNArch::initLayers()
{
    int ielem;
    int nelem = this->numElems;

    int ilayer;
    int nlayer;
    int nnode;
    int activ;
    int imemory = 0;

    int nbase = this->getSymmFunc()->getNumBasis();

    for (ielem = 0; ielem < nelem; ++ielem)
    {
        this->symmAve[ielem] = ZERO;
        this->symmDev[ielem] = -ONE;
    }

    if (this->isEnergyMode())
    {
        nlayer = this->property->getLayersEnergy();
        nnode  = this->property->getNodesEnergy();
        activ  = this->property->getActivEnergy();

        this->interLayersEnergy = new NNLayer**[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->interLayersEnergy[ielem] = new NNLayer*[nlayer];
            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersEnergy[ielem][ilayer]
                = new NNLayer(ilayer == 0 ? nbase : nnode, nnode, activ, imemory++, this->memory);
            }
        }

        this->lastLayersEnergy = new NNLayer*[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->lastLayersEnergy[ielem] = new NNLayer(nnode, 1, ACTIVATION_ASIS, imemory++, this->memory);
        }
    }

    if (this->isChargeMode())
    {
        nlayer = this->property->getLayersCharge();
        nnode  = this->property->getNodesCharge();
        activ  = this->property->getActivCharge();

        this->interLayersCharge = new NNLayer**[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->interLayersCharge[ielem] = new NNLayer*[nlayer];
            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersCharge[ielem][ilayer]
                = new NNLayer(ilayer == 0 ? nbase : nnode, nnode, activ, imemory++, this->memory);
            }
        }

        this->lastLayersCharge = new NNLayer*[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->lastLayersCharge[ielem] = new NNLayer(nnode, 1, ACTIVATION_ASIS, imemory++, this->memory);
        }
    }
}

void NNArch::goForwardOnEnergy()
{
    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;
    int nelem = this->numElems;

    int nbase = this->getSymmFunc()->getNumBasis();

    int ilayer;
    int nlayer = this->property->getLayersEnergy();

    int jbatch;

    nnpreal ave;
    nnpreal dev;

    // input symmetry functions to the first layer
    #pragma omp parallel for private(iatom, ielem, jbatch, ave, dev)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem  = this->elements[iatom];
        jbatch = this->ibatch[iatom];

        ave = this->symmAve[ielem];
        dev = this->symmDev[ielem];

        #pragma omp simd
        for (int ibase = 0; ibase < nbase; ++ibase)
        {
            this->interLayersEnergy[ielem][0]->getData()[ibase + jbatch * nbase]
            = (this->symmData[ibase + iatom * nbase] - ave) / dev;
        }
    }

    // propagate through layers
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        for (ilayer = 0; ilayer < nlayer; ++ilayer)
        {
            if (ilayer < (nlayer - 1))
            {
                this->interLayersEnergy[ielem][ilayer]->goForward(
                this->interLayersEnergy[ielem][ilayer + 1]->getData());
            }
            else
            {
                this->interLayersEnergy[ielem][ilayer]->goForward(
                this->lastLayersEnergy[ielem]->getData());
            }
        }

        this->lastLayersEnergy[ielem]->goForward(this->energyData[ielem]);
    }
}

void NNArch::goBackwardOnForce()
{
    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;
#ifdef _NNP_GPU
    int jatom;
    int lenAtoms;
    int iatomBlock = this->property->getGpuAtomBlock();
#endif

    int nneigh;
    int mneigh;
    int nneigh3;

    int ielem;
    int nelem = this->numElems;

    int nbase = this->getSymmFunc()->getNumBasis();

    int ilayer;
    int nlayer = this->property->getLayersEnergy();

    int jbatch;

    nnpreal  dev;
    nnpreal  symmScale;
    nnpreal* symmGrad;

    bool transDiff = this->getSymmFunc()->isTransDiff();

    const int     i1 = 1;
    const nnpreal a0 = ZERO;

    // derive energies by itselves, to be units
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        #pragma omp parallel for private(jbatch)
        for (jbatch = 0; jbatch < this->nbatch[ielem]; ++jbatch)
        {
            this->energyGrad[ielem][jbatch] = ONE;
        }
    }

    // propagate through layers
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        this->lastLayersEnergy[ielem]->goBackward(this->energyGrad[ielem], true);

        for (ilayer = (nlayer - 1); ilayer >= 0; --ilayer)
        {
            if (ilayer < (nlayer - 1))
            {
                this->interLayersEnergy[ielem][ilayer]->goBackward(
                this->interLayersEnergy[ielem][ilayer + 1]->getGrad(), true);
            }
            else
            {
                this->interLayersEnergy[ielem][ilayer]->goBackward(
                this->lastLayersEnergy[ielem]->getGrad(), true);
            }
        }
    }

    // calculate forces w/ derivatives of symmetry functions
    if (this->getSymmFunc()->isHiddenDiff())
    {
#ifdef _NNP_GPU
        symmGrad = this->getSymmFunc()->getSymmGrad();

        if (symmGrad == nullptr)
        {
            if ((this->idxNeighbor[natom - 1] + this->numNeighbor[natom - 1]) > 0)
            {
                stop_by_error("symmGrad is not allocated.");
            }
            else
            {
                return;
            }
        }

        for (iatom = 0; iatom < natom; iatom += iatomBlock)
        {
            lenAtoms = min(iatom + iatomBlock, natom) - iatom;

            mneigh = this->idxNeighbor[iatom];

            #pragma omp parallel for private(jatom, ielem, jbatch)
            for (jatom = 0; jatom < lenAtoms; ++jatom)
            {
                ielem  = this->elements[iatom + jatom];
                jbatch = this->ibatch  [iatom + jatom];

                #pragma omp simd
                for (int ibase = 0; ibase < nbase; ++ibase)
                {
                    symmGrad[ibase + jatom * nbase] =
                    this->interLayersEnergy[ielem][0]->getGrad()[ibase + jbatch * nbase];
                }
            }

            this->symmFunc->driveHiddenDiff(lenAtoms, &(this->numNeighbor[iatom]), &(this->idxNeighbor[iatom]),
                                            &(this->forceData[3 * mneigh]));
        }

        #pragma omp parallel for private (iatom, ielem, nneigh, mneigh, dev, symmScale)
        for (iatom = 0; iatom < natom; ++iatom)
        {
            ielem  = this->elements[iatom];

            nneigh = this->numNeighbor[iatom];
            mneigh = this->idxNeighbor[iatom];

            if (nneigh < 1)
            {
                continue;
            }

            dev = this->symmDev[ielem];
            symmScale = -ONE / dev;

            #pragma omp simd
            for (int ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                const int jneigh = ineigh + mneigh;
                this->forceData[3 * jneigh + 0] *= symmScale;
                this->forceData[3 * jneigh + 1] *= symmScale;
                this->forceData[3 * jneigh + 2] *= symmScale;
            }
        }
#else
        stop_by_error("hiddenDiff is only for GPU.");
#endif
    }
    else
    {
        #pragma omp parallel for private(iatom, ielem, jbatch, nneigh, mneigh, nneigh3, \
                                         dev, symmScale, symmGrad)
        for (iatom = 0; iatom < natom; ++iatom)
        {
            ielem  = this->elements[iatom];
            jbatch = this->ibatch  [iatom];

            nneigh = this->numNeighbor[iatom];
            mneigh = this->idxNeighbor[iatom];
            nneigh3 = 3 * nneigh;

            if (nneigh < 1)
            {
                continue;
            }

            dev = this->symmDev[ielem];
            symmScale = -ONE / dev;
            symmGrad  = &(this->interLayersEnergy[ielem][0]->getGrad()[jbatch * nbase]);

            if (transDiff)
            {
                xgemv_("N", &nneigh3, &nbase,
                       &symmScale, &(this->symmDiff[3 * mneigh * nbase]), &nneigh3,
                       &(symmGrad[0]), &i1,
                       &a0, &(this->forceData[3 * mneigh]), &i1);
            }
            else
            {
                xgemv_("T", &nbase, &nneigh3,
                       &symmScale, &(this->symmDiff[3 * mneigh * nbase]), &nbase,
                       &(symmGrad[0]), &i1,
                       &a0, &(this->forceData[3 * mneigh]), &i1);
            }
        }
    }
}

void NNArch::goForwardOnCharge()
{
    if (!this->isChargeMode())
    {
        stop_by_error("this is not charge-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;
    int nelem = this->numElems;

    int nbase = this->getSymmFunc()->getNumBasis();

    int ilayer;
    int nlayer = this->property->getLayersCharge();

    int jbatch;

    nnpreal ave;
    nnpreal dev;

    // input symmetry functions to the first layer
    #pragma omp parallel for private(iatom, ielem, jbatch, ave, dev)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem  = this->elements[iatom];
        jbatch = this->ibatch[iatom];

        ave = this->symmAve[ielem];
        dev = this->symmDev[ielem];

        #pragma omp simd
        for (int ibase = 0; ibase < nbase; ++ibase)
        {
            this->interLayersCharge[ielem][0]->getData()[ibase + jbatch * nbase]
            = (this->symmData[ibase + iatom * nbase] - ave) / dev;
        }
    }

    // propagate through layers
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        for (ilayer = 0; ilayer < nlayer; ++ilayer)
        {
            if (ilayer < (nlayer - 1))
            {
                this->interLayersCharge[ielem][ilayer]->goForward(
                this->interLayersCharge[ielem][ilayer + 1]->getData());
            }
            else
            {
                this->interLayersCharge[ielem][ilayer]->goForward(
                this->lastLayersCharge[ielem]->getData());
            }
        }

        this->lastLayersCharge[ielem]->goForward(this->chargeData[ielem]);
    }
}

void NNArch::obtainEnergies(nnpreal* energies) const
{
    if (energies == nullptr)
    {
        stop_by_error("energies is null.");
    }

    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;

    int jbatch;

    #pragma omp parallel for private(iatom, ielem, jbatch)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem  = this->elements[iatom];
        jbatch = this->ibatch[iatom];

        energies[iatom] = this->energyData[ielem][jbatch];
    }
}

void NNArch::obtainForces(nnpreal*** forces) const
{
    if (forces == nullptr)
    {
        stop_by_error("forces is null.");
    }

    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int nneigh;
    int mneigh;

    #pragma omp parallel for private (iatom, nneigh, mneigh)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh = this->numNeighbor[iatom];
        mneigh = this->idxNeighbor[iatom];

        if (nneigh < 1)
        {
            continue;
        }

        #pragma omp simd
        for (int ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            const int jneigh = ineigh + mneigh;
            forces[iatom][ineigh][0] = this->forceData[3 * jneigh + 0];
            forces[iatom][ineigh][1] = this->forceData[3 * jneigh + 1];
            forces[iatom][ineigh][2] = this->forceData[3 * jneigh + 2];
        }
    }
}

void NNArch::obtainCharges(nnpreal* charges) const
{
    if (charges == nullptr)
    {
        stop_by_error("charges is null.");
    }

    if (!this->isChargeMode())
    {
        stop_by_error("this is not charge-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;

    int jbatch;

    #pragma omp parallel for private(iatom, ielem, jbatch)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem  = this->elements[iatom];
        jbatch = this->ibatch[iatom];

        charges[iatom] = this->chargeData[ielem][jbatch];
    }
}
