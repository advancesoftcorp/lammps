/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_pot.h"

ReaxPot::ReaxPot(real rcut, real mixingRate, FILE* fp, int rank, MPI_Comm world)
{
    this->param = new ReaxParam(rcut, fp, rank, world);

    this->geometry    = NULL;
    this->geometryMap = NULL;

    this->mixingRate = mixingRate;

    this->elemNeighs = NULL;
    this->rxyzNeighs = NULL;

    this->numBonds = NULL;
    this->idxBonds = NULL;

    this->BOs_raw     = NULL;
    this->BOs_corr    = NULL;
    this->Deltas_raw  = NULL;
    this->Deltas_corr = NULL;
    this->Deltas_e    = NULL;
    this->n0lps       = NULL;
    this->nlps        = NULL;
    this->Slps        = NULL;
    this->Tlps        = NULL;

    this->dBOdrs_raw     = NULL;
    this->dBOdBOs        = NULL;
    this->dBOdDeltas     = NULL;
    this->dEdBOs_raw     = NULL;
    this->dEdBOs_corr    = NULL;
    this->dEdDeltas_raw  = NULL;
    this->dEdDeltas_corr = NULL;
    this->dEdSlps        = NULL;
    this->dn0lpdDeltas   = NULL;
    this->dnlpdDeltas    = NULL;
    this->dTlpdDeltas    = NULL;
}

ReaxPot::~ReaxPot()
{
    delete this->param;

    this->clearGeometry();
}

void ReaxPot::clearAtomData()
{
    if (this->Deltas_raw != NULL)
    {
        delete[] this->Deltas_raw;
        this->Deltas_raw = NULL;
    }

    if (this->Deltas_corr != NULL)
    {
        delete[] this->Deltas_corr;
        this->Deltas_corr = NULL;
    }

    if (this->Deltas_e != NULL)
    {
        delete[] this->Deltas_e;
        this->Deltas_e = NULL;
    }

    if (this->n0lps != NULL)
    {
        delete[] this->n0lps;
        this->n0lps = NULL;
    }

    if (this->nlps != NULL)
    {
        delete[] this->nlps;
        this->nlps = NULL;
    }

    if (this->Slps != NULL)
    {
        delete[] this->Slps;
        this->Slps = NULL;
    }

    if (this->Tlps != NULL)
    {
        delete[] this->Tlps;
        this->Tlps = NULL;
    }

    if (this->dEdDeltas_raw != NULL)
    {
        delete[] this->dEdDeltas_raw;
        this->dEdDeltas_raw = NULL;
    }

    if (this->dEdDeltas_corr != NULL)
    {
        delete[] this->dEdDeltas_corr;
        this->dEdDeltas_corr = NULL;
    }

    if (this->dEdSlps != NULL)
    {
        delete[] this->dEdSlps;
        this->dEdSlps = NULL;
    }

    if (this->dn0lpdDeltas != NULL)
    {
        delete[] this->dn0lpdDeltas;
        this->dn0lpdDeltas = NULL;
    }

    if (this->dnlpdDeltas != NULL)
    {
        delete[] this->dnlpdDeltas;
        this->dnlpdDeltas = NULL;
    }

    if (this->dTlpdDeltas != NULL)
    {
        delete[] this->dTlpdDeltas;
        this->dTlpdDeltas = NULL;
    }
}

void ReaxPot::clearPairData()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int ineigh;
    int nneigh;

    int ibond;
    int nbond;

    if (this->elemNeighs != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            delete[] this->elemNeighs[iatom];
        }

        delete[] this->elemNeighs;
        this->elemNeighs = NULL;
    }

    if (this->rxyzNeighs != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nneigh = this->geometry->getNumNeighbors(iatom);

            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                delete[] this->rxyzNeighs[iatom][ineigh];
            }

            delete[] this->rxyzNeighs[iatom];
        }

        delete[] this->rxyzNeighs;
        this->rxyzNeighs = NULL;
    }

    if (this->BOs_raw != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->BOs_raw[iatom][ibond];
            }

            delete[] this->BOs_raw[iatom];
        }

        delete[] this->BOs_raw;
        this->BOs_raw = NULL;
    }

    if (this->BOs_corr != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->BOs_corr[iatom][ibond];
            }

            delete[] this->BOs_corr[iatom];
        }

        delete[] this->BOs_corr;
        this->BOs_corr = NULL;
    }

    if (this->dBOdrs_raw != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->dBOdrs_raw[iatom][ibond];
            }

            delete[] this->dBOdrs_raw[iatom];
        }

        delete[] this->dBOdrs_raw;
        this->dBOdrs_raw = NULL;
    }

    if (this->dBOdBOs != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->dBOdBOs[iatom][ibond];
            }

            delete[] this->dBOdBOs[iatom];
        }

        delete[] this->dBOdBOs;
        this->dBOdBOs = NULL;
    }

    if (this->dBOdDeltas != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->dBOdDeltas[iatom][ibond];
            }

            delete[] this->dBOdDeltas[iatom];
        }

        delete[] this->dBOdDeltas;
        this->dBOdDeltas = NULL;
    }

    if (this->dEdBOs_raw != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->dEdBOs_raw[iatom][ibond];
            }

            delete[] this->dEdBOs_raw[iatom];
        }

        delete[] this->dEdBOs_raw;
        this->dEdBOs_raw = NULL;
    }

    if (this->dEdBOs_corr != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nbond = this->numBonds[iatom];

            for (ibond = 0; ibond < nbond; ++ibond)
            {
                delete[] this->dEdBOs_corr[iatom][ibond];
            }

            delete[] this->dEdBOs_corr[iatom];
        }

        delete[] this->dEdBOs_corr;
        this->dEdBOs_corr = NULL;
    }

    if (this->numBonds != NULL)
    {
        delete[] this->numBonds;
        this->numBonds = NULL;
    }

    if (this->idxBonds != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            delete[] this->idxBonds[iatom];
        }

        delete[] this->idxBonds;
        this->idxBonds = NULL;
    }
}

void ReaxPot::clearGeometry()
{
    if (this->geometry != NULL)
    {
        delete this->geometry;
        this->geometry = NULL;
    }

    if (this->geometryMap != NULL)
    {
        delete this->geometry;Map
        this->geometryMap = NULL;
    }
}

void ReaxPot::setGeometry(const Geometry* geometry)
{
    if (geometry == NULL)
    {
        stop_by_error("geometry is null.");
    }

    int iatom;
    int jatom;
    int natom = geometry->getNumAtoms();
    int matom;

    int ielem;
    int nelem = this->param->numElems;
    int atomNum;

    int* iatomList = new int[natom];
    int* ielemList = new int[natom];

    real x, y, z;

    this->clearGeometry();

    // count available atoms
    matom = 0;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        atomNum = geometry->getAtomNum(iatom);
        ielem   = this->param->atomNumToElement(atomNum);

        if (0 <= ielem && ielem < nelem)
        {
            iatomList[matom] = iatom;
            ielemList[matom] = ielem;
            matom++;
        }
    }

    // in case of no atoms
    if (matom < 1)
    {
        this->geometry    = NULL;
        this->geometryMap = NULL;
        return;
    }

    // create geometry
    const real* lattice = geometry->getLattice();

    this->geometry    = new Geometry(lattice, matom, true);
    this->geometryMap = new int[matom];

    for (jatom = 0; jatom < matom; ++jatom)
    {
        iatom   = iatomList[jatom];
        ielem   = ielemList[jatom];
        atomNum = geometry->getAtomNum(iatom);
        x       = geometry->getX(iatom);
        y       = geometry->getY(iatom);
        z       = geometry->getZ(iatom);

        this->geometry->setAtom(jatom, ielem, atomNum, x, y, z, ZERO, ZERO, ZERO, 0.0, ZERO);
        this->geometryMap[jatom] = iatom;
    }

    this->geometry->haveSetAtoms(-ONE, false);

    delete[] iatomList;
    delete[] ielemList;
}

void ReaxPot::removePotentialFrom(Geometry* geometry)
{
    if (geometry == NULL)
    {
        return;
    }

    this->setGeometry(geometry);

    if (this->geometry == NULL || this->geometryMap == NULL)
    {
        return;
    }

    this->calculatePotential(true);

    int iatom;
    int jatom;
    int natom = geometry->getNumAtoms();

    double energy;
    double totEnergy;
    real fx, fy, fz;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        energy = this->mixingRate * KCAL2EV_DOUBLE * this->geometry->doubleEnergy(iatom);

        fx = this->mixingRate * KCAL2EV * this->geometry->getFx(iatom);
        fy = this->mixingRate * KCAL2EV * this->geometry->getFy(iatom);
        fz = this->mixingRate * KCAL2EV * this->geometry->getFz(iatom);

        jatom = this->geometryMap[iatom];

        geometry->addEnergy(jatom, -energy);
        geometry->addForce (jatom, -fx, -fy, -fz);
    }

    totEnergy = this->mixingRate * KCAL2EV_DOUBLE * this->geometry->doubleTotEnergy();

    geometry->addTotEnergy(-totEnergy);
}

real ReaxPot::totalEnergyOfPotential(const Geometry* geometry)
{
    if (geometry == NULL)
    {
        return ZERO;
    }

    this->setGeometry(geometry);

    if (this->geometry == NULL)
    {
        return ZERO;
    }

    this->calculatePotential(false);

    real totEnergy = this->mixingRate * KCAL2EV * this->geometry->getTotEnergy();

    this->clearGeometry();

    return totEnergy;
}

real ReaxPot::energyAndForceOfPotential(const Geometry* geometry, real* energy, real* force)
{
    if (geometry == NULL)
    {
        return ZERO;
    }

    this->setGeometry(geometry);

    if (this->geometry == NULL || this->geometryMap == NULL)
    {
        return ZERO;
    }

    this->calculatePotential(true);

    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    for (iatom = 0; iatom < natom; ++iatom)
    {
        jatom = this->geometryMap[iatom];

        energy[jatom] = this->mixingRate * KCAL2EV * this->geometry->getEnergy(iatom);

        force [3 * jatom + 0] = this->mixingRate * KCAL2EV * this->geometry->getFx(iatom);
        force [3 * jatom + 1] = this->mixingRate * KCAL2EV * this->geometry->getFy(iatom);
        force [3 * jatom + 2] = this->mixingRate * KCAL2EV * this->geometry->getFz(iatom);
    }

    real totEnergy = this->mixingRate * KCAL2EV * this->geometry->getTotEnergy();

    this->clearGeometry();

    return totEnergy;
}

void ReaxPot::calculatePotential(bool withForce)
{
    if (this->geometry == NULL)
    {
        return;
    }

    int iatom;
    int natom = this->geometry->getNumAtoms();

    if (natom < 1)
    {
        this->geometry->setTotEnergy(0.0);
        return;
    }

    // initialize atomic energy and force
    for (iatom = 0; iatom < natom; ++iatom)
    {
        this->geometry->setEnergy(iatom, 0.0);
        this->geometry->setForce (iatom, ZERO, ZERO, ZERO);
    }

    // calculate bond-order and related terms
    this->geometry->updateNeighbors(this->param->rcut_bond);
    this->createNeighbors(this->param->rcut_bond);

    this->calculateBondOrder();
    this->calculateLonePairNumber();

    this->calculateBondEnergy();
    this->calculateLonePairEnergy();
    this->calculateOverCoordEnergy();

    if (withForce)
    {
        this->calculateBondOrderForce();
    }

    this->clearPairData();
    this->geometry->clearNeighbors();

    // calculate van der Waals terms
    this->geometry->updateNeighbors(this->param->rcut_vdw);
    this->createNeighbors(this->param->rcut_vdw);

    this->calculateVanDerWaalsEnergy(withForce);

    this->clearPairData();
    this->geometry->clearNeighbors();

    this->clearAtomData();

    // sum atomic energy -> total energy
    double totEnergy = 0.0;

    for (iatom = 0; iatom < natom; ++iatom)
    {
        totEnergy += this->geometry->doubleEnergy(iatom);
    }

    this->geometry->setTotEnergy(totEnergy);
}

void ReaxPot::createNeighbors(real rcut)
{
    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    int ineigh;
    int nneigh;
    const int** neighbor;

    const real* lattice = this->geometry->getLattice();

    int  ja, jb, jc;
    real ra, rb, rc;

    real x, x0, dx;
    real y, y0, dy;
    real z, z0, dz;
    real r, rr;

    int*   elemNeigh;
    real** rxyzNeigh;

    this->elemNeighs = new int*  [natom];
    this->rxyzNeighs = new real**[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh   = this->geometry->getNumNeighbors(iatom);
        neighbor = this->geometry->getNeighbors(iatom);

        elemNeigh = new int[nneigh + 1];
        rxyzNeigh = new real*[nneigh];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            rxyzNeigh[ineigh] = new real[4];
        }

        elemNeigh[0] = this->geometry->getElement(iatom);

        x0 = this->geometry->getX(iatom);
        y0 = this->geometry->getY(iatom);
        z0 = this->geometry->getZ(iatom);

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            jatom = neighbor[ineigh][0];

            elemNeigh[ineigh + 1] = this->geometry->getElement(jatom);

            x = this->geometry->getX(jatom);
            y = this->geometry->getY(jatom);
            z = this->geometry->getZ(jatom);

            ja = neighbor[ineigh][1];
            if (ja != 0)
            {
                ra = (real) ja;
                x += ra * lattice[3 * 0 + 0];
                y += ra * lattice[3 * 0 + 1];
                z += ra * lattice[3 * 0 + 2];
            }

            jb = neighbor[ineigh][2];
            if (jb != 0)
            {
                rb = (real) jb;
                x += rb * lattice[3 * 1 + 0];
                y += rb * lattice[3 * 1 + 1];
                z += rb * lattice[3 * 1 + 2];
            }

            jc = neighbor[ineigh][3];
            if (jc != 0)
            {
                rc = (real) jc;
                x += rc * lattice[3 * 2 + 0];
                y += rc * lattice[3 * 2 + 1];
                z += rc * lattice[3 * 2 + 2];
            }

            dx = x - x0;
            dy = y - y0;
            dz = z - z0;
            rr = dx * dx + dy * dy + dz * dz;
            r  = sqrt(rr);

            rxyzNeigh[ineigh][0] = r;
            rxyzNeigh[ineigh][1] = dx;
            rxyzNeigh[ineigh][2] = dy;
            rxyzNeigh[ineigh][3] = dz;
        }

        this->elemNeighs[iatom] = elemNeigh;
        this->rxyzNeighs[iatom] = rxyzNeigh;
    }
}

