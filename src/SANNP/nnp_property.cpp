/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_property.h"

Property::Property()
{
    /*
     * set default values
     */
    this->symmFunc       = SYMM_FUNC_BEHLER;

    this->m2             = 100;
    this->m3             = 10;
    this->rinner         = REAL(0.0);  // Angs
    this->router         = REAL(6.0);  // Angs

    this->numRadius      = 0;
    this->numAngle       = 0;
    this->behlerEta1     = NULL;
    this->behlerEta2     = NULL;
    this->behlerRs       = NULL;
    this->behlerZeta     = NULL;

    this->layersEnergy   = 2;
    this->nodesEnergy    = 512;
    this->activEnergy    = ACTIVATION_TANH;

    this->layersCharge   = 2;
    this->nodesCharge    = 512;
    this->activCharge    = ACTIVATION_TANH;

    this->withCharge     = 0;
}

Property::~Property()
{
    if (this->behlerEta1 != NULL)
    {
        delete[] this->behlerEta1;
    }

    if (this->behlerEta2 != NULL)
    {
        delete[] this->behlerEta2;
    }

    if (this->behlerRs != NULL)
    {
        delete[] this->behlerRs;
    }

    if (this->behlerZeta != NULL)
    {
        delete[] this->behlerZeta;
    }
}

void Property::peekProperty(FILE* fp, int rank, MPI_Comm world)
{
    int ierr;

    ierr = 0;
    if (rank == 0 && fp == NULL)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot open ffield file");

    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d", &(this->symmFunc)) == EOF)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at symmFunc");

    MPI_Bcast(&(this->symmFunc), 1, MPI_INT, 0, world);

    if (this->symmFunc == SYMM_FUNC_MANYBODY)
    {
        ierr = 0;
        if (rank == 0 && fscanf(fp, IFORM_D2_F2, &(this->m2), &(this->m3), &(this->rinner), &(this->router)) == EOF)
        {
            ierr = 1;
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at symmFunc parameter");

        MPI_Bcast(&(this->m2), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->m3), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->rinner), 1, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->router), 1, MPI_REAL0, 0, world);
    }

    else if (this->symmFunc == SYMM_FUNC_BEHLER)
    {
        ierr = 0;
        if (rank == 0 && fscanf(fp, IFORM_D2_F1, &(this->numRadius), &(this->numAngle), &(this->router)) == EOF)
        {
            ierr = 1;
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at symmFunc parameter");

        MPI_Bcast(&(this->numRadius), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->numAngle), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->router), 1, MPI_REAL0, 0, world);

        if (this->behlerEta1 != NULL)
        {
            delete[] this->behlerEta1;
        }
        if (this->behlerEta2 != NULL)
        {
            delete[] this->behlerEta2;
        }
        if (this->behlerRs != NULL)
        {
            delete[] this->behlerRs;
        }
        if (this->behlerZeta != NULL)
        {
            delete[] this->behlerZeta;
        }

        if (this->numRadius > 0)
        {
            this->behlerEta1 = new real[this->numRadius];
            this->behlerRs   = new real[this->numRadius];
        }

        if (this->numAngle > 0)
        {
            this->behlerEta2 = new real[this->numAngle];
            this->behlerZeta = new real[this->numAngle];
        }

        ierr = 0;
        for (int i = 0; i < this->numRadius; ++i)
        {
            if (rank == 0 && fscanf(fp, IFORM_F2, &(this->behlerEta1[i]), &(this->behlerRs[i])) == EOF)
            {
                ierr = 1;
                break;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at Behler parameter");

        MPI_Bcast(&(this->behlerEta1[0]), numRadius, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->behlerRs[0]), numRadius, MPI_REAL0, 0, world);

        ierr = 0;
        for (int i = 0; i < this->numAngle; ++i)
        {
            if (rank == 0 && fscanf(fp, IFORM_F2, &(this->behlerEta2[i]), &(this->behlerZeta[i])) == EOF)
            {
                ierr = 1;
                break;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at Behler parameter");

        MPI_Bcast(&(this->behlerEta2[0]), numAngle, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->behlerZeta[0]), numAngle, MPI_REAL0, 0, world);
    }

    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d", &(this->withCharge)) == EOF)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at withCharge");

    MPI_Bcast(&(this->withCharge), 1, MPI_INT, 0, world);

    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d %d %d", &(this->layersEnergy), &(this->nodesEnergy), &(this->activEnergy)) == EOF)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at layersEnergy");

    MPI_Bcast(&(this->layersEnergy), 1, MPI_INT, 0, world);
    MPI_Bcast(&(this->nodesEnergy), 1, MPI_INT, 0, world);
    MPI_Bcast(&(this->activEnergy), 1, MPI_INT, 0, world);

    if (this->withCharge != 0)
    {
        ierr = 0;
        if (rank == 0 && fscanf(fp, "%d %d %d", &(this->layersCharge), &(this->nodesCharge), &(this->activCharge)) == EOF)
        {
            ierr = 1;
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at layersCharge");

        MPI_Bcast(&(this->layersCharge), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->nodesCharge), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->activCharge), 1, MPI_INT, 0, world);
    }
}
