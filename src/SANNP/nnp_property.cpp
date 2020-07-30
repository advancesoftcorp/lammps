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
    this->symmFunc     = SYMM_FUNC_NULL;
    this->elemWeight   = 0;
    this->tanhCutoff   = 0;

    this->m2           = 0;
    this->m3           = 0;
    this->rinner       = ZERO;
    this->router       = ZERO;

    this->numRadius    = 0;
    this->numAngle     = 0;
    this->rcutRadius   = ZERO;
    this->rcutAngle    = ZERO;

    this->behlerG4     = 0;
    this->behlerEta1   = NULL;
    this->behlerEta2   = NULL;
    this->behlerRs1    = NULL;
    this->behlerRs2    = NULL;
    this->behlerZeta   = NULL;

    this->layersEnergy = 0;
    this->nodesEnergy  = 0;
    this->activEnergy  = ACTIVATION_NULL;

    this->layersCharge = 0;
    this->nodesCharge  = 0;
    this->activCharge  = ACTIVATION_NULL;

    this->withCharge   = 0;
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

    if (this->behlerRs1 != NULL)
    {
        delete[] this->behlerRs1;
    }

    if (this->behlerRs2 != NULL)
    {
        delete[] this->behlerRs2;
    }

    if (this->behlerZeta != NULL)
    {
        delete[] this->behlerZeta;
    }
}

void Property::readProperty(FILE* fp, int rank, MPI_Comm world)
{
    int ierr;

    const int lenLine = 256;
    char line[lenLine];

    ierr = 0;
    if (rank == 0 && fp == NULL)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot open ffield file");

    ierr = 0;
    if (rank == 0)
    {
        if (fgets(line, lenLine, fp) == NULL)
        {
            ierr = 1;
        }

        if (ierr == 0)
        {
            if (sscanf(line, "%d %d %d", &(this->symmFunc), &(this->elemWeight), &(this->tanhCutoff)) != 3)
            {
                if (sscanf(line, "%d", &(this->symmFunc)) != 1)
                {
                    ierr = 1;
                }
                else
                {
                    this->elemWeight = 0;
                    this->tanhCutoff = 1;
                }
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot read ffield file, at symmFunc");

    MPI_Bcast(&(this->symmFunc),   1, MPI_INT, 0, world);
    MPI_Bcast(&(this->elemWeight), 1, MPI_INT, 0, world);
    MPI_Bcast(&(this->tanhCutoff), 1, MPI_INT, 0, world);

    if (this->symmFunc == SYMM_FUNC_MANYBODY)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == NULL)
            {
                ierr = 1;
            }

            if (ierr == 0)
            {
                if (sscanf(line, IFORM_D2_F2, &(this->m2), &(this->m3), &(this->rinner), &(this->router)) != 4)
                {
                    ierr = 1;
                }
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot read ffield file, at symmFunc parameter (many-body)");

        MPI_Bcast(&(this->m2),     1, MPI_INT,   0, world);
        MPI_Bcast(&(this->m3),     1, MPI_INT,   0, world);
        MPI_Bcast(&(this->rinner), 1, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->router), 1, MPI_REAL0, 0, world);
    }

    else if (this->symmFunc == SYMM_FUNC_BEHLER)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == NULL)
            {
                ierr = 1;
            }

            if (ierr == 0)
            {
                if (sscanf(line, IFORM_D2_F2_D1, &(this->numRadius),  &(this->numAngle),
                                                 &(this->rcutRadius), &(this->rcutAngle), &(this->behlerG4)) != 5)
                {
                    if (sscanf(line, IFORM_D2_F1, &(this->numRadius), &(this->numAngle), &(this->rcutRadius)) != 3)
                    {
                    	ierr = 1;
                    }
                    else
                    {
                        this->rcutAngle = this->rcutRadius;
                        this->behlerG4  = 1;
                    }
                }
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot read ffield file, at symmFunc parameter (Behler)");

        MPI_Bcast(&(this->numRadius),  1, MPI_INT,   0, world);
        MPI_Bcast(&(this->numAngle),   1, MPI_INT,   0, world);
        MPI_Bcast(&(this->rcutRadius), 1, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->rcutAngle),  1, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->behlerG4),   1, MPI_INT,   0, world);

        if (this->behlerEta1 != NULL)
        {
            delete[] this->behlerEta1;
        }
        if (this->behlerEta2 != NULL)
        {
            delete[] this->behlerEta2;
        }
        if (this->behlerRs1 != NULL)
        {
            delete[] this->behlerRs1;
        }
        if (this->behlerRs2 != NULL)
        {
            delete[] this->behlerRs2;
        }
        if (this->behlerZeta != NULL)
        {
            delete[] this->behlerZeta;
        }

        if (this->numRadius > 0)
        {
            this->behlerEta1 = new real[this->numRadius];
            this->behlerRs1  = new real[this->numRadius];
        }

        if (this->numAngle > 0)
        {
            this->behlerEta2 = new real[this->numAngle];
            this->behlerZeta = new real[this->numAngle];
            this->behlerRs2  = new real[this->numAngle];
        }

        if (this->numRadius > 0)
        {
            ierr = 0;
            if (rank == 0)
            {
                for (int i = 0; i < this->numRadius; ++i)
                {
                    if (fgets(line, lenLine, fp) == NULL)
                    {
                        ierr = 1;
                        break;
                    }

                    if (sscanf(line, IFORM_F2, &(this->behlerEta1[i]), &(this->behlerRs1[i])) != 2)
                    {
                        ierr = 1;
                        break;
                    }
                }
            }

            MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
            if (ierr != 0) stop_by_error("cannot read ffield file, at Behler parameter");

            MPI_Bcast(&(this->behlerEta1[0]), this->numRadius, MPI_REAL0, 0, world);
            MPI_Bcast(&(this->behlerRs1[0]),  this->numRadius, MPI_REAL0, 0, world);
        }

        if (this->numAngle > 0)
        {
            ierr = 0;
            if (rank == 0)
            {
                for (int i = 0; i < this->numAngle; ++i)
                {
                    if (fgets(line, lenLine, fp) == NULL)
                    {
                        ierr = 1;
                        break;
                    }

                    if (sscanf(line, IFORM_F3, &(this->behlerEta2[i]),
                                               &(this->behlerZeta[i]), &(this->behlerRs2[i])) != 3)
                    {
                        if (sscanf(line, IFORM_F2, &(this->behlerEta2[i]), &(this->behlerZeta[i])) != 2)
                        {
                        	ierr = 1;
                        	break;
                        }
                        else
                        {
                            this->behlerRs2[i] = ZERO;
                        }
                    }
                }
            }

            MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
            if (ierr != 0) stop_by_error("cannot read ffield file, at Behler parameter");

            MPI_Bcast(&(this->behlerEta2[0]), this->numAngle, MPI_REAL0, 0, world);
            MPI_Bcast(&(this->behlerZeta[0]), this->numAngle, MPI_REAL0, 0, world);
            MPI_Bcast(&(this->behlerRs2[0]),  this->numAngle, MPI_REAL0, 0, world);
        }
    }

    else if (this->symmFunc == SYMM_FUNC_CHEBYSHEV)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == NULL)
            {
                ierr = 1;
            }

            if (ierr == 0)
            {
                if (sscanf(line, IFORM_D2_F2, &(this->numRadius),  &(this->numAngle),
                                              &(this->rcutRadius), &(this->rcutAngle)) != 4)
                {
                    ierr = 1;
                }
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot read ffield file, at symmFunc parameter (Chebyshev)");

        MPI_Bcast(&(this->numRadius),  1, MPI_INT,   0, world);
        MPI_Bcast(&(this->numAngle),   1, MPI_INT,   0, world);
        MPI_Bcast(&(this->rcutRadius), 1, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->rcutAngle),  1, MPI_REAL0, 0, world);
    }

    else
    {
        stop_by_error("cannot read ffield file, at symmFunc (unknown type)");
    }

    ierr = 0;
    if (rank == 0)
    {
        if (fgets(line, lenLine, fp) == NULL)
        {
            ierr = 1;
        }

        if (ierr == 0)
        {
            if (sscanf(line, "%d", &(this->withCharge)) != 1)
            {
                ierr = 1;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot read ffield file, at withCharge");

    MPI_Bcast(&(this->withCharge), 1, MPI_INT, 0, world);

    ierr = 0;
    if (rank == 0)
    {
        if (fgets(line, lenLine, fp) == NULL)
        {
            ierr = 1;
        }

        if (ierr == 0)
        {
            if (sscanf(line, "%d %d %d", &(this->layersEnergy),
                                         &(this->nodesEnergy), &(this->activEnergy)) != 3)
            {
                ierr = 1;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot read ffield file, at layersEnergy");

    MPI_Bcast(&(this->layersEnergy), 1, MPI_INT, 0, world);
    MPI_Bcast(&(this->nodesEnergy),  1, MPI_INT, 0, world);
    MPI_Bcast(&(this->activEnergy),  1, MPI_INT, 0, world);

    if (this->withCharge != 0)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == NULL)
            {
                ierr = 1;
            }

            if (ierr == 0)
            {
                if (sscanf(line, "%d %d %d", &(this->layersCharge),
                                             &(this->nodesCharge), &(this->activCharge)) != 3)
                {
                    ierr = 1;
                }
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot read ffield file, at layersCharge");

        MPI_Bcast(&(this->layersCharge), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->nodesCharge),  1, MPI_INT, 0, world);
        MPI_Bcast(&(this->activCharge),  1, MPI_INT, 0, world);
    }
}
