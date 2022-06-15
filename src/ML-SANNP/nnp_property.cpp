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
    this->symmFunc      = SYMM_FUNC_NULL;
    this->elemWeight    = 0;
    this->tanhCutoff    = 0;
    this->withClassical = 0;

    this->m2            = 0;
    this->m3            = 0;
    this->rinner        = ZERO;
    this->router        = ZERO;

    this->numRadius     = 0;
    this->numAngle      = 0;
    this->rcutRadius    = ZERO;
    this->rcutAngle     = ZERO;

    this->behlerG4      = 0;
    this->behlerEta1    = nullptr;
    this->behlerEta2    = nullptr;
    this->behlerRs1     = nullptr;
    this->behlerRs2     = nullptr;
    this->behlerZeta    = nullptr;

    this->layersEnergy  = 0;
    this->nodesEnergy   = 0;
    this->activEnergy   = ACTIVATION_NULL;

    this->layersCharge  = 0;
    this->nodesCharge   = 0;
    this->activCharge   = ACTIVATION_NULL;

    this->withCharge    = 0;
}

Property::~Property()
{
    if (this->behlerEta1 != nullptr)
    {
        delete[] this->behlerEta1;
    }

    if (this->behlerEta2 != nullptr)
    {
        delete[] this->behlerEta2;
    }

    if (this->behlerRs1 != nullptr)
    {
        delete[] this->behlerRs1;
    }

    if (this->behlerRs2 != nullptr)
    {
        delete[] this->behlerRs2;
    }

    if (this->behlerZeta != nullptr)
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
    if (rank == 0 && fp == nullptr)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot open ffield file");

    ierr = 0;
    if (rank == 0)
    {
        if (fgets(line, lenLine, fp) == nullptr)
        {
            ierr = 1;
        }

        if (ierr == 0)
        {
            if (sscanf(line, "%d %d %d %d", &(this->symmFunc), &(this->elemWeight),
                                            &(this->tanhCutoff), &(this->withClassical)) != 4)
            {
                if (sscanf(line, "%d %d %d", &(this->symmFunc),
                                             &(this->elemWeight), &(this->tanhCutoff)) != 3)
                {
                    if (sscanf(line, "%d", &(this->symmFunc)) != 1)
                    {
                        ierr = 1;
                    }
                    else
                    {
                        this->elemWeight    = 0;
                        this->tanhCutoff    = 1;
                        this->withClassical = 0;
                    }
                }
                else
                {
                    this->withClassical = 0;
                }
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot read ffield file, at symmFunc");

    MPI_Bcast(&(this->symmFunc),      1, MPI_INT, 0, world);
    MPI_Bcast(&(this->elemWeight),    1, MPI_INT, 0, world);
    MPI_Bcast(&(this->tanhCutoff),    1, MPI_INT, 0, world);
    MPI_Bcast(&(this->withClassical), 1, MPI_INT, 0, world);

    if (this->symmFunc == SYMM_FUNC_MANYBODY)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == nullptr)
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

        MPI_Bcast(&(this->m2),     1, MPI_INT,     0, world);
        MPI_Bcast(&(this->m3),     1, MPI_INT,     0, world);
        MPI_Bcast(&(this->rinner), 1, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->router), 1, MPI_NNPREAL, 0, world);
    }

    else if (this->symmFunc == SYMM_FUNC_BEHLER)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == nullptr)
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

        MPI_Bcast(&(this->numRadius),  1, MPI_INT,     0, world);
        MPI_Bcast(&(this->numAngle),   1, MPI_INT,     0, world);
        MPI_Bcast(&(this->rcutRadius), 1, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->rcutAngle),  1, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->behlerG4),   1, MPI_INT,     0, world);

        if (this->behlerEta1 != nullptr)
        {
            delete[] this->behlerEta1;
        }
        if (this->behlerEta2 != nullptr)
        {
            delete[] this->behlerEta2;
        }
        if (this->behlerRs1 != nullptr)
        {
            delete[] this->behlerRs1;
        }
        if (this->behlerRs2 != nullptr)
        {
            delete[] this->behlerRs2;
        }
        if (this->behlerZeta != nullptr)
        {
            delete[] this->behlerZeta;
        }

        if (this->numRadius > 0)
        {
            this->behlerEta1 = new nnpreal[this->numRadius];
            this->behlerRs1  = new nnpreal[this->numRadius];
        }

        if (this->numAngle > 0)
        {
            this->behlerEta2 = new nnpreal[this->numAngle];
            this->behlerZeta = new nnpreal[this->numAngle];
            this->behlerRs2  = new nnpreal[this->numAngle];
        }

        if (this->numRadius > 0)
        {
            ierr = 0;
            if (rank == 0)
            {
                for (int i = 0; i < this->numRadius; ++i)
                {
                    if (fgets(line, lenLine, fp) == nullptr)
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

            MPI_Bcast(&(this->behlerEta1[0]), this->numRadius, MPI_NNPREAL, 0, world);
            MPI_Bcast(&(this->behlerRs1[0]),  this->numRadius, MPI_NNPREAL, 0, world);
        }

        if (this->numAngle > 0)
        {
            ierr = 0;
            if (rank == 0)
            {
                for (int i = 0; i < this->numAngle; ++i)
                {
                    if (fgets(line, lenLine, fp) == nullptr)
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

            MPI_Bcast(&(this->behlerEta2[0]), this->numAngle, MPI_NNPREAL, 0, world);
            MPI_Bcast(&(this->behlerZeta[0]), this->numAngle, MPI_NNPREAL, 0, world);
            MPI_Bcast(&(this->behlerRs2[0]),  this->numAngle, MPI_NNPREAL, 0, world);
        }
    }

    else if (this->symmFunc == SYMM_FUNC_CHEBYSHEV)
    {
        ierr = 0;
        if (rank == 0)
        {
            if (fgets(line, lenLine, fp) == nullptr)
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

        MPI_Bcast(&(this->numRadius),  1, MPI_INT,     0, world);
        MPI_Bcast(&(this->numAngle),   1, MPI_INT,     0, world);
        MPI_Bcast(&(this->rcutRadius), 1, MPI_NNPREAL, 0, world);
        MPI_Bcast(&(this->rcutAngle),  1, MPI_NNPREAL, 0, world);
    }

    else
    {
        stop_by_error("cannot read ffield file, at symmFunc (unknown type)");
    }

    if (this->symmFunc == SYMM_FUNC_BEHLER || this->symmFunc == SYMM_FUNC_CHEBYSHEV)
    {
        if (this->numAngle < 1)
        {
            this->cutoffMode = CUTOFF_MODE_SINGLE;
        }
        else
        {
            nnpreal rcut1 = this->rcutRadius;
            nnpreal rcut2 = this->rcutAngle;

            if (fabs(rcut1 - rcut2) > NNPREAL(1.0e-4))
            {
                this->cutoffMode = CUTOFF_MODE_DOUBLE;
            }
            else
            {
                this->cutoffMode = CUTOFF_MODE_IPSO;
            }
        }
    }

    else
    {
        this->cutoffMode = CUTOFF_MODE_NULL;
    }

    MPI_Bcast(&(this->cutoffMode), 1, MPI_INT, 0, world);

    ierr = 0;
    if (rank == 0)
    {
        if (fgets(line, lenLine, fp) == nullptr)
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
        if (fgets(line, lenLine, fp) == nullptr)
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
            if (fgets(line, lenLine, fp) == nullptr)
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

    if (rank == 0)
    {
        this->printProperty();
    }
}

void Property::printProperty()
{
    FILE* fp = fopen("log.nnp", "w");
    if (fp == nullptr)
    {
        return;
    }

    fprintf(fp, "\n");

    if (this->symmFunc == SYMM_FUNC_MANYBODY)
    {
        fprintf(fp, "  %s\n", "Symmetry Function (Many-Body Method):");
        fprintf(fp, "  %s%d\n",     "  M2     = ", this->m2);
        fprintf(fp, "  %s%d\n",     "  M3     = ", this->m3);
        fprintf(fp, "  %s%.3f%s\n", "  Rinner = ", this->rinner, " Angstrom");
        fprintf(fp, "  %s%.3f%s\n", "  Router = ", this->router, " Angstrom");
        fprintf(fp, "\n");
        fprintf(fp, "  %s\n", "  [See Y.Huang, et al., Rhys. Rev. B 99, 064103 (2019)]");
        fprintf(fp, "\n");
    }

    else if (this->symmFunc == SYMM_FUNC_BEHLER)
    {
        nnpreal eta1;
        nnpreal eta1Min = ZERO;
        nnpreal eta1Max = ZERO;

        nnpreal rs1;
        nnpreal rs1Min = ZERO;
        nnpreal rs1Max = ZERO;

        if (this->numRadius > 0)
        {
            for (int i = 0; i < this->numRadius; ++i)
            {
                eta1 = this->behlerEta1[i];
                eta1Min = i == 0 ? eta1 : min(eta1Min, eta1);
                eta1Max = i == 0 ? eta1 : max(eta1Max, eta1);

                rs1 = this->behlerRs1[i];
                rs1Min = i == 0 ? rs1 : min(rs1Min, rs1);
                rs1Max = i == 0 ? rs1 : max(rs1Max, rs1);
            }
        }

        nnpreal eta2;
        nnpreal eta2Min = ZERO;
        nnpreal eta2Max = ZERO;

        nnpreal zeta;
        nnpreal zetaMin = ZERO;
        nnpreal zetaMax = ZERO;

        nnpreal rs2;
        nnpreal rs2Min = ZERO;
        nnpreal rs2Max = ZERO;

        if (this->numAngle > 0)
        {
            for (int i = 0; i < this->numAngle; ++i)
            {
                eta2 = this->behlerEta2[i];
                eta2Min = i == 0 ? eta2 : min(eta2Min, eta2);
                eta2Max = i == 0 ? eta2 : max(eta2Max, eta2);

                zeta = this->behlerZeta[i];
                zetaMin = i == 0 ? zeta : min(zetaMin, zeta);
                zetaMax = i == 0 ? zeta : max(zetaMax, zeta);

                rs2 = this->behlerRs2[i];
                rs2Min = i == 0 ? rs2 : min(rs2Min, rs2);
                rs2Max = i == 0 ? rs2 : max(rs2Max, rs2);
            }
        }

        int gType;
        if (this->behlerG4)
        {
            gType = 4;
        }
        else
        {
            gType = 3;
        }

        if (this->tanhCutoff != 0)
        {
            fprintf(fp, "  %s%d%s\n", "Symmetry Function (Behler's G2 & G", gType, " w/ tanh3-cutoff):");
        }
        else
        {
            fprintf(fp, "  %s%d%s\n", "Symmetry Function (Behler's G2 & G", gType, " w/ cosine-cutoff):");
        }

        fprintf(fp, "  %s%s\n",         "  Element Weight = ",           this->elemWeight != 0 ? "Yes" : "No");
        fprintf(fp, "  %s%d\n",         "  Number of G2s  = ",           this->numRadius);
        fprintf(fp, "  %s%d%s%d%s\n",   "  Number of G", gType, "s  = ", this->numAngle, " x 2");
        fprintf(fp, "  %s%.3f%s\n",     "  Rcut for G2    = ",           this->rcutRadius, " Angstrom");
        fprintf(fp, "  %s%d%s%.3f%s\n", "  Rcut for G", gType, "    = ", this->rcutAngle,  " Angstrom");

        nnpreal max0 = ZERO;
        max0 = max(max0, eta1Max);
        max0 = max(max0, rs1Max);
        max0 = max(max0, eta2Max);
        max0 = max(max0, zetaMax);
        max0 = max(max0, rs2Max);

        const char* form1;
        const char* form2;
        const char* form3;
        if (max0 < 9.9995)
        {
            form1 = "  %s%5.3f%s%5.3f%s\n";
            form2 = "  %s%d%s%5.3f%s%5.3f%s\n";
            form3 = "  %s%d%s%5.3f%s%5.3f\n";
        }
        else
        {
            form1 = "  %s%6.3f%s%6.3f%s\n";
            form2 = "  %s%d%s%6.3f%s%6.3f%s\n";
            form3 = "  %s%d%s%6.3f%s%6.3f\n";
        }

        fprintf(fp, form1, "  Eta  for G2    = ",           eta1Min, " ~ ", eta1Max, " Angstorm^-2");
        fprintf(fp, form1, "  Rs   for G2    = ",           rs1Min,  " ~ ", rs1Max,  " Angstrom");
        fprintf(fp, form2, "  Eta  for G", gType, "    = ", eta2Min, " ~ ", eta2Max, " Angstrom^-2");
        fprintf(fp, form3, "  Zeta for G", gType, "    = ", zetaMin, " ~ ", zetaMax);
        fprintf(fp, form2, "  Rs   for G", gType, "    = ", rs2Min,  " ~ ", rs2Max,  " Angstrom");
        fprintf(fp, "\n");
        fprintf(fp, "  %s\n", "  [See J.Behler, Int. J. Quant. Chem. 115, 1032 (2015)]");
        fprintf(fp, "  %s\n", "  [See M.Gastegger, et al., J. Chem. Phys. 148, 241709 (2018)]");
        fprintf(fp, "\n");
    }

    else if (this->symmFunc == SYMM_FUNC_CHEBYSHEV)
    {
        if (this->tanhCutoff != 0)
        {
            fprintf(fp, "  %s\n", "Symmetry Function (Chebyshev polynomial w/ tanh3-cutoff):");
        }
        else
        {
            fprintf(fp, "  %s\n", "Symmetry Function (Chebyshev polynomial w/ cosine-cutoff):");
        }

        fprintf(fp, "  %s%s\n",     "  Element Weight = ", this->elemWeight != 0 ? "Yes" : "No");
        fprintf(fp, "  %s%d\n",     "  Number of C2s  = ", this->numRadius);
        fprintf(fp, "  %s%d\n",     "  Number of C3s  = ", this->numAngle);
        fprintf(fp, "  %s%.3f%s\n", "  Rcut for C2    = ", this->rcutRadius, " Angstrom");
        fprintf(fp, "  %s%.3f%s\n", "  Rcut for C3    = ", this->rcutAngle,  " Angstrom");
        fprintf(fp, "\n");
        fprintf(fp, "  %s\n", "  [See N.Artrith, et al., Rhys. Rev. B 96, 014112 (2017)]");
        fprintf(fp, "\n");
    }

    char strActivEnergy[32] = "???";
    this->activToString(strActivEnergy, this->activEnergy);

    fprintf(fp, "  %s\n", "Neural Network for Energy:");
    fprintf(fp, "  %s%d\n", "  Number of Layers    = ", this->layersEnergy);
    fprintf(fp, "  %s%d\n", "  Number of Nodes     = ", this->nodesEnergy);
    fprintf(fp, "  %s%s\n", "  Activation Func.    = ", strActivEnergy);
    fprintf(fp, "  %s%s\n", "  With Atomic Charge  = ", this->withCharge    != 0 ? "Yes" : "No");
    fprintf(fp, "  %s%s\n", "  Classical Potential = ", this->withClassical != 0 ? "Yes" : "No");
    fprintf(fp, "\n");

    if (this->withCharge != 0)
    {
        char strActivCharge[32] = "???";
        this->activToString(strActivCharge, this->activCharge);

        fprintf(fp, "  %s\n", "Neural Network for Charge:");
        fprintf(fp, "  %s%d\n", "  Number of Layers    = ", this->layersCharge);
        fprintf(fp, "  %s%d\n", "  Number of Nodes     = ", this->nodesCharge);
        fprintf(fp, "  %s%s\n", "  Activation Func.    = ", strActivCharge);
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void Property::activToString(char* str, int activ)
{
    if (activ == ACTIVATION_ASIS)
    {
        strcpy(str, "Not Used");
    }
    else if (activ == ACTIVATION_SIGMOID)
    {
        strcpy(str, "sigmoid");
    }
    else if (activ == ACTIVATION_TANH)
    {
        strcpy(str, "tanh");
    }
    else if (activ == ACTIVATION_ELU)
    {
        strcpy(str, "eLU");
    }
    else if (activ == ACTIVATION_TWTANH)
    {
        strcpy(str, "Twisted tanh");
    }
    else if (activ == ACTIVATION_GELU)
    {
        strcpy(str, "GELU");
    }
}

