/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_reax_param.h"

#define OFFDIAG_THR   NNPREAL(1.0e-16)
#define MAX_ATOM_NUM  256

ReaxParam::ReaxParam(nnpreal rcut, FILE* fp)
{
    this->numElems = 0;
    this->atomNums = NULL;

    this->rcut_bond = rcut;
    this->rcut_vdw  = ZERO;
    this->BO_cut    = ZERO;

    this->mass = NULL;

    this->r_atom_sigma = NULL;
    this->r_atom_pi    = NULL;
    this->r_atom_pipi  = NULL;

    this->r_pair_sigma = NULL;
    this->r_pair_pi    = NULL;
    this->r_pair_pipi  = NULL;

    this->p_bo1 = NULL;
    this->p_bo2 = NULL;
    this->p_bo3 = NULL;
    this->p_bo4 = NULL;
    this->p_bo5 = NULL;
    this->p_bo6 = NULL;

    this->De_sigma = NULL;
    this->De_pi    = NULL;
    this->De_pipi  = NULL;
    this->p_be1    = NULL;
    this->p_be2    = NULL;

    this->ovc_corr = NULL;
    this->v13_corr = NULL;

    this->p_boc1      = ZERO;
    this->p_boc2      = ZERO;
    this->p_boc3_atom = NULL;
    this->p_boc4_atom = NULL;
    this->p_boc5_atom = NULL;
    this->p_boc3      = NULL;
    this->p_boc4      = NULL;
    this->p_boc5      = NULL;

    this->p_over1 = NULL;
    this->p_over2 = NULL;
    this->p_over3 = ZERO;
    this->p_over4 = ZERO;

    this->Val     = NULL;
    this->Val_e   = NULL;
    this->Val_boc = NULL;
    this->Val_ang = NULL;

    this->p_lp2    = NULL;
    this->n_lp_opt = NULL;

    this->r1_lp     = ZERO;
    this->r2_lp     = ZERO;
    this->lambda_lp = ZERO;

    this->shielding = false;
    this->innerWall = false;

    this->swa_vdw    = ZERO;
    this->swb_vdw    = ZERO;
    this->Tap_vdw[0] = ZERO;
    this->Tap_vdw[1] = ZERO;
    this->Tap_vdw[2] = ZERO;
    this->Tap_vdw[3] = ZERO;
    this->Tap_vdw[4] = ZERO;
    this->Tap_vdw[5] = ZERO;
    this->Tap_vdw[6] = ZERO;
    this->Tap_vdw[7] = ZERO;

    this->D_vdw_atom     = NULL;
    this->alpha_vdw_atom = NULL;
    this->gammaw_atom    = NULL;
    this->r_vdw_atom     = NULL;
    this->p_core1_atom   = NULL;
    this->p_core2_atom   = NULL;
    this->p_core3_atom   = NULL;

    this->D_vdw     = NULL;
    this->alpha_vdw = NULL;
    this->gammaw    = NULL;
    this->r_vdw     = NULL;
    this->p_vdw     = ZERO;
    this->p_core1   = NULL;
    this->p_core2   = NULL;
    this->p_core3   = NULL;

    this->readFFieldReax(fp);

    this->modifyParameters();

    this->atomNumMap = NULL;
    this->createAtomNumMap();
}

ReaxParam::~ReaxParam()
{
    this->deallocateAllData();

    delete[] this->atomNumMap;
}

void ReaxParam::createAtomNumMap()
{
    this->atomNumMap = new int[MAX_ATOM_NUM];

    for (int atomNum = 1; atomNum <= MAX_ATOM_NUM; ++atomNum)
    {
        this->atomNumMap[atomNum - 1] = -1;
    }

    for (int ielem = 0; ielem < this->numElems; ++ielem)
    {
        int atomNum = this->atomNums[ielem];

        if (1 <= atomNum && atomNum <= MAX_ATOM_NUM)
        {
            this->atomNumMap[atomNum - 1] = ielem;
        }
    }
}

int ReaxParam::atomNumToElement(int atomNum)
{
    if (1 <= atomNum && atomNum <= MAX_ATOM_NUM)
    {
        return this->atomNumMap[atomNum - 1];
    }
    else
    {
        return -1;
    }
}

nnpreal* ReaxParam::allocateElemData()
{
    if (this->numElems < 1) return NULL;

    nnpreal* data = new nnpreal[this->numElems];

    for (int i = 0; i < this->numElems; ++i)
    {
        data[i] = ZERO;
    }

    return data;
}

void ReaxParam::deallocateElemData(nnpreal*  data)
{
    if (data == NULL) return;

    delete[] data;
}

nnpreal** ReaxParam::allocatePairData()
{
    if (this->numElems < 1) return NULL;

    nnpreal** data = new nnpreal*[this->numElems];

    for (int i = 0; i < this->numElems; ++i)
    {
        data[i] = new nnpreal[this->numElems];

        for (int j = 0; j < this->numElems; ++j)
        {
            data[i][j] = ZERO;
        }
    }

    return data;
}

void ReaxParam::deallocatePairData(nnpreal** data)
{
    if (data == NULL) return;

    for (int i = 0; i < this->numElems; ++i)
    {
        delete[] data[i];
    }

    delete[] data;
}

void ReaxParam::allocateAllData()
{
    if (this->numElems > 0)
    {
        this->atomNums = new int[this->numElems];

        for (int ielem = 0; ielem < this->numElems; ++ielem)
        {
            this->atomNums[ielem] = 0;
        }
    }

    this->mass = this->allocateElemData();

    this->r_atom_sigma = this->allocateElemData();
    this->r_atom_pi    = this->allocateElemData();
    this->r_atom_pipi  = this->allocateElemData();

    this->r_pair_sigma = this->allocatePairData();
    this->r_pair_pi    = this->allocatePairData();
    this->r_pair_pipi  = this->allocatePairData();

    this->p_bo1 = this->allocatePairData();
    this->p_bo2 = this->allocatePairData();
    this->p_bo3 = this->allocatePairData();
    this->p_bo4 = this->allocatePairData();
    this->p_bo5 = this->allocatePairData();
    this->p_bo6 = this->allocatePairData();

    this->De_sigma = this->allocatePairData();
    this->De_pi    = this->allocatePairData();
    this->De_pipi  = this->allocatePairData();
    this->p_be1    = this->allocatePairData();
    this->p_be2    = this->allocatePairData();

    this->ovc_corr = this->allocatePairData();
    this->v13_corr = this->allocatePairData();

    this->p_boc3_atom = this->allocateElemData();
    this->p_boc4_atom = this->allocateElemData();
    this->p_boc5_atom = this->allocateElemData();
    this->p_boc3      = this->allocatePairData();
    this->p_boc4      = this->allocatePairData();
    this->p_boc5      = this->allocatePairData();

    this->p_over1 = this->allocatePairData();
    this->p_over2 = this->allocateElemData();

    this->Val     = this->allocateElemData();
    this->Val_e   = this->allocateElemData();
    this->Val_boc = this->allocateElemData();
    this->Val_ang = this->allocateElemData();

    this->p_lp2    = this->allocateElemData();
    this->n_lp_opt = this->allocateElemData();

    this->D_vdw_atom     = this->allocateElemData();
    this->alpha_vdw_atom = this->allocateElemData();
    this->gammaw_atom    = this->allocateElemData();
    this->r_vdw_atom     = this->allocateElemData();
    this->p_core1_atom   = this->allocateElemData();
    this->p_core2_atom   = this->allocateElemData();
    this->p_core3_atom   = this->allocateElemData();

    this->D_vdw     = this->allocatePairData();
    this->alpha_vdw = this->allocatePairData();
    this->gammaw    = this->allocatePairData();
    this->r_vdw     = this->allocatePairData();
    this->p_core1   = this->allocatePairData();
    this->p_core2   = this->allocatePairData();
    this->p_core3   = this->allocatePairData();
}

void ReaxParam::deallocateAllData()
{
    if (this->atomNums != NULL)
    {
        delete[] this->atomNums;
    }

    this->deallocateElemData(this->mass);

    this->deallocateElemData(this->r_atom_sigma);
    this->deallocateElemData(this->r_atom_pi);
    this->deallocateElemData(this->r_atom_pipi);

    this->deallocatePairData(this->r_pair_sigma);
    this->deallocatePairData(this->r_pair_pi);
    this->deallocatePairData(this->r_pair_pipi);

    this->deallocatePairData(this->p_bo1);
    this->deallocatePairData(this->p_bo2);
    this->deallocatePairData(this->p_bo3);
    this->deallocatePairData(this->p_bo4);
    this->deallocatePairData(this->p_bo5);
    this->deallocatePairData(this->p_bo6);

    this->deallocatePairData(this->De_sigma);
    this->deallocatePairData(this->De_pi);
    this->deallocatePairData(this->De_pipi);
    this->deallocatePairData(this->p_be1);
    this->deallocatePairData(this->p_be2);

    this->deallocatePairData(this->ovc_corr);
    this->deallocatePairData(this->v13_corr);

    this->deallocateElemData(this->p_boc3_atom);
    this->deallocateElemData(this->p_boc4_atom);
    this->deallocateElemData(this->p_boc5_atom);
    this->deallocatePairData(this->p_boc3);
    this->deallocatePairData(this->p_boc4);
    this->deallocatePairData(this->p_boc5);

    this->deallocatePairData(this->p_over1);
    this->deallocateElemData(this->p_over2);

    this->deallocateElemData(this->Val);
    this->deallocateElemData(this->Val_e);
    this->deallocateElemData(this->Val_boc);
    this->deallocateElemData(this->Val_ang);

    this->deallocateElemData(this->p_lp2);
    this->deallocateElemData(this->n_lp_opt);

    this->deallocateElemData(this->D_vdw_atom);
    this->deallocateElemData(this->alpha_vdw_atom);
    this->deallocateElemData(this->gammaw_atom);
    this->deallocateElemData(this->r_vdw_atom);
    this->deallocateElemData(this->p_core1_atom);
    this->deallocateElemData(this->p_core2_atom);
    this->deallocateElemData(this->p_core3_atom);

    this->deallocatePairData(this->D_vdw);
    this->deallocatePairData(this->alpha_vdw);
    this->deallocatePairData(this->gammaw);
    this->deallocatePairData(this->r_vdw);
    this->deallocatePairData(this->p_core1);
    this->deallocatePairData(this->p_core2);
    this->deallocatePairData(this->p_core3);
}

void ReaxParam::setPairData(nnpreal** data, int i, int j, nnpreal value)
{
    data[i][j] = value;
    data[j][i] = value;
}

void ReaxParam::readFFieldReax(FILE* fp)
{
    int i;
    int iatom;
    int jatom;
    int numGens;
    int numAtoms;
    int numBonds;
    int numOffDs;

    nnpreal genValue;
    nnpreal atomValue[1 + 32];
    nnpreal bondValue[1 + 16];
    nnpreal offDValue[1 + 6];

    const int lenLine = 1024;
    char line [lenLine];
    char token[lenLine];
    char elem[5];

    while (fgets(line, lenLine, fp) != NULL)
    {
        if (sscanf(line, "%s", token) != 1)
        {
            continue;
        }
        if (strcmp(token, ">>>") == 0)
        {
            break;
        }
    }

    // 1) comment line
    if (fgets(line, lenLine, fp) == NULL)
    {
        stop_by_error("cannot read the first comment line.");
    }

    // 2) general parameters
    if (fgets(line, lenLine, fp) == NULL || sscanf(line, "%d", &numGens) != 1)
    {
        stop_by_error("cannot read #general parameters.");
    }

    if (numGens < 1)
    {
        stop_by_error("#general parameters is not positive.");
    }

    for (i = 1; i <= numGens; ++i)
    {
        if (fgets(line, lenLine, fp) == NULL || sscanf(line, IFORM_F1, &genValue) != 1)
        {
            stop_by_error("cannot read a general parameter.");
        }

        if      (i ==  1) {this->p_boc1  = genValue;}
        else if (i ==  2) {this->p_boc2  = genValue;}
        else if (i == 12) {this->swa_vdw = genValue;}
        else if (i == 13) {this->swb_vdw = genValue;}
        else if (i == 29) {this->p_vdw   = genValue;}
        else if (i == 30) {this->BO_cut  = genValue / NNPREAL(100.0);}
        else if (i == 32) {this->p_over4 = genValue;}
        else if (i == 33) {this->p_over3 = genValue;}
    }

    // 3) atomic parameters
    if (fgets(line, lenLine, fp) == NULL || sscanf(line, "%d", &numAtoms) != 1)
    {
        stop_by_error("cannot read #atomic parameters.");
    }

    if (numAtoms < 1)
    {
        stop_by_error("#atomic parameters is not positive.");
    }

    if (fgets(line, lenLine, fp) == NULL)
    {
        stop_by_error("cannot read the header(2nd-line) of atomic parameters.");
    }
    if (fgets(line, lenLine, fp) == NULL)
    {
        stop_by_error("cannot read the header(3rd-line) of atomic parameters.");
    }
    if (fgets(line, lenLine, fp) == NULL)
    {
        stop_by_error("cannot read the header(4th-line) of atomic parameters.");
    }

    this->numElems = numAtoms;

    this->allocateAllData();

    for (i = 1; i <= numAtoms; ++i)
    {
        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_S1_F8, elem, &atomValue[1], &atomValue[2], &atomValue[3], &atomValue[4],
                                           &atomValue[5], &atomValue[6], &atomValue[7], &atomValue[8]) != 9)
        {
            char message[256];
            sprintf(message, "cannot read a atomic parameter @1st-line : iatom=%d", i);
            stop_by_error(message);
        }

        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_F8, &atomValue[9],  &atomValue[10], &atomValue[11], &atomValue[12],
                                  &atomValue[13], &atomValue[14], &atomValue[15], &atomValue[16]) != 8)
        {
            char message[256];
            sprintf(message, "cannot read a atomic parameter @2nd-line : iatom=%d", i);
            stop_by_error(message);
        }

        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_F8, &atomValue[17], &atomValue[18], &atomValue[19], &atomValue[20],
                                  &atomValue[21], &atomValue[22], &atomValue[23], &atomValue[24]) != 8)
        {
            char message[256];
            sprintf(message, "cannot read a atomic parameter @3rd-line : iatom=%d", i);
            stop_by_error(message);
        }

        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_F8, &atomValue[25], &atomValue[26], &atomValue[27], &atomValue[28],
                                  &atomValue[29], &atomValue[30], &atomValue[31], &atomValue[32]) != 8)
        {
            char message[256];
            sprintf(message, "cannot read a atomic parameter @4th-line : iatom=%d", i);
            stop_by_error(message);
        }

        iatom = i - 1;

        this->atomNums[iatom] = GeometrySet::elementToAtomNum(elem);

        if (this->atomNums[iatom] < 1)
        {
            char message[256];
            sprintf(message, "incorrect atomic number: iatom=%d", i);
            stop_by_error(message);
        }

        for (jatom = 0; jatom < iatom; ++jatom)
        {
            if (this->atomNums[iatom] == this->atomNums[jatom])
            {
                char message[256];
                sprintf(message, "duplex atomic number: iatom=%d", i);
                stop_by_error(message);
            }
        }

        this->mass          [iatom] = atomValue[3];
        this->r_atom_sigma  [iatom] = atomValue[1];
        this->r_atom_pi     [iatom] = atomValue[7];
        this->r_atom_pipi   [iatom] = atomValue[17];
        this->p_boc3_atom   [iatom] = atomValue[21];
        this->p_boc4_atom   [iatom] = atomValue[20];
        this->p_boc5_atom   [iatom] = atomValue[22];
        this->p_over2       [iatom] = atomValue[25];
        this->Val           [iatom] = atomValue[2];
        this->Val_e         [iatom] = atomValue[8];
        this->Val_boc       [iatom] = atomValue[28];
        this->Val_ang       [iatom] = atomValue[11];
        this->p_lp2         [iatom] = atomValue[18];
        this->D_vdw_atom    [iatom] = atomValue[5];
        this->alpha_vdw_atom[iatom] = atomValue[9];
        this->gammaw_atom   [iatom] = atomValue[10];
        this->r_vdw_atom    [iatom] = atomValue[4] * NNPREAL(2.0); // radius -> diameter
        this->p_core1_atom  [iatom] = atomValue[30];
        this->p_core2_atom  [iatom] = atomValue[31];
        this->p_core3_atom  [iatom] = atomValue[32];
    }

    // 4) bond's parameters
    if (fgets(line, lenLine, fp) == NULL || sscanf(line, "%d", &numBonds) != 1)
    {
        stop_by_error("cannot read #bond's parameters.");
    }

    if (numBonds < 0)
    {
        stop_by_error("#bond's parameters is negative.");
    }

    if (fgets(line, lenLine, fp) == NULL)
    {
        stop_by_error("cannot read the header(2nd-line) of bond's parameters.");
    }

    for (i = 1; i <= numBonds; ++i)
    {
        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_D2_F8, &iatom, &jatom, &bondValue[1], &bondValue[2], &bondValue[3], &bondValue[4],
                                                     &bondValue[5], &bondValue[6], &bondValue[7], &bondValue[8]) != 10)
        {
            char message[256];
            sprintf(message, "cannot read a bond's parameter @1st-line : ibond=%d", i);
            stop_by_error(message);
        }

        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_F8, &bondValue[9],  &bondValue[10], &bondValue[11], &bondValue[12],
                                  &bondValue[13], &bondValue[14], &bondValue[15], &bondValue[16]) != 8)
        {
            char message[256];
            sprintf(message, "cannot read a bond's parameter @2nd-line : ibond=%d", i);
            stop_by_error(message);
        }

        iatom--;
        jatom--;

        if (iatom < 0 || numAtoms <= iatom || jatom < 0 || numAtoms <= jatom)
        {
            char message[256];
            sprintf(message, "index of atom is out of range: ibond=%d", i);
            stop_by_error(message);
        }

        this->setPairData(this->p_bo1,    iatom, jatom, bondValue[13]);
        this->setPairData(this->p_bo2,    iatom, jatom, bondValue[14]);
        this->setPairData(this->p_bo3,    iatom, jatom, bondValue[10]);
        this->setPairData(this->p_bo4,    iatom, jatom, bondValue[11]);
        this->setPairData(this->p_bo5,    iatom, jatom, bondValue[5]);
        this->setPairData(this->p_bo6,    iatom, jatom, bondValue[7]);
        this->setPairData(this->De_sigma, iatom, jatom, bondValue[1]);
        this->setPairData(this->De_pi,    iatom, jatom, bondValue[2]);
        this->setPairData(this->De_pipi,  iatom, jatom, bondValue[3]);
        this->setPairData(this->p_be1,    iatom, jatom, bondValue[4]);
        this->setPairData(this->p_be2,    iatom, jatom, bondValue[9]);
        this->setPairData(this->ovc_corr, iatom, jatom, bondValue[15]);
        this->setPairData(this->v13_corr, iatom, jatom, bondValue[6]);
        this->setPairData(this->p_over1,  iatom, jatom, bondValue[8]);
    }

    // 5) off-diagonal parameters
    if (fgets(line, lenLine, fp) == NULL || sscanf(line, "%d", &numOffDs) != 1)
    {
        stop_by_error("cannot read #off-diagonal parameters.");
    }

    if (numOffDs < 0)
    {
        stop_by_error("#off-diagonal parameters is negative.");
    }

    for (i = 1; i <= numOffDs; ++i)
    {
        if (fgets(line, lenLine, fp) == NULL ||
           sscanf(line, IFORM_D2_F6, &iatom, &jatom, &offDValue[1], &offDValue[2], &offDValue[3],
                                                     &offDValue[4], &offDValue[5], &offDValue[6]) != 8)
        {
            char message[256];
            sprintf(message, "cannot read a off-diagonal parameter: ioffDiag=%d", i);
            stop_by_error(message);
        }

        iatom--;
        jatom--;

        if (iatom < 0 || numAtoms <= iatom || jatom < 0 || numAtoms <= jatom)
        {
            char message[256];
            sprintf(message, "index of atom is out of range: ioffDiag=%d", i);
            stop_by_error(message);
        }

        this->setPairData(this->D_vdw,        iatom, jatom, offDValue[1]);
        this->setPairData(this->alpha_vdw,    iatom, jatom, offDValue[3]);
        this->setPairData(this->r_vdw,        iatom, jatom, offDValue[2] * NNPREAL(2.0)); // radius -> diameter
        this->setPairData(this->r_pair_sigma, iatom, jatom, offDValue[4]);
        this->setPairData(this->r_pair_pi,    iatom, jatom, offDValue[5]);
        this->setPairData(this->r_pair_pipi,  iatom, jatom, offDValue[6]);
    }

    while (fgets(line, lenLine, fp) != NULL)
    {
        if (sscanf(line, "%s", token) != 1)
        {
            continue;
        }
        if (strcmp(token, "<<<") == 0)
        {
            break;
        }
    }
}

void ReaxParam::modifyParameters()
{
    this->rcut_bond = this->rcut_bond > ZERO ? this->rcut_bond : NNPREAL(5.0);
    this->rcut_vdw  = this->swb_vdw;

    if (this->rcut_vdw <= ZERO)
    {
        stop_by_error("rcut_vdw is not positive.");
    }

    this->modifyParametersBondOrder();
    this->modifyParametersLonePairNumber();
    this->modifyParametersVanDerWaalsEnergy();
}

void ReaxParam::modifyParametersBondOrder()
{
    int ielem;
    int jelem;

    // correct Val_boc for light element
    for (ielem = 0; ielem < this->numElems; ++ielem)
    {
        if (this->mass[ielem] < NNPREAL(21.0))
        {
            this->Val_boc[ielem] = this->Val_ang[ielem];
        }
    }

    // inter atomic parameters
    for (ielem = 0; ielem < this->numElems; ++ielem)
    {
        for (jelem = 0; jelem <= ielem; ++jelem)
        {
            if (this->r_pair_sigma[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->r_pair_sigma, ielem, jelem,
                                  NNPREAL(0.5) * (this->r_atom_sigma[ielem] + this->r_atom_sigma[jelem]));
            }

            if (this->r_pair_pi[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->r_pair_pi, ielem, jelem,
                                  NNPREAL(0.5) * (this->r_atom_pi[ielem] + this->r_atom_pi[jelem]));
            }

            if (this->r_pair_pipi[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->r_pair_pipi, ielem, jelem,
                                  NNPREAL(0.5) * (this->r_atom_pipi[ielem] + this->r_atom_pipi[jelem]));
            }

            if (this->p_boc3[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->p_boc3, ielem, jelem,
                                  sqrt(this->p_boc3_atom[ielem] * this->p_boc3_atom[jelem]));
            }

            if (this->p_boc4[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->p_boc4, ielem, jelem,
                                  sqrt(this->p_boc4_atom[ielem] * this->p_boc4_atom[jelem]));
            }

            if (this->p_boc5[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->p_boc5, ielem, jelem,
                                  sqrt(this->p_boc5_atom[ielem] * this->p_boc5_atom[jelem]));
            }
        }
    }
}

void ReaxParam::modifyParametersLonePairNumber()
{
    int ielem;

    // parameters of Tap, see FIG.1. of J.Chem.Phys. 153, 021102 (2020)
    this->r1_lp     = NNPREAL(1.5);
    this->r2_lp     = NNPREAL(2.3);
    this->lambda_lp = NNPREAL(0.8);

    // define n_lp_opt
    for (ielem = 0; ielem < this->numElems; ++ielem)
    {
        this->n_lp_opt[ielem] = NNPREAL(0.5) * (this->Val_e[ielem] - this->Val[ielem]);
    }
}

void ReaxParam::modifyParametersVanDerWaalsEnergy()
{
    int ielem;
    int jelem;

    // define shielding & innerWall
    bool shielding;
    bool innerWall;

    for (ielem = 0; ielem < this->numElems; ++ielem)
    {
        if (this->gammaw_atom[ielem] > NNPREAL(0.5)){
            shielding = true;
        }
        else
        {
            shielding = false;
        }

        if (ielem == 0)
        {
            this->shielding = shielding;
        }
        else if (this->shielding != shielding)
        {
            //stop_by_error("not consistent shielding");
        }

        if (this->p_core1_atom[ielem] > NNPREAL(0.01) && this->p_core3_atom[ielem] > NNPREAL(0.01))
        {
            innerWall = true;
        }
        else
        {
            innerWall = false;
        }

        if (ielem == 0)
        {
            this->innerWall = innerWall;
        }
        else if (this->innerWall != innerWall)
        {
            //stop_by_error("not consistent innerWall");
        }
    }

    // define coeff of Tap
    real swa = this->swa_vdw;
    real swb = this->swb_vdw;
    real swd = swb - swa;

    real swa2 = swa * swa;
    real swa3 = swa * swa2;

    real swb2 = swb * swb;
    real swb3 = swb * swb2;
    real swb4 = swb * swb3;
    real swb5 = swb * swb4;
    real swb6 = swb * swb5;
    real swb7 = swb * swb6;

    real swd2 = swd  * swd;
    real swd4 = swd2 * swd2;
    real swd7 = swd  * swd2 * swd4;

    this->Tap_vdw[7] =  NNPREAL(20.0) / swd7;
    this->Tap_vdw[6] = -NNPREAL(70.0)  * (swa  + swb) / swd7;
    this->Tap_vdw[5] =  NNPREAL(84.0)  * (swa2 + NNPREAL(3.0) * swa * swb + swb2) / swd7;
    this->Tap_vdw[4] = -NNPREAL(35.0)  * (swa3 + NNPREAL(9.0) * swa2 * swb + 9.0 * swb2 * swa + swb3) / swd7;
    this->Tap_vdw[3] =  NNPREAL(140.0) * (swa3 * swb + NNPREAL(3.0) * swa2 * swb2 + swb3 * swa) / swd7;
    this->Tap_vdw[2] = -NNPREAL(210.0) * (swa3 * swb2 + swb3 * swa2) / swd7;
    this->Tap_vdw[1] =  NNPREAL(140.0) * swa3 * swb3 / swd7;
    this->Tap_vdw[0] = (-NNPREAL(35.0) * swa3 * swb4 + NNPREAL(21.0) * swa2 * swb5 + NNPREAL(7.0) * swa * swb6 + swb7) / swd7;

    // inter atomic parameters
    for (ielem = 0; ielem < this->numElems; ++ielem)
    {
        for (jelem = 0; jelem <= ielem; ++jelem)
        {
            if (this->D_vdw[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->D_vdw, ielem, jelem,
                                  sqrt(this->D_vdw_atom[ielem] * this->D_vdw_atom[jelem]));
            }

            if (this->alpha_vdw[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->alpha_vdw, ielem, jelem,
                                  sqrt(this->alpha_vdw_atom[ielem] * this->alpha_vdw_atom[jelem]));
            }

            if (this->gammaw[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->gammaw, ielem, jelem,
                                  sqrt(this->gammaw_atom[ielem] * this->gammaw_atom[jelem]));
            }

            if (this->r_vdw[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->r_vdw, ielem, jelem,
                                  sqrt(this->r_vdw_atom[ielem] * this->r_vdw_atom[jelem]));
            }

            if (this->p_core1[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->p_core1, ielem, jelem,
                                  sqrt(this->p_core1_atom[ielem] * this->p_core1_atom[jelem]));
            }

            if (this->p_core2[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->p_core2, ielem, jelem,
                                  sqrt(this->p_core2_atom[ielem] * this->p_core2_atom[jelem]));
            }

            if (this->p_core3[ielem][jelem] < OFFDIAG_THR)
            {
                this->setPairData(this->p_core3, ielem, jelem,
                                  sqrt(this->p_core3_atom[ielem] * this->p_core3_atom[jelem]));
            }
        }
    }
}

