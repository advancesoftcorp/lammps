/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "ReaxPot.h"
#include <cmath>
using namespace std;

#define OVC_THR REAL(0.001)
#define V13_THR REAL(0.001)
#define BO_THR  REAL(1.0e-10)

void ReaxPot::calculateBondOrder()
{
    this->calculateBondOrderRaw();

    this->calculateBondOrderCorr();
}

void ReaxPot::calculateBondOrderRaw()
{
    int iatom;
    int natom = this->geometry->getNumAtoms();

    int  nbond;
    int  ineigh;
    int  nneigh;
    int* idxBond;

    int ielem;
    int jelem;

    int*   elemNeigh;
    real** rxyzNeigh;

    real r;
    real ri_sigma, rj_sigma, r_sigma;
    real ri_pi,    rj_pi,    r_pi;
    real ri_pipi,  rj_pipi,  r_pipi;

    real pbo1, pbo2;
    real pbo3, pbo4;
    real pbo5, pbo6;

    real BO;
    real BO_sigma;
    real BO_pi;
    real BO_pipi;
    real BO_cut = this->param->BO_cut;

    real logBO_sigma;
    real logBO_pi;
    real logBO_pipi;

    real dBOdr_sigma;
    real dBOdr_pi;
    real dBOdr_pipi;

    real** BO_raw;
    real** dBOdr_raw;
    real   Delta_raw;

    this->numBonds = new int [natom];
    this->idxBonds = new int*[natom];

    this->BOs_raw    = new real**[natom];
    this->dBOdrs_raw = new real**[natom];
    this->Deltas_raw = new real  [natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh    = this->geometry->getNumNeighbors(iatom);
        elemNeigh = this->elemNeighs[iatom];
        rxyzNeigh = this->rxyzNeighs[iatom];

        nbond   = 0;
        idxBond = new int[nneigh];

        BO_raw    = new real*[nneigh];
        dBOdr_raw = new real*[nneigh];

        ielem = elemNeigh[0];

        Delta_raw = -this->param->Val[ielem];

        ri_sigma = this->param->r_atom_sigma[ielem];
        ri_pi    = this->param->r_atom_pi   [ielem];
        ri_pipi  = this->param->r_atom_pipi [ielem];

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            jelem = elemNeigh[ineigh + 1];
            r     = rxyzNeigh[ineigh][0];

            rj_sigma = this->param->r_atom_sigma[jelem];
            rj_pi    = this->param->r_atom_pi   [jelem];
            rj_pipi  = this->param->r_atom_pipi [jelem];
            r_sigma  = this->param->r_pair_sigma[ielem][jelem];
            r_pi     = this->param->r_pair_pi   [ielem][jelem];
            r_pipi   = this->param->r_pair_pipi [ielem][jelem];
            pbo1     = this->param->p_bo1       [ielem][jelem];
            pbo2     = this->param->p_bo2       [ielem][jelem];
            pbo3     = this->param->p_bo3       [ielem][jelem];
            pbo4     = this->param->p_bo4       [ielem][jelem];
            pbo5     = this->param->p_bo5       [ielem][jelem];
            pbo6     = this->param->p_bo6       [ielem][jelem];

            if (ri_sigma > ZERO && rj_sigma > ZERO)
            {
                logBO_sigma = pbo1 * pow(r / r_sigma, pbo2);
                BO_sigma    = (ONE + BO_cut) * exp(logBO_sigma);
                dBOdr_sigma = pbo2 / r * logBO_sigma * BO_sigma;
            }
            else
            {
                BO_sigma    = ZERO;
                dBOdr_sigma = ZERO;
            }

            if (ri_pi > ZERO && rj_pi > ZERO)
            {
                logBO_pi = pbo3 * pow(r / r_pi, pbo4);
                BO_pi    = exp(logBO_pi);
                dBOdr_pi = pbo4 / r * logBO_pi * BO_pi;
            }
            else
            {
                BO_pi    = ZERO;
                dBOdr_pi = ZERO;
            }

            if (ri_pipi > ZERO && rj_pipi > ZERO)
            {
                logBO_pipi = pbo5 * pow(r / r_pipi, pbo6);
                BO_pipi    = exp(logBO_pipi);
                dBOdr_pipi = pbo6 / r * logBO_pipi * BO_pipi;
            }
            else
            {
                BO_pipi    = ZERO;
                dBOdr_pipi = ZERO;
            }

            BO = BO_sigma + BO_pi + BO_pipi;

            if (BO >= BO_cut)
            {
                BO       -= BO_cut;
                BO_sigma -= BO_cut;

                BO_raw[nbond] = new real[3];
                BO_raw[nbond][0] = BO_sigma;
                BO_raw[nbond][1] = BO_pi;
                BO_raw[nbond][2] = BO_pipi;

                dBOdr_raw[nbond] = new real[3];
                dBOdr_raw[nbond][0] = dBOdr_sigma;
                dBOdr_raw[nbond][1] = dBOdr_pi;
                dBOdr_raw[nbond][2] = dBOdr_pipi;

                idxBond[nbond] = ineigh;
                nbond++;

                Delta_raw += BO;
            }
        }

        this->numBonds[iatom] = nbond;
        this->idxBonds[iatom] = idxBond;

        this->BOs_raw   [iatom] = BO_raw;
        this->dBOdrs_raw[iatom] = dBOdr_raw;
        this->Deltas_raw[iatom] = Delta_raw;
    }
}

void ReaxPot::calculateBondOrderCorr()
{
    int iatom;
    int jatom;
    int natom = this->geometry->getNumAtoms();

    int  ibond;
    int  nbond;
    int  ineigh;
    int* idxBond;
    const int** neighbor;

    int ielem;
    int jelem;

    int* elemNeigh;

    real fac1, fac2, fac3, fac4;

    real ovc_corr;
    real v13_corr;

    // to keep accuracy of f1, f4, f5, some variables are defined as double
    double pboc1 = (double) this->param->p_boc1;
    double pboc2 = (double) this->param->p_boc2;
    double pboc3, pboc4, pboc5;

    double Vali,     Valj;
    double Vali_boc, Valj_boc;

    double Deltai,     Deltaj;
    double Deltai_boc, Deltaj_boc;
    double exp1Deltai, exp1Deltaj;
    double exp2Deltai, exp2Deltaj;

    double f1, f2, f3;
    double e4, f4, g4;
    double e5, f5, g5;

    double vif2, vif2f3;
    double vjf2, vjf2f3;

    double df4dBO;
    double df5dBO;

    double df1df2;
    double df1df3;
    double df1dDelta;
    double df2dDelta;
    double df3dDelta;
    double df4dDelta;

    double BOr_tot;
    double BOr2_tot;
    double BOr_sigma;
    double BOr_pi;
    double BOr_pipi;

    real BOc_tot;
    real BOc_pi;
    real BOc_pipi;

    real dBOdBO_tot;
    real dBOdBO_pi1;
    real dBOdBO_pipi1;
    real dBOdBO_pi2;
    real dBOdBO_pipi2;

    real dBOdDelta_tot;
    real dBOdDelta_pi;
    real dBOdDelta_pipi;

    real** BO_raw;
    real** BO_corr;
    real** dBOdBO;
    real** dBOdDelta;
    real   Delta_corr;
    real   Delta_e;

    double* exp1Deltas = new double[natom];
    double* exp2Deltas = new double[natom];

    this->BOs_corr    = new real**[natom];
    this->dBOdBOs     = new real**[natom];
    this->dBOdDeltas  = new real**[natom];
    this->Deltas_corr = new real  [natom];
    this->Deltas_e    = new real  [natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        Deltai = this->Deltas_raw[iatom];
        exp1Deltas[iatom] = exp(-pboc1 * Deltai);
        exp2Deltas[iatom] = exp(-pboc2 * Deltai);
    }

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nbond     = this->numBonds[iatom];
        idxBond   = this->idxBonds[iatom];
        neighbor  = this->geometry->getNeighbors(iatom);
        elemNeigh = this->elemNeighs[iatom];

        BO_raw    = this->BOs_raw[iatom];
        BO_corr   = new real*[nbond];
        dBOdBO    = new real*[nbond];
        dBOdDelta = new real*[nbond];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            BO_corr  [ibond] = new real[3];
            dBOdBO   [ibond] = new real[5];
            dBOdDelta[ibond] = new real[3];
        }

        ielem = elemNeigh[0];

        Delta_corr = -this->param->Val  [ielem];
        Delta_e    = -this->param->Val_e[ielem];

        Vali       = (double) this->param->Val    [ielem];
        Vali_boc   = (double) this->param->Val_boc[ielem];
        Deltai     = (double) this->Deltas_raw    [iatom];
        Deltai_boc = Deltai + Vali - Vali_boc;
        exp1Deltai = exp1Deltas          [iatom];
        exp2Deltai = exp2Deltas          [iatom];

        for (ibond = 0; ibond < nbond; ++ibond)
        {
            ineigh = idxBond[ibond];
            jatom  = neighbor[ineigh][0];
            jelem  = elemNeigh[ineigh + 1];

            BOr_sigma = (double) BO_raw[ibond][0];
            BOr_pi    = (double) BO_raw[ibond][1];
            BOr_pipi  = (double) BO_raw[ibond][2];
            BOr_tot   = BOr_sigma + BOr_pi + BOr_pipi;
            BOr2_tot  = BOr_tot * BOr_tot;

            Valj       = (double) this->param->Val     [jelem];
            Valj_boc   = (double) this->param->Val_boc [jelem];
            Deltaj     = (double) this->Deltas_raw     [jatom];
            Deltaj_boc = Deltaj + Valj - Valj_boc;
            exp1Deltaj = exp1Deltas           [jatom];
            exp2Deltaj = exp2Deltas           [jatom];
            pboc3      = (double) this->param->p_boc3  [ielem][jelem];
            pboc4      = (double) this->param->p_boc4  [ielem][jelem];
            pboc5      = (double) this->param->p_boc5  [ielem][jelem];
            ovc_corr   = this->param->ovc_corr[ielem][jelem];
            v13_corr   = this->param->v13_corr[ielem][jelem];

            // calculate f1
            if (ovc_corr >= OVC_THR)
            {
                f2 = exp1Deltai + exp1Deltaj;
                df2dDelta = -pboc1 * exp1Deltai;

                f3 = exp2Deltai + exp2Deltaj;
                f3 = -log(0.5 * f3) / pboc2;
                df3dDelta = exp2Deltai / (exp2Deltai + exp2Deltaj);

                vif2   = Vali + f2;
                vjf2   = Valj + f2;
                vif2f3 = vif2 + f3;
                vjf2f3 = vjf2 + f3;

                f1     =  0.5 * (vif2 / vif2f3 + vjf2 / vjf2f3);
                df1df3 = -0.5 * (vif2 / vif2f3 / vif2f3 + vjf2 / vjf2f3 / vjf2f3);
                df1df2 =  0.5 * (1.0 / vif2f3 + 1.0 / vjf2f3) + df1df3;
                df1dDelta = df1df2 * df2dDelta + df1df3 * df3dDelta;
            }
            else
            {
                f1 = 1.0;
                df1dDelta = 0.0;
            }

            // calculate f4 and f5
            if (v13_corr >= V13_THR)
            {
                e4 = -pboc3 * (pboc4 * BOr2_tot - Deltai_boc) + pboc5;
                e4 = exp(e4);
                f4 = 1.0 / (1.0 + e4);
                g4 = pboc3 * e4 * f4 * f4;

                e5 = -pboc3 * (pboc4 * BOr2_tot - Deltaj_boc) + pboc5;
                e5 = exp(e5);
                f5 = 1.0 / (1.0 + e5);
                g5 = pboc3 * e5 * f5 * f5;

                fac1      = 2.0 * BOr_tot * pboc4;
                df4dBO    = fac1 * g4;
                df5dBO    = fac1 * g5;
                df4dDelta = -g4;
            }
            else
            {
                f4 = 1.0;
                f5 = 1.0;
                df4dBO    = 0.0;
                df5dBO    = 0.0;
                df4dDelta = 0.0;
            }

            // correct BO
            fac1     = (real) (f1 * f4 * f5);
            fac2     = ((real) f1) * fac1;
            BOc_tot  = ((real) BOr_tot)  * fac1;
            BOc_pi   = ((real) BOr_pi)   * fac2;
            BOc_pipi = ((real) BOr_pipi) * fac2;

            fac3         = (real) (f1 * (df4dBO * f5 + f4 * df5dBO));
            fac4         = ((real) f1) * fac3;
            dBOdBO_tot   = fac1 + ((real) BOr_tot) * fac3;
            dBOdBO_pi1   = fac2;
            dBOdBO_pipi1 = fac2;
            dBOdBO_pi2   = ((real) BOr_pi)   * fac4;
            dBOdBO_pipi2 = ((real) BOr_pipi) * fac4;

            fac1           = ((real) (df1dDelta * f4 + f1 * df4dDelta) * f5);
            fac2           = ((real) f1 * (2.0 * df1dDelta * f4 + f1 * df4dDelta) * f5);
            dBOdDelta_tot  = ((real) BOr_tot)  * fac1;
            dBOdDelta_pi   = ((real) BOr_pi)   * fac2;
            dBOdDelta_pipi = ((real) BOr_pipi) * fac2;

            if (BOc_tot < BO_THR)
            {
                BOc_tot       = ZERO;
                dBOdBO_tot    = ZERO;
                dBOdDelta_tot = ZERO;
            }

            if (BOc_pi < BO_THR)
            {
                BOc_pi       = ZERO;
                dBOdBO_pi1   = ZERO;
                dBOdBO_pi2   = ZERO;
                dBOdDelta_pi = ZERO;
            }

            if (BOc_pipi < BO_THR)
            {
                BOc_pipi       = ZERO;
                dBOdBO_pipi1   = ZERO;
                dBOdBO_pipi2   = ZERO;
                dBOdDelta_pipi = ZERO;
            }

            BO_corr[ibond][0] = BOc_tot;
            BO_corr[ibond][1] = BOc_pi;
            BO_corr[ibond][2] = BOc_pipi;

            dBOdBO[ibond][0] = dBOdBO_tot;    // dBO/dBO'
            dBOdBO[ibond][1] = dBOdBO_pi1;    // dBO(pi)/dBO'(pi)
            dBOdBO[ibond][2] = dBOdBO_pipi1;  // dBO(pipi)/dBO'(pipi)
            dBOdBO[ibond][3] = dBOdBO_pi2;    // dBO(pi)/dBO'
            dBOdBO[ibond][4] = dBOdBO_pipi2;  // dBO(pipi)/dBO'

            dBOdDelta[ibond][0] = dBOdDelta_tot;
            dBOdDelta[ibond][1] = dBOdDelta_pi;
            dBOdDelta[ibond][2] = dBOdDelta_pipi;

            Delta_corr += BOc_tot;
            Delta_e    += BOc_tot;
        }

        this->BOs_corr   [iatom] = BO_corr;
        this->dBOdBOs    [iatom] = dBOdBO;
        this->dBOdDeltas [iatom] = dBOdDelta;
        this->Deltas_corr[iatom] = Delta_corr;
        this->Deltas_e   [iatom] = Delta_e;
    }

    delete[] exp1Deltas;
    delete[] exp2Deltas;
}

