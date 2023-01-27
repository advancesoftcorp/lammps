/*
 * Copyright (C) 2023 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_REAX_PARAM_H_
#define NNP_REAX_PARAM_H_

#include "nnp_common.h"

#define MAX_ATOM_NUM  256

class ReaxParam
{
public:
    ReaxParam(nnpreal rcut, FILE* fp, int rank, MPI_Comm world);
    virtual ~ReaxParam();

    int  numElems;
    int* atomNums;

    nnpreal   rcut_bond;
    nnpreal   rcut_vdw;
    nnpreal   BO_cut;

    nnpreal*  mass;

    nnpreal*  r_atom_sigma;
    nnpreal*  r_atom_pi;
    nnpreal*  r_atom_pipi;

    nnpreal** r_pair_sigma;
    nnpreal** r_pair_pi;
    nnpreal** r_pair_pipi;

    nnpreal** p_bo1;
    nnpreal** p_bo2;
    nnpreal** p_bo3;
    nnpreal** p_bo4;
    nnpreal** p_bo5;
    nnpreal** p_bo6;

    nnpreal** De_sigma;
    nnpreal** De_pi;
    nnpreal** De_pipi;
    nnpreal** p_be1;
    nnpreal** p_be2;

    nnpreal** ovc_corr;
    nnpreal** v13_corr;

    nnpreal   p_boc1;
    nnpreal   p_boc2;
    nnpreal*  p_boc3_atom;
    nnpreal*  p_boc4_atom;
    nnpreal*  p_boc5_atom;
    nnpreal** p_boc3;
    nnpreal** p_boc4;
    nnpreal** p_boc5;

    nnpreal** p_over1;
    nnpreal*  p_over2;
    nnpreal   p_over3;
    nnpreal   p_over4;

    nnpreal*  Val;
    nnpreal*  Val_e;
    nnpreal*  Val_boc;
    nnpreal*  Val_ang;

    nnpreal*  p_lp2;
    nnpreal*  n_lp_opt;

    nnpreal   r1_lp;
    nnpreal   r2_lp;
    nnpreal   lambda_lp;

    int       shielding;
    int       innerWall;

    nnpreal   swa_vdw;
    nnpreal   swb_vdw;
    nnpreal   Tap_vdw[8];

    nnpreal*  D_vdw_atom;
    nnpreal*  alpha_vdw_atom;
    nnpreal*  gammaw_atom;
    nnpreal*  r_vdw_atom;
    nnpreal*  p_core1_atom;
    nnpreal*  p_core2_atom;
    nnpreal*  p_core3_atom;

    nnpreal** D_vdw;
    nnpreal** alpha_vdw;
    nnpreal** gammaw;
    nnpreal** r_vdw;
    nnpreal   p_vdw;
    nnpreal** p_core1;
    nnpreal** p_core2;
    nnpreal** p_core3;

    int atomNumToElement(int atomNum) const
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

private:
    int* atomNumMap;

    void createAtomNumMap();

    nnpreal* allocateElemData();
    void deallocateElemData(nnpreal*  data);

    nnpreal** allocatePairData();
    void deallocatePairData(nnpreal** data);

    void allocateAllData();
    void deallocateAllData();

    void setPairData(nnpreal** data, int i, int j, nnpreal value);

    int  readFFieldReax(FILE* fp);

    int  elementToAtomNum(const char *elem);

    void toRealElement(char *elem);

    int  modifyParameters();

    void modifyParametersBondOrder();

    void modifyParametersLonePairNumber();

    void modifyParametersVanDerWaalsEnergy();

    void shareParameters(int rank, MPI_Comm world);
};

#endif /* NNP_REAX_PARAM_H_ */
