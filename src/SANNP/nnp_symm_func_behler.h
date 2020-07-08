/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_SYMM_FUNC_BEHLER_H_
#define NNP_SYMM_FUNC_BEHLER_H_

#include "nnp_common.h"
#include "nnp_symm_func.h"

class SymmFuncBehler : public SymmFunc
{
public:
    SymmFuncBehler(int numElems, int sizeRad, int sizeAng, real radiusCut,
                   const real* radiusEta, const real* radiusShift, const real* angleEta, const real* angleZeta);

    virtual ~SymmFuncBehler();

    void calculate(int numNeighbor, real** posNeighbor, int* elemNeighbor,
                   real* symmData, real* symmDiff) const;

    int getNumRadBasis() const
    {
        return this->numRadBasis;
    }

    int getNumAngBasis() const
    {
        return this->numAngBasis;
    }

private:
    int sizeRad;
    int sizeAng;

    int numRadBasis;
    int numAngBasis;

    real radiusCut;

    const real* radiusEta;
    const real* radiusShift;

    const real* angleEta;
    const real* angleZeta;
};

#endif /* NNP_SYMM_FUNC_BEHLER_H_ */
