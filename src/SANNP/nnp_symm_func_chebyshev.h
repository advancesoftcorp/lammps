/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_SYMM_FUNC_CHEBYSHEV_H_
#define NNP_SYMM_FUNC_CHEBYSHEV_H_

#include "nnp_common.h"
#include "nnp_symm_func.h"

class SymmFuncChebyshev : public SymmFunc
{
public:
    SymmFuncChebyshev(int numElems, bool tanhCutFunc, bool elemWeight,
                      int sizeRad, int sizeAng, real rcutRad, real rcutAng);

    virtual ~SymmFuncChebyshev();

    void calculate(int numNeighbor, int* elemNeighbor, real** posNeighbor,
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

    real rcutRad;
    real rcutAng;
};

#endif /* NNP_SYMM_FUNC_CHEBYSHEV_H_ */
