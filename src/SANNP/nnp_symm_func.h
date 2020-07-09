/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_SYMM_FUNC_H_
#define NNP_SYMM_FUNC_H_

#include "nnp_common.h"

class SymmFunc
{
public:
    SymmFunc(int numElemes, bool tanhCutFunc, bool elemWeight);
    virtual ~SymmFunc();

    virtual void calculate(int numNeighbor, int* elemNeighbor, real** posNeighbor,
                           real* symmData, real* symmDiff) const = 0;

    int getNumBasis() const
    {
        return this->numBasis;
    }

protected:
    int numElems;

    int numBasis;

    bool elemWeight;

    void cutoffFunction(real* fc, real* dfcdr, real r, real rc) const;

private:
    bool tanhCutFunc;
};

inline void SymmFunc::cutoffFunction(real* fc, real* dfcdr, real r, real rc) const
{
    if (this->tanhCutFunc)
    {
        real tanh1 = tanh(ONE - r / rc);
        real tanh2 = tanh1 * tanh1;
        fc[0]      = tanh1 * tanh2;
        dfcdr[0]   = -REAL(3.0) * tanh2 * (ONE - tanh2) / rc;
    }

    else
    {
        real fac = PI / rc;
        fc[0]    =  REAL(0.5) * (cos(fac * r) + ONE);
        dfcdr[0] = -REAL(0.5) * fac * sin(fac * r);
    }
}

#endif /* NNP_SYMM_FUNC_H_ */
