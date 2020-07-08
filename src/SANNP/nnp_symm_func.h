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
    SymmFunc(int numElemes);
    virtual ~SymmFunc();

    virtual void calculate(int numNeighbor, real** posNeighbor, int* elemNeighbor,
                           real* symmData, real* symmDiff) const = 0;

    int getNumBasis() const
    {
        return this->numBasis;
    }

protected:
    int numElems;
    int numBasis;
};

#endif /* NNP_SYMM_FUNC_H_ */
