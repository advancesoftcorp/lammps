/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_SYMM_FUNC_MANYBODY_H_
#define NNP_SYMM_FUNC_MANYBODY_H_

#include "nnp_common.h"
#include "nnp_symm_func.h"

class SymmFuncManyBody : public SymmFunc
{
public:
    SymmFuncManyBody(int numElems, bool elemWeight, int size2Body, int size3Body, real radiusInner, real radiusOuter);
    virtual ~SymmFuncManyBody();

    void calculate(int numNeighbor, real** posNeighbor, int* elemNeighbor,
                   real* symmData, real* symmDiff) const;

    int getNum2BodyBasis() const
    {
        return this->num2BodyBasis;
    }

    int getNum3BodyBasis() const
    {
        return this->num3BodyBasis;
    }

private:
    int size2Body;
    int size3Body;

    int num2BodyBasis;
    int num3BodyBasis;

    real radiusInner;
    real radiusOuter;

    real step2Body;
    real step3Body;
};

#endif /* NNP_SYMM_FUNC_MANYBODY_H_ */
