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


class SymmFuncManyBody : public SymmFunc
{
public:
    SymmFuncManyBody(int numElems, int size2Body, int size3Body, real radiusInner, real radiusOuter);
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

#endif /* NNP_SYMM_FUNC_H_ */
