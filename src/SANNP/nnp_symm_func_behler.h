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
    SymmFuncBehler(int numElems, bool tanhCutFunc, bool elemWeight,
                   int sizeRad, int sizeAng, real rcutRad, real rcutAng);

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

    void setRadiusData(const real* radiusEta, const real* radiusShift)
    {
        if (this->sizeRad < 1)
        {
            return;
        }

        if (radiusEta == NULL)
        {
            stop_by_error("radiusEta is null.");
        }

        if (radiusShift == NULL)
        {
            stop_by_error("radiusShift is null.");
        }

        this->radiusEta = radiusEta;

        this->radiusShift = radiusShift;
    }

    void setAngleData(bool angleMod, const real* angleEta, const real* angleZeta, const real* angleShift)
    {
        if (this->sizeAng < 1)
        {
            return;
        }

        if (angleEta == NULL)
        {
            stop_by_error("angleEta is null.");
        }

        if (angleZeta == NULL)
        {
            stop_by_error("angleZeta is null.");
        }

        if (angleShift == NULL)
        {
            stop_by_error("angleShift is null.");
        }

        this->angleMod = angleMod;

        this->angleEta = angleEta;

        this->angleZeta = angleZeta;

        this->angleShift = angleShift;
    }

private:
    int sizeRad;
    int sizeAng;

    int numRadBasis;
    int numAngBasis;

    real rcutRad;
    real rcutAng;

    real radiusCut;

    const real* radiusEta;
    const real* radiusShift;

    bool angleMod;
    const real* angleEta;
    const real* angleZeta;
    const real* angleShift;
};

#endif /* NNP_SYMM_FUNC_BEHLER_H_ */
