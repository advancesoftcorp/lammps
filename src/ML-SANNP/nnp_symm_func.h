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

    virtual void calculate(int numNeighbor, int* elemNeighbor, nnpreal** posNeighbor,
                           nnpreal* symmData, nnpreal* symmDiff) = 0;

#ifdef _NNP_GPU
    virtual void calculate(int numAtoms, int* numNeighbor, int* idxNeighbor, int** elemNeighbor, nnpreal*** posNeighbor,
                           nnpreal* symmData, nnpreal* symmDiff)
    {
        int iatom;

        for (iatom = 0; iatom < numAtoms; ++iatom)
        {
            this->calculate(numNeighbor[iatom], elemNeighbor[iatom], posNeighbor[iatom],
                            symmData, symmDiff);
        }
    }

    virtual nnpreal* getSymmGrad()
    {
        // NOP
        return nullptr;
    }

    virtual void allocHiddenDiff(int maxAtoms, int fullNeigh)
    {
        // NOP
    }

    virtual void driveHiddenDiff(int lenAtoms, int* numNeighbor, int* idxNeighbor, nnpreal* forceData)
    {
        // NOP
    }
#endif

    int getNumBasis() const
    {
        return this->numBasis;
    }

    bool isTransDiff() const
    {
        return this->transDiff;
    }

    bool isHiddenDiff() const
    {
        return this->hiddenDiff;
    }

    void cutoffFunction(nnpreal* fc, nnpreal* dfcdr, nnpreal r, nnpreal rc) const;

protected:
    int numElems;

    int numBasis;

    bool tanhCutFunc;

    bool elemWeight;

    bool transDiff;

    bool hiddenDiff;
};

inline void SymmFunc::cutoffFunction(nnpreal* fc, nnpreal* dfcdr, nnpreal r, nnpreal rc) const
{
    if (this->tanhCutFunc)
    {
        nnpreal tanh1 = tanh(ONE - r / rc);
        nnpreal tanh2 = tanh1 * tanh1;
        fc[0]      = tanh1 * tanh2;
        dfcdr[0]   = -NNPREAL(3.0) * tanh2 * (ONE - tanh2) / rc;
    }

    else
    {
        nnpreal fac = PI / rc;
        fc[0]    =  NNPREAL(0.5) * (cos(fac * r) + ONE);
        dfcdr[0] = -NNPREAL(0.5) * fac * sin(fac * r);
    }
}

#endif /* NNP_SYMM_FUNC_H_ */
