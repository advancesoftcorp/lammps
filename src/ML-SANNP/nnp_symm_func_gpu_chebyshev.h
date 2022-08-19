/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_SYMM_FUNC_GPU_CHEBYSHEV_H_
#define NNP_SYMM_FUNC_GPU_CHEBYSHEV_H_

#include "nnp_common.h
#include "nnp_symm_func_gpu.h"

class SymmFuncGPUChebyshev : public SymmFuncGPU
{
public:
    SymmFuncGPUChebyshev(int numElems, bool tanhCutFunc, bool elemWeight,
                         int sizeRad, int sizeAng, nnpreal rcutRad, nnpreal rcutAng, int cutoffMode);

    virtual ~SymmFuncGPUChebyshev() override;

protected:
    void calculateRadial(dim3 grid, dim3 block);

    void calculateAnglarElemWeight(dim3 grid, dim3 block, size_t sizeShared);

    void calculateAnglarNotElemWeight(dim3 grid, dim3 block, size_t sizeShared, int dimBasis);
};

#endif /* NNP_SYMM_FUNC_GPU_CHEBYSHEV_H_ */
