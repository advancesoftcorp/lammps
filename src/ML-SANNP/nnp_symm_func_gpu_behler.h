/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_SYMM_FUNC_GPU_BEHLER_H_
#define NNP_SYMM_FUNC_GPU_BEHLER_H_

#include "nnp_common.h
#include "nnp_symm_func_gpu.h"

class SymmFuncGPUBehler : public SymmFuncGPU
{
public:
    SymmFuncGPUBehler(int numElems, bool tanhCutFunc, bool elemWeight,
                      int sizeRad, int sizeAng, nnpreal rcutRad, nnpreal rcutAng, int cutoffMode);

    virtual ~SymmFuncGPUBehler() override;

    void setRadiusData(const nnpreal* radiusEta, const nnpreal* radiusShift)
    {
        if (this->sizeRad < 1)
        {
            return;
        }

        if (radiusEta == nullptr)
        {
            stop_by_error("radiusEta is null.");
        }

        if (radiusShift == nullptr)
        {
            stop_by_error("radiusShift is null.");
        }

        if (this->radiusEta   == nullptr) cudaMalloc(&(this->radiusEta),   sizeof(nnpreal) * this->sizeRad);
        if (this->radiusShift == nullptr) cudaMalloc(&(this->radiusShift), sizeof(nnpreal) * this->sizeRad);

        cudaMemcpy(this->radiusEta,   radiusEta,   sizeof(nnpreal) * this->sizeRad, cudaMemcpyHostToDevice);
        cudaMemcpy(this->radiusShift, radiusShift, sizeof(nnpreal) * this->sizeRad, cudaMemcpyHostToDevice);
    }

    void setAngleData(bool angleMod, const nnpreal* angleEta, const nnpreal* angleZeta, const nnpreal* angleShift)
    {
        if (this->sizeAng < 1)
        {
            return;
        }

        if (angleEta == nullptr)
        {
            stop_by_error("angleEta is null.");
        }

        if (angleZeta == nullptr)
        {
            stop_by_error("angleZeta is null.");
        }

        if (angleShift == nullptr)
        {
            stop_by_error("angleShift is null.");
        }

        this->angleMod = angleMod;

        if (this->angleEta   == nullptr) cudaMalloc(&(this->angleEta),   sizeof(nnpreal) * this->sizeAng / 2);
        if (this->angleZeta  == nullptr) cudaMalloc(&(this->angleZeta),  sizeof(nnpreal) * this->sizeAng / 2);
        if (this->angleShift == nullptr) cudaMalloc(&(this->angleShift), sizeof(nnpreal) * this->sizeAng / 2);

        cudaMemcpy(this->angleEta,   angleEta,   sizeof(nnpreal) * this->sizeAng / 2, cudaMemcpyHostToDevice);
        cudaMemcpy(this->angleZeta,  angleZeta,  sizeof(nnpreal) * this->sizeAng / 2, cudaMemcpyHostToDevice);
        cudaMemcpy(this->angleShift, angleShift, sizeof(nnpreal) * this->sizeAng / 2, cudaMemcpyHostToDevice);
    }

protected:
    void calculateRadial(dim3 grid, dim3 block);

    void calculateAnglarElemWeight(dim3 grid, dim3 block, size_t sizeShared);

    void calculateAnglarNotElemWeight(dim3 grid, dim3 block, size_t sizeShared, int dimBasis);

private:
    nnpreal* radiusEta;
    nnpreal* radiusShift;

    bool     angleMod;
    nnpreal* angleEta;
    nnpreal* angleZeta;
    nnpreal* angleShift;
};

#endif /* NNP_SYMM_FUNC_GPU_BEHLER_H_ */
