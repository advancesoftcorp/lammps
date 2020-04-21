/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NEURAL_NETWORK_POTENTIAL_H_
#define NEURAL_NETWORK_POTENTIAL_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cmath>
using namespace std;

#ifdef _SINGLE

#define real     float
#define REAL(x)  x##f

#define IFORM_F1     "%f"
#define IFORM_F2     "%f %f"
#define IFORM_S1_F2  "%s %f %f"
#define IFORM_D2_F1  "%d %d %f"
#define IFORM_D2_F2  "%d %d %f %f"

#define xgemv_  sgemv_
#define xgemm_  sgemm_

#define MPI_REAL0  MPI_FLOAT

#else

#define real     double
#define REAL(x)  x

#define IFORM_F1     "%lf"
#define IFORM_F2     "%lf %lf"
#define IFORM_S1_F2  "%s %lf %lf"
#define IFORM_D2_F1  "%d %d %lf"
#define IFORM_D2_F2  "%d %d %lf %lf"

#define xgemm_  dgemm_
#define xgemv_  dgemv_

#define MPI_REAL0  MPI_DOUBLE

#endif

#define ZERO       REAL(0.0)
#define ONE        REAL(1.0)
#define PI         REAL(3.14159265358979324)
#define PId        3.14159265358979324

extern "C"
{

int xgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
           const real* alpha, real* a, const int* lda, real* b, const int* ldb,
           const real* beta, real* c, const int* ldc);

int xgemv_(const char* trans, const int* m, const int* n,
           const real* alpha, real* a, const int* lda, real* x, const int* incx,
           const real* beta, real* y, const int* incy);

}

#define SYMM_FUNC_NULL      0
#define SYMM_FUNC_MANYBODY  1
#define SYMM_FUNC_BEHLER    2

#define ACTIVATION_NULL     0
#define ACTIVATION_ASIS     1
#define ACTIVATION_SIGMOID  2
#define ACTIVATION_TANH     3
#define ACTIVATION_ELU      4

// NNArch
#define NNARCH_MODE_BOTH    0
#define NNARCH_MODE_ENERGY  1
#define NNARCH_MODE_CHARGE  2

inline void stop_by_error(const char* message)
{
    printf("[STOP] %s\n", message);
    fflush(stdout);

    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();

    exit(1);
}

class NNLayer
{
public:
    NNLayer(int numInpNodes, int numOutNodes, int activation);
    virtual ~NNLayer();

    void setSizeOfBatch(int sizeBatch);

    void scanWeight(FILE* fp, int rank, MPI_Comm world);

    void projectWeightFrom(NNLayer* src, int* mapInpNodes);

    void goForward(real* outData) const;

    void goBackward(const real* outData, real* outGrad, bool toInpGrad);

    int getNumInpNodes() const
    {
        return this->numInpNodes;
    }

    int getNumOutNodes() const
    {
        return this->numOutNodes;
    }

    real* getData()
    {
        return this->inpData;
    }

    real* getGrad()
    {
        return this->inpGrad;
    }

private:
    int numInpNodes;
    int numOutNodes;
    int sizeBatch;

    int activation;

    real* inpData;
    real* inpGrad;

    real* weight;
    real* bias;

    void operateActivation(real* outData) const;

    void deriveActivation(const real* outData, real* outGrad) const;
};

class Property
{
public:
    Property();
    virtual ~Property();

    void peekProperty(FILE* fp, int rank, MPI_Comm world);

    int getSymmFunc() const
    {
        return this->symmFunc;
    }

    int getM2() const
    {
        return this->m2;
    }

    int getM3() const
    {
        return this->m3;
    }

    real getRinner() const
    {
        return this->rinner;
    }

    real getRouter() const
    {
        return this->router;
    }

    int getNumRadius() const
    {
        return this->numRadius;
    }

    int getNumAngle() const
    {
        return this->numAngle;
    }

    const real* getBehlerEta1() const
    {
        return this->behlerEta1;
    }

    const real* getBehlerEta2() const
    {
        return this->behlerEta2;
    }

    const real* getBehlerRs() const
    {
        return this->behlerRs;
    }

    const real* getBehlerZeta() const
    {
        return this->behlerZeta;
    }

    int getLayersEnergy() const
    {
        return this->layersEnergy;
    }

    int getNodesEnergy() const
    {
        return this->nodesEnergy;
    }

    int getActivEnergy() const
    {
        return this->activEnergy;
    }

    int getLayersCharge() const
    {
        return this->layersCharge;
    }

    int getNodesCharge() const
    {
        return this->nodesCharge;
    }

    int getActivCharge() const
    {
        return this->activCharge;
    }

    int getWithCharge() const
    {
        return this->withCharge;
    }

private:
    // about symmetry functions
    int symmFunc;

    int m2;
    int m3;
    real rinner;
    real router;

    int   numRadius;
    int   numAngle;
    real* behlerEta1;
    real* behlerEta2;
    real* behlerRs;
    real* behlerZeta;

    // about neural networks
    int  layersEnergy;
    int  nodesEnergy;
    int  activEnergy;

    int  layersCharge;
    int  nodesCharge;
    int  activCharge;

    int  withCharge;
};

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

class NNArch
{
public:
    NNArch(int mode, int numElems, const Property* property);
    virtual ~NNArch();

    void restoreNN(FILE* fp, int numElement, char** elementNames, int rank, MPI_Comm world);

    void setMapElem(int* mapElem);

    void initGeometry(int inum, int* ilist, int* type, int* typeMap, int* numneigh);

    void clearGeometry();

    void calculateSymmFuncs(int* numNeighbor, int** elemNeighbor, real*** posNeighbor);

    void renormalizeSymmFuncs();

    void initLayers();

    void goForwardOnEnergy();

    void goBackwardOnForce();

    void goForwardOnCharge();

    void obtainEnergies(real* energies) const;

    void obtainForces(real*** forces) const;

    void obtainCharges(real* charges) const;

private:
    int mode;

    int numElems;
    int numAtoms;

    const Property* property;

    int* mapElem;

    int* indexElem; // iatom -> ielem
    int* numNeighbor; // iatom -> nneigh

    int   mbatch;
    int*  nbatch;
    int*  ibatch;

    real** energyData;
    real** energyGrad;

    real*** forceData;

    real** chargeData;

    real**   symmData;
    real**   symmDiff;
    real*     symmAve;
    real*     symmDev;
    SymmFunc* symmFunc;

    NNLayer*** interLayersEnergy;
    NNLayer**  lastLayersEnergy;

    NNLayer*** interLayersCharge;
    NNLayer**  lastLayersCharge;

    bool isEnergyMode() const
    {
        return this->mode == NNARCH_MODE_BOTH || this->mode == NNARCH_MODE_ENERGY;
    }

    bool isChargeMode() const
    {
        return this->mode == NNARCH_MODE_BOTH || this->mode == NNARCH_MODE_CHARGE;
    }

    void initGeometries();

    SymmFunc* getSymmFunc();
};

#endif /* NEURAL_NETWORK_POTENTIAL_H_ */
