/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "neural_network_potential.h"

NNLayer::NNLayer(int numInpNodes, int numOutNodes, int activation)
{
    if (numInpNodes < 1)
    {
        stop_by_error("input size of neural network is not positive.");
    }

    if (numOutNodes < 1)
    {
        stop_by_error("output size of neural network is not positive.");
    }

    this->numInpNodes = numInpNodes;
    this->numOutNodes = numOutNodes;
    this->sizeBatch = 0;

    this->activation = activation;

    this->inpData = NULL;
    this->inpGrad = NULL;

    this->weight = new real[this->numInpNodes * this->numOutNodes];
    this->bias = new real[this->numOutNodes];
}

NNLayer::~NNLayer()
{
    if (this->inpData != NULL)
    {
        delete[] this->inpData;
    }
    if (this->inpGrad != NULL)
    {
        delete[] this->inpGrad;
    }

    delete[] this->weight;
    delete[] this->bias;
}

void NNLayer::setSizeOfBatch(int sizeBatch)
{
    if (sizeBatch < 1)
    {
        stop_by_error("size of batch is not positive.");
    }

    if (this->sizeBatch == sizeBatch)
    {
        return;
    }

    this->sizeBatch = sizeBatch;

    if (this->inpData != NULL)
    {
        delete[] this->inpData;
    }

    if (this->inpGrad != NULL)
    {
        delete[] this->inpGrad;
    }

    this->inpData = new real[this->numInpNodes * this->sizeBatch];
    this->inpGrad = new real[this->numInpNodes * this->sizeBatch];
}

void NNLayer::scanWeight(FILE* fp, int rank, MPI_Comm world)
{
    int iweight;
    int nweight = this->numInpNodes * this->numOutNodes;

    int ibias;
    int nbias = this->numOutNodes;

    int ierr;

    ierr = 0;
    if (rank == 0)
    {
        for (iweight = 0; iweight < nweight; ++iweight)
        {
            if (fscanf(fp, IFORM_F1, &(this->weight[iweight])) == EOF)
            {
                ierr = 1;
                break;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot scan neural network @weight");

    ierr = 0;
    if (rank == 0)
    {
        for (ibias = 0; ibias < nbias; ++ibias)
        {
            if (fscanf(fp, IFORM_F1, &(this->bias[ibias])) == EOF)
            {
                ierr = 1;
                break;
            }
        }
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot scan neural network @bias");

    MPI_Bcast(&(this->weight[0]), nweight, MPI_REAL0, 0, world);
    MPI_Bcast(&(this->bias[0]), nbias, MPI_REAL0, 0, world);
}

void NNLayer::projectWeightFrom(NNLayer* src, int* mapInpNodes)
{
    if (src == NULL)
    {
        stop_by_error("source layer is null.");
    }

    if (mapInpNodes == NULL)
    {
        stop_by_error("map of input nodes is null.");
    }

    int numInpNodes1 = src ->numInpNodes;
    int numInpNodes2 = this->numInpNodes;
    int numOutNodes1 = src ->numOutNodes;
    int numOutNodes2 = this->numOutNodes;
    int activation1  = src ->activation;
    int activation2  = this->activation;

    if (numOutNodes1 != numOutNodes2)
    {
        stop_by_error("cannot projet weight, for incorrect #out-nodes.");
    }

    if (activation1 != activation2)
    {
        stop_by_error("cannot projet weight, for incorrect activation.");
    }

    int ioutNodes;
    int iinpNodes1;
    int iinpNodes2;

    #pragma omp parallel for private (ioutNodes, iinpNodes1, iinpNodes2)
    for (ioutNodes = 0; ioutNodes < numOutNodes1; ++ioutNodes)
    {
        for (iinpNodes1 = 0; iinpNodes1 < numInpNodes1; ++iinpNodes1)
        {
            iinpNodes2 = mapInpNodes[iinpNodes1];

            if (0 <= iinpNodes2 && iinpNodes2 < numInpNodes2)
            {
                this->weight[iinpNodes2 + ioutNodes * numInpNodes2] =
                src ->weight[iinpNodes1 + ioutNodes * numInpNodes1];
            }
        }
    }
}

void NNLayer::goForward(real* outData) const
{
    if (outData == NULL)
    {
        stop_by_error("outData is null.");
    }

    if (this->inpData == NULL)
    {
        stop_by_error("inpData is null.");
    }

    if (this->sizeBatch < 1)
    {
        stop_by_error("size of batch is not positive.");
    }

    // inpData -> outData, through neural network
    real a0 = ZERO;
    real a1 = ONE;

    xgemm_("T", "N", &(this->numOutNodes), &(this->sizeBatch), &(this->numInpNodes),
           &a1, this->weight, &(this->numInpNodes), this->inpData, &(this->numInpNodes),
           &a0, outData, &(this->numOutNodes));

    int ibatch;
    int ioutNode;

    #pragma omp parallel for private (ibatch, ioutNode)
    for (ibatch = 0; ibatch < this->sizeBatch; ++ibatch)
    {
        for (ioutNode = 0; ioutNode < this->numOutNodes; ++ioutNode)
        {
            outData[ioutNode + ibatch * this->numOutNodes] += this->bias[ioutNode];
        }
    }

    // operate activation function
    this->operateActivation(outData);
}

void NNLayer::goBackward(const real* outData, real* outGrad, bool toInpGrad)
{
    if (outData == NULL)
    {
        stop_by_error("outData is null.");
    }

    if (outGrad == NULL)
    {
        stop_by_error("outGrad is null.");
    }

    if (this->sizeBatch < 1)
    {
        stop_by_error("size of batch is not positive.");
    }

    // derive activation function
    this->deriveActivation(outData, outGrad);

    real a0 = ZERO;
    real a1 = ONE;

    // outGrad -> inpGrad, through neural network
    if (toInpGrad)
    {
        if (this->inpGrad == NULL)
        {
            stop_by_error("inpGrad is null.");
        }

        xgemm_("N", "N", &(this->numInpNodes), &(this->sizeBatch), &(this->numOutNodes),
               &a1, this->weight, &(this->numInpNodes), outGrad, &(this->numOutNodes),
               &a0, this->inpGrad, &(this->numInpNodes));
    }
}

void NNLayer::operateActivation(real* outData) const
{
    real x;
    int idata;
    int ndata = this->sizeBatch * this->numOutNodes;

    if (this->activation == ACTIVATION_ASIS)
    {
        // NOP
    }

    else if (this->activation == ACTIVATION_SIGMOID)
    {
        #pragma omp parallel for private (idata, x)
        for (idata = 0; idata < ndata; ++idata)
        {
            x = outData[idata];
            outData[idata] = ONE / (ONE + exp(-x));
        }
    }

    else if (this->activation == ACTIVATION_TANH)
    {
        #pragma omp parallel for private (idata, x)
        for (idata = 0; idata < ndata; ++idata)
        {
            x = outData[idata];
            outData[idata] = tanh(x);
        }
    }

    else if (this->activation == ACTIVATION_ELU)
    {
        #pragma omp parallel for private (idata, x)
        for (idata = 0; idata < ndata; ++idata)
        {
            x = outData[idata];
            outData[idata] = (x >= ZERO) ? x : (exp(x) - ONE);
        }
    }
}

void NNLayer::deriveActivation(const real* outData, real* outGrad) const
{
    real y;
    int idata;
    int ndata = this->sizeBatch * this->numOutNodes;

    if (this->activation == ACTIVATION_ASIS)
    {
        // NOP
    }

    else if (this->activation == ACTIVATION_SIGMOID)
    {
        #pragma omp parallel for private (idata, y)
        for (idata = 0; idata < ndata; ++idata)
        {
            y = outData[idata];
            outGrad[idata] *= y * (ONE - y);
        }
    }

    else if (this->activation == ACTIVATION_TANH)
    {
        #pragma omp parallel for private (idata, y)
        for (idata = 0; idata < ndata; ++idata)
        {
            y = outData[idata];
            outGrad[idata] *= ONE - y * y;
        }
    }

    else if (this->activation == ACTIVATION_ELU)
    {
        #pragma omp parallel for private (idata, y)
        for (idata = 0; idata < ndata; ++idata)
        {
            y = outData[idata];
            outGrad[idata] *= (y >= ZERO) ? ONE : (y + ONE);
        }
    }
}

Property::Property()
{
    /*
     * set default values
     */
    this->symmFunc       = SYMM_FUNC_BEHLER;

    this->m2             = 100;
    this->m3             = 10;
    this->rinner         = REAL(0.0);  // Angs
    this->router         = REAL(6.0);  // Angs

    this->numRadius      = 0;
    this->numAngle       = 0;
    this->behlerEta1     = NULL;
    this->behlerEta2     = NULL;
    this->behlerRs       = NULL;
    this->behlerZeta     = NULL;

    this->layersEnergy   = 2;
    this->nodesEnergy    = 512;
    this->activEnergy    = ACTIVATION_TANH;

    this->layersCharge   = 2;
    this->nodesCharge    = 512;
    this->activCharge    = ACTIVATION_TANH;

    this->withCharge     = 0;
}

Property::~Property()
{
    if (this->behlerEta1 != NULL)
    {
        delete[] this->behlerEta1;
    }

    if (this->behlerEta2 != NULL)
    {
        delete[] this->behlerEta2;
    }

    if (this->behlerRs != NULL)
    {
        delete[] this->behlerRs;
    }

    if (this->behlerZeta != NULL)
    {
        delete[] this->behlerZeta;
    }
}

void Property::peekProperty(FILE* fp, int rank, MPI_Comm world)
{
    int ierr;

    ierr = 0;
    if (rank == 0 && fp == NULL)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot open ffield file");

    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d", &(this->symmFunc)) == EOF)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at symmFunc");

    MPI_Bcast(&(this->symmFunc), 1, MPI_INT, 0, world);

    if (this->symmFunc == SYMM_FUNC_MANYBODY)
    {
        ierr = 0;
        if (rank == 0 && fscanf(fp, IFORM_D2_F2, &(this->m2), &(this->m3), &(this->rinner), &(this->router)) == EOF)
        {
            ierr = 1;
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at symmFunc parameter");

        MPI_Bcast(&(this->m2), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->m3), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->rinner), 1, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->router), 1, MPI_REAL0, 0, world);
    }

    else if (this->symmFunc == SYMM_FUNC_BEHLER)
    {
        ierr = 0;
        if (rank == 0 && fscanf(fp, IFORM_D2_F1, &(this->numRadius), &(this->numAngle), &(this->router)) == EOF)
        {
            ierr = 1;
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at symmFunc parameter");

        MPI_Bcast(&(this->numRadius), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->numAngle), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->router), 1, MPI_REAL0, 0, world);

        if (this->behlerEta1 != NULL)
        {
            delete[] this->behlerEta1;
        }
        if (this->behlerEta2 != NULL)
        {
            delete[] this->behlerEta2;
        }
        if (this->behlerRs != NULL)
        {
            delete[] this->behlerRs;
        }
        if (this->behlerZeta != NULL)
        {
            delete[] this->behlerZeta;
        }

        if (this->numRadius > 0)
        {
            this->behlerEta1 = new real[this->numRadius];
            this->behlerRs   = new real[this->numRadius];
        }

        if (this->numAngle > 0)
        {
            this->behlerEta2 = new real[this->numAngle];
            this->behlerZeta = new real[this->numAngle];
        }

        ierr = 0;
        for (int i = 0; i < this->numRadius; ++i)
        {
            if (rank == 0 && fscanf(fp, IFORM_F2, &(this->behlerEta1[i]), &(this->behlerRs[i])) == EOF)
            {
                ierr = 1;
                break;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at Behler parameter");

        MPI_Bcast(&(this->behlerEta1[0]), numRadius, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->behlerRs[0]), numRadius, MPI_REAL0, 0, world);

        ierr = 0;
        for (int i = 0; i < this->numAngle; ++i)
        {
            if (rank == 0 && fscanf(fp, IFORM_F2, &(this->behlerEta2[i]), &(this->behlerZeta[i])) == EOF)
            {
                ierr = 1;
                break;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at Behler parameter");

        MPI_Bcast(&(this->behlerEta2[0]), numAngle, MPI_REAL0, 0, world);
        MPI_Bcast(&(this->behlerZeta[0]), numAngle, MPI_REAL0, 0, world);
    }

    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d", &(this->withCharge)) == EOF)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at withCharge");

    MPI_Bcast(&(this->withCharge), 1, MPI_INT, 0, world);

    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d %d %d", &(this->layersEnergy), &(this->nodesEnergy), &(this->activEnergy)) == EOF)
    {
        ierr = 1;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at layersEnergy");

    MPI_Bcast(&(this->layersEnergy), 1, MPI_INT, 0, world);
    MPI_Bcast(&(this->nodesEnergy), 1, MPI_INT, 0, world);
    MPI_Bcast(&(this->activEnergy), 1, MPI_INT, 0, world);

    if (this->withCharge != 0)
    {
        ierr = 0;
        if (rank == 0 && fscanf(fp, "%d %d %d", &(this->layersCharge), &(this->nodesCharge), &(this->activCharge)) == EOF)
        {
            ierr = 1;
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file, at layersCharge");

        MPI_Bcast(&(this->layersCharge), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->nodesCharge), 1, MPI_INT, 0, world);
        MPI_Bcast(&(this->activCharge), 1, MPI_INT, 0, world);
    }
}

SymmFunc::SymmFunc(int numElems)
{
    if (numElems < 1)
    {
        stop_by_error("number of elements is not positive.");
    }

    this->numElems = numElems;
    this->numBasis = 0;
}

SymmFunc::~SymmFunc()
{
    // NOP
}

SymmFuncBehler::SymmFuncBehler(int numElems, int sizeRad, int sizeAng, real radiusCut,
                               const real* radiusEta, const real* radiusShift,
                               const real* angleEta,  const real* angleZeta) : SymmFunc(numElems)
{
    if (sizeRad < 1)
    {
        stop_by_error("size of radius basis is not positive.");
    }

    if (sizeAng < 0)
    {
        stop_by_error("size of angle basis is negative.");
    }

    if (radiusCut <= ZERO)
    {
        stop_by_error("cutoff radius is not positive.");
    }

    if (radiusEta == NULL)
    {
        stop_by_error("radiusEta is null.");
    }

    if (radiusShift == NULL)
    {
        stop_by_error("radiusShift is null.");
    }

    if (sizeAng > 0 && angleEta == NULL)
    {
        stop_by_error("angleEta is null.");
    }

    if (sizeAng > 0 && angleZeta == NULL)
    {
        stop_by_error("angleZeta is null.");
    }

    this->sizeRad = sizeRad;
    this->sizeAng = sizeAng;

    this->numRadBasis = this->sizeRad * this->numElems;
    this->numAngBasis = this->sizeAng * 2 * (this->numElems * (this->numElems + 1) / 2);

    this->numBasis = this->numRadBasis + this->numAngBasis;

    this->radiusCut = radiusCut;

    this->radiusEta   = radiusEta;
    this->radiusShift = radiusShift;

    this->angleEta  = angleEta;
    this->angleZeta = angleZeta;
}

SymmFuncBehler::~SymmFuncBehler()
{
    // NOP
}

void SymmFuncBehler::calculate(int numNeighbor, real** posNeighbor, int* elemNeighbor,
                               real* symmData, real* symmDiff) const
{
    if (posNeighbor == NULL || elemNeighbor == NULL)
    {
        stop_by_error("neighbor is null.");
    }

    if (symmData == NULL)
    {
        stop_by_error("symmData is null.");
    }

    if (symmDiff == NULL)
    {
        stop_by_error("symmDiff is null.");
    }

    // define varialbes
    const int numFree = 3 * (1 + numNeighbor);

    int ineigh1, ineigh2;

    int jelem1, jelem2;
    int ifree1, ifree2;

    int imode;
    int ibase, jbase, kbase;

    real x1, x2;
    real y1, y2;
    real z1, z2;
    real r1, r2, dr, rr;

    real  rs, eta;
    real  zeta, zeta0;
    real* zeta1;

    int  ilambda;
    real lambda;

    real tanh1;
    real tanh2;

    real fc1, fc2;
    real dfc1dr1, dfc2dr2;
    real dfc1dx1, dfc2dx2;
    real dfc1dy1, dfc2dy2;
    real dfc1dz1, dfc2dz2;

    real fc12;
    real dfc12dx1, dfc12dx2;
    real dfc12dy1, dfc12dy2;
    real dfc12dz1, dfc12dz2;

    real gau;
    real dgaudx1, dgaudx2;
    real dgaudy1, dgaudy2;
    real dgaudz1, dgaudz2;

    real psi;
    real dpsidx1, dpsidx2;
    real dpsidy1, dpsidy2;
    real dpsidz1, dpsidz2;

    real chi;
    real chi0;
    real dchidpsi;
    const real chi0_thr = REAL(1.0e-6);

    real g;
    real dgdx1, dgdx2;
    real dgdy1, dgdy2;
    real dgdz1, dgdz2;

    real coef0, coef1, coef2;

    // initialize symmetry functions
    for (ibase = 0; ibase < this->numBasis; ++ibase)
    {
        symmData[ibase] = ZERO;
    }

    for (ifree1 = 0; ifree1 < numFree; ++ifree1)
    {
        for (ibase = 0; ibase < this->numBasis; ++ibase)
        {
            symmDiff[ibase + ifree1 * this->numBasis] = ZERO;
        }
    }

    if (numNeighbor < 1)
    {
        return;
    }

    // radial part
    for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
    {
        ifree1 = 3 * (ineigh1 + 1);
        jelem1 = elemNeighbor[ineigh1];

        jbase = jelem1 * this->sizeRad;

        r1 = posNeighbor[ineigh1][0];
        x1 = posNeighbor[ineigh1][1];
        y1 = posNeighbor[ineigh1][2];
        z1 = posNeighbor[ineigh1][3];

        tanh1   = tanh(ONE - r1 / this->radiusCut);
        tanh2   = tanh1 * tanh1;
        fc1     = tanh1 * tanh2;
        dfc1dr1 = -REAL(3.0) * tanh2 * (ONE - tanh2) / this->radiusCut;
        dfc1dx1 = x1 / r1 * dfc1dr1;
        dfc1dy1 = y1 / r1 * dfc1dr1;
        dfc1dz1 = z1 / r1 * dfc1dr1;

        for (imode = 0; imode < this->sizeRad; ++imode)
        {
            eta = this->radiusEta[imode];
            rs  = this->radiusShift[imode];

            dr  = r1 - rs;
            rr  = dr * dr;

            gau     = exp(-eta * rr);
            coef0   = -REAL(2.0) * eta * dr / r1 * gau;
            dgaudx1 = x1 * coef0;
            dgaudy1 = y1 * coef0;
            dgaudz1 = z1 * coef0;

            g     = gau * fc1;
            dgdx1 = dgaudx1 * fc1 + gau * dfc1dx1;
            dgdy1 = dgaudy1 * fc1 + gau * dfc1dy1;
            dgdz1 = dgaudz1 * fc1 + gau * dfc1dz1;

            ibase = imode + jbase;

            symmData[ibase] += g;

            symmDiff[ibase + 0 * this->numBasis] -= dgdx1;
            symmDiff[ibase + 1 * this->numBasis] -= dgdy1;
            symmDiff[ibase + 2 * this->numBasis] -= dgdz1;

            symmDiff[ibase + (ifree1 + 0) * this->numBasis] += dgdx1;
            symmDiff[ibase + (ifree1 + 1) * this->numBasis] += dgdy1;
            symmDiff[ibase + (ifree1 + 2) * this->numBasis] += dgdz1;
        }
    }

    if (numNeighbor < 2 || this->sizeAng < 1)
    {
        return;
    }

    // angular part
    zeta1 = new real[this->sizeAng];
    for (imode = 0; imode < this->sizeAng; ++imode)
    {
        zeta = this->angleZeta[imode];
        zeta1[imode] = pow(REAL(2.0), ONE - zeta);
    }

    for (ineigh2 = 0; ineigh2 < numNeighbor; ++ineigh2)
    {
        ifree2 = 3 * (ineigh2 + 1);
        jelem2 = elemNeighbor[ineigh2];

        r2 = posNeighbor[ineigh2][0];
        x2 = posNeighbor[ineigh2][1];
        y2 = posNeighbor[ineigh2][2];
        z2 = posNeighbor[ineigh2][3];

        tanh1   = tanh(ONE - r2 / this->radiusCut);
        tanh2   = tanh1 * tanh1;
        fc2     = tanh1 * tanh2;
        dfc2dr2 = -REAL(3.0) * tanh2 * (ONE - tanh2) / this->radiusCut;
        dfc2dx2 = x2 / r2 * dfc2dr2;
        dfc2dy2 = y2 / r2 * dfc2dr2;
        dfc2dz2 = z2 / r2 * dfc2dr2;

        for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
        {
            ifree1 = 3 * (ineigh1 + 1);
            jelem1 = elemNeighbor[ineigh1];

            if (jelem1 > jelem2 || (jelem1 == jelem2 && ineigh1 >= ineigh2))
            {
                continue;
            }

            kbase = (jelem1 + jelem2 * (jelem2 + 1) / 2) * 2 * this->sizeAng;

            r1 = posNeighbor[ineigh1][0];
            x1 = posNeighbor[ineigh1][1];
            y1 = posNeighbor[ineigh1][2];
            z1 = posNeighbor[ineigh1][3];

            rr = r1 * r1 + r2 * r2;

            tanh1   = tanh(ONE - r1 / this->radiusCut);
            tanh2   = tanh1 * tanh1;
            fc1     = tanh1 * tanh2;
            dfc1dr1 = -REAL(3.0) * tanh2 * (ONE - tanh2) / this->radiusCut;
            dfc1dx1 = x1 / r1 * dfc1dr1;
            dfc1dy1 = y1 / r1 * dfc1dr1;
            dfc1dz1 = z1 / r1 * dfc1dr1;

            fc12 = fc1 * fc2;
            dfc12dx1 = dfc1dx1 * fc2;
            dfc12dy1 = dfc1dy1 * fc2;
            dfc12dz1 = dfc1dz1 * fc2;
            dfc12dx2 = fc1 * dfc2dx2;
            dfc12dy2 = fc1 * dfc2dy2;
            dfc12dz2 = fc1 * dfc2dz2;

            psi     = (x1 * x2 + y1 * y2 + z1 * z2) / r1 / r2;
            coef0   = ONE / r1 / r2;
            coef1   = psi / r1 / r1;
            coef2   = psi / r2 / r2;
            dpsidx1 = coef0 * x2 - coef1 * x1;
            dpsidy1 = coef0 * y2 - coef1 * y1;
            dpsidz1 = coef0 * z2 - coef1 * z1;
            dpsidx2 = coef0 * x1 - coef2 * x2;
            dpsidy2 = coef0 * y1 - coef2 * y2;
            dpsidz2 = coef0 * z1 - coef2 * z2;

            for (ilambda = 0; ilambda < 2; ++ilambda)
            {
                lambda = (ilambda == 0) ? ONE : (-ONE);

                chi0 = ONE + lambda * psi;
                if (chi0 < chi0_thr)
                {
                    continue;
                }

                jbase = ilambda * this->sizeAng;

                for (imode = 0; imode < this->sizeAng; ++imode)
                {
                    eta   = this->angleEta[imode];
                    zeta  = this->angleZeta[imode];
                    zeta0 = zeta1[imode];

                    chi      = zeta0 * pow(chi0, zeta);
                    dchidpsi = zeta * lambda * chi / chi0;

                    gau     = exp(-eta * rr);
                    coef0   = -REAL(2.0) * eta * gau;
                    dgaudx1 = x1 * coef0;
                    dgaudy1 = y1 * coef0;
                    dgaudz1 = z1 * coef0;
                    dgaudx2 = x2 * coef0;
                    dgaudy2 = y2 * coef0;
                    dgaudz2 = z2 * coef0;

                    g     = chi * gau * fc12;
                    dgdx1 = dchidpsi * dpsidx1 * gau * fc12 + chi * dgaudx1 * fc12 + chi * gau * dfc12dx1;
                    dgdy1 = dchidpsi * dpsidy1 * gau * fc12 + chi * dgaudy1 * fc12 + chi * gau * dfc12dy1;
                    dgdz1 = dchidpsi * dpsidz1 * gau * fc12 + chi * dgaudz1 * fc12 + chi * gau * dfc12dz1;
                    dgdx2 = dchidpsi * dpsidx2 * gau * fc12 + chi * dgaudx2 * fc12 + chi * gau * dfc12dx2;
                    dgdy2 = dchidpsi * dpsidy2 * gau * fc12 + chi * dgaudy2 * fc12 + chi * gau * dfc12dy2;
                    dgdz2 = dchidpsi * dpsidz2 * gau * fc12 + chi * dgaudz2 * fc12 + chi * gau * dfc12dz2;

                    ibase = this->numRadBasis + imode + jbase + kbase;

                    symmData[ibase] += g;

                    symmDiff[ibase + 0 * this->numBasis] -= dgdx1 + dgdx2;
                    symmDiff[ibase + 1 * this->numBasis] -= dgdy1 + dgdy2;
                    symmDiff[ibase + 2 * this->numBasis] -= dgdz1 + dgdz2;

                    symmDiff[ibase + (ifree1 + 0) * this->numBasis] += dgdx1;
                    symmDiff[ibase + (ifree1 + 1) * this->numBasis] += dgdy1;
                    symmDiff[ibase + (ifree1 + 2) * this->numBasis] += dgdz1;

                    symmDiff[ibase + (ifree2 + 0) * this->numBasis] += dgdx2;
                    symmDiff[ibase + (ifree2 + 1) * this->numBasis] += dgdy2;
                    symmDiff[ibase + (ifree2 + 2) * this->numBasis] += dgdz2;
                }
            }
        }
    }

    delete[] zeta1;
}

SymmFuncManyBody::SymmFuncManyBody(int numElems, int size2Body, int size3Body,
                                   real radiusInner, real radiusOuter) : SymmFunc(numElems)
{
    if (size2Body < 1)
    {
        stop_by_error("size of 2-body is not positive.");
    }

    if (size3Body < 0)
    {
        stop_by_error("size of 3-body is negative.");
    }

    if (radiusInner < ZERO)
    {
        stop_by_error("inner radius is negative.");
    }

    if (radiusOuter <= radiusInner)
    {
        stop_by_error("outer radius is too small.");
    }

    this->size2Body = size2Body;
    this->size3Body = size3Body;

    int totalSize2 = this->numElems * this->size2Body;
    this->num2BodyBasis = totalSize2;

    int totalSize3 = this->numElems * this->size3Body;
    this->num3BodyBasis = totalSize3 * (totalSize3 + 1) / 2 * this->size3Body;

    this->numBasis = this->num2BodyBasis + this->num3BodyBasis;

    this->radiusInner = radiusInner;
    this->radiusOuter = radiusOuter;

    this->step2Body = (this->radiusOuter - this->radiusInner) / ((real) this->size2Body);

    if (this->size3Body > 0)
    {
        this->step3Body = (this->radiusOuter - this->radiusInner) / ((real) this->size3Body);
    }
    else
    {
        this->step3Body = ZERO;
    }
}

SymmFuncManyBody::~SymmFuncManyBody()
{
    // NOP
}

void SymmFuncManyBody::calculate(int numNeighbor, real** posNeighbor, int* elemNeighbor,
                                 real* symmData, real* symmDiff) const
{
    if (posNeighbor == NULL || elemNeighbor == NULL)
    {
        stop_by_error("neighbor is null.");
    }

    if (symmData == NULL)
    {
        stop_by_error("symmData is null.");
    }

    if (symmDiff == NULL)
    {
        stop_by_error("symmDiff is null.");
    }

    // define varialbes
    const int subDim3Body = this->numElems * this->size3Body;
    const int subSize3Body = subDim3Body * (subDim3Body + 1) / 2;

    const int numFree = 3 * (1 + numNeighbor);

    int ineigh1, ineigh2;

    int jelem1, jelem2;
    int ifree1, ifree2;

    int imode1, imode2, imode3;
    int staMode1, staMode2, staMode3;
    int endMode1, endMode2, endMode3;
    int endMode1_;

    int ibase;
    int ibase1, ibase2, ibase3;

    real x1, x2, dx;
    real y1, y2, dy;
    real z1, z2, dz;
    real r1, r2, r3, rr;
    real s1, s2, s3;
    real t1, t2, t3;

    real phi1, phi2, phi3;
    real dphi1dr, dphi2dr, dphi3dr;
    real dphi1dx, dphi2dx, dphi3dx;
    real dphi1dy, dphi2dy, dphi3dy;
    real dphi1dz, dphi2dz, dphi3dz;
    real dphi1dx_, dphi2dx_, dphi3dx_;
    real dphi1dy_, dphi2dy_, dphi3dy_;
    real dphi1dz_, dphi2dz_, dphi3dz_;

    // initialize symmetry functions
    for (ibase = 0; ibase < this->numBasis; ++ibase)
    {
        symmData[ibase] = ZERO;
    }

    for (ifree1 = 0; ifree1 < numFree; ++ifree1)
    {
        for (ibase = 0; ibase < this->numBasis; ++ibase)
        {
            symmDiff[ibase + ifree1 * this->numBasis] = ZERO;
        }
    }

    if (numNeighbor < 1)
    {
        return;
    }

    // 2-body
    for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
    {
        ifree1 = 3 * (ineigh1 + 1);
        jelem1 = elemNeighbor[ineigh1];

        r1 = posNeighbor[ineigh1][0];
        x1 = posNeighbor[ineigh1][1];
        y1 = posNeighbor[ineigh1][2];
        z1 = posNeighbor[ineigh1][3];

        staMode1 = (int) ((r1 - this->radiusInner) / this->step2Body);
        endMode1 = staMode1 + 1;

        staMode1 = max(staMode1, 0);
        endMode1 = min(endMode1, this->size2Body - 1);

        for (imode1 = staMode1; imode1 <= endMode1; ++imode1)
        {
            s1 = this->radiusInner + ((real) imode1) * this->step2Body;
            t1 = (r1 - s1) / this->step2Body;

            phi1 = REAL(0.5) * cos(PI * t1) + REAL(0.5);
            dphi1dr = -REAL(0.5) * PI / this->step2Body * sin(PI * t1);
            dphi1dx = x1 / r1 * dphi1dr;
            dphi1dy = y1 / r1 * dphi1dr;
            dphi1dz = z1 / r1 * dphi1dr;

            ibase = imode1 + jelem1 * this->size2Body;

            symmData[ibase] += phi1;

            symmDiff[ibase + 0 * this->numBasis] += dphi1dx;
            symmDiff[ibase + 1 * this->numBasis] += dphi1dy;
            symmDiff[ibase + 2 * this->numBasis] += dphi1dz;

            symmDiff[ibase + (ifree1 + 0) * this->numBasis] -= dphi1dx;
            symmDiff[ibase + (ifree1 + 1) * this->numBasis] -= dphi1dy;
            symmDiff[ibase + (ifree1 + 2) * this->numBasis] -= dphi1dz;
        }
    }

    if (numNeighbor < 2 || this->size3Body < 1)
    {
        return;
    }

    // 3-body
    for (ineigh2 = 0; ineigh2 < numNeighbor; ++ineigh2)
    {
        ifree2 = 3 * (ineigh2 + 1);
        jelem2 = elemNeighbor[ineigh2];

        r2 = posNeighbor[ineigh2][0];
        x2 = posNeighbor[ineigh2][1];
        y2 = posNeighbor[ineigh2][2];
        z2 = posNeighbor[ineigh2][3];

        staMode2 = (int) ((r2 - this->radiusInner) / this->step3Body);
        endMode2 = staMode2 + 1;

        staMode2 = max(staMode2, 0);
        endMode2 = min(endMode2, this->size3Body - 1);

        for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
        {
            ifree1 = 3 * (ineigh1 + 1);
            jelem1 = elemNeighbor[ineigh1];

            if (jelem1 > jelem2 || (jelem1 == jelem2 && ineigh1 >= ineigh2))
            {
                continue;
            }

            r1 = posNeighbor[ineigh1][0];
            x1 = posNeighbor[ineigh1][1];
            y1 = posNeighbor[ineigh1][2];
            z1 = posNeighbor[ineigh1][3];

            staMode1 = (int) ((r1 - this->radiusInner) / this->step3Body);
            endMode1 = staMode1 + 1;

            staMode1 = max(staMode1, 0);
            endMode1 = min(endMode1, this->size3Body - 1);

            dx = x2 - x1;
            dy = y2 - y1;
            dz = z2 - z1;
            rr = dx * dx + dy * dy + dz * dz;
            r3 = sqrt(rr);

            staMode3 = (int) ((r3 - this->radiusInner) / this->step3Body);
            endMode3 = staMode3 + 1;

            staMode3 = max(staMode3, 0);
            endMode3 = min(endMode3, this->size3Body - 1);

            for (imode3 = staMode3; imode3 <= endMode3; ++imode3)
            {
                s3 = this->radiusInner + ((real) imode3) * this->step3Body;
                t3 = (r3 - s3) / this->step3Body;

                phi3 = REAL(0.5) * cos(PI * t3) + REAL(0.5);
                dphi3dr = -REAL(0.5) * PI / this->step3Body * sin(PI * t3);
                dphi3dx = (x1 - x2) / r3 * dphi3dr;
                dphi3dy = (y1 - y2) / r3 * dphi3dr;
                dphi3dz = (z1 - z2) / r3 * dphi3dr;

                ibase3 = imode3 * subSize3Body;

                for (imode2 = staMode2; imode2 <= endMode2; ++imode2)
                {
                    s2 = this->radiusInner + ((real) imode2) * this->step3Body;
                    t2 = (r2 - s2) / this->step3Body;

                    phi2 = REAL(0.5) * cos(PI * t2) + REAL(0.5);
                    dphi2dr = -REAL(0.5) * PI / this->step3Body * sin(PI * t2);
                    dphi2dx = x2 / r2 * dphi2dr;
                    dphi2dy = y2 / r2 * dphi2dr;
                    dphi2dz = z2 / r2 * dphi2dr;

                    ibase2 = imode2 + jelem2 * this->size3Body;
                    ibase2 = ibase2 * (ibase2 + 1) / 2;

                    endMode1_ = (jelem1 == jelem2) ? min(endMode1, imode2) : endMode1;

                    for (imode1 = staMode1; imode1 <= endMode1_; ++imode1)
                    {
                        s1 = this->radiusInner + ((real) imode1) * this->step3Body;
                        t1 = (r1 - s1) / this->step3Body;

                        phi1 = REAL(0.5) * cos(PI * t1) + REAL(0.5);
                        dphi1dr = -REAL(0.5) * PI / this->step3Body * sin(PI * t1);
                        dphi1dx = x1 / r1 * dphi1dr;
                        dphi1dy = y1 / r1 * dphi1dr;
                        dphi1dz = z1 / r1 * dphi1dr;

                        ibase1 = imode1 + jelem1 * this->size3Body;

                        ibase = this->num2BodyBasis + ibase1 + ibase2 + ibase3;

                        symmData[ibase] += phi1 * phi2 * phi3;

                        dphi1dx_ = dphi1dx * phi2 * phi3;
                        dphi1dy_ = dphi1dy * phi2 * phi3;
                        dphi1dz_ = dphi1dz * phi2 * phi3;

                        dphi2dx_ = phi1 * dphi2dx * phi3;
                        dphi2dy_ = phi1 * dphi2dy * phi3;
                        dphi2dz_ = phi1 * dphi2dz * phi3;

                        dphi3dx_ = phi1 * phi2 * dphi3dx;
                        dphi3dy_ = phi1 * phi2 * dphi3dy;
                        dphi3dz_ = phi1 * phi2 * dphi3dz;

                        symmDiff[ibase + 0 * this->numBasis] += dphi1dx_ + dphi2dx_;
                        symmDiff[ibase + 1 * this->numBasis] += dphi1dy_ + dphi2dy_;
                        symmDiff[ibase + 2 * this->numBasis] += dphi1dz_ + dphi2dz_;

                        symmDiff[ibase + (ifree1 + 0) * this->numBasis] -= dphi1dx_ - dphi3dx_;
                        symmDiff[ibase + (ifree1 + 1) * this->numBasis] -= dphi1dy_ - dphi3dy_;
                        symmDiff[ibase + (ifree1 + 2) * this->numBasis] -= dphi1dz_ - dphi3dz_;

                        symmDiff[ibase + (ifree2 + 0) * this->numBasis] -= dphi2dx_ + dphi3dx_;
                        symmDiff[ibase + (ifree2 + 1) * this->numBasis] -= dphi2dy_ + dphi3dy_;
                        symmDiff[ibase + (ifree2 + 2) * this->numBasis] -= dphi2dz_ + dphi3dz_;
                    }
                }
            }
        }
    }
}

NNArch::NNArch(int mode, int numElems, const Property* property)
{
    if (numElems < 1)
    {
        stop_by_error("#elements is not positive.");
    }

    if (property == NULL)
    {
        stop_by_error("property is null.");
    }

    this->mode = mode;

    this->numElems = numElems;
    this->numAtoms = 0;

    this->property = property;

    this->mbatch = 0;
    this->nbatch = new int[this->numElems];
    this->ibatch = NULL;

    this->mapElem = NULL;

    if (this->isEnergyMode())
    {
        this->energyData = new real*[this->numElems];
        this->energyGrad = new real*[this->numElems];
    }
    else
    {
        this->energyData = NULL;
        this->energyGrad = NULL;
    }

    this->forceData = NULL;

    if (this->isChargeMode())
    {
        this->chargeData = new real*[this->numElems];
    }
    else
    {
        this->chargeData = NULL;
    }

    this->symmData    = NULL;
    this->symmDiff    = NULL;
    this->symmAve     = new real[this->numElems];
    this->symmDev     = new real[this->numElems];
    this->symmFunc    = NULL;

    this->interLayersEnergy = NULL;
    this->lastLayersEnergy  = NULL;

    this->interLayersCharge = NULL;
    this->lastLayersCharge  = NULL;
}

NNArch::~NNArch()
{
    int iatom;
    int natom = this->numAtoms;

    int ineigh;
    int nneigh;

    int ielem;
    int nelem = this->numElems;

    int ilayer;
    int nlayerEnergy = this->property->getLayersEnergy();
    int nlayerCharge = this->property->getLayersCharge();

    delete[] this->nbatch;

    if (this->energyData != NULL)
    {
        delete[] this->energyData;
    }
    if (this->energyGrad != NULL)
    {
        delete[] this->energyGrad;
    }

    if (this->forceData != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            nneigh = this->numNeighbor[iatom] + 1;

            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                delete[] this->forceData[iatom][ineigh];
            }

            delete[] this->forceData[iatom];
        }
    }

    if (this->chargeData != NULL)
    {
        delete[] this->chargeData;
    }

    delete[] this->symmAve;
    delete[] this->symmDev;

    if (this->symmFunc != NULL)
    {
        delete this->symmFunc;
    }

    if (this->interLayersEnergy != NULL)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            for (ilayer = 0; ilayer < nlayerEnergy; ++ilayer)
            {
                delete this->interLayersEnergy[ielem][ilayer];
            }
            delete[] this->interLayersEnergy[ielem];
        }
        delete[] this->interLayersEnergy;
    }

    if (this->lastLayersEnergy != NULL)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            delete this->lastLayersEnergy[ielem];
        }
        delete[] this->lastLayersEnergy;
    }

    if (this->interLayersCharge != NULL)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            for (ilayer = 0; ilayer < nlayerCharge; ++ilayer)
            {
                delete this->interLayersCharge[ielem][ilayer];
            }
            delete[] this->interLayersCharge[ielem];
        }
        delete[] this->interLayersCharge;
    }

    if (this->lastLayersCharge != NULL)
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            delete this->lastLayersCharge[ielem];
        }
        delete[] this->lastLayersCharge;
    }

    delete[] this->indexElem;
    delete[] this->numNeighbor;
}

void NNArch::restoreNN(FILE* fp, int numElement, char** elementNames, int rank, MPI_Comm world)
{
    int  symmFunc = this->property->getSymmFunc();
    int  m2 = this->property->getM2();
    int  m3 = this->property->getM3();
    int  mm2, mm3;
    int  i2;
    int  i3,  ii3;
    int  i3_, ii3_;
    int  j3,  jj3;
    int  j3_, jj3_;
    int  k3;
    int  nrad = this->property->getNumRadius();
    int  nang = this->property->getNumAngle();
    int  irad, iang;
    int  nbase, ibase, jbase;

    int ilayer;
    int nlayerEnergy = property->getLayersEnergy();
    int nlayerCharge = property->getLayersCharge();
    int nnodeEnergy = property->getNodesEnergy();
    int nnodeCharge = property->getNodesCharge();
    int activEnergy = property->getActivEnergy();
    int activCharge = property->getActivCharge();
    int withCharge = property->getWithCharge();

    int ielem,  jelem;
    int kelem,  lelem;
    int kelem_, lelem_;
    int nelemOld;
    int nelemNew = numElement;

    const int lenElemName = 32;
    char** elemNamesOld;
    char** elemNamesNew;

    int* mapElem;

    real* symmAveOld;
    real* symmDevOld;

    int* mapSymmFunc;

    NNLayer* oldLayer;

    int ierr;

    // read number of elements
    ierr = 0;
    if (rank == 0 && fscanf(fp, "%d", &nelemOld) == EOF)
    {
        ierr = 0;
    }

    MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
    if (ierr != 0) stop_by_error("cannot peek ffield file, at nelem");

    MPI_Bcast(&nelemOld, 1, MPI_INT, 0, world);

    // read element's properties
    elemNamesOld = new char*[nelemOld];
    elemNamesNew = new char*[nelemNew];

    symmAveOld = new real[nelemOld];
    symmDevOld = new real[nelemOld];

    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        elemNamesOld[ielem] = new char[lenElemName];
        symmAveOld  [ielem] = ZERO;
        symmDevOld  [ielem] = -ONE;

        ierr = 0;
        if (rank == 0)
        {
            if (fscanf(fp, IFORM_S1_F2, elemNamesOld[ielem], &symmAveOld[ielem], &symmDevOld[ielem]) == EOF)
            {
                ierr = 1;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("cannot peek ffield file.");

        ierr = 0;
        if (rank == 0)
        {
            if (symmDevOld[ielem] <= ZERO)
            {
                ierr = 1;
            }
        }

        MPI_Bcast(&ierr, 1, MPI_INT, 0, world);
        if (ierr != 0) stop_by_error("deviation of symmetry functions is not positive.");

        MPI_Bcast(elemNamesOld[ielem], lenElemName, MPI_CHAR, 0, world);
    }

    MPI_Bcast(&symmAveOld[0], nelemOld, MPI_REAL0, 0, world);
    MPI_Bcast(&symmDevOld[0], nelemOld, MPI_REAL0, 0, world);

    for (ielem = 0; ielem < nelemNew; ++ielem)
    {
        elemNamesNew[ielem] = new char[lenElemName];
        strcpy(elemNamesNew[ielem], elementNames[ielem]);
    }

    // map of elements
    mapElem = new int[nelemOld];

    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        kelem = -1;
        for (jelem = 0; jelem < nelemNew; ++jelem)
        {
            if (strcmp(elemNamesOld[ielem], elemNamesNew[jelem]) == 0)
            {
                kelem = jelem;
                break;
            }
        }

        mapElem[ielem] = kelem;
    }

    // has all elements ?
    for (jelem = 0; jelem < nelemNew; ++jelem)
    {
        bool hasElem = false;

        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            if (jelem == mapElem[ielem])
            {
                hasElem = true;
                break;;
            }
        }

        if (!hasElem)
        {
            char message[256];
            sprintf(message, "ffield file does not have the element: %s", elemNamesNew[jelem]);
            stop_by_error(message);
        }
    }

    // set symmetry function's properties
    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        kelem = mapElem[ielem];
        if (kelem < 0)
        {
            continue;
        }

        symmAve[kelem] = symmAveOld[ielem];
        symmDev[kelem] = symmDevOld[ielem];
    }

    // map of symmetry functions
    if (symmFunc == SYMM_FUNC_MANYBODY)
    {
        mm2 = nelemOld * m2;
        mm3 = nelemOld * m3;

        ibase = 0;
        nbase = mm2 + mm3 * (mm3 + 1) / 2 * m3;

        mapSymmFunc = new int[nbase];

        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            if (kelem < 0)
            {
                for (i2 = 0; i2 < m2; ++i2)
                {
                    mapSymmFunc[ibase] = -1;
                    ibase++;
                }
            }

            else
            {
                jbase = m2 * kelem;

                for (i2 = 0; i2 < m2; ++i2)
                {
                    mapSymmFunc[ibase] = i2 + jbase;
                    ibase++;
                }
            }
        }

        for (k3 = 0; k3 < m3; ++k3)
        {
            jbase = mm2 + k3 * mm3 * (mm3 + 1) / 2;

            for (jj3 = 0; jj3 < mm3; ++jj3)
            {
                j3    = jj3 % m3;
                jelem = jj3 / m3;
                lelem = mapElem[jelem];

                for (ii3 = 0; ii3 <= jj3; ++ii3)
                {
                    i3    = ii3 % m3;
                    ielem = ii3 / m3;
                    kelem = mapElem[ielem];

                    if (lelem < 0 || kelem < 0)
                    {
                        mapSymmFunc[ibase] = -1;
                    }

                    else
                    {
                        lelem_ = max(kelem, lelem);
                        kelem_ = min(kelem, lelem);

                        if (lelem_ == kelem_)
                        {
                            j3_ = max(i3, j3);
                            i3_ = min(i3, j3);
                        }
                        else
                        {
                            j3_ = j3;
                            i3_ = i3;
                        }

                        jj3_ = j3_ + m3 * lelem_;
                        ii3_ = i3_ + m3 * kelem_;

                        mapSymmFunc[ibase] = ii3_ + jj3_ * (jj3_ + 1) / 2 + jbase;
                    }

                    ibase++;
                }
            }
        }
    }

    else //if (symmFunc == SYMM_FUNC_BEHLER)
    {
        ibase = 0;
        nbase = nrad * nelemOld + nang * 2 * nelemOld * (nelemOld + 1) / 2;

        mapSymmFunc = new int[nbase];

        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            if (kelem < 0)
            {
                for (irad = 0; irad < nrad; ++irad)
                {
                    mapSymmFunc[ibase] = -1;
                    ibase++;
                }
            }

            else
            {
                jbase = nrad * kelem;

                for (irad = 0; irad < nrad; ++irad)
                {
                    mapSymmFunc[ibase] = irad + jbase;
                    ibase++;
                }
            }
        }

        for (jelem = 0; jelem < nelemOld; ++jelem)
        {
            lelem = mapElem[jelem];
            for (ielem = 0; ielem <= jelem; ++ielem)
            {
                kelem = mapElem[ielem];

                if (lelem < 0 || kelem < 0)
                {
                    for (iang = 0; iang < (nang * 2); ++iang)
                    {
                        mapSymmFunc[ibase] = -1;
                        ibase++;
                    }
                }

                else
                {
                    lelem_ = max(kelem, lelem);
                    kelem_ = min(kelem, lelem);

                    jbase = nrad * nelemNew + nang * 2 * (kelem_ + lelem_ * (lelem_ + 1) / 2);

                    for (iang = 0; iang < (nang * 2); ++iang)
                    {
                        mapSymmFunc[ibase] = iang + jbase;
                        ibase++;
                    }
                }
            }
        }
    }

    // read NN energy
    if (interLayersEnergy != NULL && lastLayersEnergy != NULL)
    {
        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            // the first layer
            oldLayer = new NNLayer(nbase, nnodeEnergy, activEnergy);
            oldLayer->scanWeight(fp, rank, world);

            if (kelem >= 0)
            {
                interLayersEnergy[kelem][0]->projectWeightFrom(oldLayer, mapSymmFunc);
            }

            delete oldLayer;

            // the 2nd ~ last layers (dummy)
            if (kelem < 0)
            {
                for (ilayer = 1; ilayer < nlayerEnergy; ++ilayer)
                {
                    oldLayer = new NNLayer(nnodeEnergy, nnodeEnergy, activEnergy);
                    oldLayer->scanWeight(fp, rank, world);
                    delete oldLayer;
                }

                oldLayer = new NNLayer(nnodeEnergy, 1, ACTIVATION_ASIS);
                oldLayer->scanWeight(fp, rank, world);
                delete oldLayer;
            }

            // the 2nd ~ last layers (real)
            else
            {
                for (ilayer = 1; ilayer < nlayerEnergy; ++ilayer)
                {
                    interLayersEnergy[kelem][ilayer]->scanWeight(fp, rank, world);
                }

                lastLayersEnergy[kelem]->scanWeight(fp, rank, world);
            }
        }
    }

    // read NN charge
    if (withCharge != 0 && interLayersCharge != NULL && lastLayersCharge != NULL)
    {
        for (ielem = 0; ielem < nelemOld; ++ielem)
        {
            kelem = mapElem[ielem];

            // the first layer
            oldLayer = new NNLayer(nbase, nnodeCharge, activCharge);
            oldLayer->scanWeight(fp, rank, world);

            if (kelem >= 0)
            {
                interLayersCharge[kelem][0]->projectWeightFrom(oldLayer, mapSymmFunc);
            }

            delete oldLayer;

            // the 2nd ~ last layers (dummy)
            if (kelem < 0)
            {
                for (ilayer = 1; ilayer < nlayerCharge; ++ilayer)
                {
                    oldLayer = new NNLayer(nnodeCharge, nnodeCharge, activCharge);
                    oldLayer->scanWeight(fp, rank, world);
                    delete oldLayer;
                }

                oldLayer = new NNLayer(nnodeCharge, 1, ACTIVATION_ASIS);
                oldLayer->scanWeight(fp, rank, world);
                delete oldLayer;
            }

            // the 2nd ~ last layers (real)
            else
            {
                for (ilayer = 1; ilayer < nlayerCharge; ++ilayer)
                {
                    interLayersCharge[kelem][ilayer]->scanWeight(fp, rank, world);
                }

                lastLayersCharge[kelem]->scanWeight(fp, rank, world);
            }
        }
    }

    // release memory
    for (ielem = 0; ielem < nelemOld; ++ielem)
    {
        delete[] elemNamesOld[ielem];
    }

    for (ielem = 0; ielem < nelemNew; ++ielem)
    {
        delete[] elemNamesNew[ielem];
    }

    delete[] elemNamesOld;
    delete[] elemNamesNew;

    delete[] mapElem;

    delete[] symmAveOld;
    delete[] symmDevOld;

    delete[] mapSymmFunc;
}

void NNArch::initGeometry(int inum, int* ilist, int* type, int* typeMap, int* numNeighbor)
{
    this->numAtoms = inum;

    int iatom;
    int natom = inum;

    int ielem;
    int nelem = this->numElems;

    int ineigh;
    int nneigh;

    int ilayer;
    int nlayer;

    int jbatch;

    // allocate memory
    this->indexElem = new int[natom];
    this->numNeighbor = new int[natom];

    // generate indexElem
    for (iatom = 0; iatom < natom; ++iatom)
    {
        this->indexElem[iatom] = typeMap[type[ilist[iatom]]] - 1;
    }

    // count size of batch
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        this->nbatch[ielem] = 0;
    }

    this->ibatch = new int[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->indexElem[iatom];
        this->ibatch[iatom] = this->nbatch[ielem];
        this->nbatch[ielem]++;
    }

    this->mbatch = 0;
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        this->mbatch += this->nbatch[ielem];
    }

    if (this->mbatch < 1)
    {
        return;
    }

    if (this->isEnergyMode())
    {
        nlayer = this->property->getLayersEnergy();

        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->nbatch[ielem] < 1)
            {
                continue;
            }

            jbatch = this->nbatch[ielem];

            this->energyData[ielem] = new real[jbatch];
            this->energyGrad[ielem] = new real[jbatch];

            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersEnergy[ielem][ilayer]->setSizeOfBatch(jbatch);
            }

            this->lastLayersEnergy[ielem]->setSizeOfBatch(jbatch);
        }

        this->forceData = new real**[natom];

        for (iatom = 0; iatom < natom; ++iatom)
        {
            nneigh = numNeighbor[iatom] + 1;

            this->forceData[iatom] = new real*[nneigh];

            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                this->forceData[iatom][ineigh] = new real[3];
            }
        }
    }

    if (this->isChargeMode())
    {
        nlayer = this->property->getLayersCharge();

        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->nbatch[ielem] < 1)
            {
                continue;
            }

            jbatch = this->nbatch[ielem];

            this->chargeData[ielem] = new real[jbatch];

            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersCharge[ielem][ilayer]->setSizeOfBatch(jbatch);
            }

            this->lastLayersCharge[ielem]->setSizeOfBatch(jbatch);
        }
    }
}

void NNArch::clearGeometry()
{
    int iatom;
    int natom = this->numAtoms;

    int ineigh;
    int nneigh;

    int ielem;
    int nelem = this->numElems;

    // release memory
    if (this->ibatch != NULL)
    {
        delete[] this->ibatch;
    }

    if (this->isEnergyMode())
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->nbatch[ielem] < 1)
            {
                continue;
            }

            if (this->energyData[ielem] != NULL)
            {
                delete[] this->energyData[ielem];
            }
            if (this->energyGrad[ielem] != NULL)
            {
                delete[] this->energyGrad[ielem];
            }
        }

        if (this->forceData != NULL)
        {
            for (iatom = 0; iatom < natom; ++iatom)
            {
                nneigh = this->numNeighbor[iatom] + 1;

                for (ineigh = 0; ineigh < nneigh; ++ineigh)
                {
                    delete[] this->forceData[iatom][ineigh];
                }

                delete[] this->forceData[iatom];
            }
        }
    }

    if (this->isChargeMode())
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            if (this->nbatch[ielem] < 1)
            {
                continue;
            }

            if (this->chargeData[ielem] != NULL)
            {
                delete[] this->chargeData[ielem];
            }
        }
    }

    if (this->symmData != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            delete[] this->symmData[iatom];
        }
        delete[] this->symmData;
    }

    if (this->symmDiff != NULL)
    {
        for (iatom = 0; iatom < natom; ++iatom)
        {
            delete[] this->symmDiff[iatom];
        }
        delete[] this->symmDiff;
    }

    // initialize memory
    this->mbatch = 0;
    this->ibatch = NULL;
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        this->nbatch[ielem] = 0;
    }

    if (this->isEnergyMode())
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->energyData[ielem] = NULL;
            this->energyGrad[ielem] = NULL;
        }

        this->forceData = NULL;
    }

    if (this->isChargeMode())
    {
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->chargeData[ielem] = NULL;
        }
    }

    this->symmData = NULL;
    this->symmDiff = NULL;
}

SymmFunc* NNArch::getSymmFunc()
{
    if (this->symmFunc == NULL)
    {
        if (this->numElems < 1)
        {
            stop_by_error("#elements is not positive.");
        }

        if (this->property == NULL)
        {
            stop_by_error("property is null.");
        }

        if (this->property->getSymmFunc() == SYMM_FUNC_MANYBODY)
        {
            int  m2 = this->property->getM2();
            int  m3 = this->property->getM3();

            real rinner = this->property->getRinner();
            real router = this->property->getRouter();

            this->symmFunc = new SymmFuncManyBody(this->numElems, m2, m3, rinner, router);
        }

        else if (this->property->getSymmFunc() == SYMM_FUNC_BEHLER)
        {
            int  nrad = this->property->getNumRadius();
            int  nang = this->property->getNumAngle();
            real rcut = this->property->getRouter();

            const real* eta1 = this->property->getBehlerEta1();
            const real* eta2 = this->property->getBehlerEta2();
            const real* rs   = this->property->getBehlerRs();
            const real* zeta = this->property->getBehlerZeta();

            this->symmFunc = new SymmFuncBehler(this->numElems, nrad, nang, rcut, eta1, rs, eta2, zeta);
        }

        if (this->symmFunc == NULL)
        {
            stop_by_error("cannot create symmFunc.");
        }

        int nbase = this->symmFunc->getNumBasis();
        if (nbase < 1)
        {
            stop_by_error("there are no symmetry functions.");
        }
    }

    return this->symmFunc;
}

void NNArch::calculateSymmFuncs(int* numNeighbor, int** elemNeighbor, real*** posNeighbor)
{
    int iatom;
    int natom = this->numAtoms;

    int nneigh;

    int nbase = this->getSymmFunc()->getNumBasis();

    real** symmData;
    real** symmDiff;

    // allocate memory
    symmData = new real*[natom];
    symmDiff = new real*[natom];

    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh = numNeighbor[iatom] + 1;

        symmData[iatom] = new real[nbase];
        symmDiff[iatom] = new real[nbase * 3 * nneigh];

        this->numNeighbor[iatom] = numNeighbor[iatom];
    }

    // calculate symmetry functions
    #pragma omp parallel for private(iatom)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        this->symmFunc->calculate(numNeighbor[iatom], posNeighbor[iatom], elemNeighbor[iatom],
                                  symmData[iatom], symmDiff[iatom]);
    }

    this->symmData = symmData;
    this->symmDiff = symmDiff;
}

void NNArch::renormalizeSymmFuncs()
{
    int iatom;
    int natom = this->numAtoms;

    int ineigh;
    int nneigh;
    int nneigh3;

    int ielem;
    int nelem = this->numElems;

    int ibase;
    int nbase = this->getSymmFunc()->getNumBasis();

    real ave;
    real dev;

    real** symmData;
    real** symmDiff;

    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->symmDev[ielem] <= ZERO)
        {
            stop_by_error("symmDev have not been prepared.");
        }
    }

    symmData = this->symmData;
    symmDiff = this->symmDiff;

    #pragma omp parallel for private (iatom, ielem, ibase, \
                                      nneigh, nneigh3, ineigh, ave, dev)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->indexElem[iatom];
        nneigh = this->numNeighbor[iatom] + 1;
        nneigh3 = 3 * nneigh;

        ave = this->symmAve[ielem];
        dev = this->symmDev[ielem];

        for (ibase = 0; ibase < nbase; ++ibase)
        {
            symmData[iatom][ibase] -= ave;
            symmData[iatom][ibase] /= dev;
        }

        for (ineigh = 0; ineigh < nneigh3; ++ineigh)
        {
            for (ibase = 0; ibase < nbase; ++ibase)
            {
                symmDiff[iatom][ibase + ineigh * nbase] /= dev;
            }
        }
    }
}

void NNArch::initLayers()
{
    int ielem;
    int nelem = this->numElems;

    int ilayer;
    int nlayer;
    int nnode;
    int activ;

    int nbase = this->getSymmFunc()->getNumBasis();

    for (ielem = 0; ielem < nelem; ++ielem)
    {
        this->symmAve[ielem] = ZERO;
        this->symmDev[ielem] = -ONE;
    }

    if (this->isEnergyMode())
    {
        nlayer = this->property->getLayersEnergy();
        nnode  = this->property->getNodesEnergy();
        activ  = this->property->getActivEnergy();

        this->interLayersEnergy = new NNLayer**[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->interLayersEnergy[ielem] = new NNLayer*[nlayer];
            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersEnergy[ielem][ilayer]
                = new NNLayer(ilayer == 0 ? nbase : nnode, nnode, activ);
            }
        }

        this->lastLayersEnergy = new NNLayer*[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->lastLayersEnergy[ielem]
            = new NNLayer(nnode, 1, ACTIVATION_ASIS);
        }
    }

    if (this->isChargeMode())
    {
        nlayer = this->property->getLayersCharge();
        nnode  = this->property->getNodesCharge();
        activ  = this->property->getActivCharge();

        this->interLayersCharge = new NNLayer**[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->interLayersCharge[ielem] = new NNLayer*[nlayer];
            for (ilayer = 0; ilayer < nlayer; ++ilayer)
            {
                this->interLayersCharge[ielem][ilayer]
                = new NNLayer(ilayer == 0 ? nbase : nnode, nnode, activ);
            }
        }

        this->lastLayersCharge = new NNLayer*[nelem];
        for (ielem = 0; ielem < nelem; ++ielem)
        {
            this->lastLayersCharge[ielem]
            = new NNLayer(nnode, 1, ACTIVATION_ASIS);
        }
    }
}

void NNArch::goForwardOnEnergy()
{
    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;
    int nelem = this->numElems;

    int ibase;
    int nbase = this->getSymmFunc()->getNumBasis();

    int ilayer;
    int nlayer = this->property->getLayersEnergy();

    int jbatch;

    // input symmetry functions to the first layer
    #pragma omp parallel for private(iatom, ielem, jbatch, ibase)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->indexElem[iatom];
        jbatch = this->ibatch[iatom];

        for (ibase = 0; ibase < nbase; ++ibase)
        {
            this->interLayersEnergy[ielem][0]->getData()[ibase + jbatch * nbase]
            = this->symmData[iatom][ibase];
        }
    }

    // propagate through layers
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        for (ilayer = 0; ilayer < nlayer; ++ilayer)
        {
            if (ilayer < (nlayer - 1))
            {
                this->interLayersEnergy[ielem][ilayer]->goForward(
                this->interLayersEnergy[ielem][ilayer + 1]->getData());
            }
            else
            {
                this->interLayersEnergy[ielem][ilayer]->goForward(
                this->lastLayersEnergy[ielem]->getData());
            }
        }

        this->lastLayersEnergy[ielem]->goForward(this->energyData[ielem]);
    }
}

void NNArch::goBackwardOnForce()
{
    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ineigh;
    int mneigh;
    int nneigh;
    int nneigh3;

    int ielem;
    int nelem = this->numElems;

    int ibase;
    int nbase = this->getSymmFunc()->getNumBasis();

    int ilayer;
    int nlayer = this->property->getLayersEnergy();

    int jbatch;

    real*  symmGrad;
    real*  forceNeigh;

    const int  i1 = 1;
    const real a0 = ZERO;
    const real a1 = ONE;

    // derive energies by itselves, to be units
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        for (jbatch = 0; jbatch < this->nbatch[ielem]; ++jbatch)
        {
            this->energyGrad[ielem][jbatch] = ONE;
        }
    }

    // propagate through layers
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        this->lastLayersEnergy[ielem]->goBackward(
                this->energyData[ielem],
                this->energyGrad[ielem], true);

        for (ilayer = (nlayer - 1); ilayer >= 0; --ilayer)
        {
            if (ilayer < (nlayer - 1))
            {
                this->interLayersEnergy[ielem][ilayer]->goBackward(
                this->interLayersEnergy[ielem][ilayer + 1]->getData(),
                this->interLayersEnergy[ielem][ilayer + 1]->getGrad(), true);
            }
            else
            {
                this->interLayersEnergy[ielem][ilayer]->goBackward(
                this->lastLayersEnergy[ielem]->getData(),
                this->lastLayersEnergy[ielem]->getGrad(), true);
            }
        }
    }

    // calculate forces w/ derivatives of symmetry functions
    mneigh = 0;
    #pragma omp parallel for private(iatom, nneigh) reduction(max : mneigh)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh = this->numNeighbor[iatom] + 1;
        mneigh = max(mneigh, nneigh);
    }

    #pragma omp parallel private(iatom, ielem, jbatch, ibase, \
                             nneigh, nneigh3, ineigh, forceNeigh, symmGrad)
    {
        forceNeigh = new real[3 * mneigh];
        symmGrad = new real[nbase];

        #pragma omp for
        for (iatom = 0; iatom < natom; ++iatom)
        {
            ielem = this->indexElem[iatom];
            jbatch = this->ibatch[iatom];

            nneigh = this->numNeighbor[iatom] + 1;
            nneigh3 = 3 * nneigh;

            for (ibase = 0; ibase < nbase; ++ibase)
            {
                symmGrad[ibase] =
                    this->interLayersEnergy[ielem][0]->getGrad()[ibase + jbatch * nbase];
            }

            xgemv_("T", &nbase, &nneigh3,
                   &a1, &(this->symmDiff[iatom][0]), &nbase,
                   &(symmGrad[0]), &i1,
                   &a0, &(forceNeigh[0]), &i1);

            for (ineigh = 0; ineigh < nneigh; ++ineigh)
            {
                this->forceData[iatom][ineigh][0] = -forceNeigh[3 * ineigh + 0];
                this->forceData[iatom][ineigh][1] = -forceNeigh[3 * ineigh + 1];
                this->forceData[iatom][ineigh][2] = -forceNeigh[3 * ineigh + 2];
            }
        }

        delete[] forceNeigh;
        delete[] symmGrad;
    }
}

void NNArch::goForwardOnCharge()
{
    if (!this->isChargeMode())
    {
        stop_by_error("this is not charge-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;
    int nelem = this->numElems;

    int ibase;
    int nbase = this->getSymmFunc()->getNumBasis();

    int ilayer;
    int nlayer = this->property->getLayersCharge();

    int jbatch;

    // input symmetry functions to the first layer
    #pragma omp parallel for private(iatom, ielem, jbatch, ibase)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->indexElem[iatom];
        jbatch = this->ibatch[iatom];

        for (ibase = 0; ibase < nbase; ++ibase)
        {
            this->interLayersCharge[ielem][0]->getData()[ibase + jbatch * nbase] = this->symmData[iatom][ibase];
        }
    }

    // propagate through layers
    for (ielem = 0; ielem < nelem; ++ielem)
    {
        if (this->nbatch[ielem] < 1)
        {
            continue;
        }

        for (ilayer = 0; ilayer < nlayer; ++ilayer)
        {
            if (ilayer < (nlayer - 1))
            {
                this->interLayersCharge[ielem][ilayer]->goForward(
                this->interLayersCharge[ielem][ilayer + 1]->getData());
            }
            else
            {
                this->interLayersCharge[ielem][ilayer]->goForward(
                this->lastLayersCharge[ielem]->getData());
            }
        }

        this->lastLayersCharge[ielem]->goForward(this->chargeData[ielem]);
    }
}

void NNArch::obtainEnergies(real* energies) const
{
    if (energies == NULL)
    {
        stop_by_error("energies is null.");
    }

    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;

    int jbatch;

    #pragma omp parallel for private(iatom, ielem, jbatch)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->indexElem[iatom];
        jbatch = this->ibatch[iatom];

        energies[iatom] = this->energyData[ielem][jbatch];
    }
}

void NNArch::obtainForces(real*** forces) const
{
    if (forces == NULL)
    {
        stop_by_error("forces is null.");
    }

    if (!this->isEnergyMode())
    {
        stop_by_error("this is not energy-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ineigh;
    int nneigh;

    #pragma omp parallel for private (iatom, ineigh, nneigh)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        nneigh = this->numNeighbor[iatom] + 1;

        for (ineigh = 0; ineigh < nneigh; ++ineigh)
        {
            forces[iatom][ineigh][0] = this->forceData[iatom][ineigh][0];
            forces[iatom][ineigh][1] = this->forceData[iatom][ineigh][1];
            forces[iatom][ineigh][2] = this->forceData[iatom][ineigh][2];
        }
    }
}

void NNArch::obtainCharges(real* charges) const
{
    if (charges == NULL)
    {
        stop_by_error("charges is null.");
    }

    if (!this->isChargeMode())
    {
        stop_by_error("this is not charge-mode.");
    }

    int iatom;
    int natom = this->numAtoms;

    int ielem;

    int jbatch;

    #pragma omp parallel for private(iatom, ielem, jbatch)
    for (iatom = 0; iatom < natom; ++iatom)
    {
        ielem = this->indexElem[iatom];
        jbatch = this->ibatch[iatom];

        charges[iatom] = this->chargeData[ielem][jbatch];
    }
}
