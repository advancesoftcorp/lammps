/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func_manybody.h"

SymmFuncManyBody::SymmFuncManyBody(int numElems, bool elemWeight, int size2Body, int size3Body,
                                   nnpreal radiusInner, nnpreal radiusOuter) : SymmFunc(numElems, false, elemWeight)
{
    if (this->elemWeight)
    {
        stop_by_error("many-body does not support elemWeight.");
    }

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

    this->step2Body = (this->radiusOuter - this->radiusInner) / ((nnpreal) this->size2Body);

    if (this->size3Body > 0)
    {
        this->step3Body = (this->radiusOuter - this->radiusInner) / ((nnpreal) this->size3Body);
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

void SymmFuncManyBody::calculate(int numNeighbor, int* elemNeighbor, nnpreal** posNeighbor,
                                 nnpreal* symmData, nnpreal* symmDiff)
{
    if (elemNeighbor == nullptr || posNeighbor == nullptr)
    {
        stop_by_error("neighbor is null.");
    }

    if (symmData == nullptr)
    {
        stop_by_error("symmData is null.");
    }

    if (symmDiff == nullptr)
    {
        stop_by_error("symmDiff is null.");
    }

    // define varialbes
    const int subDim3Body = this->numElems * this->size3Body;
    const int subSize3Body = subDim3Body * (subDim3Body + 1) / 2;

    const int numFree = 3 * numNeighbor;

    int ineigh1, ineigh2;

    int jelem1, jelem2;
    int ifree1, ifree2;

    int imode1, imode2, imode3;
    int staMode1, staMode2, staMode3;
    int endMode1, endMode2, endMode3;
    int endMode1_;

    int ibase;
    int ibase1, ibase2, ibase3;

    nnpreal x1, x2, dx;
    nnpreal y1, y2, dy;
    nnpreal z1, z2, dz;
    nnpreal r1, r2, r3, rr;
    nnpreal s1, s2, s3;
    nnpreal t1, t2, t3;

    nnpreal phi1, phi2, phi3;
    nnpreal dphi1dr, dphi2dr, dphi3dr;
    nnpreal dphi1dx, dphi2dx, dphi3dx;
    nnpreal dphi1dy, dphi2dy, dphi3dy;
    nnpreal dphi1dz, dphi2dz, dphi3dz;
    nnpreal dphi1dx_, dphi2dx_, dphi3dx_;
    nnpreal dphi1dy_, dphi2dy_, dphi3dy_;
    nnpreal dphi1dz_, dphi2dz_, dphi3dz_;

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
        ifree1 = 3 * ineigh1;
        jelem1 = elemNeighbor[ineigh1];

        r1 = posNeighbor[ineigh1][0];
        if (r1 >= this->radiusOuter)
        {
            continue;
        }

        x1 = posNeighbor[ineigh1][1];
        y1 = posNeighbor[ineigh1][2];
        z1 = posNeighbor[ineigh1][3];

        staMode1 = (int) ((r1 - this->radiusInner) / this->step2Body);
        endMode1 = staMode1 + 1;

        staMode1 = max(staMode1, 0);
        endMode1 = min(endMode1, this->size2Body - 1);

        for (imode1 = staMode1; imode1 <= endMode1; ++imode1)
        {
            s1 = this->radiusInner + ((nnpreal) imode1) * this->step2Body;
            t1 = (r1 - s1) / this->step2Body;

            phi1 = NNPREAL(0.5) * cos(PI * t1) + NNPREAL(0.5);
            dphi1dr = -NNPREAL(0.5) * PI / this->step2Body * sin(PI * t1);
            dphi1dx = -x1 / r1 * dphi1dr;
            dphi1dy = -y1 / r1 * dphi1dr;
            dphi1dz = -z1 / r1 * dphi1dr;

            ibase = imode1 + jelem1 * this->size2Body;

            symmData[ibase] += phi1;

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
        ifree2 = 3 * ineigh2;
        jelem2 = elemNeighbor[ineigh2];

        r2 = posNeighbor[ineigh2][0];
        if (r2 >= this->radiusOuter)
        {
            continue;
        }

        x2 = posNeighbor[ineigh2][1];
        y2 = posNeighbor[ineigh2][2];
        z2 = posNeighbor[ineigh2][3];

        staMode2 = (int) ((r2 - this->radiusInner) / this->step3Body);
        endMode2 = staMode2 + 1;

        staMode2 = max(staMode2, 0);
        endMode2 = min(endMode2, this->size3Body - 1);

        for (ineigh1 = 0; ineigh1 < numNeighbor; ++ineigh1)
        {
            ifree1 = 3 * ineigh1;
            jelem1 = elemNeighbor[ineigh1];

            if (jelem1 > jelem2 || (jelem1 == jelem2 && ineigh1 >= ineigh2))
            {
                continue;
            }

            r1 = posNeighbor[ineigh1][0];
            if (r1 >= this->radiusOuter)
            {
                continue;
            }

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

            if (rr >= this->radiusOuter * this->radiusOuter)
            {
                continue;
            }

            r3 = sqrt(rr);

            staMode3 = (int) ((r3 - this->radiusInner) / this->step3Body);
            endMode3 = staMode3 + 1;

            staMode3 = max(staMode3, 0);
            endMode3 = min(endMode3, this->size3Body - 1);

            for (imode3 = staMode3; imode3 <= endMode3; ++imode3)
            {
                s3 = this->radiusInner + ((nnpreal) imode3) * this->step3Body;
                t3 = (r3 - s3) / this->step3Body;

                phi3 = NNPREAL(0.5) * cos(PI * t3) + NNPREAL(0.5);
                dphi3dr = -NNPREAL(0.5) * PI / this->step3Body * sin(PI * t3);
                dphi3dx = (x1 - x2) / r3 * dphi3dr;
                dphi3dy = (y1 - y2) / r3 * dphi3dr;
                dphi3dz = (z1 - z2) / r3 * dphi3dr;

                ibase3 = imode3 * subSize3Body;

                for (imode2 = staMode2; imode2 <= endMode2; ++imode2)
                {
                    s2 = this->radiusInner + ((nnpreal) imode2) * this->step3Body;
                    t2 = (r2 - s2) / this->step3Body;

                    phi2 = NNPREAL(0.5) * cos(PI * t2) + NNPREAL(0.5);
                    dphi2dr = -NNPREAL(0.5) * PI / this->step3Body * sin(PI * t2);
                    dphi2dx = -x2 / r2 * dphi2dr;
                    dphi2dy = -y2 / r2 * dphi2dr;
                    dphi2dz = -z2 / r2 * dphi2dr;

                    ibase2 = imode2 + jelem2 * this->size3Body;
                    ibase2 = ibase2 * (ibase2 + 1) / 2;

                    endMode1_ = (jelem1 == jelem2) ? min(endMode1, imode2) : endMode1;

                    for (imode1 = staMode1; imode1 <= endMode1_; ++imode1)
                    {
                        s1 = this->radiusInner + ((nnpreal) imode1) * this->step3Body;
                        t1 = (r1 - s1) / this->step3Body;

                        phi1 = NNPREAL(0.5) * cos(PI * t1) + NNPREAL(0.5);
                        dphi1dr = -NNPREAL(0.5) * PI / this->step3Body * sin(PI * t1);
                        dphi1dx = -x1 / r1 * dphi1dr;
                        dphi1dy = -y1 / r1 * dphi1dr;
                        dphi1dz = -z1 / r1 * dphi1dr;

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
