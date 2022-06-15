/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifndef NNP_COMMON_H_
#define NNP_COMMON_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cmath>
using namespace std;

#ifdef _SINGLE

#define nnpreal     float
#define NNPREAL(x)  x##f

#define IFORM_F1        "%f"
#define IFORM_F2        "%f %f"
#define IFORM_F3        "%f %f %f"
#define IFORM_S1_F2     "%s %f %f"
#define IFORM_S1_F2_D1  "%s %f %f %d"
#define IFORM_S2_F4     "%s %s %f %f %f %f"
#define IFORM_D2_F1     "%d %d %f"
#define IFORM_D2_F2     "%d %d %f %f"
#define IFORM_D2_F2_D1  "%d %d %f %f %d"

#define xgemv_  sgemv_
#define xgemm_  sgemm_

#define MPI_NNPREAL  MPI_FLOAT

#else

#define nnpreal     double
#define NNPREAL(x)  x

#define IFORM_F1        "%lf"
#define IFORM_F2        "%lf %lf"
#define IFORM_F3        "%lf %lf %lf"
#define IFORM_S1_F2     "%s %lf %lf"
#define IFORM_S1_F2_D1  "%s %lf %lf %d"
#define IFORM_S2_F4     "%s %s %lf %lf %lf %lf"
#define IFORM_D2_F1     "%d %d %lf"
#define IFORM_D2_F2     "%d %d %lf %lf"
#define IFORM_D2_F2_D1  "%d %d %lf %lf %d"

#define xgemm_  dgemm_
#define xgemv_  dgemv_

#define MPI_NNPREAL  MPI_DOUBLE

#endif

#define ZERO    NNPREAL(0.0)
#define ONE     NNPREAL(1.0)
#define PI      NNPREAL(3.14159265358979324)
#define PId     3.14159265358979324
#define ROOT2   NNPREAL(1.41421356237309505)
#define ROOTPI  NNPREAL(1.77245385090551603)

extern "C"
{

int xgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
           const nnpreal* alpha, nnpreal* a, const int* lda, nnpreal* b, const int* ldb,
           const nnpreal* beta, nnpreal* c, const int* ldc);

int xgemv_(const char* trans, const int* m, const int* n,
           const nnpreal* alpha, nnpreal* a, const int* lda, nnpreal* x, const int* incx,
           const nnpreal* beta, nnpreal* y, const int* incy);

}

#define SYMM_FUNC_NULL       0
#define SYMM_FUNC_MANYBODY   1
#define SYMM_FUNC_BEHLER     2
#define SYMM_FUNC_CHEBYSHEV  3

#define ACTIVATION_NULL      0
#define ACTIVATION_ASIS      1
#define ACTIVATION_SIGMOID   2
#define ACTIVATION_TANH      3
#define ACTIVATION_ELU       4
#define ACTIVATION_TWTANH    5
#define ACTIVATION_GELU      6

// SymmFunc
#define CUTOFF_MODE_NULL     0
#define CUTOFF_MODE_SINGLE   1
#define CUTOFF_MODE_DOUBLE   2
#define CUTOFF_MODE_IPSO     3

// NNArch
#define NNARCH_MODE_BOTH     0
#define NNARCH_MODE_ENERGY   1
#define NNARCH_MODE_CHARGE   2

inline void stop_by_error(const char* message)
{
    printf("[STOP] %s\n", message);
    fflush(stdout);

    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();

    exit(1);
}

#endif /* NNP_COMMON_H_ */
