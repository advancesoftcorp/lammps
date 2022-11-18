/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef PAIR_CLASS

PairStyle(m3gnet, PairM3GNet)

#else

#ifndef LMP_PAIR_M3GNET_H_
#define LMP_PAIR_M3GNET_H_

#include <Python.h>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "pair.h"
#include "force.h"
#include "update.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "domain.h"

namespace LAMMPS_NS
{

class PairM3GNet: public Pair
{
public:
	PairM3GNet(class LAMMPS*);

    virtual ~PairM3GNet() override;

    void compute(int, int) override;

    void settings(int, char **) override;

    void coeff(int, char **) override;

    double init_one(int, int) override;

    void init_style() override;

private:
    int*      atomNumMap;
    int*      atomNums;
    double**  cell;
    double**  positions;
    double**  forces;
    double**  stress;

    int       maxinum;
    int       initializedPython;
    double    cutoff;

    int       npythonPath;
    char**    pythonPaths;

    PyObject* pyModule;
    PyObject* pyFunc;

    void allocate();

    void prepareGNN();

    void performGNN();

    void finalizePython();

    double initializePython(const char *name, int gpu);

    double calculatePython();

    int elementToAtomNum(const char *elem);

    void toRealElement(char *elem);
};

}  // namespace LAMMPS_NS

#endif /* LMP_PAIR_M3GNET_H_ */
#endif
