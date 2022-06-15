/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#ifdef PAIR_CLASS

PairStyle(oc20, PairOC20)

#else

#ifndef LMP_PAIR_OC20_H_
#define LMP_PAIR_OC20_H_

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

class PairOC20: public Pair
{
public:
    PairOC20(class LAMMPS*);

    virtual ~PairOC20() override;

    void compute(int, int) override;

    void settings(int, char **) override;

    void coeff(int, char **) override;

    double init_one(int, int) override;

    void init_style() override;

protected:
    virtual int withGPU();

private:
    int*      atomNumMap;
    int*      atomNums;
    double**  cell;
    double**  positions;
    double**  forces;

    int       maxinum;
    int       initializedPython;
    int       virialWarning;
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

#endif /* LMP_PAIR_OC20_H_ */
#endif
