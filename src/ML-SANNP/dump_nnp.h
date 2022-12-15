/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(nnp,DumpNNP);
// clang-format on
#else

#ifndef LMP_DUMP_NNP_H
#define LMP_DUMP_NNP_H

#include "dump.h"

namespace LAMMPS_NS
{

class DumpNNP : public Dump
{
public:
    DumpNNP(class LAMMPS *, int, char **);
    ~DumpNNP() override;

protected:
    void init_style() override;
    void write_header(bigint) override;
    int count() override;
    void pack(tagint *) override;
    int convert_string(int, double *) override;
    void write_data(int, double *) override;

private:
    double x2ryd, e2ryd, f2ryd, q2ryd;
    int nevery;
    class Compute *pe;
    class Compute *peatom;
};

} // namespace LAMMPS_NS

#endif
#endif
