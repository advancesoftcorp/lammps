/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "dump_nnp.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include <cfloat>

using namespace LAMMPS_NS;

#define ONELINE 128
#define DELTA 1048576

#define BOHR    0.52917720859
#define RYDBERG 13.605691930242388
#define BOLTZ   8.617343e-5

#define FOR_SANNP 0

const char *DumpNNP::elemname[] = {
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Rm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu"
};

const double DumpNNP::elemmass[] = {
      1.00794,   4.00260,
      6.94100,   9.01218,  10.81100,  12.01070,  14.00674,  15.99940,  18.99840,  20.17970,
     22.98977,  24.30500,  26.98154,  28.08550,  30.97376,  32.06600,  35.45270,  39.94800,
     39.09830,  40.07800,  44.95591,  47.86700,  50.94150,  51.99610,  54.93805,  55.84500,  58.93320,  58.69340,  63.54600,  65.39000,  69.72300,  72.61000,  74.92160,  78.96000,  79.90400,  83.80000,
     85.46780,  87.62000,  88.90585,  91.22400,  92.90638,  95.94000,  98.00000, 101.07000, 102.90550, 106.42000, 107.86820, 112.41100, 114.81800, 118.71000, 121.76000, 127.60000, 126.90447, 131.29000,
    132.90545, 137.32700,
    138.90550, 140.11600, 140.90765, 144.24000, 145.00000, 150.36000, 151.96400, 157.25000, 158.92534, 162.50000, 164.93032, 167.26000, 168.93421, 173.04000, 174.96700,
    178.49000, 180.94790, 183.84000, 186.20700, 190.23000, 192.21700, 195.07800, 196.96655, 200.59000, 204.38330, 207.20000, 208.98038, 209.00000, 210.00000, 222.00000,
    223.00000, 226.00000,
    227.00000, 232.03810, 231.03588, 238.02890, 237.00000, 244.00000
};

const int DumpNNP::nelem = sizeof(elemmass) / sizeof(double);

DumpNNP::DumpNNP(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg), typenames(nullptr)
{
    if (narg != 5) error->all(FLERR, "Illegal dump nnp command");
    if (binary || multiproc) error->all(FLERR, "Invalid dump nnp filename");

    sort_flag = 1;
    sortcol = 0;
    size_one = 14;

    x2ryd = 1.0 / (BOHR * force->angstrom);
    e2ryd = BOLTZ / (RYDBERG * force->boltz);
    f2ryd = e2ryd / x2ryd;

    nevery = utils::inumeric(FLERR, arg[3], false, lmp);
    if (nevery <= 0) error->all(FLERR, "Illegal dump custom command");

    delete[] format_default;

    format_default = utils::strdup(
    "  %-3s %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E %19.12E");

    ntypes = atom->ntypes;
    typenames = nullptr;

    pe = nullptr;
}

DumpNNP::~DumpNNP()
{
    delete[] format_default;
    format_default = nullptr;

    if (typenames)
    {
        for (int i = 1; i <= ntypes; i++)
            delete [] typenames[i];

        delete [] typenames;
        typenames = nullptr;
    }
}

void DumpNNP::init_style()
{
    format = utils::strdup(fmt::format("{}\n", format_default));

    pe = modify->get_compute_by_id("thermo_pe");

    if (multifile == 0) openfile();
}

int DumpNNP::modify_param(int narg, char **arg)
{
    if (strcmp(arg[0], "element") == 0)
    {
        if (narg < ntypes + 1)
        {
            error->all(FLERR, "Dump modify element names do not match atom types");
        }

        if (typenames)
        {
            for (int i = 1; i <= ntypes; i++)
                delete [] typenames[i];

            delete [] typenames;
            typenames = nullptr;
        }

        typenames = new char*[ntypes + 1];
        for (int itype = 1; itype <= ntypes; itype++)
        {
            typenames[itype] = utils::strdup(arg[itype]);
        }

        return ntypes + 1;
    }

    return 0;
}

void DumpNNP::write_header(bigint n)
{
    if (!typenames)
    {
        typenames = new char*[ntypes + 1];

        for (int itype = 1; itype <= ntypes; itype++)
        {
            double mass = atom->mass_setflag[itype] ? atom->mass[itype] : 0.0;
            typenames[itype] = utils::strdup(detectElementByMass(mass));
        }
    }

    if (me == 0)
    {
        std::string header = fmt::format("{:8}{:8}     {:19.12E}\n", n, FOR_SANNP, pe->scalar);

        double xdim = (domain->boxhi[0] - domain->boxlo[0]) * x2ryd;
        double ydim = (domain->boxhi[1] - domain->boxlo[1]) * x2ryd;
        double zdim = (domain->boxhi[2] - domain->boxlo[2]) * x2ryd;

        double xy = 0.0;
        double xz = 0.0;
        double yz = 0.0;

        if (domain->triclinic)
        {
            xy = domain->xy * x2ryd;
            xz = domain->xz * x2ryd;
            yz = domain->yz * x2ryd;
        }

        header += fmt::format(" {:19.12E} {:19.12E} {:19.12E}\n", xdim, 0.0,  0.0 );
        header += fmt::format(" {:19.12E} {:19.12E} {:19.12E}\n", xy,   ydim, 0.0 );
        header += fmt::format(" {:19.12E} {:19.12E} {:19.12E}\n", xz,   yz,   zdim);

        fmt::print(fp, header);
    }
}

int DumpNNP::count()
{
    if (update->whichflag == 0)
    {
        if (pe->invoked_peratom != update->ntimestep)
        {
            error->all(FLERR, "Compute used in dump between runs is not current");
        }
    }
    else
    {
        if (pe->invoked_peratom != update->ntimestep)
        {
            pe->compute_scalar();
        }
    }

    pe->addstep(update->ntimestep + nevery);

    return Dump::count();
}

void DumpNNP::pack(tagint *ids)
{
    int m, n;

    tagint *tag = atom->tag;
    int *type = atom->type;
    int *mask = atom->mask;
    double **x = atom->x;
    double **f = atom->f;
    int nlocal = atom->nlocal;

    m = 0;
    n = 0;
    for (int i = 0; i < nlocal; i++)
    {
        if (mask[i] & groupbit)
        {
            buf[m++] = tag[i];
            buf[m++] = type[i];
            buf[m++] = x[i][0] * x2ryd;
            buf[m++] = x[i][1] * x2ryd;
            buf[m++] = x[i][2] * x2ryd;
            buf[m++] = 0.0; // atomic energy
            buf[m++] = f[i][0] * f2ryd;
            buf[m++] = f[i][1] * f2ryd;
            buf[m++] = f[i][2] * f2ryd;
            buf[m++] = 0.0; // atomic charge
            buf[m++] = 0.0; // coulomb energy
            buf[m++] = 0.0; // coulomb force x
            buf[m++] = 0.0; // coulomb force y
            buf[m++] = 0.0; // coulomb force z
            if (ids) ids[n++] = tag[i];
        }
    }
}

int DumpNNP::convert_string(int n, double *mybuf)
{
    int offset = 0;
    int m = 0;

    for (int i = 0; i < n; i++)
    {
        if (offset + ONELINE > maxsbuf)
        {
            if ((bigint) maxsbuf + DELTA > MAXSMALLINT) return -1;
            maxsbuf += DELTA;
            memory->grow(sbuf, maxsbuf, "dump:sbuf");
        }

        offset += sprintf(&sbuf[offset], format,
                          typenames[static_cast<int> (mybuf[m +  1])],
                          mybuf[m +  2], mybuf[m +  3], mybuf[m +  4],
                          mybuf[m +  5],
                          mybuf[m +  6], mybuf[m +  7], mybuf[m +  8],
                          mybuf[m +  9], mybuf[m + 10],
                          mybuf[m + 11], mybuf[m + 12], mybuf[m + 13]);

        offset += sprintf(&sbuf[offset],"\n");

        m += size_one;
    }

    return offset;
}

void DumpNNP::write_data(int n, double *mybuf)
{
    if (buffer_flag == 1)
    {
        if (mybuf) fwrite(mybuf, sizeof(char), n, fp);
    }
    else
    {
        int m = 0;

        for (int i = 0; i < n; i++)
        {
            fprintf(fp, format,
                    typenames[static_cast<int> (mybuf[m +  1])],
                    mybuf[m +  2], mybuf[m +  3], mybuf[m +  4],
                    mybuf[m +  5],
                    mybuf[m +  6], mybuf[m +  7], mybuf[m +  8],
                    mybuf[m +  9], mybuf[m + 10],
                    mybuf[m + 11], mybuf[m + 12], mybuf[m + 13]);

            m += size_one;
        }
    }
}

const char *DumpNNP::detectElementByMass(double mass)
{
    if (mass <= 0.0)
    {
        return "X";
    }

    int ielemMin = -1;
    double dmassMin = DBL_MAX;

    for (int ielem = 0; ielem < nelem; ielem++)
    {
        double dmass = abs(elemmass[ielem] - mass);
        if (dmass < dmassMin)
        {
            ielemMin = ielem;
            dmassMin = dmass;
        }
    }

    if (dmassMin > 5.0 || ielemMin == -1)
    {
        return "X";
    }

    return elemname[ielemMin];
}
