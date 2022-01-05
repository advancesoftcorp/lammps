/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_oc20.h"

using namespace LAMMPS_NS;

PairOC20::PairOC20(LAMMPS *lmp) : Pair(lmp)
{
    this->atomNumMap        = NULL;
    this->maxinum           = 10;
    this->initializedPython = 0;
    this->cutoff            = 0.0;
    this->pyModule          = NULL;
    this->pyFunc            = NULL;
}

PairOC20::~PairOC20()
{
    if (copymode)
    {
        return;
    }

    if (this->atomNumMap != NULL)
    {
        delete[] this->atomNumMap;
    }

    if (allocated)
    {
        memory->destroy(cutsq);
        memory->destroy(setflag);
        memory->destroy(this->cell);
        memory->destroy(this->atomNums);
        memory->destroy(this->positions);
        memory->destroy(this->forces);
    }

    if (this->initializedPython)
    {
        this->finalizePython();
    }
}

void PairOC20::allocate()
{
    allocated = 1;

    const int ntypes = atom->ntypes;

    memory->create(cutsq,   ntypes + 1, ntypes + 1,   "pair:cutsq");
    memory->create(setflag, ntypes + 1, ntypes + 1,   "pair:setflag");

    memory->create(this->cell,      3, 3,             "pair:cell");
    memory->create(this->atomNums,  this->maxinum,    "pair:atomNums");
    memory->create(this->positions, this->maxinum, 3, "pair:positions");
    memory->create(this->forces,    this->maxinum, 3, "pair:forces");
}

void PairOC20::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);

    if (eflag_atom)
    {
        error->all(FLERR, "Pair style OC20 does not support atomic energy");
    }

    if (vflag)
    {
        error->all(FLERR, "Pair style OC20 does currently not support virial pressure");
    }

    if (vflag_atom)
    {
        error->all(FLERR, "Pair style OC20 does not support atomic virial pressure");
    }

    if (vflag_global)
    {
        if (vflag_fdotr == 0)
        {
            error->all(FLERR, "Pair style OC20 does not support explicit virial pressure");
        }
    }

    this->prepareGNN();

    this->performGNN();

    if (vflag_fdotr)
    {
        virial_fdotr_compute();
    }
}

void PairOC20::prepareGNN()
{
    int i;
    int iatom;

    int*  type = atom->type;
    double** x = atom->x;

    int  inum  = list->inum;
    int* ilist = list->ilist;

    double* boxlo = domain->boxlo;

    // grow with inum and nneigh
    if (inum > this->maxinum)
    {
        this->maxinum = inum + this->maxinum / 2;

        memory->grow(this->atomNums,  this->maxinum,    "pair:atomNums");
        memory->grow(this->positions, this->maxinum, 3, "pair:positions");
        memory->grow(this->forces,    this->maxinum, 3, "pair:forces");
    }

    // set cell
    this->cell[0][0] = domain->h[0]; // xx
    this->cell[1][1] = domain->h[1]; // yy
    this->cell[2][2] = domain->h[2]; // zz
    this->cell[1][2] = domain->h[3]; // yz
    this->cell[0][2] = domain->h[4]; // xz
    this->cell[0][1] = domain->h[5]; // xy
    this->cell[1][0] = this->cell[0][1];
    this->cell[2][0] = this->cell[0][2];
    this->cell[2][1] = this->cell[1][2];

    // set atomNums and positions
    #pragma omp parallel for private(iatom, i)
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        this->atomNums[iatom] = this->atomNumMap[type[i]];

        this->positions[iatom][0] = x[i][0] - boxlo[0];
        this->positions[iatom][1] = x[i][1] - boxlo[1];
        this->positions[iatom][2] = x[i][2] - boxlo[2];
    }
}

void PairOC20::performGNN()
{
    int i;
    int iatom;

    double** f = atom->f;

    int  inum  = list->inum;
    int* ilist = list->ilist;

    double evdwl = 0.0;

    // perform Graph Neural Network Potential of OC20
    evdwl = this->calculatePython();

    // set total energy
    if (eflag_global)
    {
        eng_vdwl += evdwl;
    }

    // set atomic forces
    for (iatom = 0; iatom < inum; ++iatom)
    {
        i = ilist[iatom];

        f[i][0] += forces[iatom][0];
        f[i][1] += forces[iatom][1];
        f[i][2] += forces[iatom][2];
    }
}

void PairOC20::settings(int narg, char **arg)
{
    if (narg != 0)
    {
        error->all(FLERR, "pair_style oc20 command has unnecessary argument(s).");
    }
}

void PairOC20::coeff(int narg, char **arg)
{
    int i, j;
    int count;
    double r, rr;

    int ntypes = atom->ntypes;
    int ntypesEff;

    int gpu = withGPU();

    if (narg != (3 + ntypes))
    {
        error->all(FLERR, "Incorrect number of arguments for pair_coeff.");
    }

    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    {
        error->all(FLERR, "Only wildcard asterisk is allowed in place of atom types for pair_coeff.");
    }

    if (this->atomNumMap != NULL)
    {
        delete this->atomNumMap;
    }

    this->atomNumMap = new int[ntypes + 1];

    ntypesEff = 0;
    for (i = 0; i < ntypes; ++i)
    {
        if (strcmp(arg[i + 3], "NULL") == 0)
        {
            this->atomNumMap[i + 1] = 0;
        }
        else
        {
            this->atomNumMap[i + 1] = this->elementToAtomNum(arg[i + 3]);
            ntypesEff++;
        }
    }

    if (ntypesEff < 1)
    {
        error->all(FLERR, "There are no elements for pair_coeff of OC20.");
    }

    if (!allocated)
    {
        allocate();
    }

    if (this->initializedPython)
    {
        this->finalizePython();
    }

    this->cutoff = this->initializePython(arg[2], gpu);

    if (this->cutoff <= 0.0)
    {
        error->all(FLERR, "Cutoff is not positive for pair_coeff of OC20.");
    }

    count = 0;
    r = this->cutoff;
    rr = r * r;

    for (i = 1; i <= ntypes; ++i)
    {
        for (j = i; j <= ntypes; ++j)
        {
            if (this->atomNumMap[i] > 0 && this->atomNumMap[j] > 0)
            {
                cutsq[i][j] = rr;
                setflag[i][j] = 1;
                count++;
            }
            else
            {
                setflag[i][j] = 0;
            }
        }
    }

    if (count == 0)
    {
        error->all(FLERR, "Incorrect args for pair coefficients");
    }
}

double PairOC20::init_one(int i, int j)
{
    if (setflag[i][j] == 0)
    {
        error->all(FLERR, "All pair coeffs are not set");
    }

    return this->cutoff;
}

void PairOC20::init_style()
{
    if (strcmp(update->unit_style, "metal") != 0)
    {
        error->all(FLERR, "Pair style OC20 requires 'units metal'");
    }

    int* periodicity = domain->periodicity;

    if (!(periodicity[0] && periodicity[1] && periodicity[2]))
    {
        error->all(FLERR, "Pair style OC20 requires periodic boundary condition");
    }

    if (comm->nprocs > 1)
    {
        error->all(FLERR, "Pair style OC20 does not support MPI parallelization");
    }

    neighbor->request(this, instance_me);
}

int PairOC20::withGPU()
{
    return 0;
}

void PairOC20::finalizePython()
{
    if (this->initializedPython == 0)
    {
        return;
    }

    Py_XDECREF(this->pyFunc);
    Py_XDECREF(this->pyModule);

    Py_Finalize();
}

double PairOC20::initializePython(const char *name, int gpu)
{
    if (this->initializedPython != 0)
    {
        return this->cutoff;
    }

    double cutoff = -1.0;

    PyObject* pySys    = NULL;
    PyObject* pyPath   = NULL;
    PyObject* pyName   = NULL;
    PyObject* pyModule = NULL;
    PyObject* pyFunc   = NULL;
    PyObject* pyArgs   = NULL;
    PyObject* pyArg1   = NULL;
    PyObject* pyArg2   = NULL;
    PyObject* pyValue  = NULL;

    Py_Initialize();

    pySys  = PyImport_ImportModule("sys");
    pyPath = PyObject_GetAttrString(pySys, "path");
    PyList_Append(pyPath, PyUnicode_DecodeFSDefault("."));

    pyName = PyUnicode_DecodeFSDefault("oc20_driver");

    if (pyName != NULL)
    {
        pyModule = PyImport_Import(pyName);
        Py_DECREF(pyName);
    }

    if (pyModule != NULL)
    {
        pyFunc = PyObject_GetAttrString(pyModule, "oc20_initialize");

        if (pyFunc != NULL && PyCallable_Check(pyFunc))
        {
            pyArg1 = PyUnicode_FromString(name);
            pyArg2 = PyBool_FromLong(gpu);

            pyArgs = PyTuple_New(2);
            PyTuple_SetItem(pyArgs, 0, pyArg1);
            PyTuple_SetItem(pyArgs, 1, pyArg2);

            pyValue = PyObject_CallObject(pyFunc, pyArgs);

            Py_DECREF(pyArgs);

            if (pyValue != NULL && PyFloat_Check(pyValue))
            {
                this->initializedPython = 1;
                cutoff = PyFloat_AsDouble(pyValue);
            }
            else
            {
                if (PyErr_Occurred()) PyErr_Print();
            }

            Py_XDECREF(pyValue);
        }

        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }

        Py_XDECREF(pyFunc);

        pyFunc = PyObject_GetAttrString(pyModule, "oc20_get_energy_and_forces");

        if (pyFunc != NULL && PyCallable_Check(pyFunc))
        {
            // NOP
        }
        else
        {
            this->initializedPython = 0;
            if (PyErr_Occurred()) PyErr_Print();
        }

        //Py_XDECREF(pyFunc);
        //Py_DECREF(pyModule);
    }

    else
    {
        if (PyErr_Occurred()) PyErr_Print();
    }

    if (this->initializedPython == 0)
    {
        Py_XDECREF(pyFunc);
        Py_XDECREF(pyModule);

        Py_Finalize();

        error->all(FLERR, "Cannot initialize python for pair_coeff of OC20.");
    }

    this->pyModule = pyModule;
    this->pyFunc   = pyFunc;

    return cutoff;
}

double PairOC20::calculatePython()
{
    int i;
    int iatom;
    int natom = list->inum;

    double energy = 0.0;
    int hasEnergy = 0;
    int hasForces = 0;

    PyObject* pyFunc  = this->pyFunc;
    PyObject* pyArgs  = NULL;
    PyObject* pyArg1  = NULL;
    PyObject* pyArg2  = NULL;
    PyObject* pyArg3  = NULL;
    PyObject* pyAsub  = NULL;
    PyObject* pyValue = NULL;
    PyObject* pyVal1  = NULL;
    PyObject* pyVal2  = NULL;
    PyObject* pyVsub  = NULL;
    PyObject* pyVobj  = NULL;

    // set cell -> pyArgs1
    pyArg1 = PyList_New(3);

    for (i = 0; i < 3; ++i)
    {
        pyAsub = PyList_New(3);
        PyList_SetItem(pyAsub, 0, PyFloat_FromDouble(this->cell[i][0]));
        PyList_SetItem(pyAsub, 1, PyFloat_FromDouble(this->cell[i][1]));
        PyList_SetItem(pyAsub, 2, PyFloat_FromDouble(this->cell[i][2]));
        PyList_SetItem(pyArg1, i, pyAsub);
        Py_DECREF(pyAsub);
    }

    Py_DECREF(pyArg1);

    // set atomNums -> pyArgs2
    pyArg2 = PyList_New(natom);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        PyList_SetItem(pyArg2, iatom, PyLong_FromLong(this->atomNums[iatom]));
    }

    Py_DECREF(pyArg2);

    // set positions -> pyArgs3
    pyArg3 = PyList_New(natom);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        pyAsub = PyList_New(3);
        PyList_SetItem(pyAsub, 0, PyFloat_FromDouble(this->positions[iatom][0]));
        PyList_SetItem(pyAsub, 1, PyFloat_FromDouble(this->positions[iatom][1]));
        PyList_SetItem(pyAsub, 2, PyFloat_FromDouble(this->positions[iatom][2]));
        PyList_SetItem(pyArg3, iatom, pyAsub);
        Py_DECREF(pyAsub);
    }

    Py_DECREF(pyArg3);

    // call function
    pyArgs = PyTuple_New(3);
    PyTuple_SetItem(pyArgs, 0, pyArg1);
    PyTuple_SetItem(pyArgs, 1, pyArg2);
    PyTuple_SetItem(pyArgs, 2, pyArg3);

    pyValue = PyObject_CallObject(pyFunc, pyArgs);

    Py_DECREF(pyArgs);

    if (pyValue != NULL && PyTuple_Check(pyValue) && PyTuple_Size(pyValue) >= 2)
    {
        // get energy <- pyValue
        pyVal1 = PyTuple_GetItem(pyValue, 0);
        if (pyVal1 != NULL && PyFloat_Check(pyVal1))
        {
            hasEnergy = 1;
            energy = PyFloat_AsDouble(pyVal1);
        }
        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }

        // get forces <- pyValue
        pyVal2 = PyTuple_GetItem(pyValue, 1);
        if (pyVal2 != NULL && PyList_Check(pyVal2) && PyList_Size(pyVal2) >= natom)
        {
            hasForces = 1;

            for (iatom = 0; iatom < natom; ++iatom)
            {
                pyVsub = PyList_GetItem(pyVal2, iatom);
                if (pyVsub != NULL && PyList_Check(pyVsub) && PyList_Size(pyVsub) >= 3)
                {
                    for (i = 0; i < 3; ++i)
                    {
                        pyVobj = PyList_GetItem(pyVsub, i);
                        if (pyVobj != NULL && PyFloat_Check(pyVobj))
                        {
                            this->forces[iatom][i] = PyFloat_AsDouble(pyVobj);
                        }
                        else
                        {
                            if (PyErr_Occurred()) PyErr_Print();
                            hasForces = 0;
                            break;
                        }
                    }
                }
                else
                {
                    if (PyErr_Occurred()) PyErr_Print();
                    hasForces = 0;
                    break;
                }

                if (hasForces == 0)
                {
                    break;
                }
            }
        }
        else
        {
            if (PyErr_Occurred()) PyErr_Print();
        }
    }

    else
    {
        if (PyErr_Occurred()) PyErr_Print();
    }

    Py_XDECREF(pyValue);

    if (hasEnergy == 0 || hasForces == 0)
    {
        error->all(FLERR, "Cannot calculate energy and forces by python of OC20.");
    }

    return energy;
}

const int NUM_ELEMENTS = 118;

const char* ALL_ELEMENTS[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Rm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};

int PairOC20::elementToAtomNum(const char *elem)
{
    int n = strlen(elem);
    n = n > 2 ? 2 : n;

    int m = n;
    char elem1[16];

    strcpy(elem1, elem);

    this->toRealElement(elem1);

    if (strlen(elem1) > 0)
    {
        for (int i = 0; i < NUM_ELEMENTS; ++i)
        {
            if (strcasecmp(elem1, ALL_ELEMENTS[i]) == 0)
            {
                return (i + 1);
            }
        }
    }

    char estr[256];
    sprintf(estr, "Incorrect name of element: %s", elem);
    error->all(FLERR, estr);

    return 0;
}

void PairOC20::toRealElement(char *elem)
{
    int n = strlen(elem);
    n = n > 2 ? 2 : n;

    int m = n;

    for (int i = 0; i < n; ++i)
    {
        char c = elem[i];
        if (c == '0' || c == '1' || c == '2' || c == '3' || c == '4' ||
            c == '5' || c == '6' || c == '7' || c == '8' || c == '9' || c == ' ' ||
            c == '_' || c == '-' || c == '+' || c == '*' || c == '~' || c == ':' || c == '#')
        {
            m = i;
            break;
        }

        elem[i] = c;
    }

    elem[m] = '\0';
}

