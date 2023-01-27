/*
 * Copyright (C) 2022 AdvanceSoft Corporation
 *
 * This source code is licensed under the GNU General Public License Version 2
 * found in the LICENSE file in the root directory of this source tree.
 */

#include "pair_m3gnet.h"

using namespace LAMMPS_NS;

#define GPA_TO_EVA3  160.21766208

PairM3GNet::PairM3GNet(LAMMPS *lmp) : Pair(lmp)
{
    single_enable           = 0;
    restartinfo             = 0;
    one_coeff               = 1;
    manybody_flag           = 1;
    no_virial_fdotr_compute = 1;
    centroidstressflag      = CENTROID_NOTAVAIL;

    this->atomNumMap        = nullptr;
    this->maxinum           = 10;
    this->initializedPython = 0;
    this->cutoff            = 0.0;
    this->npythonPath       = 0;
    this->pythonPaths       = nullptr;
    this->pyModule          = nullptr;
    this->pyFunc            = nullptr;
}

PairM3GNet::~PairM3GNet()
{
    if (copymode)
    {
        return;
    }

    if (this->atomNumMap != nullptr)
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
        memory->destroy(this->stress);
    }

    if (this->pythonPaths != nullptr)
    {
        for (int i = 0; i < this->npythonPath; ++i)
        {
            delete[] this->pythonPaths[i];
        }

        delete[] this->pythonPaths;
    }

    if (this->initializedPython)
    {
        this->finalizePython();
    }
}

void PairM3GNet::allocate()
{
    allocated = 1;

    const int ntypes = atom->ntypes;

    memory->create(cutsq,   ntypes + 1, ntypes + 1,   "pair:cutsq");
    memory->create(setflag, ntypes + 1, ntypes + 1,   "pair:setflag");

    memory->create(this->cell,      3, 3,             "pair:cell");
    memory->create(this->atomNums,  this->maxinum,    "pair:atomNums");
    memory->create(this->positions, this->maxinum, 3, "pair:positions");
    memory->create(this->forces,    this->maxinum, 3, "pair:forces");
    memory->create(this->stress,    6,                "pair:stress");
}

void PairM3GNet::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);

    if (eflag_atom)
    {
        error->all(FLERR, "Pair style M3GNet does not support atomic energy");
    }

    if (vflag_atom)
    {
        error->all(FLERR, "Pair style M3GNet does not support atomic virial pressure");
    }

    this->prepareGNN();

    this->performGNN();
}

void PairM3GNet::prepareGNN()
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
    this->cell[2][1] = domain->h[3]; // yz
    this->cell[2][0] = domain->h[4]; // xz
    this->cell[1][0] = domain->h[5]; // xy
    this->cell[0][1] = 0.0;
    this->cell[0][2] = 0.0;
    this->cell[1][2] = 0.0;

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

void PairM3GNet::performGNN()
{
    int i;
    int iatom;

    double** f = atom->f;

    int  inum  = list->inum;
    int* ilist = list->ilist;

    double volume;
    double factor;
    double evdwl = 0.0;

    // perform Graph Neural Network Potential of M3GNet
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

        f[i][0] += this->forces[iatom][0];
        f[i][1] += this->forces[iatom][1];
        f[i][2] += this->forces[iatom][2];
    }

    // set virial pressure
    if (vflag_global)
    {
        // GPa -> eV/A^3
        volume = domain->xprd * domain->yprd * domain->zprd;
        factor = volume / GPA_TO_EVA3;

        virial[0] -= factor * this->stress[0]; // xx
        virial[1] -= factor * this->stress[1]; // yy
        virial[2] -= factor * this->stress[2]; // zz
        virial[3] -= factor * this->stress[3]; // yz
        virial[4] -= factor * this->stress[4]; // xz
        virial[5] -= factor * this->stress[5]; // xy
    }
}

void PairM3GNet::settings(int narg, char **arg)
{
    if (comm->nprocs > 1)
    {
        error->all(FLERR, "Pair style M3GNet does not support MPI parallelization");
    }

    if (narg < 1)
    {
        return;
    }

    this->npythonPath = narg;
    this->pythonPaths = new char*[this->npythonPath];

    for (int i = 0; i < this->npythonPath; ++i)
    {
        this->pythonPaths[i] = new char[512];
        strcpy(this->pythonPaths[i], arg[i]);
    }
}

void PairM3GNet::coeff(int narg, char **arg)
{
    int i, j;
    int count;

    int ntypes = atom->ntypes;
    int ntypesEff;

    int dftd3 = withDFTD3();

    if (narg != (3 + ntypes) && narg != (4 + ntypes))
    {
        error->all(FLERR, "Incorrect number of arguments for pair_coeff.");
    }

    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    {
        error->all(FLERR, "Only wildcard asterisk is allowed in place of atom types for pair_coeff.");
    }

    if (this->atomNumMap != nullptr)
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
        error->all(FLERR, "There are no elements for pair_coeff of M3GNet.");
    }

    if (!allocated)
    {
        allocate();
    }

    if (this->initializedPython)
    {
        this->finalizePython();
    }

    this->cutoff = this->initializePython(arg[2], dftd3);

    if (this->cutoff <= 0.0)
    {
        error->all(FLERR, "Cutoff is not positive for pair_coeff of M3GNet.");
    }

    count = 0;

    for (i = 1; i <= ntypes; ++i)
    {
        for (j = i; j <= ntypes; ++j)
        {
            if (this->atomNumMap[i] > 0 && this->atomNumMap[j] > 0)
            {
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

double PairM3GNet::init_one(int i, int j)
{
    if (setflag[i][j] == 0)
    {
        error->all(FLERR, "All pair coeffs are not set");
    }

    double r, rr;

    r = this->cutoff;
    rr = r * r;

    cutsq[i][j] = rr;
    cutsq[j][i] = rr;

    return r;
}

void PairM3GNet::init_style()
{
    if (strcmp(update->unit_style, "metal") != 0)
    {
        error->all(FLERR, "Pair style M3GNet requires 'units metal'");
    }

    int* periodicity = domain->periodicity;

    if (!(periodicity[0] && periodicity[1] && periodicity[2]))
    {
        error->all(FLERR, "Pair style M3GNet requires periodic boundary condition");
    }

    neighbor->add_request(this, NeighConst::REQ_FULL);
}

int PairM3GNet::withDFTD3()
{
    return 0;
}

void PairM3GNet::finalizePython()
{
    if (this->initializedPython == 0)
    {
        return;
    }

    Py_XDECREF(this->pyFunc);
    Py_XDECREF(this->pyModule);

    Py_Finalize();
}

double PairM3GNet::initializePython(const char *name, int dftd3)
{
    if (this->initializedPython != 0)
    {
        return this->cutoff;
    }

    double cutoff = -1.0;

    PyObject* pySys    = nullptr;
    PyObject* pyPath   = nullptr;
    PyObject* pyName   = nullptr;
    PyObject* pyModule = nullptr;
    PyObject* pyFunc   = nullptr;
    PyObject* pyArgs   = nullptr;
    PyObject* pyArg1   = nullptr;
    PyObject* pyArg2   = nullptr;
    PyObject* pyValue  = nullptr;

    Py_Initialize();

    pySys  = PyImport_ImportModule("sys");
    pyPath = PyObject_GetAttrString(pySys, "path");

    pyName = PyUnicode_DecodeFSDefault(".");
    if (pyName != nullptr)
    {
        PyList_Append(pyPath, pyName);
        Py_DECREF(pyName);
    }

    if (this->pythonPaths != nullptr)
    {
        for (int i = 0; i < this->npythonPath; ++i)
        {
            pyName = PyUnicode_DecodeFSDefault(this->pythonPaths[i]);
            if (pyName != nullptr)
            {
                PyList_Append(pyPath, pyName);
                Py_DECREF(pyName);
            }
        }
    }

    pyName = PyUnicode_DecodeFSDefault("m3gnet_driver");
    if (pyName != nullptr)
    {
        pyModule = PyImport_Import(pyName);
        Py_DECREF(pyName);
    }

    if (pyModule != nullptr)
    {
        pyFunc = PyObject_GetAttrString(pyModule, "m3gnet_initialize");

        if (pyFunc != nullptr && PyCallable_Check(pyFunc))
        {
            pyArg1 = PyUnicode_FromString(name);
            pyArg2 = PyBool_FromLong(dftd3);

            pyArgs = PyTuple_New(2);
            PyTuple_SetItem(pyArgs, 0, pyArg1);
            PyTuple_SetItem(pyArgs, 1, pyArg2);

            pyValue = PyObject_CallObject(pyFunc, pyArgs);

            Py_DECREF(pyArgs);

            if (pyValue != nullptr && PyFloat_Check(pyValue))
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

        pyFunc = PyObject_GetAttrString(pyModule, "m3gnet_get_energy_forces_stress");

        if (pyFunc != nullptr && PyCallable_Check(pyFunc))
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

        error->all(FLERR, "Cannot initialize python for pair_coeff of M3GNet.");
    }

    this->pyModule = pyModule;
    this->pyFunc   = pyFunc;

    return cutoff;
}

double PairM3GNet::calculatePython()
{
    int i;
    int iatom;
    int natom = list->inum;

    double energy = 0.0;
    int hasEnergy = 0;
    int hasForces = 0;
    int hasStress = 0;

    PyObject* pyFunc  = this->pyFunc;
    PyObject* pyArgs  = nullptr;
    PyObject* pyArg1  = nullptr;
    PyObject* pyArg2  = nullptr;
    PyObject* pyArg3  = nullptr;
    PyObject* pyAsub  = nullptr;
    PyObject* pyValue = nullptr;
    PyObject* pyVal1  = nullptr;
    PyObject* pyVal2  = nullptr;
    PyObject* pyVal3  = nullptr;
    PyObject* pyVsub  = nullptr;
    PyObject* pyVobj  = nullptr;

    // set cell -> pyArgs1
    pyArg1 = PyList_New(3);

    for (i = 0; i < 3; ++i)
    {
        pyAsub = PyList_New(3);
        PyList_SetItem(pyAsub, 0, PyFloat_FromDouble(this->cell[i][0]));
        PyList_SetItem(pyAsub, 1, PyFloat_FromDouble(this->cell[i][1]));
        PyList_SetItem(pyAsub, 2, PyFloat_FromDouble(this->cell[i][2]));
        PyList_SetItem(pyArg1, i, pyAsub);
    }

    // set atomNums -> pyArgs2
    pyArg2 = PyList_New(natom);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        PyList_SetItem(pyArg2, iatom, PyLong_FromLong(this->atomNums[iatom]));
    }

    // set positions -> pyArgs3
    pyArg3 = PyList_New(natom);

    for (iatom = 0; iatom < natom; ++iatom)
    {
        pyAsub = PyList_New(3);
        PyList_SetItem(pyAsub, 0, PyFloat_FromDouble(this->positions[iatom][0]));
        PyList_SetItem(pyAsub, 1, PyFloat_FromDouble(this->positions[iatom][1]));
        PyList_SetItem(pyAsub, 2, PyFloat_FromDouble(this->positions[iatom][2]));
        PyList_SetItem(pyArg3, iatom, pyAsub);
    }

    // call function
    pyArgs = PyTuple_New(3);
    PyTuple_SetItem(pyArgs, 0, pyArg1);
    PyTuple_SetItem(pyArgs, 1, pyArg2);
    PyTuple_SetItem(pyArgs, 2, pyArg3);

    pyValue = PyObject_CallObject(pyFunc, pyArgs);

    Py_DECREF(pyArgs);

    if (pyValue != nullptr && PyTuple_Check(pyValue) && PyTuple_Size(pyValue) >= 3)
    {
        // get energy <- pyValue
        pyVal1 = PyTuple_GetItem(pyValue, 0);
        if (pyVal1 != nullptr && PyFloat_Check(pyVal1))
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
        if (pyVal2 != nullptr && PyList_Check(pyVal2) && PyList_Size(pyVal2) >= natom)
        {
            hasForces = 1;

            for (iatom = 0; iatom < natom; ++iatom)
            {
                pyVsub = PyList_GetItem(pyVal2, iatom);
                if (pyVsub != nullptr && PyList_Check(pyVsub) && PyList_Size(pyVsub) >= 3)
                {
                    for (i = 0; i < 3; ++i)
                    {
                        pyVobj = PyList_GetItem(pyVsub, i);
                        if (pyVobj != nullptr && PyFloat_Check(pyVobj))
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

        // get stress <- pyValue
        pyVal3 = PyTuple_GetItem(pyValue, 2);
        if (pyVal3 != nullptr && PyList_Check(pyVal3) && PyList_Size(pyVal3) >= 6)
        {
            hasStress = 1;

            for (i = 0; i < 6; ++i)
            {
                pyVobj = PyList_GetItem(pyVal3, i);
                if (pyVobj != nullptr && PyFloat_Check(pyVobj))
                {
                    this->stress[i] = PyFloat_AsDouble(pyVobj);
                }
                else
                {
                    if (PyErr_Occurred()) PyErr_Print();
                    hasStress = 0;
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

    if (hasEnergy == 0 || hasForces == 0 || hasStress == 0)
    {
        error->all(FLERR, "Cannot calculate energy, forces and stress by python of M3GNet.");
    }

    return energy;
}

static const int NUM_ELEMENTS = 118;

static const char* ALL_ELEMENTS[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Rm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};

int PairM3GNet::elementToAtomNum(const char *elem)
{
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

void PairM3GNet::toRealElement(char *elem)
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

