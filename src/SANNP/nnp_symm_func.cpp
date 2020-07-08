/*
 * Copyright (C) 2020 AdvanceSoft Corporation
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 */

#include "nnp_symm_func.h"

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
