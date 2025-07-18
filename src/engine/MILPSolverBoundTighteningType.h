/*********************                                                        */
/*! \file MILPSolverBoundTighteningType.h
** \verbatim
** Top contributors (to current version):
**   Guy Katz
** This file is part of the Marabou project.
** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
** in the top-level source directory) and their institutional affiliations.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**
** [[ Add lengthier description here ]]

**/

#ifndef __MILPSolverBoundTighteningType_h__
#define __MILPSolverBoundTighteningType_h__

/*
  MILP solver bound tightening options
*/
enum class MILPSolverBoundTighteningType {
    // Only encode pure linear constraints in the underlying
    // solver, in a way that over-approximates the query
    LP_RELAXATION = 0,
    LP_RELAXATION_INCREMENTAL = 1,
    // Encode linear and integer constraints in the underlying
    // solver, in a way that completely captures the query but is
    // more expensive to solve
    MILP_ENCODING = 2,
    MILP_ENCODING_INCREMENTAL = 3,
    // Encode full queries and tries to fix relus until fix point
    ITERATIVE_PROPAGATION = 4,
    // Perform backward analysis
    BACKWARD_ANALYSIS_ONCE = 5,
    BACKWARD_ANALYSIS_CONVERGE = 6,
    // Perform backward analysis using the PreimageApproximation Algorithm (arXiv:2305.03686v4
    // [cs.SE])
    BACKWARD_ANALYSIS_PREIMAGE_APPROX = 7,
    // Perform backward analysis using INVPROP (arXiv:2302.01404v4 [cs.LG])
    BACKWARD_ANALYSIS_INVPROP = 8,
    // Perform backward analysis using PMNR with random neuron selection.
    BACKWARD_ANALYSIS_PMNR_RANDOM = 9,
    // Perform backward analysis using PMNR with maximum gradient neuron selection (arXiv:1804.10829
    // [cs.AI]).
    BACKWARD_ANALYSIS_PMNR_GRADIENT = 10,
    // Perform backward analysis using PMNR with BBPS-based neuron selection (arXiv:2405.21063v3
    // [cs.LG]).
    BACKWARD_ANALYSIS_PMNR_BBPS = 11,
    // Option to have no MILP bound tightening performed
    NONE = 20,
};

#endif // __MILPSolverBoundTighteningType_h__
