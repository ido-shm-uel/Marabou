/*********************                                                        */
/*! \file GlobalConfiguration.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Parth Shah, Derek Huang
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "GlobalConfiguration.h"

#include "DivideStrategy.h"
#include "MString.h"

#include <cstdio>


// The exponential moving average is calculated as
//     ema = current * alpha + previous * (1 - alpha)
const double GlobalConfiguration::EXPONENTIAL_MOVING_AVERAGE_ALPHA = 0.5;

// Whether to use SoI instead of Reluplex for local search for satisfying assignments
// to non-linear constraint.
bool GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH = true;

const double GlobalConfiguration::SCORE_BUMP_FOR_PL_CONSTRAINTS_NOT_IN_SOI = 5;

// Use the polarity metrics to decide which branch to take first in a case split
// and how to repair a ReLU constraint.
const bool GlobalConfiguration::USE_POLARITY_BASED_DIRECTION_HEURISTICS = true;

const double GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS = 0.0000000001;
const unsigned GlobalConfiguration::DEFAULT_DOUBLE_TO_STRING_PRECISION = 10;
const unsigned GlobalConfiguration::STATISTICS_PRINTING_FREQUENCY = 10000;
const unsigned GlobalConfiguration::STATISTICS_PRINTING_FREQUENCY_GUROBI = 100;
const double GlobalConfiguration::BOUND_COMPARISON_ADDITIVE_TOLERANCE = 0.0000001;
const double GlobalConfiguration::BOUND_COMPARISON_MULTIPLICATIVE_TOLERANCE = 0.001 * 0.0000001;
const double GlobalConfiguration::PIVOT_CHANGE_COLUMN_TOLERANCE = 0.000000001;
const double GlobalConfiguration::PIVOT_ROW_AND_COLUMN_TOLERANCE = 0.01;
const double GlobalConfiguration::ENTRY_ELIGIBILITY_TOLERANCE = 0.00000001;
const double GlobalConfiguration::RATIO_CONSTRAINT_ADDITIVE_TOLERANCE = 0.0000001 * 0.3;
const double GlobalConfiguration::RATIO_CONSTRAINT_MULTIPLICATIVE_TOLERANCE =
    0.001 * 0.0000001 * 0.3;
const double GlobalConfiguration::HARRIS_RATIO_CONSTRAINT_ADDITIVE_TOLERANCE = 0.0000001 * 0.5;
const double GlobalConfiguration::HARRIS_RATIO_CONSTRAINT_MULTIPLICATIVE_TOLERANCE =
    0.001 * 0.0000001 * 0.5;
const double GlobalConfiguration::BASIC_COSTS_ADDITIVE_TOLERANCE = 0.0000001;
const double GlobalConfiguration::BASIC_COSTS_MULTIPLICATIVE_TOLERANCE = 0.001 * 0.0000001;
const double GlobalConfiguration::SPARSE_FORREST_TOMLIN_DIAGONAL_ELEMENT_TOLERANCE = 0.00001;
const unsigned GlobalConfiguration::DEGRADATION_CHECKING_FREQUENCY = 100;
const double GlobalConfiguration::DEGRADATION_THRESHOLD = 0.1;
const double GlobalConfiguration::ACCEPTABLE_SIMPLEX_PIVOT_THRESHOLD = 0.0001;
const bool GlobalConfiguration::USE_COLUMN_MERGING_EQUATIONS = false;
const double GlobalConfiguration::GAUSSIAN_ELIMINATION_PIVOT_SCALE_THRESHOLD = 0.1;
const unsigned GlobalConfiguration::MAX_SIMPLEX_PIVOT_SEARCH_ITERATIONS = 5;
const DivideStrategy GlobalConfiguration::SPLITTING_HEURISTICS = DivideStrategy::ReLUViolation;
const unsigned GlobalConfiguration::INTERVAL_SPLITTING_FREQUENCY = 10;
const unsigned GlobalConfiguration::INTERVAL_SPLITTING_THRESHOLD = 10;
const unsigned GlobalConfiguration::BOUND_TIGHTING_ON_CONSTRAINT_MATRIX_FREQUENCY = 100;
const unsigned GlobalConfiguration::ROW_BOUND_TIGHTENER_SATURATION_ITERATIONS = 20;
const double GlobalConfiguration::COST_FUNCTION_ERROR_THRESHOLD = 0.0000000001;

const unsigned GlobalConfiguration::SIMULATION_RANDOM_SEED = 1;

const unsigned GlobalConfiguration::VOLUME_ESTIMATION_RANDOM_SEED = 1;
const unsigned GlobalConfiguration::VOLUME_ESTIMATION_ITERATIONS = 25000;
const unsigned GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_RANDOM_SEED = 1;
const unsigned GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_MAX_ITERATIONS = 25;
const double GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_STEP_SIZE = 0.025;
const double GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_LEARNING_RATE = 0.25;
const double GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_WEIGHT_DECAY = 0;

const unsigned GlobalConfiguration::INVPROP_MAX_ITERATIONS = 25000;
const double GlobalConfiguration::INVPROP_STEP_SIZE = 0.0025;
const double GlobalConfiguration::INVPROP_LEARNING_RATE = 0.025;
const double GlobalConfiguration::INVPROP_WEIGHT_DECAY = 0;
const double GlobalConfiguration::INVPROP_INITIAL_ALPHA = 0.5;
const double GlobalConfiguration::INVPROP_INITIAL_GAMMA = 0.025;

const unsigned GlobalConfiguration::PMNR_RANDOM_SEED = 1;
const unsigned GlobalConfiguration::PMNR_MAX_ITERATIONS = 100;
const unsigned GlobalConfiguration::PMNR_SELECTED_NEURONS = 3;

const bool GlobalConfiguration::USE_HARRIS_RATIO_TEST = true;

const double GlobalConfiguration::SYMBOLIC_TIGHTENING_ROUNDING_CONSTANT = 0.00000000001;
const double GlobalConfiguration::LP_TIGHTENING_ROUNDING_CONSTANT = 0.00000001;

const double GlobalConfiguration::SIGMOID_CUTOFF_CONSTANT = 20;

const bool GlobalConfiguration::PREPROCESS_INPUT_QUERY = true;
const bool GlobalConfiguration::PREPROCESSOR_ELIMINATE_VARIABLES = true;
const bool GlobalConfiguration::PL_CONSTRAINTS_ADD_AUX_EQUATIONS_AFTER_PREPROCESSING = true;
const bool GlobalConfiguration::NL_CONSTRAINTS_ADD_AUX_EQUATIONS_AFTER_PREPROCESSING = true;
const double GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD = 0.00001;

const unsigned GlobalConfiguration::PREPROCESSSING_MAX_TIGHTEING_ROUND = 1000;

const bool GlobalConfiguration::WARM_START = false;

const unsigned GlobalConfiguration::MAX_ITERATIONS_WITHOUT_PROGRESS = 10000;

const unsigned GlobalConfiguration::PSE_ITERATIONS_BEFORE_RESET = 1000;
const double GlobalConfiguration::PSE_GAMMA_ERROR_THRESHOLD = 0.001;
const double GlobalConfiguration::PSE_GAMMA_UPDATE_TOLERANCE = 0.000000001;

const double GlobalConfiguration::CONSTRAINT_COMPARISON_TOLERANCE = 0.00001;

const double GlobalConfiguration::SOFTMAX_LSE2_THRESHOLD = 0.6;

const bool GlobalConfiguration::ONLY_AUX_INITIAL_BASIS = false;

const GlobalConfiguration::ExplicitBasisBoundTighteningType
    GlobalConfiguration::EXPLICIT_BASIS_BOUND_TIGHTENING_TYPE =
        GlobalConfiguration::COMPUTE_INVERTED_BASIS_MATRIX;
const bool GlobalConfiguration::EXPLICIT_BOUND_TIGHTENING_UNTIL_SATURATION = false;
const double GlobalConfiguration::EXPLICIT_BASIS_BOUND_TIGHTENING_ROUNDING_CONSTANT = 1e-6;

const unsigned GlobalConfiguration::REFACTORIZATION_THRESHOLD = 100;
const GlobalConfiguration::BasisFactorizationType GlobalConfiguration::BASIS_FACTORIZATION_TYPE =
    GlobalConfiguration::SPARSE_FORREST_TOMLIN_FACTORIZATION;

const unsigned GlobalConfiguration::BABSR_CANDIDATES_THRESHOLD = 5;
const unsigned GlobalConfiguration::POLARITY_CANDIDATES_THRESHOLD = 5;

const unsigned GlobalConfiguration::DNC_DEPTH_THRESHOLD = 5;

const double GlobalConfiguration::MINIMAL_COEFFICIENT_FOR_TIGHTENING = 0.01;
const double GlobalConfiguration::LEMMA_CERTIFICATION_TOLERANCE = 0.000001;
const bool GlobalConfiguration::WRITE_JSON_PROOF = false;

const unsigned GlobalConfiguration::BACKWARD_BOUND_PROPAGATION_DEPTH = 3;
const unsigned GlobalConfiguration::MAX_ROUNDS_OF_BACKWARD_ANALYSIS = 10;
const unsigned GlobalConfiguration::MAX_ROUNDS_OF_PMNR_BACKWARD_ANALYSIS = 10;

#ifdef ENABLE_GUROBI
const unsigned GlobalConfiguration::GUROBI_NUMBER_OF_THREADS = 1;
const bool GlobalConfiguration::GUROBI_LOGGING = false;
#endif // ENABLE_GUROBI

// Logging - note that it is enabled only in Debug mode
const bool GlobalConfiguration::DNC_MANAGER_LOGGING = false;
const bool GlobalConfiguration::ENGINE_LOGGING = false;
const bool GlobalConfiguration::TABLEAU_LOGGING = false;
const bool GlobalConfiguration::SMT_CORE_LOGGING = false;
const bool GlobalConfiguration::DANTZIGS_RULE_LOGGING = false;
const bool GlobalConfiguration::BASIS_FACTORIZATION_LOGGING = false;
const bool GlobalConfiguration::PREPROCESSOR_LOGGING = false;
const bool GlobalConfiguration::INPUT_QUERY_LOGGING = false;
const bool GlobalConfiguration::PROJECTED_STEEPEST_EDGE_LOGGING = false;
const bool GlobalConfiguration::GAUSSIAN_ELIMINATION_LOGGING = false;
const bool GlobalConfiguration::QUERY_LOADER_LOGGING = false;
const bool GlobalConfiguration::SYMBOLIC_BOUND_TIGHTENER_LOGGING = false;
const bool GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING = false;
const bool GlobalConfiguration::MPS_PARSER_LOGGING = false;
const bool GlobalConfiguration::ONNX_PARSER_LOGGING = false;
const bool GlobalConfiguration::SOI_LOGGING = false;
const bool GlobalConfiguration::SCORE_TRACKER_LOGGING = false;
const bool GlobalConfiguration::CEGAR_LOGGING = false;

const bool GlobalConfiguration::USE_SMART_FIX = false;
const bool GlobalConfiguration::USE_LEAST_FIX = false;

void GlobalConfiguration::print()
{
    printf( "****************************\n" );
    printf( "*** Global Configuraiton ***\n" );
    printf( "****************************\n" );
    printf( "  DEFAULT_EPSILON_FOR_COMPARISONS: %.15lf\n", DEFAULT_EPSILON_FOR_COMPARISONS );
    printf( "  DEFAULT_DOUBLE_TO_STRING_PRECISION: %u\n", DEFAULT_DOUBLE_TO_STRING_PRECISION );
    printf( "  STATISTICS_PRINTING_FREQUENCY: %u\n", STATISTICS_PRINTING_FREQUENCY );
    printf( "  BOUND_COMPARISON_ADDITIVE_TOLERANCE: %.15lf\n",
            BOUND_COMPARISON_ADDITIVE_TOLERANCE );
    printf( "  BOUND_COMPARISON_MULTIPLICATIVE_TOLERANCE: %.15lf\n",
            BOUND_COMPARISON_MULTIPLICATIVE_TOLERANCE );
    printf( "  PIVOT_CHANGE_COLUMN_TOLERANCE: %.15lf\n", PIVOT_CHANGE_COLUMN_TOLERANCE );
    printf( "  RATIO_CONSTRAINT_ADDITIVE_TOLERANCE: %.15lf\n",
            RATIO_CONSTRAINT_ADDITIVE_TOLERANCE );
    printf( "  RATIO_CONSTRAINT_MULTIPLICATIVE_TOLERANCE: %.15lf\n",
            RATIO_CONSTRAINT_MULTIPLICATIVE_TOLERANCE );
    printf( "  BASIC_COSTS_ADDITIVE_TOLERANCE: %.15lf\n", BASIC_COSTS_ADDITIVE_TOLERANCE );
    printf( "  BASIC_COSTS_MULTIPLICATIVE_TOLERANCE: %.15lf\n",
            BASIC_COSTS_MULTIPLICATIVE_TOLERANCE );
    printf( "  DEGRADATION_CHECKING_FREQUENCY: %u\n", DEGRADATION_CHECKING_FREQUENCY );
    printf( "  DEGRADATION_THRESHOLD: %.15lf\n", DEGRADATION_THRESHOLD );
    printf( "  ACCEPTABLE_SIMPLEX_PIVOT_THRESHOLD: %.15lf\n", ACCEPTABLE_SIMPLEX_PIVOT_THRESHOLD );
    printf( "  USE_COLUMN_MERGING_EQUATIONS: %s\n", USE_COLUMN_MERGING_EQUATIONS ? "Yes" : "No" );
    printf( "  GAUSSIAN_ELIMINATION_PIVOT_SCALE_THRESHOLD: %.15lf\n",
            GAUSSIAN_ELIMINATION_PIVOT_SCALE_THRESHOLD );
    printf( "  MAX_SIMPLEX_PIVOT_SEARCH_ITERATIONS: %u\n", MAX_SIMPLEX_PIVOT_SEARCH_ITERATIONS );
    printf( "  BOUND_TIGHTING_ON_CONSTRAINT_MATRIX_FREQUENCY: %u\n",
            BOUND_TIGHTING_ON_CONSTRAINT_MATRIX_FREQUENCY );
    printf( "  COST_FUNCTION_ERROR_THRESHOLD: %.15lf\n", COST_FUNCTION_ERROR_THRESHOLD );
    printf( "  USE_HARRIS_RATIO_TEST: %s\n", USE_HARRIS_RATIO_TEST ? "Yes" : "No" );

    printf( "  PREPROCESS_INPUT_QUERY: %s\n", PREPROCESS_INPUT_QUERY ? "Yes" : "No" );
    printf( "  PREPROCESSOR_ELIMINATE_VARIABLES: %s\n",
            PREPROCESSOR_ELIMINATE_VARIABLES ? "Yes" : "No" );
    printf( "  PSE_ITERATIONS_BEFORE_RESET: %u\n", PSE_ITERATIONS_BEFORE_RESET );
    printf( "  PSE_GAMMA_ERROR_THRESHOLD: %.15lf\n", PSE_GAMMA_ERROR_THRESHOLD );
    printf( "  CONSTRAINT_COMPARISON_TOLERANCE: %.15lf\n", CONSTRAINT_COMPARISON_TOLERANCE );

    String basisBoundTighteningType;
    switch ( EXPLICIT_BASIS_BOUND_TIGHTENING_TYPE )
    {
    case COMPUTE_INVERTED_BASIS_MATRIX:
        basisBoundTighteningType = "Compute inverted basis matrix";
        break;

    case USE_IMPLICIT_INVERTED_BASIS_MATRIX:
        basisBoundTighteningType = "Use implicit inverted basis matrix";
        break;

    default:
        basisBoundTighteningType = "Unknown";
        break;
    }

    printf( "  EXPLICIT_BASIS_BOUND_TIGHTENING_INVERT_BASIS: %s\n",
            basisBoundTighteningType.ascii() );
    printf( "  EXPLICIT_BOUND_TIGHTENING_UNTIL_SATURATION: %s\n",
            EXPLICIT_BOUND_TIGHTENING_UNTIL_SATURATION ? "Yes" : "No" );
    printf( "  REFACTORIZATION_THRESHOLD: %u\n", REFACTORIZATION_THRESHOLD );

    String basisFactorizationType;
    if ( GlobalConfiguration::BASIS_FACTORIZATION_TYPE == GlobalConfiguration::LU_FACTORIZATION )
        basisFactorizationType = "LU_FACTORIZATION";
    else if ( GlobalConfiguration::BASIS_FACTORIZATION_TYPE ==
              GlobalConfiguration::SPARSE_LU_FACTORIZATION )
        basisFactorizationType = "SPARSE_LU_FACTORIZATION";
    else if ( GlobalConfiguration::BASIS_FACTORIZATION_TYPE ==
              GlobalConfiguration::FORREST_TOMLIN_FACTORIZATION )
        basisFactorizationType = "FORREST_TOMLIN_FACTORIZATION";
    else
        basisFactorizationType = "Unknown";

    printf( "  BASIS_FACTORIZATION_TYPE: %s\n", basisFactorizationType.ascii() );
    printf( "****************************\n" );
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
