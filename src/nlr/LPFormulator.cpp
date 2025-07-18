/*********************                                                        */
/*! \file LPFormulator.cpp
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

#include "LPFormulator.h"

#include "DeepPolySoftmaxElement.h"
#include "GurobiWrapper.h"
#include "InfeasibleQueryException.h"
#include "Layer.h"
#include "MStringf.h"
#include "NLRError.h"
#include "Options.h"
#include "TimeUtils.h"
#include "Vector.h"

namespace NLR {

LPFormulator::LPFormulator( LayerOwner *layerOwner )
    : _layerOwner( layerOwner )
    , _cutoffInUse( false )
    , _cutoffValue( 0 )
{
}

LPFormulator::~LPFormulator()
{
}

double LPFormulator::solveLPRelaxation( GurobiWrapper &gurobi,
                                        const Map<unsigned, Layer *> &layers,
                                        MinOrMax minOrMax,
                                        String variableName,
                                        unsigned lastLayer )
{
    gurobi.resetModel();
    createLPRelaxation( layers, gurobi, lastLayer );
    return optimizeWithGurobi( gurobi, minOrMax, variableName, _cutoffValue );
}

double LPFormulator::optimizeWithGurobi( GurobiWrapper &gurobi,
                                         MinOrMax minOrMax,
                                         String variableName,
                                         double cutoffValue,
                                         std::atomic_bool *infeasible )
{
    List<GurobiWrapper::Term> terms;
    terms.append( GurobiWrapper::Term( 1, variableName ) );

    if ( minOrMax == MAX )
        gurobi.setObjective( terms );
    else
        gurobi.setCost( terms );

    gurobi.setTimeLimit( FloatUtils::infinity() );

    gurobi.solve();

    if ( gurobi.infeasible() )
    {
        if ( infeasible )
        {
            *infeasible = true;
            return FloatUtils::infinity();
        }
        else
            throw InfeasibleQueryException();
    }

    if ( gurobi.cutoffOccurred() )
        return cutoffValue;

    if ( gurobi.optimal() )
    {
        Map<String, double> dontCare;
        double result = 0;
        gurobi.extractSolution( dontCare, result );
        return result;
    }
    else if ( gurobi.timeout() )
    {
        return gurobi.getObjectiveBound();
    }

    throw NLRError( NLRError::UNEXPECTED_RETURN_STATUS_FROM_GUROBI );
}

void LPFormulator::optimizeBoundsWithIncrementalLpRelaxation( const Map<unsigned, Layer *> &layers )
{
    GurobiWrapper gurobi;

    List<GurobiWrapper::Term> terms;
    Map<String, double> dontCare;
    double lb = 0;
    double ub = 0;
    double currentLb = 0;
    double currentUb = 0;

    unsigned tighterBoundCounter = 0;
    unsigned signChanges = 0;
    unsigned cutoffs = 0;

    struct timespec gurobiStart;
    (void)gurobiStart;
    struct timespec gurobiEnd;
    (void)gurobiEnd;

    gurobiStart = TimeUtils::sampleMicro();

    for ( unsigned i = 0; i < _layerOwner->getNumberOfLayers(); ++i )
    {
        /*
          Go over the layers, one by one. Each time encode the layer,
          and then issue queries on each of its variables
        */
        ASSERT( layers.exists( i ) );
        Layer *layer = layers[i];
        addLayerToModel( gurobi, layer, false );

        for ( unsigned j = 0; j < layer->getSize(); ++j )
        {
            if ( layer->neuronEliminated( j ) )
                continue;

            currentLb = layer->getLb( j );
            currentUb = layer->getUb( j );

            if ( _cutoffInUse && ( currentLb >= _cutoffValue || currentUb <= _cutoffValue ) )
                continue;

            unsigned variable = layer->neuronToVariable( j );
            Stringf variableName( "x%u", variable );

            terms.clear();
            terms.append( GurobiWrapper::Term( 1, variableName ) );

            // Maximize
            gurobi.reset();
            gurobi.setObjective( terms );
            gurobi.solve();

            if ( gurobi.infeasible() )
                throw InfeasibleQueryException();

            if ( gurobi.cutoffOccurred() )
            {
                ub = _cutoffValue;
            }
            else if ( gurobi.optimal() )
            {
                gurobi.extractSolution( dontCare, ub );
            }
            else if ( gurobi.timeout() )
            {
                ub = gurobi.getObjectiveBound();
            }
            else
            {
                throw NLRError( NLRError::UNEXPECTED_RETURN_STATUS_FROM_GUROBI );
            }

            // If the bound is tighter, store it
            if ( ub < currentUb )
            {
                gurobi.setUpperBound( variableName, ub );

                if ( FloatUtils::isPositive( currentUb ) && !FloatUtils::isPositive( ub ) )
                    ++signChanges;

                layer->setUb( j, ub );
                _layerOwner->receiveTighterBound( Tightening( variable, ub, Tightening::UB ) );
                ++tighterBoundCounter;

                if ( _cutoffInUse && ub < _cutoffValue )
                {
                    ++cutoffs;
                    continue;
                }
            }

            // Minimize
            gurobi.reset();
            gurobi.setCost( terms );
            gurobi.solve();

            if ( gurobi.infeasible() )
                throw InfeasibleQueryException();

            if ( gurobi.cutoffOccurred() )
            {
                lb = _cutoffValue;
            }
            else if ( gurobi.optimal() )
            {
                gurobi.extractSolution( dontCare, lb );
            }
            else if ( gurobi.timeout() )
            {
                lb = gurobi.getObjectiveBound();
            }
            else
            {
                throw NLRError( NLRError::UNEXPECTED_RETURN_STATUS_FROM_GUROBI );
            }

            // If the bound is tighter, store it
            if ( lb > currentLb )
            {
                gurobi.setLowerBound( variableName, lb );

                if ( FloatUtils::isNegative( currentLb ) && !FloatUtils::isNegative( lb ) )
                    ++signChanges;

                layer->setLb( j, lb );
                _layerOwner->receiveTighterBound( Tightening( variable, lb, Tightening::LB ) );
                ++tighterBoundCounter;

                if ( _cutoffInUse && lb >= _cutoffValue )
                {
                    ++cutoffs;
                    continue;
                }
            }
        }
    }

    gurobiEnd = TimeUtils::sampleMicro();

    LPFormulator_LOG(
        Stringf( "Number of tighter bounds found by Gurobi: %u. Sign changes: %u. Cutoffs: %u\n",
                 tighterBoundCounter,
                 signChanges,
                 cutoffs )
            .ascii() );
    LPFormulator_LOG( Stringf( "Seconds spent Gurobiing: %llu\n",
                               TimeUtils::timePassed( gurobiStart, gurobiEnd ) / 1000000 )
                          .ascii() );
}

void LPFormulator::optimizeBoundsWithLpRelaxation(
    const Map<unsigned, Layer *> &layers,
    bool backward,
    const Map<unsigned, Vector<double>> &layerIndicesToParameters,
    const Vector<PolygonalTightening> &polygonal_tightenings )
{
    unsigned numberOfWorkers = Options::get()->getInt( Options::NUM_WORKERS );

    Map<GurobiWrapper *, unsigned> solverToIndex;
    // Create a queue of free workers
    // When a worker is working, it is popped off the queue, when it is done, it
    // is added back to the queue.
    SolverQueue freeSolvers( numberOfWorkers );
    for ( unsigned i = 0; i < numberOfWorkers; ++i )
    {
        GurobiWrapper *gurobi = new GurobiWrapper();
        solverToIndex[gurobi] = i;
        enqueueSolver( freeSolvers, gurobi );
    }

    boost::thread *threads = new boost::thread[numberOfWorkers];
    std::mutex mtx;
    std::atomic_bool infeasible( false );

    std::atomic_uint tighterBoundCounter( 0 );
    std::atomic_uint signChanges( 0 );
    std::atomic_uint cutoffs( 0 );

    struct timespec gurobiStart;
    (void)gurobiStart;
    struct timespec gurobiEnd;
    (void)gurobiEnd;

    gurobiStart = TimeUtils::sampleMicro();

    unsigned startIndex = backward ? layers.size() - 1 : 0;
    unsigned endIndex = backward ? 0 : layers.size();
    for ( unsigned layerIndex = startIndex; layerIndex != endIndex;
          backward ? --layerIndex : ++layerIndex )
    {
        LPFormulator_LOG( Stringf( "Tightening bound for layer %u...", layerIndex ).ascii() );
        Layer *layer = layers[layerIndex];

        ThreadArgument argument( layer,
                                 &layers,
                                 std::ref( freeSolvers ),
                                 std::ref( mtx ),
                                 std::ref( infeasible ),
                                 std::ref( tighterBoundCounter ),
                                 std::ref( signChanges ),
                                 std::ref( cutoffs ),
                                 layer->getLayerIndex(),
                                 layerIndex,
                                 threads,
                                 &solverToIndex );

        // optimize every neuron of layer
        optimizeBoundsOfNeuronsWithLpRelaxation(
            argument, backward, layerIndicesToParameters, polygonal_tightenings );
        LPFormulator_LOG( Stringf( "Tightening bound for layer %u - done", layerIndex ).ascii() );
    }

    for ( unsigned i = 0; i < numberOfWorkers; ++i )
    {
        threads[i].join();
    }

    gurobiEnd = TimeUtils::sampleMicro();

    LPFormulator_LOG(
        Stringf( "Number of tighter bounds found by Gurobi: %u. Sign changes: %u. Cutoffs: %u\n",
                 tighterBoundCounter.load(),
                 signChanges.load(),
                 cutoffs.load() )
            .ascii() );
    LPFormulator_LOG( Stringf( "Seconds spent Gurobiing: %llu\n",
                               TimeUtils::timePassed( gurobiStart, gurobiEnd ) / 1000000 )
                          .ascii() );

    // Clean up
    clearSolverQueue( freeSolvers );

    if ( threads )
    {
        delete[] threads;
        threads = NULL;
    }

    if ( infeasible )
        throw InfeasibleQueryException();
}

void LPFormulator::optimizeBoundsOfOneLayerWithLpRelaxation( const Map<unsigned, Layer *> &layers,
                                                             unsigned targetIndex )
{
    unsigned numberOfWorkers = Options::get()->getInt( Options::NUM_WORKERS );

    Map<GurobiWrapper *, unsigned> solverToIndex;
    // Create a queue of free workers
    // When a worker is working, it is popped off the queue, when it is done, it
    // is added back to the queue.
    SolverQueue freeSolvers( numberOfWorkers );
    for ( unsigned i = 0; i < numberOfWorkers; ++i )
    {
        GurobiWrapper *gurobi = new GurobiWrapper();
        solverToIndex[gurobi] = i;
        enqueueSolver( freeSolvers, gurobi );
    }

    boost::thread *threads = new boost::thread[numberOfWorkers];
    std::mutex mtx;
    std::atomic_bool infeasible( false );

    std::atomic_uint tighterBoundCounter( 0 );
    std::atomic_uint signChanges( 0 );
    std::atomic_uint cutoffs( 0 );

    struct timespec gurobiStart;
    (void)gurobiStart;
    struct timespec gurobiEnd;
    (void)gurobiEnd;

    gurobiStart = TimeUtils::sampleMicro();

    Layer *layer = layers[targetIndex];

    ThreadArgument argument( layer,
                             &layers,
                             std::ref( freeSolvers ),
                             std::ref( mtx ),
                             std::ref( infeasible ),
                             std::ref( tighterBoundCounter ),
                             std::ref( signChanges ),
                             std::ref( cutoffs ),
                             layers.size() - 1,
                             targetIndex,
                             threads,
                             &solverToIndex );

    // optimize every neuron of layer
    optimizeBoundsOfNeuronsWithLpRelaxation( argument, false );

    for ( unsigned i = 0; i < numberOfWorkers; ++i )
    {
        threads[i].join();
    }

    gurobiEnd = TimeUtils::sampleMicro();

    LPFormulator_LOG(
        Stringf( "Number of tighter bounds found by Gurobi: %u. Sign changes: %u. Cutoffs: %u\n",
                 tighterBoundCounter.load(),
                 signChanges.load(),
                 cutoffs.load() )
            .ascii() );
    LPFormulator_LOG( Stringf( "Seconds spent Gurobiing: %llu\n",
                               TimeUtils::timePassed( gurobiStart, gurobiEnd ) / 1000000 )
                          .ascii() );

    clearSolverQueue( freeSolvers );

    if ( infeasible )
        throw InfeasibleQueryException();
}

void LPFormulator::optimizeBoundsOfNeuronsWithLpRelaxation(
    ThreadArgument &args,
    bool backward,
    const Map<unsigned, Vector<double>> &layerIndicesToParameters,
    const Vector<PolygonalTightening> &polygonal_tightenings )
{
    unsigned numberOfWorkers = Options::get()->getInt( Options::NUM_WORKERS );

    // Time to wait if no idle worker is availble
    boost::chrono::milliseconds waitTime( numberOfWorkers - 1 );

    Layer *layer = args._layer;
    const Map<unsigned, Layer *> layers = *args._layers;
    unsigned targetIndex = args._targetIndex;
    unsigned lastIndexOfRelaxation = args._lastIndexOfRelaxation;

    const Map<GurobiWrapper *, unsigned> solverToIndex = *args._solverToIndex;
    SolverQueue &freeSolvers = args._freeSolvers;
    std::mutex &mtx = args._mtx;
    std::atomic_bool &infeasible = args._infeasible;
    std::atomic_uint &tighterBoundCounter = args._tighterBoundCounter;
    std::atomic_uint &signChanges = args._signChanges;
    std::atomic_uint &cutoffs = args._cutoffs;
    boost::thread *threads = args._threads;

    double currentLb;
    double currentUb;

    bool skipTightenLb = false; // If true, skip lower bound tightening
    bool skipTightenUb = false; // If true, skip upper bound tightening

    // declare simulations as local var to avoid a problem which can happen due to multi thread
    // process.
    const Vector<Vector<double>> *simulations =
        _layerOwner->getLayer( targetIndex )->getSimulations();

    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( layer->neuronEliminated( i ) )
            continue;

        currentLb = layer->getLb( i );
        currentUb = layer->getUb( i );

        if ( _cutoffInUse && ( currentLb >= _cutoffValue || currentUb <= _cutoffValue ) )
            continue;

        skipTightenLb = false;
        skipTightenUb = false;

        // Loop for simulation
        for ( const auto &simValue : ( *simulations ).get( i ) )
        {
            if ( _cutoffInUse && _cutoffValue < simValue ) // If x_lower < 0 < x_sim, do not try to
                                                           // call tightning upper bound.
                skipTightenUb = true;

            if ( _cutoffInUse && simValue < _cutoffValue ) // If x_sim < 0 < x_upper, do not try to
                                                           // call tightning lower bound.
                skipTightenLb = true;

            if ( skipTightenUb && skipTightenLb )
                break;
        }

        // If no tightning is needed, continue
        if ( skipTightenUb && skipTightenLb )
        {
            LPFormulator_LOG(
                Stringf(
                    "Skip tightening lower and upper bounds for layer %d index %u", targetIndex, i )
                    .ascii() );
            continue;
        }
        else if ( skipTightenUb )
        {
            LPFormulator_LOG(
                Stringf( "Skip tightening upper bound for layer %u index %u", targetIndex, i )
                    .ascii() );
        }
        else if ( skipTightenLb )
        {
            LPFormulator_LOG(
                Stringf( "Skip tightening lower bound for layer %u index %u", targetIndex, i )
                    .ascii() );
        }

        if ( infeasible )
        {
            // infeasibility is derived, interupt all active threads
            for ( unsigned i = 0; i < numberOfWorkers; ++i )
            {
                threads[i].interrupt();
                threads[i].join();
            }

            // Clean up
            clearSolverQueue( freeSolvers );

            if ( threads )
            {
                delete[] threads;
                threads = NULL;
            }

            throw InfeasibleQueryException();
        }

        // Wait until there is an idle solver
        GurobiWrapper *freeSolver;
        while ( !freeSolvers.pop( freeSolver ) )
            boost::this_thread::sleep_for( waitTime );

        freeSolver->resetModel();

        mtx.lock();
        if ( backward )
            createLPRelaxationAfter( layers,
                                     *freeSolver,
                                     lastIndexOfRelaxation,
                                     layerIndicesToParameters,
                                     polygonal_tightenings );
        else
            createLPRelaxation( layers,
                                *freeSolver,
                                lastIndexOfRelaxation,
                                layerIndicesToParameters,
                                polygonal_tightenings );
        mtx.unlock();

        // spawn a thread to tighten the bounds for the current variable
        ThreadArgument argument( freeSolver,
                                 layer,
                                 i,
                                 currentLb,
                                 currentUb,
                                 _cutoffInUse,
                                 _cutoffValue,
                                 _layerOwner,
                                 std::ref( freeSolvers ),
                                 std::ref( mtx ),
                                 std::ref( infeasible ),
                                 std::ref( tighterBoundCounter ),
                                 std::ref( signChanges ),
                                 std::ref( cutoffs ),
                                 skipTightenLb,
                                 skipTightenUb );

        if ( numberOfWorkers == 1 )
            tightenSingleVariableBoundsWithLPRelaxation( argument );
        else
            threads[solverToIndex[freeSolver]] =
                boost::thread( tightenSingleVariableBoundsWithLPRelaxation, argument );
    }
}

void LPFormulator::tightenSingleVariableBoundsWithLPRelaxation( ThreadArgument &argument )
{
    try
    {
        GurobiWrapper *gurobi = argument._gurobi;
        Layer *layer = argument._layer;
        unsigned index = argument._index;
        double currentLb = argument._currentLb;
        double currentUb = argument._currentUb;
        bool cutoffInUse = argument._cutoffInUse;
        double cutoffValue = argument._cutoffValue;
        LayerOwner *layerOwner = argument._layerOwner;
        SolverQueue &freeSolvers = argument._freeSolvers;
        std::mutex &mtx = argument._mtx;
        std::atomic_bool &infeasible = argument._infeasible;
        std::atomic_uint &tighterBoundCounter = argument._tighterBoundCounter;
        std::atomic_uint &signChanges = argument._signChanges;
        std::atomic_uint &cutoffs = argument._cutoffs;
        bool skipTightenLb = argument._skipTightenLb;
        bool skipTightenUb = argument._skipTightenUb;

        LPFormulator_LOG(
            Stringf( "Tightening bounds for layer %u index %u", layer->getLayerIndex(), index )
                .ascii() );

        unsigned variable = layer->neuronToVariable( index );
        Stringf variableName( "x%u", variable );

        if ( !skipTightenUb )
        {
            LPFormulator_LOG( Stringf( "Computing upperbound..." ).ascii() );
            double ub = optimizeWithGurobi(
                            *gurobi, MinOrMax::MAX, variableName, cutoffValue, &infeasible ) +
                        GlobalConfiguration::LP_TIGHTENING_ROUNDING_CONSTANT;
            ;
            LPFormulator_LOG( Stringf( "Upperbound computed %f", ub ).ascii() );

            // Store the new bound if it is tighter
            if ( ub < currentUb )
            {
                if ( FloatUtils::isPositive( currentUb ) && !FloatUtils::isPositive( ub ) )
                    ++signChanges;

                mtx.lock();
                layer->setUb( index, ub );
                layerOwner->receiveTighterBound( Tightening( variable, ub, Tightening::UB ) );
                mtx.unlock();

                ++tighterBoundCounter;

                if ( cutoffInUse && ub < cutoffValue )
                {
                    ++cutoffs;
                    enqueueSolver( freeSolvers, gurobi );
                    return;
                }
            }
        }

        if ( !skipTightenLb )
        {
            LPFormulator_LOG( Stringf( "Computing lowerbound..." ).ascii() );
            gurobi->reset();
            double lb = optimizeWithGurobi(
                            *gurobi, MinOrMax::MIN, variableName, cutoffValue, &infeasible ) -
                        GlobalConfiguration::LP_TIGHTENING_ROUNDING_CONSTANT;
            LPFormulator_LOG( Stringf( "Lowerbound computed: %f", lb ).ascii() );
            // Store the new bound if it is tighter
            if ( lb > currentLb )
            {
                if ( FloatUtils::isNegative( currentLb ) && !FloatUtils::isNegative( lb ) )
                    ++signChanges;

                mtx.lock();
                layer->setLb( index, lb );
                layerOwner->receiveTighterBound( Tightening( variable, lb, Tightening::LB ) );
                mtx.unlock();
                ++tighterBoundCounter;

                if ( cutoffInUse && lb > cutoffValue )
                    ++cutoffs;
            }
        }
        enqueueSolver( freeSolvers, gurobi );
    }
    catch ( boost::thread_interrupted & )
    {
        enqueueSolver( argument._freeSolvers, argument._gurobi );
    }
}

void LPFormulator::createLPRelaxation(
    const Map<unsigned, Layer *> &layers,
    GurobiWrapper &gurobi,
    unsigned lastLayer,
    const Map<unsigned, Vector<double>> &layerIndicesToParameters,
    const Vector<PolygonalTightening> &polygonal_tightenings )
{
    for ( const auto &layer : layers )
    {
        unsigned currentLayerIndex = layer.second->getLayerIndex();
        if ( currentLayerIndex > lastLayer )
            continue;

        if ( layerIndicesToParameters.empty() )
            addLayerToModel( gurobi, layer.second, false );
        else
        {
            const Vector<double> &currentLayerCoeffs = layerIndicesToParameters[currentLayerIndex];
            addLayerToParameterisedModel( gurobi, layer.second, false, currentLayerCoeffs );
        }
    }
    addPolyognalTighteningsToLpRelaxation( gurobi, layers, 0, lastLayer, polygonal_tightenings );
}

void LPFormulator::createLPRelaxationAfter(
    const Map<unsigned, Layer *> &layers,
    GurobiWrapper &gurobi,
    unsigned firstLayer,
    const Map<unsigned, Vector<double>> &layerIndicesToParameters,
    const Vector<PolygonalTightening> &polygonal_tightenings )
{
    unsigned depth = GlobalConfiguration::BACKWARD_BOUND_PROPAGATION_DEPTH;
    std::priority_queue<unsigned, std::vector<unsigned>, std::greater<unsigned>> layersToAdd;
    Map<unsigned, unsigned> layerToDepth;

    layersToAdd.push( firstLayer );
    layerToDepth[firstLayer] = 0;
    while ( !layersToAdd.empty() )
    {
        unsigned currentLayerIndex = layersToAdd.top();
        Layer *currentLayer = layers[currentLayerIndex];
        unsigned currentDepth = layerToDepth[currentLayerIndex];
        layersToAdd.pop();
        if ( currentDepth > depth )
            continue;
        else
        {
            if ( layerIndicesToParameters.empty() )
                addLayerToModel( gurobi, currentLayer, true );
            else
            {
                const Vector<double> &currentLayerCoeffs =
                    layerIndicesToParameters[currentLayerIndex];
                addLayerToParameterisedModel( gurobi, currentLayer, true, currentLayerCoeffs );
            }

            for ( const auto &nextLayer : currentLayer->getSuccessorLayers() )
            {
                if ( layerToDepth.exists( nextLayer ) )
                    continue;
                layersToAdd.push( nextLayer );
                layerToDepth[nextLayer] = currentDepth + 1;
            }
        }
    }
    addPolyognalTighteningsToLpRelaxation(
        gurobi, layers, firstLayer, layersToAdd.top(), polygonal_tightenings );
}

void LPFormulator::addLayerToModel( GurobiWrapper &gurobi,
                                    const Layer *layer,
                                    bool createVariables )
{
    switch ( layer->getLayerType() )
    {
    case Layer::INPUT:
        addInputLayerToLpRelaxation( gurobi, layer );
        break;

    case Layer::RELU:
        addReluLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::WEIGHTED_SUM:
        addWeightedSumLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::ROUND:
        addRoundLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::LEAKY_RELU:
        addLeakyReluLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::ABSOLUTE_VALUE:
        addAbsoluteValueLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::SIGN:
        addSignLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::MAX:
        addMaxLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::SIGMOID:
        addSigmoidLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::SOFTMAX:
        addSoftmaxLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    case Layer::BILINEAR:
        addBilinearLayerToLpRelaxation( gurobi, layer, createVariables );
        break;

    default:
        throw NLRError( NLRError::LAYER_TYPE_NOT_SUPPORTED, "LPFormulator" );
        break;
    }
}

void LPFormulator::addInputLayerToLpRelaxation( GurobiWrapper &gurobi, const Layer *layer )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        unsigned variable = layer->neuronToVariable( i );
        gurobi.addVariable( Stringf( "x%u", variable ), layer->getLb( i ), layer->getUb( i ) );
    }
}

void LPFormulator::addReluLayerToLpRelaxation( GurobiWrapper &gurobi,
                                               const Layer *layer,
                                               bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = sourceValue > 0 ? sourceValue : 0;

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            gurobi.addVariable( Stringf( "x%u", targetVariable ), 0, layer->getUb( i ) );

            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // The ReLU is active, y = x
                if ( sourceLb < 0 )
                    sourceLb = 0;

                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else if ( !FloatUtils::isPositive( sourceUb ) )
            {
                // The ReLU is inactive, y = 0
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else
            {
                /*
                  The phase of this ReLU is not yet fixed.

                  For y = ReLU(x), we add the following triangular relaxation:

                  1. y >= 0
                  2. y >= x
                  3. y is below the line the crosses (x.lb,0) and (x.ub,x.ub)
                */

                // y >= 0
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y >= x, i.e. y - x >= 0
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                /*
                         u        ul
                  y <= ----- x - -----
                       u - l     u - l
                */
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -sourceUb / ( sourceUb - sourceLb ),
                                                   Stringf( "x%u", sourceVariable ) ) );
                gurobi.addLeqConstraint( terms,
                                         ( -sourceUb * sourceLb ) / ( sourceUb - sourceLb ) );
            }
        }
    }
}

void LPFormulator::addRoundLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                const Layer *layer,
                                                bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = FloatUtils::round( sourceValue );

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            double ub = std::min( FloatUtils::round( sourceUb ), layer->getUb( i ) );
            double lb = std::max( FloatUtils::round( sourceLb ), layer->getLb( i ) );

            gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

            // If u = l:  y = round(u)
            if ( FloatUtils::areEqual( sourceUb, sourceLb ) )
            {
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addEqConstraint( terms, ub );
            }

            else
            {
                List<GurobiWrapper::Term> terms;
                // y <= x + 0.5, i.e. y - x <= 0.5
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addLeqConstraint( terms, 0.5 );

                // y >= x - 0.5, i.e. y - x >= -0.5
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, -0.5 );
            }
        }
    }
}

void LPFormulator::addAbsoluteValueLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                        const Layer *layer,
                                                        bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = sourceValue > 0 ? sourceValue : -sourceValue;

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // The AbsoluteValue is active, y = x
                if ( sourceLb < 0 )
                    sourceLb = 0;

                double ub = std::min( sourceUb, layer->getUb( i ) );
                double lb = std::max( sourceLb, layer->getLb( i ) );
                gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else if ( !FloatUtils::isPositive( sourceUb ) )
            {
                double ub = std::min( -sourceLb, layer->getUb( i ) );
                double lb = std::max( -sourceUb, layer->getLb( i ) );
                gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

                // The AbsoluteValue is inactive, y = -x, i.e. y + x = 0
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else
            {
                double ub = std::min( std::max( -sourceLb, sourceUb ), layer->getUb( i ) );
                double lb = std::max( 0.0, layer->getLb( i ) );
                gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

                // The phase of this AbsoluteValue is not yet fixed, 0 <= y <= max(-lb, ub).
                // y >= 0
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y <= max(-lb, ub)
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addLeqConstraint( terms, ub );
            }
        }
    }
}

void LPFormulator::addSigmoidLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                  const Layer *layer,
                                                  bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = SigmoidConstraint::sigmoid( sourceValue );

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            double sourceUbSigmoid = SigmoidConstraint::sigmoid( sourceUb );
            double sourceLbSigmoid = SigmoidConstraint::sigmoid( sourceLb );

            double ub = std::min( sourceUbSigmoid, layer->getUb( i ) );
            double lb = std::max( sourceLbSigmoid, layer->getLb( i ) );

            gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

            // If u = l:  y = sigmoid(u)
            if ( FloatUtils::areEqual( sourceUb, sourceLb ) )
            {
                List<GurobiWrapper::Term> terms;
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addEqConstraint( terms, ub );
            }

            else
            {
                List<GurobiWrapper::Term> terms;
                double lambda = ( ub - lb ) / ( sourceUb - sourceLb );
                double lambdaPrime = std::min( SigmoidConstraint::sigmoidDerivative( sourceLb ),
                                               SigmoidConstraint::sigmoidDerivative( sourceUb ) );

                // update lower bound
                if ( FloatUtils::isPositive( sourceLb ) )
                {
                    // y >= lambda * (x - l) + sigmoid(lb), i.e. y - lambda * x >= sigmoid(lb) -
                    // lambda * l
                    terms.clear();
                    terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                    terms.append(
                        GurobiWrapper::Term( -lambda, Stringf( "x%u", sourceVariable ) ) );
                    gurobi.addGeqConstraint( terms, sourceLbSigmoid - sourceLb * lambda );
                }

                else
                {
                    // y >= lambda' * (x - l) + sigmoid(lb), i.e. y - lambda' * x >= sigmoid(lb) -
                    // lambda' * l
                    terms.clear();
                    terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                    terms.append(
                        GurobiWrapper::Term( -lambdaPrime, Stringf( "x%u", sourceVariable ) ) );
                    gurobi.addGeqConstraint( terms, sourceLbSigmoid - sourceLb * lambdaPrime );
                }

                // update upper bound
                if ( !FloatUtils::isPositive( sourceUb ) )
                {
                    // y <= lambda * (x - u) + sigmoid(ub), i.e. y - lambda * x <= sigmoid(ub) -
                    // lambda * u
                    terms.clear();
                    terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                    terms.append(
                        GurobiWrapper::Term( -lambda, Stringf( "x%u", sourceVariable ) ) );
                    gurobi.addLeqConstraint( terms, sourceUbSigmoid - sourceUb * lambda );
                }
                else
                {
                    // y <= lambda' * (x - u) + sigmoid(ub), i.e. y - lambda' * x <= sigmoid(ub) -
                    // lambda' * u
                    terms.clear();
                    terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                    terms.append(
                        GurobiWrapper::Term( -lambdaPrime, Stringf( "x%u", sourceVariable ) ) );
                    gurobi.addLeqConstraint( terms, sourceUbSigmoid - sourceUb * lambdaPrime );
                }
            }
        }
    }
}

void LPFormulator::addSignLayerToLpRelaxation( GurobiWrapper &gurobi,
                                               const Layer *layer,
                                               bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( layer->neuronEliminated( i ) )
            continue;

        unsigned targetVariable = layer->neuronToVariable( i );

        List<NeuronIndex> sources = layer->getActivationSources( i );
        const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
        unsigned sourceNeuron = sources.begin()->_neuron;

        if ( sourceLayer->neuronEliminated( sourceNeuron ) )
        {
            // If the source neuron has been eliminated, this neuron is constant
            double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
            double targetValue = FloatUtils::isNegative( sourceValue ) ? -1 : 1;

            gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

            continue;
        }

        unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
        double sourceLb = sourceLayer->getLb( sourceNeuron );
        double sourceUb = sourceLayer->getUb( sourceNeuron );
        String sourceName = Stringf( "x%u", sourceVariable );
        if ( createVariables && !gurobi.containsVariable( sourceName ) )
            gurobi.addVariable( sourceName, sourceLb, sourceUb );

        if ( !FloatUtils::isNegative( sourceLb ) )
        {
            // The Sign is positive, y = 1
            gurobi.addVariable( Stringf( "x%u", targetVariable ), 1, 1 );
        }
        else if ( FloatUtils::isNegative( sourceUb ) )
        {
            // The Sign is negative, y = -1
            gurobi.addVariable( Stringf( "x%u", targetVariable ), -1, -1 );
        }
        else
        {
            /*
              The phase of this Sign is not yet fixed.

              For y = Sign(x), we add the following parallelogram relaxation:

              1. y >= -1
              2. y <= -1
              3. y is below the line the crosses (x.lb,-1) and (0,1)
              4. y is above the line the crosses (0,-1) and (x.ub,1)
            */

            // -1 <= y <= 1
            gurobi.addVariable( Stringf( "x%u", targetVariable ), -1, 1 );

            /*
                     2
              y <= ----- x + 1
                    - l
            */
            List<GurobiWrapper::Term> terms;
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term( 2.0 / sourceLb, Stringf( "x%u", sourceVariable ) ) );
            gurobi.addLeqConstraint( terms, 1 );

            /*
                     2
              y >= ----- x - 1
                     u
            */
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append(
                GurobiWrapper::Term( -2.0 / sourceUb, Stringf( "x%u", sourceVariable ) ) );
            gurobi.addGeqConstraint( terms, -1 );
        }
    }
}


void LPFormulator::addMaxLayerToLpRelaxation( GurobiWrapper &gurobi,
                                              const Layer *layer,
                                              bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( layer->neuronEliminated( i ) )
            continue;

        unsigned targetVariable = layer->neuronToVariable( i );
        gurobi.addVariable(
            Stringf( "x%u", targetVariable ), layer->getLb( i ), layer->getUb( i ) );

        List<NeuronIndex> sources = layer->getActivationSources( i );

        bool haveFixedSourceValue = false;
        double maxFixedSourceValue = FloatUtils::negativeInfinity();

        double maxConcreteUb = FloatUtils::negativeInfinity();

        List<GurobiWrapper::Term> terms;

        for ( const auto &source : sources )
        {
            const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
            unsigned sourceNeuron = source._neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                haveFixedSourceValue = true;
                double value = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                if ( value > maxFixedSourceValue )
                    maxFixedSourceValue = value;
                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            // Target is at least source: target - source >= 0
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
            gurobi.addGeqConstraint( terms, 0 );

            // Find maximal concrete upper bound
            if ( sourceUb > maxConcreteUb )
                maxConcreteUb = sourceUb;
        }

        if ( haveFixedSourceValue && ( maxConcreteUb < maxFixedSourceValue ) )
        {
            // At least one of the sources has a fixed value,
            // and this fixed value dominates other sources.
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            gurobi.addEqConstraint( terms, maxFixedSourceValue );
        }
        else
        {
            // If we have a fixed value, it's a lower bound
            if ( haveFixedSourceValue )
            {
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addGeqConstraint( terms, maxFixedSourceValue );
            }

            // Target must be smaller than greatest concrete upper bound
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            gurobi.addLeqConstraint( terms, maxConcreteUb );
        }
    }
}

void LPFormulator::addSoftmaxLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                  const Layer *layer,
                                                  bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( layer->neuronEliminated( i ) )
            continue;

        Set<unsigned> handledInputNeurons;
        List<NeuronIndex> sources = layer->getActivationSources( i );

        Vector<double> sourceLbs;
        Vector<double> sourceUbs;
        Vector<double> sourceMids;
        Vector<double> targetLbs;
        Vector<double> targetUbs;
        for ( const auto &source : sources )
        {
            const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
            unsigned sourceNeuron = source._neuron;
            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            sourceLbs.append( sourceLb - GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
            sourceUbs.append( sourceUb + GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
            sourceMids.append( ( sourceLb + sourceUb ) / 2 );
            targetLbs.append( layer->getLb( i ) );
            targetUbs.append( layer->getUb( i ) );
        }

        // Find the index of i in the softmax
        unsigned index = 0;
        for ( const auto &source : sources )
        {
            if ( handledInputNeurons.exists( source._neuron ) )
                ++index;
            else
            {
                handledInputNeurons.insert( source._neuron );
                break;
            }
        }

        double ub =
            std::min( Layer::linearUpperBound( sourceLbs, sourceUbs, index ), layer->getUb( i ) );
        double lb =
            std::max( Layer::linearLowerBound( sourceLbs, sourceUbs, index ), layer->getLb( i ) );
        targetLbs[index] = lb;
        targetUbs[index] = ub;

        unsigned targetVariable = layer->neuronToVariable( i );
        gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

        double bias;
        SoftmaxBoundType boundType = Options::get()->getSoftmaxBoundType();
        List<GurobiWrapper::Term> terms;
        if ( FloatUtils::areEqual( lb, ub ) )
        {
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            gurobi.addEqConstraint( terms, ub );
        }
        else
        {
            // Compute symbolic bound
            if ( boundType == SoftmaxBoundType::LOG_SUM_EXP_DECOMPOSITION )
            {
                bool useLSE2 = false;
                for ( const auto &lb : targetLbs )
                {
                    if ( lb > GlobalConfiguration::SOFTMAX_LSE2_THRESHOLD )
                        useLSE2 = true;
                }
                unsigned inputIndex = 0;
                if ( !useLSE2 )
                {
                    terms.clear();
                    terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                    bias = Layer::LSELowerBound( sourceMids, sourceLbs, sourceUbs, index );
                    for ( const auto &source : sources )
                    {
                        const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
                        unsigned sourceNeuron = source._neuron;
                        unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
                        double dldj = Layer::dLSELowerBound(
                            sourceMids, sourceLbs, sourceUbs, index, inputIndex );
                        terms.append(
                            GurobiWrapper::Term( -dldj, Stringf( "x%u", sourceVariable ) ) );
                        bias -= dldj * sourceMids[inputIndex];
                        ++inputIndex;
                    }
                    gurobi.addGeqConstraint( terms, bias );
                }
                else
                {
                    terms.clear();
                    terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                    bias = Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, index );
                    for ( const auto &source : sources )
                    {
                        const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
                        unsigned sourceNeuron = source._neuron;
                        unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
                        double dldj = Layer::dLSELowerBound2(
                            sourceMids, sourceLbs, sourceUbs, index, inputIndex );
                        terms.append(
                            GurobiWrapper::Term( -dldj, Stringf( "x%u", sourceVariable ) ) );
                        bias -= dldj * sourceMids[inputIndex];
                        ++inputIndex;
                    }
                    gurobi.addGeqConstraint( terms, bias );
                }

                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                bias = Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, index );
                inputIndex = 0;
                for ( const auto &source : sources )
                {
                    const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
                    unsigned sourceNeuron = source._neuron;
                    unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
                    double dudj = Layer::dLSEUpperbound(
                        sourceMids, targetLbs, targetUbs, index, inputIndex );
                    terms.append( GurobiWrapper::Term( -dudj, Stringf( "x%u", sourceVariable ) ) );
                    bias -= dudj * sourceMids[inputIndex];
                    ++inputIndex;
                }
                gurobi.addLeqConstraint( terms, bias );
            }
            else if ( boundType == SoftmaxBoundType::EXPONENTIAL_RECIPROCAL_DECOMPOSITION )
            {
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                bias = Layer::ERLowerBound( sourceMids, sourceLbs, sourceUbs, index );
                unsigned inputIndex = 0;
                for ( const auto &source : sources )
                {
                    const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
                    unsigned sourceNeuron = source._neuron;
                    unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
                    double dldj =
                        Layer::dERLowerBound( sourceMids, sourceLbs, sourceUbs, index, inputIndex );
                    terms.append( GurobiWrapper::Term( -dldj, Stringf( "x%u", sourceVariable ) ) );
                    bias -= dldj * sourceMids[inputIndex];
                    ++inputIndex;
                }
                gurobi.addGeqConstraint( terms, bias );

                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                bias = Layer::ERUpperBound( sourceMids, targetLbs, targetUbs, index );
                inputIndex = 0;
                for ( const auto &source : sources )
                {
                    const Layer *sourceLayer = _layerOwner->getLayer( source._layer );
                    unsigned sourceNeuron = source._neuron;
                    unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
                    double dudj =
                        Layer::dERUpperBound( sourceMids, targetLbs, targetUbs, index, inputIndex );
                    terms.append( GurobiWrapper::Term( -dudj, Stringf( "x%u", sourceVariable ) ) );
                    bias -= dudj * sourceMids[inputIndex];
                    ++inputIndex;
                }
                gurobi.addLeqConstraint( terms, bias );
            }
        }
    }
}

void LPFormulator::addBilinearLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                   const Layer *layer,
                                                   bool createVariables )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );

            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );

            Vector<double> sourceLbs;
            Vector<double> sourceUbs;
            Vector<double> sourceValues;
            Vector<unsigned> sourceNeurons;
            bool allConstant = true;
            for ( const auto &sourceIndex : sources )
            {
                unsigned sourceNeuron = sourceIndex._neuron;
                double sourceLb = sourceLayer->getLb( sourceNeuron );
                double sourceUb = sourceLayer->getUb( sourceNeuron );
                String sourceName = Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeuron ) );

                sourceNeurons.append( sourceNeuron );
                sourceLbs.append( sourceLb );
                sourceUbs.append( sourceUb );

                if ( createVariables && !gurobi.containsVariable( sourceName ) )
                    gurobi.addVariable( sourceName, sourceLb, sourceUb );

                if ( !sourceLayer->neuronEliminated( sourceNeuron ) )
                {
                    allConstant = false;
                }
                else
                {
                    double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                    sourceValues.append( sourceValue );
                }
            }

            if ( allConstant )
            {
                // If the both source neurons have been eliminated, this neuron is constant
                double targetValue = sourceValues[0] * sourceValues[1];
                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );
                continue;
            }

            double lb = FloatUtils::infinity();
            double ub = FloatUtils::negativeInfinity();
            List<double> values = { sourceLbs[0] * sourceLbs[1],
                                    sourceLbs[0] * sourceUbs[1],
                                    sourceUbs[0] * sourceLbs[1],
                                    sourceUbs[0] * sourceUbs[1] };
            for ( const auto &v : values )
            {
                if ( v < lb )
                    lb = v;
                if ( v > ub )
                    ub = v;
            }

            gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

            // Lower bound: out >= l_y * x + l_x * y - l_x * l_y
            List<GurobiWrapper::Term> terms;
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term(
                -sourceLbs[1],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[0] ) ) ) );
            terms.append( GurobiWrapper::Term(
                -sourceLbs[0],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[1] ) ) ) );
            gurobi.addGeqConstraint( terms, -sourceLbs[0] * sourceLbs[1] );

            // Upper bound: out <= u_y * x + l_x * y - l_x * u_y
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term(
                -sourceUbs[1],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[0] ) ) ) );
            terms.append( GurobiWrapper::Term(
                -sourceLbs[0],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[1] ) ) ) );
            gurobi.addLeqConstraint( terms, -sourceLbs[0] * sourceUbs[1] );
        }
    }
}

void LPFormulator::addWeightedSumLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                      const Layer *layer,
                                                      bool createVariables )
{
    if ( createVariables )
    {
        for ( const auto &sourceLayerPair : layer->getSourceLayers() )
        {
            const Layer *sourceLayer = _layerOwner->getLayer( sourceLayerPair.first );
            unsigned sourceLayerSize = sourceLayerPair.second;

            for ( unsigned j = 0; j < sourceLayerSize; ++j )
            {
                if ( !sourceLayer->neuronEliminated( j ) )
                {
                    Stringf sourceVariableName( "x%u", sourceLayer->neuronToVariable( j ) );
                    if ( !gurobi.containsVariable( sourceVariableName ) )
                    {
                        gurobi.addVariable(
                            sourceVariableName, sourceLayer->getLb( j ), sourceLayer->getUb( j ) );
                    }
                }
            }
        }
    }

    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned variable = layer->neuronToVariable( i );

            gurobi.addVariable( Stringf( "x%u", variable ), layer->getLb( i ), layer->getUb( i ) );

            List<GurobiWrapper::Term> terms;
            terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", variable ) ) );

            double bias = -layer->getBias( i );

            for ( const auto &sourceLayerPair : layer->getSourceLayers() )
            {
                const Layer *sourceLayer = _layerOwner->getLayer( sourceLayerPair.first );
                unsigned sourceLayerSize = sourceLayerPair.second;

                for ( unsigned j = 0; j < sourceLayerSize; ++j )
                {
                    double weight = layer->getWeight( sourceLayerPair.first, j, i );
                    if ( !sourceLayer->neuronEliminated( j ) )
                    {
                        Stringf sourceVariableName( "x%u", sourceLayer->neuronToVariable( j ) );
                        terms.append( GurobiWrapper::Term( weight, sourceVariableName ) );
                    }
                    else
                    {
                        bias -= weight * sourceLayer->getEliminatedNeuronValue( j );
                    }
                }
            }

            gurobi.addEqConstraint( terms, bias );
        }
    }
}

void LPFormulator::addLeakyReluLayerToLpRelaxation( GurobiWrapper &gurobi,
                                                    const Layer *layer,
                                                    bool createVariables )
{
    double slope = layer->getAlpha();
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = sourceValue > 0 ? sourceValue : 0;

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );

            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            gurobi.addVariable(
                Stringf( "x%u", targetVariable ), layer->getLb( i ), layer->getUb( i ) );

            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // The LeakyReLU is active, y = x

                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else if ( !FloatUtils::isPositive( sourceUb ) )
            {
                // The LeakyReLU is inactive, y = alpha * x
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -slope, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else
            {
                double width = sourceUb - sourceLb;
                double weight = ( sourceUb - slope * sourceLb ) / width;
                double bias = ( ( slope - 1 ) * sourceUb * sourceLb ) / width;

                /*
                  The phase of this LeakyReLU is not yet fixed.
                  For y = LeakyReLU(x), we add the following triangular relaxation:
                  1. y >= alpha * x
                  2. y >= x
                  3. y is below the line the crosses (x.lb,0) and (x.ub,x.ub)
                */

                // y >= alpha * x
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -slope, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y >= x, i.e. y - x >= 0
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -weight, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addLeqConstraint( terms, bias );
            }
        }
    }
}

void LPFormulator::addLayerToParameterisedModel( GurobiWrapper &gurobi,
                                                 const Layer *layer,
                                                 bool createVariables,
                                                 const Vector<double> &coeffs )
{
    switch ( layer->getLayerType() )
    {
    case Layer::RELU:
        addReluLayerToParameterisedLpRelaxation( gurobi, layer, createVariables, coeffs );
        break;

    case Layer::LEAKY_RELU:
        addLeakyReluLayerToParameterisedLpRelaxation( gurobi, layer, createVariables, coeffs );
        break;

    case Layer::SIGN:
        addSignLayerToParameterisedLpRelaxation( gurobi, layer, createVariables, coeffs );
        break;

    case Layer::BILINEAR:
        addBilinearLayerToParameterisedLpRelaxation( gurobi, layer, createVariables, coeffs );
        break;

    default:
        addLayerToModel( gurobi, layer, createVariables );
        break;
    }
}

void LPFormulator::addReluLayerToParameterisedLpRelaxation( GurobiWrapper &gurobi,
                                                            const Layer *layer,
                                                            bool createVariables,
                                                            const Vector<double> &coeffs )
{
    double coeff = coeffs[0];
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = sourceValue > 0 ? sourceValue : 0;

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            gurobi.addVariable( Stringf( "x%u", targetVariable ), 0, layer->getUb( i ) );

            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // The ReLU is active, y = x
                if ( sourceLb < 0 )
                    sourceLb = 0;

                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else if ( !FloatUtils::isPositive( sourceUb ) )
            {
                // The ReLU is inactive, y = 0
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else
            {
                /*
                  The phase of this ReLU is not yet fixed.

                  For y = ReLU(x), we add the following relaxation:

                  1. y >= 0
                  2. y >= x
                  2. y >= coeff * x
                  3. y is below the line the crosses (x.lb,0) and (x.ub,x.ub)
                */

                // y >= 0
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y >= x, i.e. y - x >= 0.
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y >= coeff * x, i.e. y - coeff * x >= 0 (varies continuously between y >= 0 and
                // y >= alpha * x).
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -coeff, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                /*
                         u        ul
                  y <= ----- x - -----
                       u - l     u - l
                */
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -sourceUb / ( sourceUb - sourceLb ),
                                                   Stringf( "x%u", sourceVariable ) ) );
                gurobi.addLeqConstraint( terms,
                                         ( -sourceUb * sourceLb ) / ( sourceUb - sourceLb ) );
            }
        }
    }
}

void LPFormulator::addSignLayerToParameterisedLpRelaxation( GurobiWrapper &gurobi,
                                                            const Layer *layer,
                                                            bool createVariables,
                                                            const Vector<double> &coeffs )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( layer->neuronEliminated( i ) )
            continue;

        unsigned targetVariable = layer->neuronToVariable( i );

        List<NeuronIndex> sources = layer->getActivationSources( i );
        const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
        unsigned sourceNeuron = sources.begin()->_neuron;

        if ( sourceLayer->neuronEliminated( sourceNeuron ) )
        {
            // If the source neuron has been eliminated, this neuron is constant
            double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
            double targetValue = FloatUtils::isNegative( sourceValue ) ? -1 : 1;

            gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

            continue;
        }

        unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
        double sourceLb = sourceLayer->getLb( sourceNeuron );
        double sourceUb = sourceLayer->getUb( sourceNeuron );
        String sourceName = Stringf( "x%u", sourceVariable );
        if ( createVariables && !gurobi.containsVariable( sourceName ) )
            gurobi.addVariable( sourceName, sourceLb, sourceUb );

        if ( !FloatUtils::isNegative( sourceLb ) )
        {
            // The Sign is positive, y = 1
            gurobi.addVariable( Stringf( "x%u", targetVariable ), 1, 1 );
        }
        else if ( FloatUtils::isNegative( sourceUb ) )
        {
            // The Sign is negative, y = -1
            gurobi.addVariable( Stringf( "x%u", targetVariable ), -1, -1 );
        }
        else
        {
            /*
              The phase of this Sign is not yet fixed.

              For y = Sign(x), we add the following parallelogram relaxation:

              1. y >= -1
              2. y <= -1
              3. y is below the line the crosses (x.lb,-1) and (0,1)
              4. y is above the line the crosses (0,-1) and (x.ub,1)
            */

            // -1 <= y <= 1
            gurobi.addVariable( Stringf( "x%u", targetVariable ), -1, 1 );

            /*
                     2
              y <= ----- * coeff[0] * x + 1
                    - l
              Varies continuously between y <= 1 and y <= -2/l * x + 1.
            */
            List<GurobiWrapper::Term> terms;
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term( 2.0 / sourceLb * coeffs[0],
                                               Stringf( "x%u", sourceVariable ) ) );
            gurobi.addLeqConstraint( terms, 1 );

            /*
                     2
              y >= ----- * coeffs[1] * x - 1
                     u
              Varies continuously between y >= -1 and y >= 2/u * x - 1.
            */
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term( -2.0 / sourceUb * coeffs[1],
                                               Stringf( "x%u", sourceVariable ) ) );
            gurobi.addGeqConstraint( terms, -1 );
        }
    }
}

void LPFormulator::addLeakyReluLayerToParameterisedLpRelaxation( GurobiWrapper &gurobi,
                                                                 const Layer *layer,
                                                                 bool createVariables,
                                                                 const Vector<double> &coeffs )
{
    double slope = layer->getAlpha();
    double coeff = coeffs[0];
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );
            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );
            unsigned sourceNeuron = sources.begin()->_neuron;

            if ( sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                // If the source neuron has been eliminated, this neuron is constant
                double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                double targetValue = sourceValue > 0 ? sourceValue : 0;

                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );

                continue;
            }

            unsigned sourceVariable = sourceLayer->neuronToVariable( sourceNeuron );
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );

            String sourceName = Stringf( "x%u", sourceVariable );
            if ( createVariables && !gurobi.containsVariable( sourceName ) )
                gurobi.addVariable( sourceName, sourceLb, sourceUb );

            gurobi.addVariable(
                Stringf( "x%u", targetVariable ), layer->getLb( i ), layer->getUb( i ) );

            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // The LeakyReLU is active, y = x

                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else if ( !FloatUtils::isPositive( sourceUb ) )
            {
                // The LeakyReLU is inactive, y = alpha * x
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -slope, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addEqConstraint( terms, 0 );
            }
            else
            {
                double width = sourceUb - sourceLb;
                double weight = ( sourceUb - slope * sourceLb ) / width;
                double bias = ( ( slope - 1 ) * sourceUb * sourceLb ) / width;

                /*
                  The phase of this LeakyReLU is not yet fixed.
                  For y = LeakyReLU(x), we add the following relaxation:
                  1. y >= ((1 - alpha) * coeff + alpha) * x   (varies continuously between y >=
                  alpha * x and y >= x).
                  2. y >= x
                  3. y >= alpha * x
                  4. y is below the line the crosses (x.lb,0) and (x.ub,x.ub)
                */

                // y >= ((1 - alpha) * coeff + alpha) * x
                List<GurobiWrapper::Term> terms;
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -slope - ( 1 - slope ) * coeff,
                                                   Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y >= x, i.e. y - x >= 0
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -1, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                // y >= alpha * x, i.e. y - alpha * x >= 0
                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -slope, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addGeqConstraint( terms, 0 );

                terms.clear();
                terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
                terms.append( GurobiWrapper::Term( -weight, Stringf( "x%u", sourceVariable ) ) );
                gurobi.addLeqConstraint( terms, bias );
            }
        }
    }
}

void LPFormulator::addBilinearLayerToParameterisedLpRelaxation( GurobiWrapper &gurobi,
                                                                const Layer *layer,
                                                                bool createVariables,
                                                                const Vector<double> &coeffs )
{
    for ( unsigned i = 0; i < layer->getSize(); ++i )
    {
        if ( !layer->neuronEliminated( i ) )
        {
            unsigned targetVariable = layer->neuronToVariable( i );

            List<NeuronIndex> sources = layer->getActivationSources( i );

            const Layer *sourceLayer = _layerOwner->getLayer( sources.begin()->_layer );

            Vector<double> sourceLbs;
            Vector<double> sourceUbs;
            Vector<double> sourceValues;
            Vector<unsigned> sourceNeurons;
            bool allConstant = true;
            for ( const auto &sourceIndex : sources )
            {
                unsigned sourceNeuron = sourceIndex._neuron;
                double sourceLb = sourceLayer->getLb( sourceNeuron );
                double sourceUb = sourceLayer->getUb( sourceNeuron );
                String sourceName = Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeuron ) );

                sourceNeurons.append( sourceNeuron );
                sourceLbs.append( sourceLb );
                sourceUbs.append( sourceUb );

                if ( createVariables && !gurobi.containsVariable( sourceName ) )
                    gurobi.addVariable( sourceName, sourceLb, sourceUb );

                if ( !sourceLayer->neuronEliminated( sourceNeuron ) )
                {
                    allConstant = false;
                }
                else
                {
                    double sourceValue = sourceLayer->getEliminatedNeuronValue( sourceNeuron );
                    sourceValues.append( sourceValue );
                }
            }

            if ( allConstant )
            {
                // If the both source neurons have been eliminated, this neuron is constant
                double targetValue = sourceValues[0] * sourceValues[1];
                gurobi.addVariable( Stringf( "x%u", targetVariable ), targetValue, targetValue );
                continue;
            }

            double lb = FloatUtils::infinity();
            double ub = FloatUtils::negativeInfinity();
            List<double> values = { sourceLbs[0] * sourceLbs[1],
                                    sourceLbs[0] * sourceUbs[1],
                                    sourceUbs[0] * sourceLbs[1],
                                    sourceUbs[0] * sourceUbs[1] };
            for ( const auto &v : values )
            {
                if ( v < lb )
                    lb = v;
                if ( v > ub )
                    ub = v;
            }

            gurobi.addVariable( Stringf( "x%u", targetVariable ), lb, ub );

            // Billinear linear relaxation (arXiv:2405.21063v2 [cs.LG])
            // Lower bound: out >= a_l * x + b_l * y + c_l, where
            // a_l = alpha1 * l_y + ( 1 - alpha1 ) * u_y
            // b_l = alpha1 * l_x + ( 1 - alpha1 ) * u_x
            // c_l = -alpha1 * l_x * l_y - ( 1 - alpha1 ) * u_x * u_y

            // Upper bound: out <= a_u * x + b_u * y + c_u, where
            // a_u = alpha2 * u_y + ( 1 - alpha2 ) * l_y
            // b_u = alpha2 * l_x + ( 1 - alpha2 ) * u_x
            // c_u = -alpha2 * -l_x * u_y - ( 1 - alpha2 ) * u_x * l_y

            List<GurobiWrapper::Term> terms;
            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term(
                -coeffs[0] * sourceLbs[1] - ( 1 - coeffs[0] ) * sourceUbs[1],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[0] ) ) ) );
            terms.append( GurobiWrapper::Term(
                -coeffs[0] * sourceLbs[0] - ( 1 - coeffs[0] ) * sourceUbs[0],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[1] ) ) ) );
            gurobi.addGeqConstraint( terms,
                                     -coeffs[0] * sourceLbs[0] * sourceLbs[1] -
                                         ( 1 - coeffs[0] ) * sourceUbs[0] * sourceUbs[1] );

            terms.clear();
            terms.append( GurobiWrapper::Term( 1, Stringf( "x%u", targetVariable ) ) );
            terms.append( GurobiWrapper::Term(
                -coeffs[1] * sourceUbs[1] - ( 1 - coeffs[1] ) * sourceLbs[1],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[0] ) ) ) );
            terms.append( GurobiWrapper::Term(
                -coeffs[1] * sourceLbs[0] - ( 1 - coeffs[1] ) * sourceUbs[0],
                Stringf( "x%u", sourceLayer->neuronToVariable( sourceNeurons[1] ) ) ) );
            gurobi.addLeqConstraint( terms,
                                     -coeffs[1] * sourceLbs[0] * sourceUbs[1] -
                                         ( 1 - coeffs[1] ) * sourceUbs[0] * sourceLbs[1] );
        }
    }
}

void LPFormulator::addPolyognalTighteningsToLpRelaxation(
    GurobiWrapper &gurobi,
    const Map<unsigned, Layer *> &layers,
    unsigned firstLayer,
    unsigned lastLayer,
    const Vector<PolygonalTightening> &polygonal_tightenings )
{
    List<GurobiWrapper::Term> terms;
    for ( const auto &tightening : polygonal_tightenings )
    {
        Map<NeuronIndex, double> neuronToCoefficient = tightening._neuronToCoefficient;
        PolygonalTightening::PolygonalBoundType type = tightening._type;
        double value = tightening._value;

        bool out_of_bounds = false;
        for ( const auto &pair : neuronToCoefficient )
        {
            unsigned currentLayerIndex = pair.first._layer;
            if ( currentLayerIndex < firstLayer || currentLayerIndex > lastLayer )
            {
                out_of_bounds = true;
            }
        }
        if ( out_of_bounds )
        {
            continue;
        }

        terms.clear();
        for ( const auto &pair : neuronToCoefficient )
        {
            unsigned currentLayerIndex = pair.first._layer;
            unsigned i = pair.first._neuron;
            double coeff = pair.second;
            Layer *layer = layers[currentLayerIndex];

            if ( !layer->neuronEliminated( i ) )
            {
                unsigned variable = layer->neuronToVariable( i );
                String variableName = Stringf( "x%u", variable );
                if ( !gurobi.containsVariable( variableName ) )
                {
                    gurobi.addVariable( variableName, layer->getLb( i ), layer->getUb( i ) );
                }
                terms.append( GurobiWrapper::Term( coeff, Stringf( "x%u", variable ) ) );
            }
            else
            {
                value -= coeff * layer->getEliminatedNeuronValue( i );
            }
        }

        if ( type == PolygonalTightening::UB )
        {
            gurobi.addLeqConstraint( terms, value );
        }
        else
        {
            gurobi.addGeqConstraint( terms, value );
        }
    }
}

void LPFormulator::setCutoff( double cutoff )
{
    _cutoffInUse = true;
    _cutoffValue = cutoff;
}

} // namespace NLR
