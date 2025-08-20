/*********************                                                        */
/*! \file NetworkLevelReasoner.cpp
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

#include "NetworkLevelReasoner.h"

#include "AbsoluteValueConstraint.h"
#include "Debug.h"
#include "FloatUtils.h"
#include "InfeasibleQueryException.h"
#include "IterativePropagator.h"
#include "MILPFormulator.h"
#include "MStringf.h"
#include "MarabouError.h"
#include "MatrixMultiplication.h"
#include "MaxConstraint.h"
#include "NLRError.h"
#include "Options.h"
#include "Query.h"
#include "ReluConstraint.h"
#include "SignConstraint.h"

#include <cstring>

#define NLR_LOG( x, ... ) LOG( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING, "NLR: %s\n", x )

namespace NLR {

NetworkLevelReasoner::NetworkLevelReasoner()
    : _tableau( NULL )
    , _deepPolyAnalysis( nullptr )
{
}

NetworkLevelReasoner::~NetworkLevelReasoner()
{
    freeMemoryIfNeeded();
}

bool NetworkLevelReasoner::functionTypeSupported( PiecewiseLinearFunctionType type )
{
    if ( type == PiecewiseLinearFunctionType::RELU )
        return true;

    if ( type == PiecewiseLinearFunctionType::ABSOLUTE_VALUE )
        return true;

    if ( type == PiecewiseLinearFunctionType::SIGN )
        return true;

    return false;
}

void NetworkLevelReasoner::addLayer( unsigned layerIndex, Layer::Type type, unsigned layerSize )
{
    Layer *layer = new Layer( layerIndex, type, layerSize, this );
    _layerIndexToLayer[layerIndex] = layer;
}

void NetworkLevelReasoner::addLayerDependency( unsigned sourceLayer, unsigned targetLayer )
{
    _layerIndexToLayer[targetLayer]->addSourceLayer( sourceLayer,
                                                     _layerIndexToLayer[sourceLayer]->getSize() );
}

void NetworkLevelReasoner::removeLayerDependency( unsigned sourceLayer, unsigned targetLayer )
{
    _layerIndexToLayer[targetLayer]->removeSourceLayer( sourceLayer );
}

void NetworkLevelReasoner::computeSuccessorLayers()
{
    for ( unsigned i = 0; i < _layerIndexToLayer.size(); ++i )
    {
        for ( const auto &pair : _layerIndexToLayer[i]->getSourceLayers() )
        {
            _layerIndexToLayer[pair.first]->addSuccessorLayer( i );
        }
    }
}

void NetworkLevelReasoner::setWeight( unsigned sourceLayer,
                                      unsigned sourceNeuron,
                                      unsigned targetLayer,
                                      unsigned targetNeuron,
                                      double weight )
{
    _layerIndexToLayer[targetLayer]->setWeight( sourceLayer, sourceNeuron, targetNeuron, weight );
}

void NetworkLevelReasoner::setBias( unsigned layer, unsigned neuron, double bias )
{
    _layerIndexToLayer[layer]->setBias( neuron, bias );
}

void NetworkLevelReasoner::addActivationSource( unsigned sourceLayer,
                                                unsigned sourceNeuron,
                                                unsigned targetLayer,
                                                unsigned targetNeuron )
{
    _layerIndexToLayer[targetLayer]->addActivationSource( sourceLayer, sourceNeuron, targetNeuron );
}

const Layer *NetworkLevelReasoner::getLayer( unsigned index ) const
{
    return _layerIndexToLayer[index];
}

Layer *NetworkLevelReasoner::getLayer( unsigned index )
{
    return _layerIndexToLayer[index];
}

void NetworkLevelReasoner::evaluate( double *input, double *output )
{
    _layerIndexToLayer[0]->setAssignment( input );
    for ( unsigned i = 1; i < _layerIndexToLayer.size(); ++i )
        _layerIndexToLayer[i]->computeAssignment();

    const Layer *outputLayer = _layerIndexToLayer[_layerIndexToLayer.size() - 1];
    memcpy( output, outputLayer->getAssignment(), sizeof( double ) * outputLayer->getSize() );
}

void NetworkLevelReasoner::concretizeInputAssignment( Map<unsigned, double> &assignment )
{
    Layer *inputLayer = _layerIndexToLayer[0];
    ASSERT( inputLayer->getLayerType() == Layer::INPUT );

    unsigned inputLayerSize = inputLayer->getSize();
    ASSERT( inputLayerSize > 0 );

    double *input = new double[inputLayerSize];

    // First obtain the input assignment from the _tableau
    for ( unsigned index = 0; index < inputLayerSize; ++index )
    {
        if ( !inputLayer->neuronEliminated( index ) )
        {
            unsigned variable = inputLayer->neuronToVariable( index );
            double value = _tableau->getValue( variable );
            input[index] = value;
            assignment[variable] = value;
        }
        else
            input[index] = inputLayer->getEliminatedNeuronValue( index );
    }

    _layerIndexToLayer[0]->setAssignment( input );

    // Evaluate layers iteratively and store the results in "assignment"
    for ( unsigned i = 1; i < _layerIndexToLayer.size(); ++i )
    {
        Layer *currentLayer = _layerIndexToLayer[i];
        currentLayer->computeAssignment();
        for ( unsigned index = 0; index < currentLayer->getSize(); ++index )
        {
            if ( !currentLayer->neuronEliminated( index ) )
                assignment[currentLayer->neuronToVariable( index )] =
                    currentLayer->getAssignment( index );
        }
    }

    delete[] input;
}

void NetworkLevelReasoner::simulate( Vector<Vector<double>> *input )
{
    _layerIndexToLayer[0]->setSimulations( input );
    for ( unsigned i = 1; i < _layerIndexToLayer.size(); ++i )
        _layerIndexToLayer[i]->computeSimulations();
}

void NetworkLevelReasoner::setNeuronVariable( NeuronIndex index, unsigned variable )
{
    _layerIndexToLayer[index._layer]->setNeuronVariable( index._neuron, variable );
}

void NetworkLevelReasoner::receiveTighterBound( Tightening tightening )
{
    _boundTightenings.append( tightening );
}

void NetworkLevelReasoner::getConstraintTightenings( List<Tightening> &tightenings )
{
    tightenings = _boundTightenings;
    _boundTightenings.clear();
}

void NetworkLevelReasoner::clearConstraintTightenings()
{
    _boundTightenings.clear();
}

void NetworkLevelReasoner::receivePolygonalTighterBound( PolygonalTightening polygonalTightening )
{
    _polygonalBoundTightenings.append( polygonalTightening );
}

void NetworkLevelReasoner::getConstraintPolygonalTightenings(
    List<PolygonalTightening> &polygonalTightenings )
{
    polygonalTightenings = _polygonalBoundTightenings;
    _polygonalBoundTightenings.clear();
}

void NetworkLevelReasoner::clearConstraintPolygonalTightenings()
{
    _polygonalBoundTightenings.clear();
}

void NetworkLevelReasoner::symbolicBoundPropagation()
{
    for ( unsigned i = 0; i < _layerIndexToLayer.size(); ++i )
        _layerIndexToLayer[i]->computeSymbolicBounds();
}

void NetworkLevelReasoner::parameterisedSymbolicBoundPropagation( const Vector<double> &coeffs )
{
    Map<unsigned, Vector<double>> layerIndicesToParameters = getParametersForLayers( coeffs );
    for ( unsigned i = 0; i < _layerIndexToLayer.size(); ++i )
    {
        const Vector<double> &currentLayerCoeffs = layerIndicesToParameters[i];
        _layerIndexToLayer[i]->computeParameterisedSymbolicBounds( currentLayerCoeffs, true );
    }
}

void NetworkLevelReasoner::deepPolyPropagation()
{
    if ( _deepPolyAnalysis == nullptr )
        _deepPolyAnalysis = std::unique_ptr<DeepPolyAnalysis>( new DeepPolyAnalysis( this ) );
    _deepPolyAnalysis->run();
}

void NetworkLevelReasoner::lpRelaxationPropagation()
{
    LPFormulator lpFormulator( this );
    lpFormulator.setCutoff( 0 );

    if ( Options::get()->getMILPSolverBoundTighteningType() ==
             MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_ONCE ||
         Options::get()->getMILPSolverBoundTighteningType() ==
             MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_CONVERGE )
        lpFormulator.optimizeBoundsWithLpRelaxation( _layerIndexToLayer, true );
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PREIMAGE_APPROX )
        optimizeBoundsWithPreimageApproximation( lpFormulator );
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
                  MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_INVPROP ||
              Options::get()->getMILPSolverBoundTighteningType() ==
                  MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_RANDOM ||
              Options::get()->getMILPSolverBoundTighteningType() ==
                  MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_GRADIENT ||
              Options::get()->getMILPSolverBoundTighteningType() ==
                  MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_BBPS )
        optimizeBoundsWithPMNR( lpFormulator );
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::LP_RELAXATION )
        lpFormulator.optimizeBoundsWithLpRelaxation( _layerIndexToLayer );
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::LP_RELAXATION_INCREMENTAL )
        lpFormulator.optimizeBoundsWithIncrementalLpRelaxation( _layerIndexToLayer );
}

void NetworkLevelReasoner::optimizeBoundsWithPreimageApproximation( LPFormulator &lpFormulator )
{
    const Vector<double> &optimalCoeffs = OptimalParameterisedSymbolicBoundTightening();
    Map<unsigned, Vector<double>> layerIndicesToParameters =
        getParametersForLayers( optimalCoeffs );
    lpFormulator.optimizeBoundsWithLpRelaxation(
        _layerIndexToLayer, false, layerIndicesToParameters );
    lpFormulator.optimizeBoundsWithLpRelaxation(
        _layerIndexToLayer, true, layerIndicesToParameters );
}

void NetworkLevelReasoner::optimizeBoundsWithPMNR( LPFormulator &lpFormulator )
{
    const Vector<PolygonalTightening> &polygonalTightenings =
        OptimizeParameterisedPolygonalTightening();
    unsigned parameterCount = getNumberOfParameters();
    const Vector<double> &coeffs = Vector<double>( parameterCount, 0 );
    Map<unsigned, Vector<double>> layerIndicesToParameters = getParametersForLayers( coeffs );
    lpFormulator.optimizeBoundsWithLpRelaxation(
        _layerIndexToLayer, false, layerIndicesToParameters, polygonalTightenings );
    lpFormulator.optimizeBoundsWithLpRelaxation(
        _layerIndexToLayer, true, layerIndicesToParameters, polygonalTightenings );
}

void NetworkLevelReasoner::LPTighteningForOneLayer( unsigned targetIndex )
{
    LPFormulator lpFormulator( this );
    lpFormulator.setCutoff( 0 );

    if ( Options::get()->getMILPSolverBoundTighteningType() ==
         MILPSolverBoundTighteningType::LP_RELAXATION )
        lpFormulator.optimizeBoundsOfOneLayerWithLpRelaxation( _layerIndexToLayer, targetIndex );

    // TODO: implement for LP_RELAXATION_INCREMENTAL
}

void NetworkLevelReasoner::MILPPropagation()
{
    MILPFormulator milpFormulator( this );
    milpFormulator.setCutoff( 0 );

    if ( Options::get()->getMILPSolverBoundTighteningType() ==
         MILPSolverBoundTighteningType::MILP_ENCODING )
        milpFormulator.optimizeBoundsWithMILPEncoding( _layerIndexToLayer );
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::MILP_ENCODING_INCREMENTAL )
        milpFormulator.optimizeBoundsWithIncrementalMILPEncoding( _layerIndexToLayer );
}

void NetworkLevelReasoner::MILPTighteningForOneLayer( unsigned targetIndex )
{
    MILPFormulator milpFormulator( this );
    milpFormulator.setCutoff( 0 );

    if ( Options::get()->getMILPSolverBoundTighteningType() ==
         MILPSolverBoundTighteningType::MILP_ENCODING )
        milpFormulator.optimizeBoundsOfOneLayerWithMILPEncoding( _layerIndexToLayer, targetIndex );

    // TODO: implement for MILP_ENCODING_INCREMENTAL
}

void NetworkLevelReasoner::iterativePropagation()
{
    IterativePropagator iterativePropagator( this );
    iterativePropagator.setCutoff( 0 );
    iterativePropagator.optimizeBoundsWithIterativePropagation( _layerIndexToLayer );
}

void NetworkLevelReasoner::intervalArithmeticBoundPropagation()
{
    for ( unsigned i = 1; i < _layerIndexToLayer.size(); ++i )
        _layerIndexToLayer[i]->computeIntervalArithmeticBounds();
}

void NetworkLevelReasoner::freeMemoryIfNeeded()
{
    for ( const auto &layer : _layerIndexToLayer )
        delete layer.second;
    _layerIndexToLayer.clear();
}

void NetworkLevelReasoner::storeIntoOther( NetworkLevelReasoner &other ) const
{
    other.freeMemoryIfNeeded();

    for ( const auto &layer : _layerIndexToLayer )
    {
        const Layer *thisLayer = layer.second;
        Layer *newLayer = new Layer( thisLayer );
        newLayer->setLayerOwner( &other );
        other._layerIndexToLayer[newLayer->getLayerIndex()] = newLayer;
    }

    // Other has fresh copies of the PLCs, so its topological order
    // shouldn't contain any stale data
    other._constraintsInTopologicalOrder.clear();
}

void NetworkLevelReasoner::updateVariableIndices( const Map<unsigned, unsigned> &oldIndexToNewIndex,
                                                  const Map<unsigned, unsigned> &mergedVariables )
{
    for ( auto &layer : _layerIndexToLayer )
        layer.second->updateVariableIndices( oldIndexToNewIndex, mergedVariables );
}

void NetworkLevelReasoner::obtainCurrentBounds( const Query &inputQuery )
{
    for ( const auto &layer : _layerIndexToLayer )
        layer.second->obtainCurrentBounds( inputQuery );
}

void NetworkLevelReasoner::obtainCurrentBounds()
{
    ASSERT( _tableau );
    for ( const auto &layer : _layerIndexToLayer )
        layer.second->obtainCurrentBounds();
}

void NetworkLevelReasoner::setTableau( const ITableau *tableau )
{
    _tableau = tableau;
}

const ITableau *NetworkLevelReasoner::getTableau() const
{
    return _tableau;
}

void NetworkLevelReasoner::eliminateVariable( unsigned variable, double value )
{
    for ( auto &layer : _layerIndexToLayer )
        layer.second->eliminateVariable( variable, value );
}

void NetworkLevelReasoner::dumpTopology( bool dumpLayerDetails ) const
{
    printf( "Number of layers: %u. Sizes:\n", _layerIndexToLayer.size() );
    for ( unsigned i = 0; i < _layerIndexToLayer.size(); ++i )
    {
        printf( "\tLayer %u: %u \t[%s]",
                i,
                _layerIndexToLayer[i]->getSize(),
                Layer::typeToString( _layerIndexToLayer[i]->getLayerType() ).ascii() );
        printf( "\tSource layers:" );
        for ( const auto &sourceLayer : _layerIndexToLayer[i]->getSourceLayers() )
            printf( " %u", sourceLayer.first );
        printf( "\n" );
    }
    if ( dumpLayerDetails )
        for ( const auto &layer : _layerIndexToLayer )
            layer.second->dump();
}

unsigned NetworkLevelReasoner::getNumberOfLayers() const
{
    return _layerIndexToLayer.size();
}

List<PiecewiseLinearConstraint *> NetworkLevelReasoner::getConstraintsInTopologicalOrder()
{
    return _constraintsInTopologicalOrder;
}

void NetworkLevelReasoner::addConstraintInTopologicalOrder( PiecewiseLinearConstraint *constraint )
{
    _constraintsInTopologicalOrder.append( constraint );
}

void NetworkLevelReasoner::removeConstraintFromTopologicalOrder(
    PiecewiseLinearConstraint *constraint )
{
    if ( _constraintsInTopologicalOrder.exists( constraint ) )
        _constraintsInTopologicalOrder.erase( constraint );
}

void NetworkLevelReasoner::encodeAffineLayers( Query &inputQuery )
{
    for ( const auto &pair : _layerIndexToLayer )
        if ( pair.second->getLayerType() == Layer::WEIGHTED_SUM )
            generateQueryForWeightedSumLayer( inputQuery, pair.second );
}

void NetworkLevelReasoner::generateQuery( Query &result )
{
    // Number of variables
    unsigned numberOfVariables = 0;
    for ( const auto &it : _layerIndexToLayer )
    {
        unsigned maxVariable = it.second->getMaxVariable();
        if ( maxVariable > numberOfVariables )
            numberOfVariables = maxVariable;
    }
    ++numberOfVariables;
    result.setNumberOfVariables( numberOfVariables );

    // Handle the various layers
    for ( const auto &it : _layerIndexToLayer )
        generateQueryForLayer( result, *it.second );

    // Mark the input variables
    const Layer *inputLayer = _layerIndexToLayer[0];
    for ( unsigned i = 0; i < inputLayer->getSize(); ++i )
        result.markInputVariable( inputLayer->neuronToVariable( i ), i );

    // Mark the output variables
    const Layer *outputLayer = _layerIndexToLayer[_layerIndexToLayer.size() - 1];
    for ( unsigned i = 0; i < outputLayer->getSize(); ++i )
        result.markOutputVariable( outputLayer->neuronToVariable( i ), i );

    // Store any known bounds of all layers
    for ( const auto &layerPair : _layerIndexToLayer )
    {
        const Layer *layer = layerPair.second;
        for ( unsigned i = 0; i < layer->getSize(); ++i )
        {
            unsigned variable = layer->neuronToVariable( i );
            result.setLowerBound( variable, layer->getLb( i ) );
            result.setUpperBound( variable, layer->getUb( i ) );
        }
    }
}

void NetworkLevelReasoner::reindexNeurons()
{
    unsigned index = 0;
    for ( auto &it : _layerIndexToLayer )
    {
        for ( unsigned i = 0; i < it.second->getSize(); ++i )
        {
            it.second->setNeuronVariable( i, index );
            ++index;
        }
    }
}

void NetworkLevelReasoner::generateQueryForLayer( Query &inputQuery, const Layer &layer )
{
    switch ( layer.getLayerType() )
    {
    case Layer::INPUT:
        break;

    case Layer::WEIGHTED_SUM:
        generateQueryForWeightedSumLayer( inputQuery, layer );
        break;

    case Layer::RELU:
        generateQueryForReluLayer( inputQuery, layer );
        break;

    case Layer::SIGMOID:
        generateQueryForSigmoidLayer( inputQuery, layer );
        break;

    case Layer::SIGN:
        generateQueryForSignLayer( inputQuery, layer );
        break;

    case Layer::ABSOLUTE_VALUE:
        generateQueryForAbsoluteValueLayer( inputQuery, layer );
        break;

    case Layer::MAX:
        generateQueryForMaxLayer( inputQuery, layer );
        break;

    default:
        throw NLRError( NLRError::LAYER_TYPE_NOT_SUPPORTED,
                        Stringf( "Layer %u not yet supported", layer.getLayerType() ).ascii() );
        break;
    }
}

void NetworkLevelReasoner::generateQueryForReluLayer( Query &inputQuery, const Layer &layer )
{
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        NeuronIndex sourceIndex = *layer.getActivationSources( i ).begin();
        const Layer *sourceLayer = _layerIndexToLayer[sourceIndex._layer];
        ReluConstraint *relu = new ReluConstraint(
            sourceLayer->neuronToVariable( sourceIndex._neuron ), layer.neuronToVariable( i ) );
        inputQuery.addPiecewiseLinearConstraint( relu );
    }
}

void NetworkLevelReasoner::generateQueryForSigmoidLayer( Query &inputQuery, const Layer &layer )
{
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        NeuronIndex sourceIndex = *layer.getActivationSources( i ).begin();
        const Layer *sourceLayer = _layerIndexToLayer[sourceIndex._layer];
        SigmoidConstraint *sigmoid = new SigmoidConstraint(
            sourceLayer->neuronToVariable( sourceIndex._neuron ), layer.neuronToVariable( i ) );
        inputQuery.addNonlinearConstraint( sigmoid );
    }
}

void NetworkLevelReasoner::generateQueryForSignLayer( Query &inputQuery, const Layer &layer )
{
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        NeuronIndex sourceIndex = *layer.getActivationSources( i ).begin();
        const Layer *sourceLayer = _layerIndexToLayer[sourceIndex._layer];
        SignConstraint *sign = new SignConstraint(
            sourceLayer->neuronToVariable( sourceIndex._neuron ), layer.neuronToVariable( i ) );
        inputQuery.addPiecewiseLinearConstraint( sign );
    }
}

void NetworkLevelReasoner::generateQueryForAbsoluteValueLayer( Query &inputQuery,
                                                               const Layer &layer )
{
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        NeuronIndex sourceIndex = *layer.getActivationSources( i ).begin();
        const Layer *sourceLayer = _layerIndexToLayer[sourceIndex._layer];
        AbsoluteValueConstraint *absoluteValue = new AbsoluteValueConstraint(
            sourceLayer->neuronToVariable( sourceIndex._neuron ), layer.neuronToVariable( i ) );
        inputQuery.addPiecewiseLinearConstraint( absoluteValue );
    }
}

void NetworkLevelReasoner::generateQueryForMaxLayer( Query &inputQuery, const Layer &layer )
{
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        Set<unsigned> elements;
        for ( const auto &source : layer.getActivationSources( i ) )
        {
            const Layer *sourceLayer = _layerIndexToLayer[source._layer];
            elements.insert( sourceLayer->neuronToVariable( source._neuron ) );
        }

        MaxConstraint *max = new MaxConstraint( layer.neuronToVariable( i ), elements );
        inputQuery.addPiecewiseLinearConstraint( max );
    }
}

void NetworkLevelReasoner::generateQueryForWeightedSumLayer( Query &inputQuery, const Layer &layer )
{
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        Equation eq;
        eq.setScalar( -layer.getBias( i ) );
        eq.addAddend( -1, layer.neuronToVariable( i ) );

        for ( const auto &it : layer.getSourceLayers() )
        {
            const Layer *sourceLayer = _layerIndexToLayer[it.first];

            for ( unsigned j = 0; j < sourceLayer->getSize(); ++j )
            {
                double coefficient = layer.getWeight( sourceLayer->getLayerIndex(), j, i );
                if ( !FloatUtils::isZero( coefficient ) )
                    eq.addAddend( coefficient, sourceLayer->neuronToVariable( j ) );
            }
        }
        inputQuery.addEquation( eq );
    }
}

void NetworkLevelReasoner::generateLinearExpressionForWeightedSumLayer(
    Map<unsigned, LinearExpression> &variableToExpression,
    const Layer &layer )
{
    ASSERT( layer.getLayerType() == Layer::WEIGHTED_SUM );
    for ( unsigned i = 0; i < layer.getSize(); ++i )
    {
        LinearExpression exp;
        exp._constant = layer.getBias( i );
        for ( const auto &it : layer.getSourceLayers() )
        {
            const Layer *sourceLayer = _layerIndexToLayer[it.first];
            for ( unsigned j = 0; j < sourceLayer->getSize(); ++j )
            {
                double coefficient = layer.getWeight( sourceLayer->getLayerIndex(), j, i );
                if ( !FloatUtils::isZero( coefficient ) )
                {
                    unsigned var = sourceLayer->neuronToVariable( j );
                    if ( exp._addends.exists( var ) )
                        exp._addends[var] += coefficient;
                    else
                        exp._addends[var] = coefficient;
                }
            }
        }
        variableToExpression[layer.neuronToVariable( i )] = exp;
    }
}

/*
    Initialize and fill ReLU Constraint to previous bias map
    for BaBSR Heuristic
*/
void NetworkLevelReasoner::initializePreviousBiasMap()
{
    // Clear the previous bias map
    _previousBiases.clear();

    // Track accumulated ReLU neurons across layers
    unsigned accumulatedNeurons = 0;

    // Iterate through layers to find ReLU layers and their sources
    for ( const auto &layerPair : _layerIndexToLayer )
    {
        const NLR::Layer *layer = layerPair.second;

        if ( layer->getLayerType() == Layer::RELU )
        {
            // Get source layer info
            const auto &sourceLayers = layer->getSourceLayers();
            unsigned sourceLayerIndex = sourceLayers.begin()->first;
            const NLR::Layer *sourceLayer = getLayer( sourceLayerIndex );

            // Match ReLU constraints to their source layer biases
            unsigned layerSize = layer->getSize();

            // Iterate through constraints
            auto constraintIterator = _constraintsInTopologicalOrder.begin();
            for ( unsigned currentIndex = 0;
                  currentIndex < accumulatedNeurons &&
                  constraintIterator != _constraintsInTopologicalOrder.end();
                  ++currentIndex, ++constraintIterator )
            {
            }

            // Now at correct starting position
            for ( unsigned i = 0;
                  i < layerSize && constraintIterator != _constraintsInTopologicalOrder.end();
                  ++i, ++constraintIterator )
            {
                if ( auto reluConstraint =
                         dynamic_cast<const ReluConstraint *>( *constraintIterator ) )
                {
                    // Store bias in map
                    _previousBiases[reluConstraint] = sourceLayer->getBias( i );
                }
            }

            accumulatedNeurons += layerSize;
        }
    }
}

/*
    Get previous layer bias of a ReLU neuron
    for BaBSR Heuristic
*/
double NetworkLevelReasoner::getPreviousBias( const ReluConstraint *reluConstraint ) const
{
    // Initialize map if empty
    if ( _previousBiases.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializePreviousBiasMap();
    }

    // Look up pre-computed bias
    if ( !_previousBiases.exists( reluConstraint ) )
    {
        throw NLRError( NLRError::RELU_NOT_FOUND, "ReluConstraint not found in bias map." );
    }

    return _previousBiases[reluConstraint];
}

Vector<double> NetworkLevelReasoner::getOutputLayerSymbolicLb( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _outputLayerSymbolicLb.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_outputLayerSymbolicLb.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputLayerSymbolicLb[layerIndex];
}

Vector<double> NetworkLevelReasoner::getOutputLayerSymbolicUb( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _outputLayerSymbolicUb.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_outputLayerSymbolicUb.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputLayerSymbolicUb[layerIndex];
}

Vector<double> NetworkLevelReasoner::getOutputLayerSymbolicLowerBias( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _outputLayerSymbolicLowerBias.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_outputLayerSymbolicLowerBias.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputLayerSymbolicLowerBias[layerIndex];
}

Vector<double> NetworkLevelReasoner::getOutputLayerSymbolicUpperBias( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _outputLayerSymbolicUpperBias.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_outputLayerSymbolicUpperBias.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputLayerSymbolicUpperBias[layerIndex];
}

Vector<double> NetworkLevelReasoner::getSymbolicLbInTermsOfPredecessor( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _symbolicLbInTermsOfPredecessor.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_symbolicLbInTermsOfPredecessor.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _symbolicLbInTermsOfPredecessor[layerIndex];
}

Vector<double> NetworkLevelReasoner::getSymbolicUbInTermsOfPredecessor( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _symbolicUbInTermsOfPredecessor.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_symbolicUbInTermsOfPredecessor.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _symbolicUbInTermsOfPredecessor[layerIndex];
}

Vector<double>
NetworkLevelReasoner::getSymbolicLowerBiasInTermsOfPredecessor( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _symbolicLowerBiasInTermsOfPredecessor.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_symbolicLowerBiasInTermsOfPredecessor.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _symbolicLowerBiasInTermsOfPredecessor[layerIndex];
}

Vector<double>
NetworkLevelReasoner::getSymbolicUpperBiasInTermsOfPredecessor( unsigned layerIndex ) const
{
    // Initialize map if empty.
    if ( _symbolicUpperBiasInTermsOfPredecessor.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeSymbolicBoundsMaps();
    }

    if ( !_symbolicUpperBiasInTermsOfPredecessor.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _symbolicUpperBiasInTermsOfPredecessor[layerIndex];
}

Map<NeuronIndex, double> NetworkLevelReasoner::getBBPSBranchingPoint( NeuronIndex index ) const
{
    // Initialize map if empty.
    if ( _neuronToBBPSBranchingPoints.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeBBPSMaps();
    }

    if ( !_neuronToBBPSBranchingPoints.exists( index ) )
    {
        throw NLRError( NLRError::NEURON_NOT_FOUND,
                        "Neuron not found in BBPS branching points map." );
    }

    return _neuronToBBPSBranchingPoints[index];
}

double NetworkLevelReasoner::getBBPSScore( NeuronIndex index ) const
{
    // Initialize map if empty.
    if ( _neuronToBBPSScores.empty() )
    {
        const_cast<NetworkLevelReasoner *>( this )->initializeBBPSMaps();
    }

    if ( !_neuronToBBPSScores.exists( index ) )
    {
        throw NLRError( NLRError::NEURON_NOT_FOUND, "Neuron not found in BBPS scores map." );
    }

    return _neuronToBBPSScores[index];
}

unsigned
NetworkLevelReasoner::mergeConsecutiveWSLayers( const Map<unsigned, double> &lowerBounds,
                                                const Map<unsigned, double> &upperBounds,
                                                const Set<unsigned> &varsInUnhandledConstraints,
                                                Map<unsigned, LinearExpression> &eliminatedNeurons )
{
    // Iterate over all layers, except the input layer
    unsigned layer = 1;

    unsigned numberOfMergedLayers = 0;
    while ( layer < _layerIndexToLayer.size() )
    {
        if ( suitableForMerging( layer, lowerBounds, upperBounds, varsInUnhandledConstraints ) )
        {
            NLR_LOG( Stringf( "Merging layer %u with its predecessor...", layer ).ascii() );
            mergeWSLayers( layer, eliminatedNeurons );
            ++numberOfMergedLayers;
            NLR_LOG( Stringf( "Merging layer %u with its predecessor - done", layer ).ascii() );
        }
        else
            ++layer;
    }
    return numberOfMergedLayers;
}

bool NetworkLevelReasoner::suitableForMerging(
    unsigned secondLayerIndex,
    const Map<unsigned, double> &lowerBounds,
    const Map<unsigned, double> &upperBounds,
    const Set<unsigned> &varsInConstraintsUnhandledByNLR )
{
    NLR_LOG( Stringf( "Checking whether layer %u is suitable for merging...", secondLayerIndex )
                 .ascii() );

    /*
      The given layer index is a candidate layer. We now check whether
      it is an eligible second WS layer that can be merged with its
      predecessor
    */
    const Layer *secondLayer = _layerIndexToLayer[secondLayerIndex];

    // Layer should be a Weighted Sum layer
    if ( secondLayer->getLayerType() != Layer::WEIGHTED_SUM )
        return false;

    // Layer should have a single source
    if ( secondLayer->getSourceLayers().size() != 1 )
        return false;

    // Grab the predecessor layer
    unsigned firstLayerIndex = secondLayer->getSourceLayers().begin()->first;
    const Layer *firstLayer = _layerIndexToLayer[firstLayerIndex];

    // First layer should also be a weighted sum
    if ( firstLayer->getLayerType() != Layer::WEIGHTED_SUM )
        return false;

    // First layer should not feed into any other layer
    unsigned count = 0;
    for ( unsigned i = 0; i < getNumberOfLayers(); ++i )
    {
        const Layer *layer = _layerIndexToLayer[i];

        if ( layer->getSourceLayers().exists( firstLayerIndex ) )
            ++count;
    }
    if ( count > 1 )
        return false;

    // If there are bounds on the predecessor layer or if
    // the predecessor layer participates in any constraints
    // (equations, piecewise-linear constraints, nonlinear-constraints)
    // unaccounted for in the NLR cannot merge
    for ( unsigned i = 0; i < firstLayer->getSize(); ++i )
    {
        unsigned variable = firstLayer->neuronToVariable( i );
        if ( ( lowerBounds.exists( variable ) && FloatUtils::isFinite( lowerBounds[variable] ) ) ||
             ( upperBounds.exists( variable ) && FloatUtils::isFinite( upperBounds[variable] ) ) ||
             varsInConstraintsUnhandledByNLR.exists( variable ) )
            return false;
    }
    return true;
}

const Vector<double> NetworkLevelReasoner::OptimalParameterisedSymbolicBoundTightening()
{
    // Search over coeffs in [0, 1]^number_of_parameters with projected grdient descent.
    unsigned maxIterations =
        GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_MAX_ITERATIONS;
    double stepSize = GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_STEP_SIZE;
    double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS;
    double weightDecay = GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_WEIGHT_DECAY;
    double lr = GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_LEARNING_RATE;
    unsigned dimension = getNumberOfParameters();
    bool maximize = false;

    Vector<double> lowerBounds( dimension );
    Vector<double> upperBounds( dimension );
    for ( size_t j = 0; j < dimension; ++j )
    {
        lowerBounds[j] = 0;
        upperBounds[j] = 1;
    }

    // Initialize initial guess uniformly.
    Vector<double> guess( dimension );
    std::mt19937_64 rng( GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_RANDOM_SEED );
    std::uniform_real_distribution<double> dis( 0, 1 );
    for ( size_t j = 0; j < dimension; ++j )
    {
        double lb = lowerBounds[j];
        double ub = upperBounds[j];
        guess[j] = lb + dis( rng ) * ( ub - lb );
    }

    Vector<Vector<double>> candidates( dimension );
    Vector<double> gradient( dimension );

    for ( size_t i = 0; i < maxIterations; ++i )
    {
        double currentCost = EstimateVolume( guess );
        for ( size_t j = 0; j < dimension; ++j )
        {
            candidates[j] = Vector<double>( guess );
            candidates[j][j] += stepSize;

            if ( candidates[j][j] > upperBounds[j] || candidates[j][j] < lowerBounds[j] )
            {
                gradient[j] = 0;
                continue;
            }

            size_t sign = ( maximize == false ? 1 : -1 );
            double cost = EstimateVolume( candidates[j] );
            gradient[j] = sign * ( cost - currentCost ) / stepSize + weightDecay * guess[j];
        }

        bool gradientIsZero = true;
        for ( size_t j = 0; j < dimension; ++j )
        {
            if ( FloatUtils::abs( gradient[j] ) > epsilon )
            {
                gradientIsZero = false;
            }
        }
        if ( gradientIsZero )
        {
            break;
        }

        for ( size_t j = 0; j < dimension; ++j )
        {
            guess[j] -= lr * gradient[j];

            if ( guess[j] > upperBounds[j] )
            {
                guess[j] = upperBounds[j];
            }

            if ( guess[j] < lowerBounds[j] )
            {
                guess[j] = lowerBounds[j];
            }
        }
    }

    const Vector<double> optimalCoeffs( guess );
    return optimalCoeffs;
}

double NetworkLevelReasoner::EstimateVolume( const Vector<double> &coeffs )
{
    // First, run parameterised symbolic bound propagation.
    Map<unsigned, Vector<double>> layerIndicesToParameters = getParametersForLayers( coeffs );
    for ( unsigned i = 0; i < _layerIndexToLayer.size(); ++i )
    {
        ASSERT( _layerIndexToLayer.exists( i ) );
        const Vector<double> &currentLayerCoeffs = layerIndicesToParameters[i];
        _layerIndexToLayer[i]->computeParameterisedSymbolicBounds( currentLayerCoeffs );
    }

    std::mt19937_64 rng( GlobalConfiguration::VOLUME_ESTIMATION_RANDOM_SEED );
    double logBoxVolume = 0;
    double sigmoidSum = 0;

    unsigned inputLayerIndex = 0;
    unsigned outputLayerIndex = _layerIndexToLayer.size() - 1;
    Layer *inputLayer = _layerIndexToLayer[inputLayerIndex];
    Layer *outputLayer = _layerIndexToLayer[outputLayerIndex];

    // Calculate volume of input variables' bounding box.
    for ( unsigned index = 0; index < inputLayer->getSize(); ++index )
    {
        if ( inputLayer->neuronEliminated( index ) )
            continue;

        double lb = inputLayer->getLb( index );
        double ub = inputLayer->getUb( index );

        if ( lb == ub )
            continue;

        logBoxVolume += std::log( ub - lb );
    }

    for ( unsigned i = 0; i < GlobalConfiguration::VOLUME_ESTIMATION_ITERATIONS; ++i )
    {
        // Sample input point from known bounds.
        Map<unsigned, double> point;
        for ( unsigned j = 0; j < inputLayer->getSize(); ++j )
        {
            if ( inputLayer->neuronEliminated( j ) )
            {
                point.insert( j, 0 );
            }
            else
            {
                double lb = inputLayer->getLb( j );
                double ub = inputLayer->getUb( j );
                std::uniform_real_distribution<> dis( lb, ub );
                point.insert( j, dis( rng ) );
            }
        }

        // Calculate sigmoid of maximum margin from output symbolic bounds.
        double maxMargin = 0;
        for ( unsigned j = 0; j < outputLayer->getSize(); ++j )
        {
            if ( outputLayer->neuronEliminated( j ) )
                continue;

            double margin = calculateDifferenceFromSymbolic( outputLayer, point, j );
            maxMargin = std::max( maxMargin, margin );
        }
        sigmoidSum += SigmoidConstraint::sigmoid( maxMargin );
    }

    return std::exp( logBoxVolume + std::log( sigmoidSum ) ) /
           GlobalConfiguration::VOLUME_ESTIMATION_ITERATIONS;
}

double NetworkLevelReasoner::calculateDifferenceFromSymbolic( const Layer *layer,
                                                              Map<unsigned, double> &point,
                                                              unsigned i ) const
{
    unsigned size = layer->getSize();
    unsigned inputLayerSize = _layerIndexToLayer[0]->getSize();
    double lowerSum = layer->getSymbolicLowerBias()[i];
    double upperSum = layer->getSymbolicUpperBias()[i];

    for ( unsigned j = 0; j < inputLayerSize; ++j )
    {
        lowerSum += layer->getSymbolicLb()[j * size + i] * point[j];
        upperSum += layer->getSymbolicUb()[j * size + i] * point[j];
    }

    return std::max( layer->getUb( i ) - upperSum, lowerSum - layer->getLb( i ) );
}

const Vector<PolygonalTightening> NetworkLevelReasoner::OptimizeParameterisedPolygonalTightening()
{
    computeSuccessorLayers();
    const Vector<PolygonalTightening> &selectedTightenings = generatePolygonalTightenings();
    const unsigned size = selectedTightenings.size();
    Vector<PolygonalTightening> optimizedTightenings = Vector<PolygonalTightening>( {} );
    for ( unsigned i = 0; i < size; ++i )
    {
        PolygonalTightening tightening = selectedTightenings[i];
        double lowerBound =
            OptimizeSingleParameterisedPolygonalTightening( tightening, optimizedTightenings );
        tightening._value = lowerBound;
        optimizedTightenings.append( tightening );
    }
    const Vector<PolygonalTightening> tightenings( optimizedTightenings );
    return tightenings;
}

double NetworkLevelReasoner::OptimizeSingleParameterisedPolygonalTightening(
    PolygonalTightening &tightening,
    Vector<PolygonalTightening> &prevTightenings )
{
    // Search over coeffs in [0, 1]^numberOfParameters, gamma in
    // [0, inf)^sizeOfPrevTightenings with PGD.
    // unsigned maxIterations = GlobalConfiguration::INVPROP_MAX_ITERATIONS;
    unsigned maxIterations = 1000;
    // double coeffsStepSize = GlobalConfiguration::INVPROP_STEP_SIZE;
    // double gammaStepSize = GlobalConfiguration::INVPROP_STEP_SIZE;
    double coeffsStepSize = 0.025;
    double gammaStepSize = 0.0025;
    double epsilon = GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS;
    double weightDecay = GlobalConfiguration::INVPROP_WEIGHT_DECAY;
    double lr = GlobalConfiguration::INVPROP_LEARNING_RATE;
    unsigned coeffsDimension = getNumberOfParameters();
    unsigned gammaDimension = prevTightenings.size();
    bool maximize = ( tightening._type == PolygonalTightening::LB );
    int sign = ( maximize ? 1 : -1 );
    double bestBound = sign * tightening._value;

    Vector<double> coeffsLowerBounds( coeffsDimension, 0 );
    Vector<double> coeffsUpperBounds( coeffsDimension, 1 );
    Vector<double> gammaLowerBounds( gammaDimension, 0 );

    Vector<double> coeffs( coeffsDimension, GlobalConfiguration::INVPROP_INITIAL_ALPHA );
    Vector<double> gamma( gammaDimension, GlobalConfiguration::INVPROP_INITIAL_GAMMA );

    Vector<Vector<double>> coeffsCandidates( coeffsDimension );
    Vector<double> coeffsGradient( coeffsDimension );
    Vector<Vector<double>> gammaCandidates( gammaDimension );
    Vector<double> gammaGradient( gammaDimension );

    for ( unsigned i = 0; i < maxIterations; ++i )
    {
        double cost = getParameterisdPolygonalTighteningLowerBound(
            coeffs, gamma, tightening, prevTightenings );
        for ( unsigned j = 0; j < coeffsDimension; ++j )
        {
            coeffsCandidates[j] = Vector<double>( coeffs );
            coeffsCandidates[j][j] += coeffsStepSize;

            if ( coeffs[j] <= coeffsLowerBounds[j] || coeffs[j] >= coeffsUpperBounds[j] ||
                 coeffsCandidates[j][j] > coeffsUpperBounds[j] ||
                 coeffsCandidates[j][j] < coeffsLowerBounds[j] )
            {
                coeffsGradient[j] = 0;
                continue;
            }

            double currentCost = getParameterisdPolygonalTighteningLowerBound(
                coeffsCandidates[j], gamma, tightening, prevTightenings );
            coeffsGradient[j] = ( currentCost - cost ) / coeffsStepSize + weightDecay * coeffs[j];
            bestBound = ( maximize ? std::max( bestBound, currentCost )
                                   : std::min( bestBound, currentCost ) );
        }

        for ( unsigned j = 0; j < gammaDimension; ++j )
        {
            gammaCandidates[j] = Vector<double>( gamma );
            gammaCandidates[j][j] += gammaStepSize;

            if ( gamma[j] <= gammaLowerBounds[j] || gammaCandidates[j][j] < gammaLowerBounds[j] )
            {
                gammaGradient[j] = 0;
                continue;
            }

            double currentCost = getParameterisdPolygonalTighteningLowerBound(
                coeffs, gammaCandidates[j], tightening, prevTightenings );
            gammaGradient[j] = ( currentCost - cost ) / gammaStepSize + weightDecay * gamma[j];
            bestBound = ( maximize ? std::max( bestBound, currentCost )
                                   : std::min( bestBound, currentCost ) );
        }

        bool gradientIsZero = true;
        for ( unsigned j = 0; j < coeffsDimension; ++j )
        {
            if ( FloatUtils::abs( coeffsGradient[j] ) > epsilon )
            {
                gradientIsZero = false;
            }
        }
        for ( unsigned j = 0; j < gammaDimension; ++j )
        {
            if ( FloatUtils::abs( gammaGradient[j] ) > epsilon )
            {
                gradientIsZero = false;
            }
        }
        if ( gradientIsZero )
        {
            break;
        }

        for ( unsigned j = 0; j < coeffsDimension; ++j )
        {
            coeffs[j] += sign * lr * coeffsGradient[j];
            coeffs[j] = std::min( coeffs[j], coeffsUpperBounds[j] );
            coeffs[j] = std::max( coeffs[j], coeffsLowerBounds[j] );
        }
        for ( unsigned j = 0; j < gammaDimension; ++j )
        {
            gamma[j] += sign * lr * gammaGradient[j];
            gamma[j] = std::max( gamma[j], gammaLowerBounds[j] );
        }
    }

    return bestBound;
}

double NetworkLevelReasoner::getParameterisdPolygonalTighteningLowerBound(
    const Vector<double> &coeffs,
    const Vector<double> &gamma,
    PolygonalTightening &tightening,
    Vector<PolygonalTightening> &prevTightenings )
{
    // First, run parameterised symbolic bound propagation and compue successor layers.
    Map<unsigned, Vector<double>> layerIndicesToParameters = getParametersForLayers( coeffs );
    for ( unsigned i = 0; i < _layerIndexToLayer.size(); ++i )
    {
        const Vector<double> &currentLayerCoeffs = layerIndicesToParameters[i];
        _layerIndexToLayer[i]->computeParameterisedSymbolicBounds( currentLayerCoeffs );
    }

    // Recursively compute mu, muHat for every layer.
    const unsigned numLayers = _layerIndexToLayer.size();
    const unsigned maxLayer = _layerIndexToLayer.size() - 1;
    const unsigned prevTigheningsCount = prevTightenings.size();
    const unsigned inputLayerSize = _layerIndexToLayer[0]->getSize();

    Vector<Vector<double>> mu( numLayers );
    Vector<Vector<double>> muHat( numLayers );

    for ( int index = maxLayer; index >= 0; --index )
    {
        Layer *layer = _layerIndexToLayer[index];
        const unsigned layerSize = layer->getSize();
        const unsigned layerIndex = layer->getLayerIndex();

        mu[layerIndex] = Vector<double>( layerSize, 0 );
        muHat[layerIndex] = Vector<double>( layerSize, 0 );

        if ( layerIndex < maxLayer )
        {
            for ( unsigned i = 0; i < layerSize; ++i )
            {
                NeuronIndex neuron( layerIndex, i );
                for ( const unsigned successorLayerIndex : layer->getSuccessorLayers() )
                {
                    const Layer *successorLayer = getLayer( successorLayerIndex );
                    const unsigned successorLayerSize = successorLayer->getSize();
                    const Layer::Type successorLayerType = successorLayer->getLayerType();

                    if ( successorLayerType == Layer::WEIGHTED_SUM )
                    {
                        const double *successorWeights =
                            successorLayer->getWeightMatrix( layerIndex );

                        for ( unsigned j = 0; j < successorLayerSize; ++j )
                        {
                            if ( !successorLayer->neuronEliminated( j ) )
                            {
                                muHat[layerIndex][i] +=
                                    mu[successorLayerIndex][j] *
                                    successorWeights[i * successorLayerSize + j];
                            }
                        }
                    }
                    else
                    {
                        for ( unsigned j = 0; j < successorLayerSize; ++j )
                        {
                            // Find the index of the current neuron in the activation source list.
                            unsigned predecessorIndex = 0;
                            bool found = false;
                            List<NeuronIndex> sources = successorLayer->getActivationSources( j );
                            for ( const auto &sourceIndex : sources )
                            {
                                if ( sourceIndex != neuron )
                                {
                                    if ( !found )
                                    {
                                        ++predecessorIndex;
                                    }
                                }
                                else
                                {
                                    found = true;
                                }
                            }

                            if ( found )
                            {
                                if ( !successorLayer->neuronEliminated( j ) )
                                {
                                    if ( mu[successorLayerIndex][j] >= 0 )
                                    {
                                        muHat[layerIndex][i] +=
                                            mu[successorLayerIndex][j] *
                                            getSymbolicUbInTermsOfPredecessor( successorLayerIndex )
                                                [successorLayerSize * predecessorIndex + j];
                                    }
                                    else
                                    {
                                        muHat[layerIndex][i] -=
                                            mu[successorLayerIndex][j] *
                                            getSymbolicLbInTermsOfPredecessor( successorLayerIndex )
                                                [successorLayerSize * predecessorIndex + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if ( layerIndex > 0 )
        {
            // Compute mu from muHat.
            mu[layerIndex] += muHat[layerIndex];
            for ( unsigned i = 0; i < layerSize; ++i )
            {
                mu[layerIndex][i] -= tightening.getCoeff( NeuronIndex( layerIndex, i ) );
                for ( unsigned j = 0; j < prevTigheningsCount; ++j )
                {
                    const PolygonalTightening pt = prevTightenings[j];
                    double prevCoeff = pt.getCoeff( NeuronIndex( layerIndex, i ) );
                    if ( pt._type == PolygonalTightening::LB )
                    {
                        mu[layerIndex][i] -= gamma[j] * prevCoeff;
                    }
                    else
                    {
                        mu[layerIndex][i] += gamma[j] * prevCoeff;
                    }
                }
            }
        }
    }

    // Compute global bound for input space minimization problem.
    Vector<double> inputLayerBound( inputLayerSize, 0 );
    for ( unsigned i = 0; i < inputLayerSize; ++i )
    {
        inputLayerBound[i] += tightening.getCoeff( NeuronIndex( 0, i ) ) - muHat[0][i];
        for ( unsigned j = 0; j < prevTigheningsCount; ++j )
        {
            const PolygonalTightening pt = prevTightenings[j];
            double prevCoeff = pt.getCoeff( NeuronIndex( 0, i ) );
            if ( pt._type == PolygonalTightening::LB )
            {
                inputLayerBound[i] += gamma[j] * prevCoeff;
            }
            else
            {
                inputLayerBound[i] -= gamma[j] * prevCoeff;
            }
        }
    }

    // Compute lower bound for polygonal tightening bias using mu and inputLayerBound.
    double lowerBound = 0;
    for ( unsigned i = 0; i < prevTigheningsCount; ++i )
    {
        const PolygonalTightening pt = prevTightenings[i];
        if ( pt._type == PolygonalTightening::LB )
        {
            lowerBound -= gamma[i] * pt._value;
        }
        else
        {
            lowerBound += gamma[i] * pt._value;
        }
    }


    for ( int index = maxLayer; index > 0; --index )
    {
        Layer *layer = _layerIndexToLayer[index];
        const unsigned layerSize = layer->getSize();
        const unsigned layerIndex = layer->getLayerIndex();

        if ( layer->getLayerType() == Layer::WEIGHTED_SUM )
        {
            const double *biases = layer->getBiases();
            for ( unsigned i = 0; i < layerSize; ++i )
            {
                if ( !layer->neuronEliminated( i ) )
                {
                    lowerBound -= mu[layerIndex][i] * biases[i];
                }
                else
                {
                    lowerBound -= mu[layerIndex][i] * layer->getEliminatedNeuronValue( i );
                }
            }
        }
        else
        {
            for ( unsigned i = 0; i < layerSize; ++i )
            {
                if ( !layer->neuronEliminated( i ) )
                {
                    if ( mu[layerIndex][i] > 0 )
                    {
                        lowerBound -= mu[layerIndex][i] *
                                      getSymbolicUpperBiasInTermsOfPredecessor( layerIndex )[i];
                    }
                    else
                    {
                        lowerBound += mu[layerIndex][i] *
                                      getSymbolicLowerBiasInTermsOfPredecessor( layerIndex )[i];
                    }
                }
                else
                {
                    lowerBound -=
                        FloatUtils::abs( mu[layerIndex][i] ) * layer->getEliminatedNeuronValue( i );
                }
            }
        }
    }


    Layer *inputLayer = _layerIndexToLayer[0];
    const double *inputLbs = inputLayer->getLbs();
    const double *inputUbs = inputLayer->getUbs();
    for ( unsigned i = 0; i < inputLayerSize; ++i )
    {
        if ( inputLayerBound[i] > 0 )
        {
            lowerBound += inputLayerBound[i] * inputUbs[i];
        }
        else
        {
            lowerBound += inputLayerBound[i] * inputLbs[i];
        }
    }
    return lowerBound;
}

const Vector<PolygonalTightening> NetworkLevelReasoner::generatePolygonalTightenings()
{
    if ( Options::get()->getMILPSolverBoundTighteningType() ==
             MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_RANDOM ||
         Options::get()->getMILPSolverBoundTighteningType() ==
             MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_GRADIENT ||
         Options::get()->getMILPSolverBoundTighteningType() ==
             MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_BBPS )
    {
        return generatePolygonalTighteningsForPMNR();
    }
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_INVPROP )
    {
        return generatePolygonalTighteningsForInvprop();
    }

    const Vector<PolygonalTightening> tighteningsVector = Vector<PolygonalTightening>( {} );
    return tighteningsVector;
}

const Vector<PolygonalTightening> NetworkLevelReasoner::generatePolygonalTighteningsForPMNR()
{
    Vector<PolygonalTightening> tightenings = Vector<PolygonalTightening>( {} );
    unsigned neuronCount = GlobalConfiguration::PMNR_SELECTED_NEURONS;
    Vector<PolygonalTightening> lowerBoundTightenings = Vector<PolygonalTightening>( {} );
    Vector<PolygonalTightening> upperBoundTightenings = Vector<PolygonalTightening>( {} );
    const Vector<NeuronIndex> constraints = selectConstraints();

    for ( const auto &pair : constraints )
    {
        unsigned layerIndex = pair._layer;
        unsigned neuron = pair._neuron;
        Layer *layer = _layerIndexToLayer[layerIndex];
        unsigned layerSize = layer->getSize();

        Map<NeuronIndex, double> neuronToLowerCoefficient = Map<NeuronIndex, double>( {} );
        Map<NeuronIndex, double> neuronToUpperCoefficient = Map<NeuronIndex, double>( {} );
        neuronToLowerCoefficient[pair] = -1;
        neuronToUpperCoefficient[pair] = -1;

        List<NeuronIndex> sources = layer->getActivationSources( neuron );
        unsigned predecessorIndex = 0;
        for ( const auto &sourceIndex : sources )
        {
            neuronToLowerCoefficient[sourceIndex] = getSymbolicLbInTermsOfPredecessor(
                layerIndex )[predecessorIndex * layerSize + neuron];
            neuronToUpperCoefficient[sourceIndex] = getSymbolicUbInTermsOfPredecessor(
                layerIndex )[predecessorIndex * layerSize + neuron];
            ++predecessorIndex;
        }
        PolygonalTightening lowerTightening( neuronToLowerCoefficient, 0, PolygonalTightening::UB );
        PolygonalTightening upperTightening( neuronToLowerCoefficient, 0, PolygonalTightening::LB );
        lowerBoundTightenings.append( lowerTightening );
        upperBoundTightenings.append( upperTightening );
    }

    int range = ( 1 << neuronCount ) - 1;
    for ( int i = 0; i < range; ++i )
    {
        Map<NeuronIndex, double> neuronToLowerCoefficient = Map<NeuronIndex, double>( {} );
        Map<NeuronIndex, double> neuronToUpperCoefficient = Map<NeuronIndex, double>( {} );
        for ( unsigned j = 0; j < neuronCount; ++j )
        {
            int flag = ( i >> j ) % 2;
            if ( flag > 0 )
            {
                for ( const auto &pair : lowerBoundTightenings[j]._neuronToCoefficient )
                {
                    neuronToLowerCoefficient[pair.first] = pair.second;
                }
                for ( const auto &pair : upperBoundTightenings[j]._neuronToCoefficient )
                {
                    neuronToUpperCoefficient[pair.first] = pair.second;
                }
            }
        }
        PolygonalTightening lowerTightening( neuronToLowerCoefficient, 0, PolygonalTightening::UB );
        PolygonalTightening upperTightening( neuronToLowerCoefficient, 0, PolygonalTightening::LB );
        tightenings.append( lowerTightening );
        tightenings.append( upperTightening );
    }
    const Vector<PolygonalTightening> tighteningsVector =
        Vector<PolygonalTightening>( tightenings );
    return tighteningsVector;
}

const Vector<PolygonalTightening> NetworkLevelReasoner::generatePolygonalTighteningsForInvprop()
{
    Vector<PolygonalTightening> tightenings = Vector<PolygonalTightening>( {} );
    for ( const auto &pair : _layerIndexToLayer )
    {
        unsigned layerIndex = pair.first;
        Layer *layer = pair.second;
        const Vector<unsigned> &nonFixedNeurons = getNonFixedNeurons( layer );
        for ( const unsigned neuron : nonFixedNeurons )
        {
            NeuronIndex index( layerIndex, neuron );
            Map<NeuronIndex, double> neuronToLowerCoefficient = Map<NeuronIndex, double>( {} );
            Map<NeuronIndex, double> neuronToUpperCoefficient = Map<NeuronIndex, double>( {} );
            neuronToLowerCoefficient[index] = 1;
            neuronToUpperCoefficient[index] = 1;
            PolygonalTightening lowerTightening(
                neuronToLowerCoefficient, layer->getUb( neuron ), PolygonalTightening::UB );
            PolygonalTightening upperTightening(
                neuronToLowerCoefficient, layer->getLb( neuron ), PolygonalTightening::LB );
            tightenings.append( lowerTightening );
            tightenings.append( upperTightening );
        }
    }
    const Vector<PolygonalTightening> tighteningsVector =
        Vector<PolygonalTightening>( tightenings );
    return tighteningsVector;
}

const Vector<NeuronIndex> NetworkLevelReasoner::selectConstraints()
{
    if ( Options::get()->getMILPSolverBoundTighteningType() ==
         MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_RANDOM )
    {
        return selectConstraintsForPMNRRandom();
    }
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_GRADIENT )
    {
        return selectConstraintsForPMNRGradient();
    }
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::BACKWARD_ANALYSIS_PMNR_BBPS )
    {
        return selectConstraintsForPMNRBBPS();
    }

    const Vector<NeuronIndex> vect( GlobalConfiguration::PMNR_SELECTED_NEURONS );
    return vect;
}

const Vector<NeuronIndex> NetworkLevelReasoner::selectConstraintsForPMNRRandom()
{
    unsigned neuronCount = GlobalConfiguration::PMNR_SELECTED_NEURONS;
    Vector<NeuronIndex> neuronVector = Vector<NeuronIndex>( neuronCount );

    const Vector<unsigned> &candidateLayers = getLayersWithNonFixedNeurons();
    std::mt19937_64 rng( GlobalConfiguration::PMNR_RANDOM_SEED );
    std::uniform_int_distribution<unsigned> disLayer( 0, candidateLayers.size() - 1 );
    unsigned entry = disLayer( rng );
    unsigned index = candidateLayers[entry];

    Layer *layer = _layerIndexToLayer[index];
    const Vector<unsigned> &candidateNeurons = getNonFixedNeurons( layer );
    std::vector candidateNeuronsVector = candidateNeurons.getContainer();
    std::shuffle( candidateNeuronsVector.begin(), candidateNeuronsVector.end(), rng );
    for ( unsigned i = 0; i < neuronCount; ++i )
    {
        unsigned neuron = candidateNeurons[i];
        NeuronIndex idx = NeuronIndex( index, neuron );
        neuronVector[i] = idx;
    }

    const Vector<NeuronIndex> vect( neuronVector );
    return vect;
}

const Vector<NeuronIndex> NetworkLevelReasoner::selectConstraintsForPMNRGradient()
{
    unsigned neuronCount = GlobalConfiguration::PMNR_SELECTED_NEURONS;
    unsigned outputLayerSize = _layerIndexToLayer[getNumberOfLayers() - 1]->getSize();
    Vector<NeuronIndex> neuronVector = Vector<NeuronIndex>( neuronCount );
    double maxScore = 0;
    unsigned maxScoreIndex = 0;
    Map<NeuronIndex, double> neuronIndexToScore;
    for ( const auto &pair : _layerIndexToLayer )
    {
        double score = 0;
        unsigned index = pair.first;
        Layer *layer = pair.second;
        unsigned layerSize = layer->getSize();

        if ( getNonFixedNeurons( layer ).size() == 0 )
        {
            continue;
        }

        for ( unsigned i = 0; i < layerSize; ++i )
        {
            if ( isNeuronNonFixed( layer, i ) )
            {
                double neuronScore = 0;
                for ( unsigned j = 0; i < outputLayerSize; ++j )
                {
                    neuronScore +=
                        std::pow( ( getOutputLayerSymbolicLb( index )[j * outputLayerSize + i] +
                                    getOutputLayerSymbolicUb( index )[j * outputLayerSize + i] ) /
                                      2.0,
                                  2 );
                }
                neuronIndexToScore.insert( NeuronIndex( index, i ), neuronScore );
                score += neuronScore;
            }
        }

        if ( score > maxScore )
        {
            maxScore = score;
            maxScoreIndex = index;
        }
    }

    Layer *layer = _layerIndexToLayer[maxScoreIndex];
    unsigned layerSize = layer->getSize();
    std::priority_queue<std::pair<double, unsigned>,
                        std::vector<std::pair<double, unsigned>>,
                        std::less<std::pair<double, unsigned>>>
        maxQueue;
    for ( unsigned i = 0; i < layerSize; ++i )
    {
        if ( isNeuronNonFixed( layer, i ) )
        {
            double neuronScore = neuronIndexToScore[NeuronIndex( maxScoreIndex, i )];
            maxQueue.push( std::pair( neuronScore, i ) );
        }
    }

    for ( unsigned i = 0; i < neuronCount; ++i )
    {
        unsigned neuron = maxQueue.top().second;
        NeuronIndex idx = NeuronIndex( maxScoreIndex, neuron );
        neuronVector[i] = idx;
        maxQueue.pop();
    }

    const Vector<NeuronIndex> vect( neuronVector );
    return vect;
}

const Vector<NeuronIndex> NetworkLevelReasoner::selectConstraintsForPMNRBBPS()
{
    unsigned neuronCount = GlobalConfiguration::PMNR_SELECTED_NEURONS;
    Vector<NeuronIndex> neuronVector = Vector<NeuronIndex>( neuronCount );

    double maxScore = 0;
    unsigned maxScoreIndex = 0;
    Map<NeuronIndex, double> neuronIndexToScore;
    for ( const auto &pair : _layerIndexToLayer )
    {
        double score = 0;
        unsigned index = pair.first;
        Layer *layer = pair.second;
        unsigned layerSize = layer->getSize();

        if ( getNonFixedNeurons( layer ).size() == 0 )
        {
            continue;
        }

        for ( unsigned i = 0; i < layerSize; ++i )
        {
            if ( isNeuronNonFixed( layer, i ) )
            {
                double neuronScore = getBBPSScore( NeuronIndex( index, i ) );
                neuronIndexToScore.insert( NeuronIndex( index, i ), neuronScore );
                score += neuronScore;
            }
        }

        if ( score > maxScore )
        {
            maxScore = score;
            maxScoreIndex = index;
        }
    }

    Layer *layer = _layerIndexToLayer[maxScoreIndex];
    unsigned layerSize = layer->getSize();
    std::priority_queue<std::pair<double, unsigned>,
                        std::vector<std::pair<double, unsigned>>,
                        std::less<std::pair<double, unsigned>>>
        maxQueue;
    for ( unsigned i = 0; i < layerSize; ++i )
    {
        if ( isNeuronNonFixed( layer, i ) )
        {
            double neuronScore = neuronIndexToScore[NeuronIndex( maxScoreIndex, i )];
            maxQueue.push( std::pair( neuronScore, i ) );
        }
    }

    for ( unsigned i = 0; i < neuronCount; ++i )
    {
        unsigned neuron = maxQueue.top().second;
        NeuronIndex idx = NeuronIndex( maxScoreIndex, neuron );
        neuronVector[i] = idx;
        maxQueue.pop();
    }

    const Vector<NeuronIndex> vect( neuronVector );
    return vect;
}

void NetworkLevelReasoner::initializeBBPSMaps()
{
    for ( const auto &pair : _layerIndexToLayer )
    {
        unsigned index = pair.first;
        Layer *layer = pair.second;
        unsigned layerSize = layer->getSize();

        if ( getNonFixedNeurons( layer ).size() == 0 )
        {
            continue;
        }

        for ( unsigned i = 0; i < layerSize; ++i )
        {
            if ( isNeuronNonFixed( layer, i ) )
            {
                double score = 0;
                _neuronToBBPSScores.insert( NeuronIndex( index, i ), score );
            }
        }
    }
}

Map<NeuronIndex, double> NetworkLevelReasoner::getBranchingPoint( Layer *layer,
                                                                  unsigned neuron ) const
{
    ASSERT( isNeuronNonFixed( layer, neuron ) );
    Map<NeuronIndex, double> point;

    double lb = layer->getLb( neuron );
    double ub = layer->getUb( neuron );

    double branchingPoint;
    if ( layer->getLayerType() == Layer::RELU || layer->getLayerType() == Layer::LEAKY_RELU ||
         layer->getLayerType() == Layer::SIGN || layer->getLayerType() == Layer::ABSOLUTE_VALUE )
    {
        branchingPoint = 0;
    }
    else
    {
        branchingPoint = ( lb + ub ) / 2.0;
    }

    branchingPoint = branchingPoint;

    return point;
}

const Map<unsigned, Vector<double>>
NetworkLevelReasoner::getParametersForLayers( const Vector<double> &coeffs ) const
{
    unsigned totalCoeffsCount = getNumberOfParameters();
    ASSERT( coeffs.size() == totalCoeffsCount );
    unsigned index = 0;
    Map<unsigned, Vector<double>> layerIndicesToParameters;
    for ( const auto &pair : _layerIndexToLayer )
    {
        unsigned layerIndex = pair.first;
        Layer *layer = pair.second;
        unsigned coeffsCount = getNumberOfParametersPerType( layer->getLayerType() );
        Vector<double> currentCoeffs( coeffsCount );
        for ( unsigned i = 0; i < coeffsCount; ++i )
        {
            currentCoeffs[i] = coeffs[index + i];
        }
        layerIndicesToParameters.insert( layerIndex, currentCoeffs );
        index += coeffsCount;
    }
    const Map<unsigned, Vector<double>> parametersForLayers( layerIndicesToParameters );
    return parametersForLayers;
}

unsigned NetworkLevelReasoner::getNumberOfParameters() const
{
    unsigned num = 0;
    for ( const auto &pair : _layerIndexToLayer )
    {
        Layer *layer = pair.second;
        num += getNumberOfParametersPerType( layer->getLayerType() );
    }
    return num;
}

unsigned NetworkLevelReasoner::getNumberOfParametersPerType( Layer::Type t ) const
{
    if ( t == Layer::RELU || t == Layer::LEAKY_RELU )
        return 1;

    if ( t == Layer::SIGN || t == Layer::BILINEAR )
        return 2;

    return 0;
}

const Vector<unsigned> NetworkLevelReasoner::getLayersWithNonFixedNeurons() const
{
    Vector<unsigned> layerWithNonFixedNeurons = Vector<unsigned>( {} );
    for ( const auto &pair : _layerIndexToLayer )
    {
        unsigned layerIndex = pair.first;
        Layer *layer = pair.second;
        const Vector<unsigned> &nonFixedNeurons = getNonFixedNeurons( layer );
        if ( nonFixedNeurons.size() > 0 )
        {
            layerWithNonFixedNeurons.append( layerIndex );
        }
    }
    const Vector<unsigned> layerList = Vector<unsigned>( layerWithNonFixedNeurons );
    return layerList;
}

const Vector<unsigned> NetworkLevelReasoner::getNonFixedNeurons( Layer *layer ) const
{
    Vector<unsigned> nonFixedNeurons = Vector<unsigned>( {} );
    unsigned size = layer->getSize();
    for ( unsigned i = 0; i < size; ++i )
    {
        if ( isNeuronNonFixed( layer, i ) )
        {
            nonFixedNeurons.append( i );
        }
    }
    const Vector<unsigned> neuronList = Vector<unsigned>( nonFixedNeurons );
    return neuronList;
}

bool NetworkLevelReasoner::isNeuronNonFixed( Layer *layer, unsigned neuron ) const
{
    if ( layer->neuronEliminated( neuron ) )
    {
        return false;
    }

    bool nonFixed = false;
    switch ( layer->getLayerType() )
    {
    case Layer::RELU:
    case Layer::LEAKY_RELU:
    {
        double lb = layer->getLb( neuron );
        double ub = layer->getUb( neuron );
        nonFixed = !FloatUtils::isPositive( lb ) && !FloatUtils::isZero( ub );
        break;
    }
    case Layer::SIGN:
    {
        double lb = layer->getLb( neuron );
        double ub = layer->getUb( neuron );
        nonFixed = FloatUtils::isNegative( lb ) && !FloatUtils::isNegative( ub );
        break;
    }
    case Layer::ABSOLUTE_VALUE:
    {
        NeuronIndex sourceIndex = *layer->getActivationSources( neuron ).begin();
        const Layer *sourceLayer = getLayer( sourceIndex._layer );
        double sourceLb = sourceLayer->getLb( sourceIndex._neuron );
        double sourceUb = sourceLayer->getUb( sourceIndex._neuron );
        nonFixed = sourceLb < 0 && sourceUb > 0;
        break;
    }
    case Layer::SIGMOID:
    {
        NeuronIndex sourceIndex = *layer->getActivationSources( neuron ).begin();
        const Layer *sourceLayer = getLayer( sourceIndex._layer );
        double sourceLb = sourceLayer->getLb( sourceIndex._neuron );
        double sourceUb = sourceLayer->getUb( sourceIndex._neuron );
        nonFixed = !FloatUtils::areEqual( sourceLb, sourceUb );
        break;
    }
    case Layer::ROUND:
    {
        NeuronIndex sourceIndex = *layer->getActivationSources( neuron ).begin();
        const Layer *sourceLayer = getLayer( sourceIndex._layer );
        double sourceLb = sourceLayer->getLb( sourceIndex._neuron );
        double sourceUb = sourceLayer->getUb( sourceIndex._neuron );
        nonFixed =
            !FloatUtils::areEqual( FloatUtils::round( sourceUb ), FloatUtils::round( sourceLb ) );
        break;
    }
    case Layer::MAX:
    {
        List<NeuronIndex> sources = layer->getActivationSources( neuron );
        const Layer *sourceLayer = getLayer( sources.begin()->_layer );
        NeuronIndex indexOfMaxLowerBound = *( sources.begin() );
        double maxLowerBound = FloatUtils::negativeInfinity();
        double maxUpperBound = FloatUtils::negativeInfinity();

        Map<NeuronIndex, double> sourceUbs;
        for ( const auto &sourceIndex : sources )
        {
            unsigned sourceNeuron = sourceIndex._neuron;
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            sourceUbs[sourceIndex] = sourceUb;
            if ( maxLowerBound < sourceLb )
            {
                indexOfMaxLowerBound = sourceIndex;
                maxLowerBound = sourceLb;
            }
            if ( maxUpperBound < sourceUb )
            {
                maxUpperBound = sourceUb;
            }
        }

        bool phaseFixed = true;
        for ( const auto &sourceIndex : sources )
        {
            if ( sourceIndex != indexOfMaxLowerBound &&
                 FloatUtils::gt( sourceUbs[sourceIndex], maxLowerBound ) )
            {
                phaseFixed = false;
                break;
            }
        }
        nonFixed = !phaseFixed;
        break;
    }
    case Layer::SOFTMAX:
    {
        List<NeuronIndex> sources = layer->getActivationSources( neuron );
        const Layer *sourceLayer = getLayer( sources.begin()->_layer );
        Vector<double> sourceLbs;
        Vector<double> sourceUbs;
        Set<unsigned> handledInputNeurons;
        for ( const auto &sourceIndex : sources )
        {
            unsigned sourceNeuron = sourceIndex._neuron;
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            sourceLbs.append( sourceLb - GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
            sourceUbs.append( sourceUb + GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS );
        }

        unsigned index = 0;
        for ( const auto &sourceIndex : sources )
        {
            if ( handledInputNeurons.exists( sourceIndex._neuron ) )
                ++index;
            else
            {
                handledInputNeurons.insert( sourceIndex._neuron );
                break;
            }
        }

        double lb = std::max( layer->getLb( neuron ),
                              Layer::linearLowerBound( sourceLbs, sourceUbs, index ) );
        double ub = std::max( layer->getUb( neuron ),
                              Layer::linearUpperBound( sourceLbs, sourceUbs, index ) );
        nonFixed = !FloatUtils::areEqual( lb, ub );
        break;
    }
    case Layer::BILINEAR:
    {
        List<NeuronIndex> sources = layer->getActivationSources( neuron );
        const Layer *sourceLayer = getLayer( sources.begin()->_layer );
        bool eitherConstant = false;
        for ( const auto &sourceIndex : sources )
        {
            unsigned sourceNeuron = sourceIndex._neuron;
            double sourceLb = sourceLayer->getLb( sourceNeuron );
            double sourceUb = sourceLayer->getUb( sourceNeuron );
            if ( !sourceLayer->neuronEliminated( sourceNeuron ) )
            {
                eitherConstant = true;
            }
            if ( FloatUtils::areEqual( sourceLb, sourceUb ) )
            {
                eitherConstant = true;
            }
        }
        nonFixed = !eitherConstant;
        break;
    }
    case Layer::WEIGHTED_SUM:
    case Layer::INPUT:
    {
        nonFixed = false;
        break;
    }
    default:
    {
        nonFixed = false;
        break;
    }
    }
    return nonFixed;
}

void NetworkLevelReasoner::initializeSymbolicBoundsMaps( const Vector<double> &coeffs )
{
    // Clear the previous symbolic bound maps.
    _outputLayerSymbolicLb.clear();
    _outputLayerSymbolicUb.clear();
    _outputLayerSymbolicLowerBias.clear();
    _outputLayerSymbolicUpperBias.clear();

    _symbolicLbInTermsOfPredecessor.clear();
    _symbolicUbInTermsOfPredecessor.clear();
    _symbolicLowerBiasInTermsOfPredecessor.clear();
    _symbolicUpperBiasInTermsOfPredecessor.clear();

    // Temporarily add weighted sum layer to the NLR of the same size of the output layer.
    Layer *outputLayer = _layerIndexToLayer[getNumberOfLayers() - 1];
    unsigned outputLayerIndex = outputLayer->getLayerIndex();
    unsigned outputLayerSize = outputLayer->getSize();
    unsigned newLayerIndex = outputLayerIndex + 1;

    addLayer( newLayerIndex, Layer::WEIGHTED_SUM, outputLayerSize );
    addLayerDependency( outputLayerIndex, newLayerIndex );
    Layer *newLayer = _layerIndexToLayer[newLayerIndex];

    for ( unsigned i = 0; i < outputLayerSize; ++i )
    {
        setWeight( outputLayerIndex, i, newLayerIndex, i, 1 );
        newLayer->setLb( i, FloatUtils::infinity() );
        newLayer->setUb( i, FloatUtils::negativeInfinity() );
    }

    // Initialize maps with zero vectors.
    unsigned maxLayerSize = getMaxLayerSize();
    for ( const auto &pair : _layerIndexToLayer )
    {
        unsigned layerIndex = pair.first;
        Layer *layer = pair.second;
        unsigned layerSize = layer->getSize();
        Layer::Type layerType = layer->getLayerType();

        _outputLayerSymbolicLb[layerIndex] = Vector<double>( outputLayerSize * layerSize, 0 );
        _outputLayerSymbolicUb[layerIndex] = Vector<double>( outputLayerSize * layerSize, 0 );
        _outputLayerSymbolicLowerBias[layerIndex] = Vector<double>( outputLayerSize, 0 );
        _outputLayerSymbolicUpperBias[layerIndex] = Vector<double>( outputLayerSize, 0 );

        if ( layerType != Layer::WEIGHTED_SUM && layerType != Layer::INPUT )
        {
            _symbolicLbInTermsOfPredecessor[layerIndex] =
                Vector<double>( layerSize * maxLayerSize, 0 );
            _symbolicUbInTermsOfPredecessor[layerIndex] =
                Vector<double>( layerSize * maxLayerSize, 0 );
            _symbolicLowerBiasInTermsOfPredecessor[layerIndex] = Vector<double>( layerSize, 0 );
            _symbolicUpperBiasInTermsOfPredecessor[layerIndex] = Vector<double>( layerSize, 0 );
        }
    }

    // Populate symbolic bounds maps via DeepPoly.
    bool useParameterisedSBT = coeffs.size() > 0;
    Map<unsigned, Vector<double>> layerIndicesToParameters = Map<unsigned, Vector<double>>( {} );
    if ( useParameterisedSBT )
    {
        layerIndicesToParameters = getParametersForLayers( coeffs );
    }

    _deepPolyAnalysis = std::unique_ptr<DeepPolyAnalysis>(
        new DeepPolyAnalysis( this,
                              true,
                              true,
                              useParameterisedSBT,
                              &layerIndicesToParameters,
                              &_outputLayerSymbolicLb,
                              &_outputLayerSymbolicUb,
                              &_outputLayerSymbolicLowerBias,
                              &_outputLayerSymbolicUpperBias,
                              &_symbolicLbInTermsOfPredecessor,
                              &_symbolicUbInTermsOfPredecessor,
                              &_symbolicLowerBiasInTermsOfPredecessor,
                              &_symbolicUpperBiasInTermsOfPredecessor ) );
    _deepPolyAnalysis->run();
    _deepPolyAnalysis = nullptr;

    // Remove new weighted sum layer.
    removeLayerDependency( outputLayerIndex, newLayerIndex );
    _layerIndexToLayer.erase( newLayerIndex );
    if ( newLayer )
    {
        delete newLayer;
        newLayer = NULL;
    }
}

void NetworkLevelReasoner::mergeWSLayers( unsigned secondLayerIndex,
                                          Map<unsigned, LinearExpression> &eliminatedNeurons )
{
    Layer *secondLayer = _layerIndexToLayer[secondLayerIndex];
    unsigned firstLayerIndex = secondLayer->getSourceLayers().begin()->first;
    Layer *firstLayer = _layerIndexToLayer[firstLayerIndex];
    unsigned lastLayerIndex = _layerIndexToLayer.size() - 1;

    // Iterate over all inputs to the first layer
    for ( const auto &pair : firstLayer->getSourceLayers() )
    {
        unsigned previousToFirstLayerIndex = pair.first;
        const Layer *inputLayerToFirst = _layerIndexToLayer[previousToFirstLayerIndex];

        unsigned inputDimension = inputLayerToFirst->getSize();
        unsigned middleDimension = firstLayer->getSize();
        unsigned outputDimension = secondLayer->getSize();

        // Compute new weights
        const double *firstLayerMatrix = firstLayer->getWeightMatrix( previousToFirstLayerIndex );
        const double *secondLayerMatrix = secondLayer->getWeightMatrix( firstLayerIndex );
        double *newWeightMatrix = multiplyWeights(
            firstLayerMatrix, secondLayerMatrix, inputDimension, middleDimension, outputDimension );
        // Update bias for second layer
        for ( unsigned targetNeuron = 0; targetNeuron < secondLayer->getSize(); ++targetNeuron )
        {
            double newBias = secondLayer->getBias( targetNeuron );
            for ( unsigned sourceNeuron = 0; sourceNeuron < firstLayer->getSize(); ++sourceNeuron )
            {
                newBias +=
                    ( firstLayer->getBias( sourceNeuron ) *
                      secondLayer->getWeight( firstLayerIndex, sourceNeuron, targetNeuron ) );
            }

            secondLayer->setBias( targetNeuron, newBias );
        }

        // Adjust the sources of the second layer
        secondLayer->addSourceLayer( previousToFirstLayerIndex, inputLayerToFirst->getSize() );
        for ( unsigned sourceNeuron = 0; sourceNeuron < inputDimension; ++sourceNeuron )
        {
            for ( unsigned targetNeuron = 0; targetNeuron < outputDimension; ++targetNeuron )
            {
                double weight = newWeightMatrix[sourceNeuron * outputDimension + targetNeuron];
                secondLayer->setWeight(
                    previousToFirstLayerIndex, sourceNeuron, targetNeuron, weight );
            }
        }
        delete[] newWeightMatrix;
    }

    // Remove the first layer from second layer's sources
    secondLayer->removeSourceLayer( firstLayerIndex );

    generateLinearExpressionForWeightedSumLayer( eliminatedNeurons, firstLayer );

    // Finally, remove the first layer from the map and delete it
    _layerIndexToLayer.erase( firstLayerIndex );
    delete firstLayer;

    // Adjust the indices of all layers starting from secondLayerIndex
    // and higher
    for ( unsigned i = secondLayerIndex; i <= lastLayerIndex; ++i )
        reduceLayerIndex( i, secondLayerIndex );
}

double *NetworkLevelReasoner::multiplyWeights( const double *firstMatrix,
                                               const double *secondMatrix,
                                               unsigned inputDimension,
                                               unsigned middleDimension,
                                               unsigned outputDimension )
{
    double *newMatrix = new double[inputDimension * outputDimension];
    std::fill_n( newMatrix, inputDimension * outputDimension, 0 );
    matrixMultiplication(
        firstMatrix, secondMatrix, newMatrix, inputDimension, middleDimension, outputDimension );
    return newMatrix;
}

void NetworkLevelReasoner::reduceLayerIndex( unsigned layer, unsigned startIndex )
{
    // update Layer-level maps
    _layerIndexToLayer[layer]->reduceIndexFromAllMaps( startIndex );
    _layerIndexToLayer[layer]->reduceIndexAfterMerge( startIndex );

    // Update the mapping in the NLR
    _layerIndexToLayer[layer - 1] = _layerIndexToLayer[layer];
    _layerIndexToLayer.erase( layer );
}

void NetworkLevelReasoner::dumpBounds()
{
    obtainCurrentBounds();

    for ( const auto &layer : _layerIndexToLayer )
        layer.second->dumpBounds();
}

unsigned NetworkLevelReasoner::getMaxLayerSize() const
{
    unsigned maxSize = 0;
    for ( const auto &layer : _layerIndexToLayer )
    {
        unsigned currentSize = layer.second->getSize();
        if ( currentSize > maxSize )
            maxSize = currentSize;
    }
    ASSERT( maxSize > 0 );
    return maxSize;
}

const Map<unsigned, Layer *> &NetworkLevelReasoner::getLayerIndexToLayer() const
{
    return _layerIndexToLayer;
}

} // namespace NLR
