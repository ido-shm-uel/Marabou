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

void NetworkLevelReasoner::receivePolygonalTightening( PolygonalTightening &polygonalTightening )
{
    _polygonalBoundTightenings.append( polygonalTightening );
}

void NetworkLevelReasoner::getPolygonalTightenings(
    List<PolygonalTightening> &polygonalTightenings )
{
    polygonalTightenings = _polygonalBoundTightenings;
    _polygonalBoundTightenings.clear();
}

void NetworkLevelReasoner::clearPolygonalTightenings()
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

void NetworkLevelReasoner::parameterisedDeepPoly( bool storeSymbolicBounds,
                                                  const Vector<double> &coeffs )
{
    if ( !storeSymbolicBounds )
    {
        bool useParameterisedSBT = coeffs.size() > 0;
        Map<unsigned, Vector<double>> layerIndicesToParameters =
            Map<unsigned, Vector<double>>( {} );
        if ( useParameterisedSBT )
        {
            layerIndicesToParameters = getParametersForLayers( coeffs );
        }

        _deepPolyAnalysis =
            std::unique_ptr<DeepPolyAnalysis>( new DeepPolyAnalysis( this,
                                                                     storeSymbolicBounds,
                                                                     storeSymbolicBounds,
                                                                     useParameterisedSBT,
                                                                     &layerIndicesToParameters ) );

        // Clear deepPolyAnalysis pointer after running.
        _deepPolyAnalysis->run();
        _deepPolyAnalysis = nullptr;
    }
    else
    {
        // Clear the previous symbolic bound maps.
        _outputSymbolicLb.clear();
        _outputSymbolicUb.clear();
        _outputSymbolicLowerBias.clear();
        _outputSymbolicUpperBias.clear();

        _predecessorSymbolicLb.clear();
        _predecessorSymbolicUb.clear();
        _predecessorSymbolicLowerBias.clear();
        _predecessorSymbolicUpperBias.clear();

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
        for ( const auto &pair : _layerIndexToLayer )
        {
            unsigned layerIndex = pair.first;
            Layer *layer = pair.second;
            unsigned size = layer->getSize();
            Layer::Type layerType = layer->getLayerType();

            _outputSymbolicLb[layerIndex] = Vector<double>( outputLayerSize * size, 0 );
            _outputSymbolicUb[layerIndex] = Vector<double>( outputLayerSize * size, 0 );
            _outputSymbolicLowerBias[layerIndex] = Vector<double>( outputLayerSize, 0 );
            _outputSymbolicUpperBias[layerIndex] = Vector<double>( outputLayerSize, 0 );

            if ( layerType != Layer::WEIGHTED_SUM && layerType != Layer::INPUT )
            {
                unsigned maxSourceSize = 0;
                for ( unsigned i = 0; i < size; ++i )
                {
                    unsigned sourceSize = layer->getActivationSources( i ).size();
                    maxSourceSize = sourceSize > maxSourceSize ? sourceSize : maxSourceSize;
                }
                _predecessorSymbolicLb[layerIndex] = Vector<double>( size * maxSourceSize, 0 );
                _predecessorSymbolicUb[layerIndex] = Vector<double>( size * maxSourceSize, 0 );
                _predecessorSymbolicLowerBias[layerIndex] = Vector<double>( size, 0 );
                _predecessorSymbolicUpperBias[layerIndex] = Vector<double>( size, 0 );
            }
        }

        // Populate symbolic bounds maps via DeepPoly.
        bool useParameterisedSBT = coeffs.size() > 0;
        Map<unsigned, Vector<double>> layerIndicesToParameters =
            Map<unsigned, Vector<double>>( {} );
        if ( useParameterisedSBT )
        {
            layerIndicesToParameters = getParametersForLayers( coeffs );
        }

        _deepPolyAnalysis = std::unique_ptr<DeepPolyAnalysis>(
            new DeepPolyAnalysis( this,
                                  storeSymbolicBounds,
                                  storeSymbolicBounds,
                                  useParameterisedSBT,
                                  &layerIndicesToParameters,
                                  &_outputSymbolicLb,
                                  &_outputSymbolicUb,
                                  &_outputSymbolicLowerBias,
                                  &_outputSymbolicUpperBias,
                                  &_predecessorSymbolicLb,
                                  &_predecessorSymbolicUb,
                                  &_predecessorSymbolicLowerBias,
                                  &_predecessorSymbolicUpperBias ) );

        // Clear deepPolyAnalysis pointer after running.
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
    {
        const Vector<double> optimalCoeffs = OptimalParameterisedSymbolicBoundTightening();
        Map<unsigned, Vector<double>> layerIndicesToParameters =
            getParametersForLayers( optimalCoeffs );
        lpFormulator.optimizeBoundsWithLpRelaxation(
            _layerIndexToLayer, false, layerIndicesToParameters );
        lpFormulator.optimizeBoundsWithLpRelaxation(
            _layerIndexToLayer, true, layerIndicesToParameters );
    }
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::LP_RELAXATION )
        lpFormulator.optimizeBoundsWithLpRelaxation( _layerIndexToLayer );
    else if ( Options::get()->getMILPSolverBoundTighteningType() ==
              MILPSolverBoundTighteningType::LP_RELAXATION_INCREMENTAL )
        lpFormulator.optimizeBoundsWithIncrementalLpRelaxation( _layerIndexToLayer );
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

Vector<double> NetworkLevelReasoner::getOutputSymbolicLb( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _outputSymbolicLb.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_outputSymbolicLb.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputSymbolicLb[layerIndex];
}

Vector<double> NetworkLevelReasoner::getOutputSymbolicUb( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _outputSymbolicUb.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_outputSymbolicUb.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputSymbolicUb[layerIndex];
}

Vector<double> NetworkLevelReasoner::getOutputSymbolicLowerBias( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _outputSymbolicLowerBias.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_outputSymbolicLowerBias.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputSymbolicLowerBias[layerIndex];
}

Vector<double> NetworkLevelReasoner::getOutputSymbolicUpperBias( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _outputSymbolicUpperBias.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_outputSymbolicUpperBias.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in output layer symbolic bounds map." );
    }

    return _outputSymbolicUpperBias[layerIndex];
}

Vector<double> NetworkLevelReasoner::getPredecessorSymbolicLb( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _predecessorSymbolicLb.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_predecessorSymbolicLb.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _predecessorSymbolicLb[layerIndex];
}

Vector<double> NetworkLevelReasoner::getPredecessorSymbolicUb( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _predecessorSymbolicUb.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_predecessorSymbolicUb.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _predecessorSymbolicUb[layerIndex];
}

Vector<double> NetworkLevelReasoner::getPredecessorSymbolicLowerBias( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _predecessorSymbolicLowerBias.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_predecessorSymbolicLowerBias.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _predecessorSymbolicLowerBias[layerIndex];
}

Vector<double> NetworkLevelReasoner::getPredecessorSymbolicUpperBias( unsigned layerIndex )
{
    // Initialize map if empty.
    if ( _predecessorSymbolicUpperBias.empty() )
    {
        parameterisedDeepPoly( true );
    }

    if ( !_predecessorSymbolicUpperBias.exists( layerIndex ) )
    {
        throw NLRError( NLRError::LAYER_NOT_FOUND,
                        "Layer not found in predecessor layer symbolic bounds map." );
    }

    return _predecessorSymbolicUpperBias[layerIndex];
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
    double sign = ( maximize ? 1 : -1 );

    Vector<double> lowerBounds( dimension, 0 );
    Vector<double> upperBounds( dimension, 1 );

    // Initialize initial guess uniformly.
    Vector<double> guess( dimension );
    std::mt19937_64 rng( GlobalConfiguration::PREIMAGE_APPROXIMATION_OPTIMIZATION_RANDOM_SEED );
    std::uniform_real_distribution<double> dis( 0, 1 );
    for ( unsigned j = 0; j < dimension; ++j )
    {
        double lb = lowerBounds[j];
        double ub = upperBounds[j];
        guess[j] = lb + dis( rng ) * ( ub - lb );
    }

    Vector<Vector<double>> candidates( dimension );
    Vector<double> gradient( dimension );

    for ( unsigned i = 0; i < maxIterations; ++i )
    {
        double currentCost = EstimateVolume( guess );
        for ( unsigned j = 0; j < dimension; ++j )
        {
            candidates[j] = Vector<double>( guess );
            candidates[j][j] += stepSize;

            if ( candidates[j][j] > upperBounds[j] || candidates[j][j] < lowerBounds[j] )
            {
                gradient[j] = 0;
                continue;
            }

            double cost = EstimateVolume( candidates[j] );
            gradient[j] = ( cost - currentCost ) / stepSize + weightDecay * guess[j];
        }

        bool gradientIsZero = true;
        for ( unsigned j = 0; j < dimension; ++j )
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

        for ( unsigned j = 0; j < dimension; ++j )
        {
            guess[j] += sign * lr * gradient[j];

            guess[j] = std::min( guess[j], upperBounds[j] );
            guess[j] = std::max( guess[j], lowerBounds[j] );
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

const Map<unsigned, Vector<double>>
NetworkLevelReasoner::getParametersForLayers( const Vector<double> &coeffs ) const
{
    ASSERT( coeffs.size() == getNumberOfParameters() );
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

const Vector<unsigned> NetworkLevelReasoner::getLayersWithNonfixedNeurons() const
{
    Vector<unsigned> layerWithNonfixedNeurons = Vector<unsigned>( {} );
    for ( const auto &pair : _layerIndexToLayer )
    {
        if ( !pair.second->getNonfixedNeurons().empty() )
        {
            layerWithNonfixedNeurons.append( pair.first );
        }
    }
    const Vector<unsigned> layerList = Vector<unsigned>( layerWithNonfixedNeurons );
    return layerList;
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
