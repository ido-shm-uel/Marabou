/*********************                                                        */
/*! \file NetworkLevelReasoner.h
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

#ifndef __NetworkLevelReasoner_h__
#define __NetworkLevelReasoner_h__

#include "DeepPolyAnalysis.h"
#include "ITableau.h"
#include "LPFormulator.h"
#include "Layer.h"
#include "LayerOwner.h"
#include "Map.h"
#include "MatrixMultiplication.h"
#include "NeuronIndex.h"
#include "PiecewiseLinearFunctionType.h"
#include "PolygonalTightening.h"
#include "Tightening.h"
#include "Vector.h"

#include <memory>

namespace NLR {

/*
  A class for performing operations that require knowledge of network
  level structure and topology.
*/

class NetworkLevelReasoner : public LayerOwner
{
public:
    NetworkLevelReasoner();
    ~NetworkLevelReasoner();

    static bool functionTypeSupported( PiecewiseLinearFunctionType type );

    /*
      Populate the NLR by specifying the network's topology.
    */
    void addLayer( unsigned layerIndex, Layer::Type type, unsigned layerSize );
    void addLayerDependency( unsigned sourceLayer, unsigned targetLayer );
    void removeLayerDependency( unsigned sourceLayer, unsigned targetLayer );
    void computeSuccessorLayers();
    void setWeight( unsigned sourceLayer,
                    unsigned sourceNeuron,
                    unsigned targetLayer,
                    unsigned targetNeuron,
                    double weight );
    void setBias( unsigned layer, unsigned neuron, double bias );
    void addActivationSource( unsigned sourceLayer,
                              unsigned sourceNeuron,
                              unsigned targetLeyer,
                              unsigned targetNeuron );

    unsigned getNumberOfLayers() const;
    const Layer *getLayer( unsigned index ) const;
    Layer *getLayer( unsigned index );

    /*
      Bind neurons in the NLR to the Tableau variables that represent them.
    */
    void setNeuronVariable( NeuronIndex index, unsigned variable );

    /*
      Perform an evaluation of the network for a specific input.
    */
    void evaluate( double *input, double *output );

    /*
      Perform an evaluation of the network for the current input variable
      assignment and store the resulting variable assignment in the assignment.
    */
    void concretizeInputAssignment( Map<unsigned, double> &assignment );

    /*
      Perform a simulation of the network for a specific input
    */
    void simulate( Vector<Vector<double>> *input );

    /*
      Bound propagation methods:

        - obtainCurrentBounds: make the NLR obtain the current bounds
          on all variables from the tableau.

        - Interval arithmetic: compute the bounds of a layer's neurons
          based on the concrete bounds of the previous layer.

        - Symbolic: for each neuron in the network, we compute lower
          and upper bounds on the lower and upper bounds of the
          neuron. This bounds are expressed as linear combinations of
          the input neurons. Sometimes these bounds let us simplify
          expressions and obtain tighter bounds (e.g., if the upper
          bound on the upper bound of a ReLU node is negative, that
          ReLU is inactive and its output can be set to 0.

        - Parametrised Symbolic: For certain activation functions, there
          is a continuum of valid symbolic bounds. We receive a map of
          coefficients in range [0, 1] for every layer index, then compute
          the parameterised symbolic bounds (or default to regular
          symbolic bounds if parameterised bounds not implemented).

        - LP Relaxation: invoking an LP solver on a series of LP
          relaxations of the problem we're trying to solve, and
          optimizing the lower and upper bounds of each of the
          varibales.

        - receiveTighterBound: this is a callback from the layer
          objects, through which they report tighter bounds.

        - getConstraintTightenings: this is the function that an
          external user calls in order to collect the tighter bounds
          discovered by the NLR.

        - receiveTighterPolygonalBound: this is a callback from the layer
          objects, through which they report tighter polygonal bounds.

        - getConstraintPolygonalTightenings: this is the function that an
          external user calls in order to collect the tighter polygonal bounds
          discovered by the NLR.
    */

    void setTableau( const ITableau *tableau );
    const ITableau *getTableau() const;

    void obtainCurrentBounds( const Query &inputQuery );
    void obtainCurrentBounds();
    void intervalArithmeticBoundPropagation();
    void symbolicBoundPropagation();
    void parameterisedSymbolicBoundPropagation( const Vector<double> &coeffs );
    void deepPolyPropagation();
    void lpRelaxationPropagation();
    void LPTighteningForOneLayer( unsigned targetIndex );
    void MILPPropagation();
    void MILPTighteningForOneLayer( unsigned targetIndex );
    void iterativePropagation();

    void initializeSymbolicBoundsMaps( const Vector<double> &coeffs = Vector<double>( {} ) );

    // Return optimizable parameters which minimize parameterised SBT bounds' volume.
    const Vector<double> OptimalParameterisedSymbolicBoundTightening();

    // Optimize biases of generated parameterised polygonal tightenings.
    const Vector<PolygonalTightening> OptimizeParameterisedPolygonalTightening();

    // Get total number of optimizable parameters for parameterised SBT relaxation.
    unsigned getNumberOfParameters() const;

    void receiveTighterBound( Tightening tightening );
    void getConstraintTightenings( List<Tightening> &tightenings );
    void clearConstraintTightenings();

    void receivePolygonalTighterBound( PolygonalTightening polygonalTightening );
    void getConstraintPolygonalTightenings( List<PolygonalTightening> &polygonalTightening );
    void clearConstraintPolygonalTightenings();

    /*
      For debugging purposes: dump the network topology
    */
    void dumpTopology( bool dumpLayerDetails = true ) const;

    /*
      Duplicate the reasoner
    */
    void storeIntoOther( NetworkLevelReasoner &other ) const;

    /*
      Methods that are typically invoked by the preprocessor, to
      inform us of changes in variable indices or if a variable has
      been eliminated
    */
    void eliminateVariable( unsigned variable, double value );
    void updateVariableIndices( const Map<unsigned, unsigned> &oldIndexToNewIndex,
                                const Map<unsigned, unsigned> &mergedVariables );

    /*
      The various piecewise-linear constraints, sorted in topological
      order. The sorting is done externally.
    */
    List<PiecewiseLinearConstraint *> getConstraintsInTopologicalOrder();
    void addConstraintInTopologicalOrder( PiecewiseLinearConstraint *constraint );
    void removeConstraintFromTopologicalOrder( PiecewiseLinearConstraint *constraint );

    /*
      Add an ecoding of all the affine layers as equations in the given Query
    */
    void encodeAffineLayers( Query &inputQuery );

    /*
      Generate an input query from this NLR, according to the
      discovered network topology
    */
    void generateQuery( Query &query );

    /*
      Given a ReLU Constraint, get the previous layer bias
      for the BaBSR Heuristic
    */
    double getPreviousBias( const ReluConstraint *reluConstraint ) const;

    Vector<double> getOutputSymbolicLb( unsigned layerIndex ) const;
    Vector<double> getOutputSymbolicUb( unsigned layerIndex ) const;
    Vector<double> getOutputSymbolicLowerBias( unsigned layerIndex ) const;
    Vector<double> getOutputSymbolicUpperBias( unsigned layerIndex ) const;

    Vector<double> getPredecessorSymbolicLb( unsigned layerIndex ) const;
    Vector<double> getPredecessorSymbolicUb( unsigned layerIndex ) const;
    Vector<double> getPredecessorSymbolicLowerBias( unsigned layerIndex ) const;
    Vector<double> getPredecessorSymbolicUpperBias( unsigned layerIndex ) const;

    Map<NeuronIndex, double> getBBPSBranchingPoint( NeuronIndex index ) const;
    double getPMNRScore( NeuronIndex index ) const;

    /*
      Finds logically consecutive WS layers and merges them, in order
      to reduce the total number of layers and variables in the
      network
    */
    unsigned mergeConsecutiveWSLayers( const Map<unsigned, double> &lowerBounds,
                                       const Map<unsigned, double> &upperBounds,
                                       const Set<unsigned> &varsInUnhandledConstraints,
                                       Map<unsigned, LinearExpression> &eliminatedNeurons );

    /*
      Print the bounds of variables layer by layer
    */
    void dumpBounds();

    /*
      Get the size of the widest layer
    */
    unsigned getMaxLayerSize() const;

    const Map<unsigned, Layer *> &getLayerIndexToLayer() const;

private:
    Map<unsigned, Layer *> _layerIndexToLayer;
    const ITableau *_tableau;

    // Tightenings and Polyognal Tightenings discovered by the various layers
    List<Tightening> _boundTightenings;
    List<PolygonalTightening> _polygonalBoundTightenings;

    std::unique_ptr<DeepPolyAnalysis> _deepPolyAnalysis;

    void freeMemoryIfNeeded();

    List<PiecewiseLinearConstraint *> _constraintsInTopologicalOrder;

    // Map each neuron to a linear expression representing its weighted sum
    void generateLinearExpressionForWeightedSumLayer(
        Map<unsigned, LinearExpression> &variableToExpression,
        const Layer &layer );

    // Helper functions for generating an input query
    void generateQueryForLayer( Query &inputQuery, const Layer &layer );
    void generateQueryForWeightedSumLayer( Query &inputQuery, const Layer &layer );
    void generateEquationsForWeightedSumLayer( List<Equation> &equations, const Layer &layer );
    void generateQueryForReluLayer( Query &inputQuery, const Layer &layer );
    void generateQueryForSigmoidLayer( Query &inputQuery, const Layer &layer );
    void generateQueryForSignLayer( Query &inputQuery, const Layer &layer );
    void generateQueryForAbsoluteValueLayer( Query &inputQuery, const Layer &layer );
    void generateQueryForMaxLayer( Query &inputQuery, const Layer &layer );

    bool suitableForMerging( unsigned secondLayerIndex,
                             const Map<unsigned, double> &lowerBounds,
                             const Map<unsigned, double> &upperBounds,
                             const Set<unsigned> &varsInConstraintsUnhandledByNLR );
    void mergeWSLayers( unsigned secondLayerIndex,
                        Map<unsigned, LinearExpression> &eliminatedNeurons );
    double *multiplyWeights( const double *firstMatrix,
                             const double *secondMatrix,
                             unsigned inputDimension,
                             unsigned middleDimension,
                             unsigned outputDimension );
    void reduceLayerIndex( unsigned layer, unsigned startIndex );

    void optimizeBoundsWithPreimageApproximation( LPFormulator &lpFormulator );

    // Estimate Volume of parameterised symbolic bound tightening.
    double EstimateVolume( const Vector<double> &coeffs );

    void optimizeBoundsWithPMNR( LPFormulator &lpFormulator );

    // Optimize biases of generated parameterised polygonal tightenings.
    double
    OptimizeSingleParameterisedPolygonalTightening( PolygonalTightening &tightening,
                                                    Vector<PolygonalTightening> &prevTightenings );

    // Get current lower bound for selected parameterised polygonal tightenings' biases.
    double
    getParameterisdPolygonalTighteningLowerBound( const Vector<double> &coeffs,
                                                  const Vector<double> &gamma,
                                                  PolygonalTightening &tightening,
                                                  Vector<PolygonalTightening> &prevTightenings );

    const Vector<PolygonalTightening> generatePolygonalTightenings();
    const Vector<PolygonalTightening> generatePolygonalTighteningsForPMNR();
    const Vector<PolygonalTightening> generatePolygonalTighteningsForInvprop();

    // Heuristically select neurons and polygonal tightenings for PMNR.
    const Vector<NeuronIndex> selectConstraints();


    double getPMNRGradientScore( NeuronIndex index ) const;
    double getPMNRBBPSScore( NeuronIndex index ) const;

    const Vector<NeuronIndex> selectConstraintsForPMNRRandom();
    const Vector<NeuronIndex> selectConstraintsForPMNRGradient();
    const Vector<NeuronIndex> selectConstraintsForPMNRBBPS();

    // Return difference between given point and upper and lower bounds determined by parameterised
    // SBT relaxation.
    double calculateDifferenceFromSymbolic( const Layer *layer,
                                            Map<unsigned, double> &point,
                                            unsigned i ) const;

    // Get map containing vector of optimizable parameters for parameterised SBT relaxation for
    // every layer index.
    const Map<unsigned, Vector<double>>
    getParametersForLayers( const Vector<double> &coeffs ) const;

    // Get number of optimizable parameters for parameterised SBT relaxation per layer type.
    unsigned getNumberOfParametersPerType( Layer::Type t ) const;

    // Get all indices of active non-weighted sum layers for INVPROP.
    const Vector<unsigned> getLayersWithNonFixedNeurons() const;

    const Vector<NeuronIndex> getNonFixedNeurons( const Layer *layer ) const;

    bool isNeuronNonFixed( NeuronIndex index ) const;

    /*
      Store previous biases for each ReLU neuron in a map for getPreviousBias()
      and BaBSR heuristic
    */
    Map<const ReluConstraint *, double> _previousBiases;
    void initializePreviousBiasMap();

    void initializePMNRScoreMap();
    void initializeBBPSBranchingMap();

    double calculateNeuronPMNRScore( NeuronIndex index ) const;
    double calculatePMNRGradientScore( NeuronIndex index ) const;
    double calculatePMNRBBPSScore( NeuronIndex index ) const;
    Map<NeuronIndex, double> calculateBranchingPoint( NeuronIndex index ) const;

    /*
      If the NLR is manipulated manually in order to generate a new
      input query, this method can be used to assign variable indices
      to all neurons in the network
    */
    void reindexNeurons();

    Map<NeuronIndex, Map<NeuronIndex, double>> _neuronToBBPSBranchingPoints;
    Map<NeuronIndex, double> _neuronToPMNRScores;

    Map<unsigned, Vector<double>> _outputSymbolicLb;
    Map<unsigned, Vector<double>> _outputSymbolicUb;
    Map<unsigned, Vector<double>> _outputSymbolicLowerBias;
    Map<unsigned, Vector<double>> _outputSymbolicUpperBias;

    Map<unsigned, Vector<double>> _predecessorSymbolicLb;
    Map<unsigned, Vector<double>> _predecessorSymbolicUb;
    Map<unsigned, Vector<double>> _predecessorSymbolicLowerBias;
    Map<unsigned, Vector<double>> _predecessorSymbolicUpperBias;
};

} // namespace NLR

#endif // __NetworkLevelReasoner_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
