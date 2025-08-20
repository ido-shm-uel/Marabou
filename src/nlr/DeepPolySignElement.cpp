/*********************                                                        */
/*! \file DeepPolySignElement.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze Andrew Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "DeepPolySignElement.h"

#include "FloatUtils.h"

namespace NLR {

DeepPolySignElement::DeepPolySignElement( Layer *layer )
{
    _layer = layer;
    _size = layer->getSize();
    _layerIndex = layer->getLayerIndex();
}

DeepPolySignElement::~DeepPolySignElement()
{
    freeMemoryIfNeeded();
}

void DeepPolySignElement::execute( const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore )
{
    log( "Executing..." );
    ASSERT( hasPredecessor() );
    allocateMemory();

    // Update the symbolic and concrete upper- and lower- bounds
    // of each neuron
    for ( unsigned i = 0; i < _size; ++i )
    {
        NeuronIndex sourceIndex = *( _layer->getActivationSources( i ).begin() );
        DeepPolyElement *predecessor = deepPolyElementsBefore[sourceIndex._layer];
        double sourceLb = predecessor->getLowerBound( sourceIndex._neuron );
        double sourceUb = predecessor->getUpperBound( sourceIndex._neuron );

        if ( !_useParameterisedSBT )
        {
            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // Phase positive
                // Symbolic bound: 1 <= x_f <= 1
                // Concrete bound: 1 <= x_f <= 1
                _symbolicUb[i] = 0;
                _symbolicUpperBias[i] = 1;
                _ub[i] = 1;

                _symbolicLb[i] = 0;
                _symbolicLowerBias[i] = 1;
                _lb[i] = 1;
            }
            else if ( FloatUtils::isNegative( sourceUb ) )
            {
                // Phase negative
                // Symbolic bound: -1 <= x_f <= -1
                // Concrete bound: -1 <= x_f <= -1
                _symbolicUb[i] = 0;
                _symbolicUpperBias[i] = -1;
                _ub[i] = -1;

                _symbolicLb[i] = 0;
                _symbolicLowerBias[i] = -1;
                _lb[i] = -1;
            }
            else
            {
                // Sign not fixed
                // Use the relaxation defined in https://arxiv.org/pdf/2011.02948.pdf
                // Symbolic upper bound: x_f <= -2 / l * x_b + 1
                // Concrete upper bound: x_f <= 1
                _symbolicUb[i] = -2 / sourceLb;
                _symbolicUpperBias[i] = 1;
                _ub[i] = 1;

                // Symbolic lower bound: x_f >= (2 / u) * x_b - 1
                // Concrete lower bound: x_f >= -1
                _symbolicLb[i] = 2 / sourceUb;
                _symbolicLowerBias[i] = -1;
                _lb[i] = -1;
            }
        }
        else
        {
            Vector<double> coeffs = ( *_layerIndicesToParameters )[_layerIndex];
            ASSERT( coeffs.size() == 2 );
            ASSERT( coeffs[0] >= 0 && coeffs[0] <= 1 );
            ASSERT( coeffs[1] >= 0 && coeffs[1] <= 1 );
            if ( !FloatUtils::isNegative( sourceLb ) )
            {
                // Phase positive
                // Symbolic bound: 1 <= x_f <= 1
                // Concrete bound: 1 <= x_f <= 1
                _symbolicUb[i] = 0;
                _symbolicUpperBias[i] = 1;
                _ub[i] = 1;

                _symbolicLb[i] = 0;
                _symbolicLowerBias[i] = 1;
                _lb[i] = 1;
            }
            else if ( FloatUtils::isNegative( sourceUb ) )
            {
                // Phase negative
                // Symbolic bound: -1 <= x_f <= -1
                // Concrete bound: -1 <= x_f <= -1
                _symbolicUb[i] = 0;
                _symbolicUpperBias[i] = -1;
                _ub[i] = -1;

                _symbolicLb[i] = 0;
                _symbolicLowerBias[i] = -1;
                _lb[i] = -1;
            }
            else
            {
                // Sign not fixed
                // The upper bound's phase is not fixed, use parameterised
                // parallelogram approximation: y <= - 2 / l * coeffs[0] * x + 1
                // (varies continuously between y <= 1 and y <= -2 / l * x + 1).
                // Concrete upper bound: x_f <= 1
                _symbolicUb[i] = -2.0 / sourceLb * coeffs[0];
                _symbolicUpperBias[i] = 1;
                _ub[i] = 1;

                // The lower bound's phase is not fixed, use parameterised
                // parallelogram approximation: y >= 2 / u * coeffs[1] * x - 1
                // (varies continuously between y >= -1 and y >= 2 / u * x - 1).
                // Symbolic lower bound: x_f >= (2 / u) * x_b - 1
                // Concrete lower bound: x_f >= -1
                _symbolicLb[i] = 2.0 / sourceUb * coeffs[1];
                _symbolicLowerBias[i] = -1;
                _lb[i] = -1;
            }
        }
        log( Stringf( "Neuron%u LB: %f b + %f, UB: %f b + %f",
                      i,
                      _symbolicLb[i],
                      _symbolicLowerBias[i],
                      _symbolicUb[i],
                      _symbolicUpperBias[i] ) );
        log( Stringf( "Neuron%u LB: %f, UB: %f", i, _lb[i], _ub[i] ) );
    }

    if ( _storeSymbolicBoundsInTermsOfPredecessor )
    {
        storePredecessorSymbolicBounds();
    }

    log( "Executing - done" );
}

void DeepPolySignElement::storePredecessorSymbolicBounds()
{
    for ( unsigned i = 0; i < _size; ++i )
    {
        NeuronIndex sourceIndex = *( _layer->getActivationSources( i ).begin() );
        ( *_symbolicLbInTermsOfPredecessor )[_layerIndex][_size * sourceIndex._neuron + i] =
            _symbolicLb[i];
        ( *_symbolicUbInTermsOfPredecessor )[_layerIndex][_size * sourceIndex._neuron + i] =
            _symbolicUb[i];
        ( *_symbolicLowerBiasInTermsOfPredecessor )[_layerIndex][i] = _symbolicLowerBias[i];
        ( *_symbolicUpperBiasInTermsOfPredecessor )[_layerIndex][i] = _symbolicUpperBias[i];
    }
}

void DeepPolySignElement::symbolicBoundInTermsOfPredecessor( const double *symbolicLb,
                                                             const double *symbolicUb,
                                                             double *symbolicLowerBias,
                                                             double *symbolicUpperBias,
                                                             double *symbolicLbInTermsOfPredecessor,
                                                             double *symbolicUbInTermsOfPredecessor,
                                                             unsigned targetLayerSize,
                                                             DeepPolyElement *predecessor )
{
    log( Stringf( "Computing symbolic bounds with respect to layer %u...",
                  predecessor->getLayerIndex() ) );

    /*
      We have the symbolic bound of the target layer in terms of the
      Sign outputs, the goal is to compute the symbolic bound of the target
      layer in terms of the Sign inputs.
    */
    for ( unsigned i = 0; i < _size; ++i )
    {
        NeuronIndex sourceIndex = *( _layer->getActivationSources( i ).begin() );
        unsigned sourceNeuronIndex = sourceIndex._neuron;
        DEBUG( { ASSERT( predecessor->getLayerIndex() == sourceIndex._layer ); } );

        /*
          Take symbolic upper bound as an example.
          Suppose the symbolic upper bound of the j-th neuron in the
          target layer is ... + a_i * f_i + ...,
          and the symbolic bounds of f_i in terms of b_i is
          m * b_i + n <= f_i <= p * b_i + q.
          If a_i >= 0, replace f_i with p * b_i + q, otherwise,
          replace f_i with m * b_i + n
        */

        // Symbolic bounds of the Sign output in terms of the Sign input
        // coeffLb * b_i + lowerBias <= f_i <= coeffUb * b_i + upperBias
        double coeffLb = _symbolicLb[i];
        double coeffUb = _symbolicUb[i];
        double lowerBias = _symbolicLowerBias[i];
        double upperBias = _symbolicUpperBias[i];

        // Substitute the Sign input for the Sign output
        for ( unsigned j = 0; j < targetLayerSize; ++j )
        {
            // The symbolic lower- and upper- bounds of the j-th neuron in the
            // target layer are ... + weightLb * f_i + ...
            // and ... + weightUb * f_i + ..., respectively.
            unsigned newIndex = sourceNeuronIndex * targetLayerSize + j;
            unsigned oldIndex = i * targetLayerSize + j;

            // Update the symbolic lower bound
            double weightLb = symbolicLb[oldIndex];
            if ( weightLb >= 0 )
            {
                symbolicLbInTermsOfPredecessor[newIndex] += weightLb * coeffLb;
                symbolicLowerBias[j] += weightLb * lowerBias;
            }
            else
            {
                symbolicLbInTermsOfPredecessor[newIndex] += weightLb * coeffUb;
                symbolicLowerBias[j] += weightLb * upperBias;
            }

            // Update the symbolic upper bound
            double weightUb = symbolicUb[oldIndex];
            if ( weightUb >= 0 )
            {
                symbolicUbInTermsOfPredecessor[newIndex] += weightUb * coeffUb;
                symbolicUpperBias[j] += weightUb * upperBias;
            }
            else
            {
                symbolicUbInTermsOfPredecessor[newIndex] += weightUb * coeffLb;
                symbolicUpperBias[j] += weightUb * lowerBias;
            }
        }
    }
}

void DeepPolySignElement::allocateMemory()
{
    freeMemoryIfNeeded();

    DeepPolyElement::allocateMemory();

    _symbolicLb = new double[_size];
    _symbolicUb = new double[_size];

    std::fill_n( _symbolicLb, _size, 0 );
    std::fill_n( _symbolicUb, _size, 0 );

    _symbolicLowerBias = new double[_size];
    _symbolicUpperBias = new double[_size];

    std::fill_n( _symbolicLowerBias, _size, 0 );
    std::fill_n( _symbolicUpperBias, _size, 0 );
}

void DeepPolySignElement::freeMemoryIfNeeded()
{
    DeepPolyElement::freeMemoryIfNeeded();
    if ( _symbolicLb )
    {
        delete[] _symbolicLb;
        _symbolicLb = NULL;
    }
    if ( _symbolicUb )
    {
        delete[] _symbolicUb;
        _symbolicUb = NULL;
    }
    if ( _symbolicLowerBias )
    {
        delete[] _symbolicLowerBias;
        _symbolicLowerBias = NULL;
    }
    if ( _symbolicUpperBias )
    {
        delete[] _symbolicUpperBias;
        _symbolicUpperBias = NULL;
    }
}

void DeepPolySignElement::log( const String &message )
{
    if ( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
        printf( "DeepPolySignElement: %s\n", message.ascii() );
}

} // namespace NLR
