/*********************                                                        */
/*! \file DeepPolyElement.cpp
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

#include "DeepPolyElement.h"

namespace NLR {

DeepPolyElement::DeepPolyElement()
    : _layer( NULL )
    , _size( 0 )
    , _layerIndex( 0 )
    , _storeOutputLayerSymbolicBounds( false )
    , _storeSymbolicBoundsInTermsOfPredecessor( false )
    , _symbolicLb( NULL )
    , _symbolicUb( NULL )
    , _symbolicLowerBias( NULL )
    , _symbolicUpperBias( NULL )
    , _lb( NULL )
    , _ub( NULL )
    , _work1SymbolicLb( NULL )
    , _work1SymbolicUb( NULL )
    , _work2SymbolicLb( NULL )
    , _work2SymbolicUb( NULL )
    , _workSymbolicLowerBias( NULL )
    , _workSymbolicUpperBias( NULL ){};

unsigned DeepPolyElement::getSize() const
{
    return _size;
}

unsigned DeepPolyElement::getLayerIndex() const
{
    return _layerIndex;
}

Layer::Type DeepPolyElement::getLayerType() const
{
    return _layer->getLayerType();
}

bool DeepPolyElement::hasPredecessor()
{
    return !_layer->getSourceLayers().empty();
}

const Map<unsigned, unsigned> &DeepPolyElement::getPredecessorIndices() const
{
    const Map<unsigned, unsigned> &sourceLayers = _layer->getSourceLayers();
    return sourceLayers;
}

double *DeepPolyElement::getSymbolicLb() const
{
    return _symbolicLb;
}

double *DeepPolyElement::getSymbolicUb() const
{
    return _symbolicUb;
}

double *DeepPolyElement::getSymbolicLowerBias() const
{
    return _symbolicLowerBias;
}

double *DeepPolyElement::getSymbolicUpperBias() const
{
    return _symbolicUpperBias;
}

double DeepPolyElement::getLowerBound( unsigned index ) const
{
    ASSERT( index < getSize() );
    return _lb[index];
}

double DeepPolyElement::getUpperBound( unsigned index ) const
{
    ASSERT( index < getSize() );
    return _ub[index];
}

void DeepPolyElement::setStoreOutputLayerSymbolicBounds( bool storeOutputLayerSymbolicBounds )
{
    _storeOutputLayerSymbolicBounds = storeOutputLayerSymbolicBounds;
}

void DeepPolyElement::setStoreSymbolicBoundsInTermsOfPredecessor(
    bool storeSymbolicBoundsInTermsOfPredecessor )
{
    _storeSymbolicBoundsInTermsOfPredecessor = storeSymbolicBoundsInTermsOfPredecessor;
}

double DeepPolyElement::getLowerBoundFromLayer( unsigned index ) const
{
    ASSERT( index < getSize() );
    return _layer->getLb( index );
}

double DeepPolyElement::getUpperBoundFromLayer( unsigned index ) const
{
    ASSERT( index < getSize() );
    return _layer->getUb( index );
}

void DeepPolyElement::getConcreteBounds()
{
    unsigned size = getSize();
    for ( unsigned i = 0; i < size; ++i )
    {
        _lb[i] = _layer->getLb( i );
        _ub[i] = _layer->getUb( i );
    }
}

void DeepPolyElement::allocateMemory()
{
    freeMemoryIfNeeded();

    unsigned size = getSize();
    _lb = new double[size];
    _ub = new double[size];

    std::fill_n( _lb, size, FloatUtils::negativeInfinity() );
    std::fill_n( _ub, size, FloatUtils::infinity() );
}

void DeepPolyElement::freeMemoryIfNeeded()
{
    if ( _lb )
    {
        delete[] _lb;
        _lb = NULL;
    }

    if ( _ub )
    {
        delete[] _ub;
        _ub = NULL;
    }
}

void DeepPolyElement::setWorkingMemory( double *work1SymbolicLb,
                                        double *work1SymbolicUb,
                                        double *work2SymbolicLb,
                                        double *work2SymbolicUb,
                                        double *workSymbolicLowerBias,
                                        double *workSymbolicUpperBias )
{
    _work1SymbolicLb = work1SymbolicLb;
    _work1SymbolicUb = work1SymbolicUb;
    _work2SymbolicLb = work2SymbolicLb;
    _work2SymbolicUb = work2SymbolicUb;
    _workSymbolicLowerBias = workSymbolicLowerBias;
    _workSymbolicUpperBias = workSymbolicUpperBias;
}

void DeepPolyElement::setSymbolicBoundsMemory(
    Map<unsigned, double *> *outputLayerSymbolicLb,
    Map<unsigned, double *> *outputLayerSymbolicUb,
    Map<unsigned, double *> *outputLayerSymbolicLowerBias,
    Map<unsigned, double *> *outputLayerSymbolicUpperBias,
    Map<unsigned, double *> *symbolicLbInTermsOfPredecessor,
    Map<unsigned, double *> *symbolicUbInTermsOfPredecessor,
    Map<unsigned, double *> *symbolicLowerBiasInTermsOfPredecessor,
    Map<unsigned, double *> *symbolicUpperBiasInTermsOfPredecessor )
{
    _outputLayerSymbolicLb = outputLayerSymbolicLb;
    _outputLayerSymbolicUb = outputLayerSymbolicUb;
    _outputLayerSymbolicLowerBias = outputLayerSymbolicLowerBias;
    _outputLayerSymbolicUpperBias = outputLayerSymbolicUpperBias;
    _symbolicLbInTermsOfPredecessor = symbolicLbInTermsOfPredecessor;
    _symbolicUbInTermsOfPredecessor = symbolicUbInTermsOfPredecessor;
    _symbolicLowerBiasInTermsOfPredecessor = symbolicLowerBiasInTermsOfPredecessor;
    _symbolicUpperBiasInTermsOfPredecessor = symbolicUpperBiasInTermsOfPredecessor;
}

void DeepPolyElement::storeOutputSymbolicBounds(
    unsigned sourceLayerSize,
    double *work1SymbolicLb,
    double *work1SymbolicUb,
    double *workSymbolicLowerBias,
    double *workSymbolicUpperBias,
    Map<unsigned, double *> &residualLb,
    Map<unsigned, double *> &residualUb,
    Set<unsigned> &residualLayerIndices,
    const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore )
{
    double *workSymbolicLowerBias2 = new double[_size];
    double *workSymbolicUpperBias2 = new double[_size];
    memcpy( workSymbolicLowerBias2, workSymbolicLowerBias, _size * sizeof( double ) );
    memcpy( workSymbolicUpperBias2, workSymbolicUpperBias, _size * sizeof( double ) );

    for ( const auto &residualLayerIndex : residualLayerIndices )
    {
        DeepPolyElement *residualElement = deepPolyElementsBefore[residualLayerIndex];
        double *currentResidualLb = residualLb[residualLayerIndex];
        double *currentResidualUb = residualUb[residualLayerIndex];

        // Get concrete bounds
        for ( unsigned i = 0; i < residualElement->getSize(); ++i )
        {
            double sourceLb = residualElement->getLowerBoundFromLayer( i ) -
                              GlobalConfiguration::SYMBOLIC_TIGHTENING_ROUNDING_CONSTANT;
            double sourceUb = residualElement->getUpperBoundFromLayer( i ) +
                              GlobalConfiguration::SYMBOLIC_TIGHTENING_ROUNDING_CONSTANT;

            for ( unsigned j = 0; j < _size; ++j )
            {
                // Compute lower bound
                double weight = currentResidualLb[i * _size + j];
                if ( weight >= 0 )
                {
                    workSymbolicLowerBias2[j] += ( weight * sourceLb );
                }
                else
                {
                    workSymbolicLowerBias2[j] += ( weight * sourceUb );
                }

                // Compute upper bound
                weight = currentResidualUb[i * _size + j];
                if ( weight >= 0 )
                {
                    workSymbolicUpperBias2[j] += ( weight * sourceUb );
                }
                else
                {
                    workSymbolicUpperBias2[j] += ( weight * sourceLb );
                }
            }
        }
    }

    double *currentSymbolicLb = ( *_outputLayerSymbolicLb )[_layerIndex];
    double *currentSymbolicUb = ( *_outputLayerSymbolicUb )[_layerIndex];
    double *currentSymbolicLowerBias = ( *_outputLayerSymbolicLowerBias )[_layerIndex];
    double *currentSymbolicUpperBias = ( *_outputLayerSymbolicUpperBias )[_layerIndex];
    memcpy( currentSymbolicLb, work1SymbolicLb, _size * sourceLayerSize * sizeof( double ) );
    memcpy( currentSymbolicUb, work1SymbolicUb, _size * sourceLayerSize * sizeof( double ) );
    memcpy( currentSymbolicLowerBias, workSymbolicLowerBias2, _size * sizeof( double ) );
    memcpy( currentSymbolicUpperBias, workSymbolicUpperBias2, _size * sizeof( double ) );

    if ( workSymbolicLowerBias2 )
    {
        delete[] workSymbolicLowerBias2;
        workSymbolicLowerBias2 = NULL;
    }

    if ( workSymbolicUpperBias2 )
    {
        delete[] workSymbolicUpperBias2;
        workSymbolicUpperBias2 = NULL;
    }
}

} // namespace NLR
