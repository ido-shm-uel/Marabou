/*********************                                                        */
/*! \file DeepPolyElement.h
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

#ifndef __DeepPolyElement_h__
#define __DeepPolyElement_h__

#include "Layer.h"
#include "MStringf.h"
#include "Map.h"
#include "NLRError.h"

#include <climits>

namespace NLR {

class DeepPolyElement
{
public:
    DeepPolyElement( bool storeSymbolicBounds = false );
    virtual ~DeepPolyElement(){};

    // execute the abstract layer based on the abstract layers topologically
    // before it.
    virtual void execute( const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore ) = 0;

    /*
      Given the symbolic bounds of some layer Y (of size layerSize) in terms of
      this layer, add (to the last four arugment) the symbolic bounds of layer
      Y in terms of an immediate predecessor of this layer.
    */
    virtual void symbolicBoundInTermsOfPredecessor( const double *symbolicLb,
                                                    const double *symbolicUb,
                                                    double *symbolicLowerBias,
                                                    double *symbolicUpperBias,
                                                    double *symbolicLbInTermsOfPredecessor,
                                                    double *symbolicUbInTermsOfPredecessor,
                                                    unsigned targetLayerSize,
                                                    DeepPolyElement *predecessor ) = 0;

    /*
      Returns whether this abstract element has a predecessor.
    */
    bool hasPredecessor();

    /*
      Returns the layer index corresponding to the predecessor of this element.
    */
    const Map<unsigned, unsigned> &getPredecessorIndices() const;

    unsigned getSize() const;
    unsigned getLayerIndex() const;
    Layer::Type getLayerType() const;
    double *getSymbolicLb() const;
    double *getSymbolicUb() const;
    double *getSymbolicLowerBias() const;
    double *getSymbolicUpperBias() const;
    double getLowerBound( unsigned index ) const;
    double getUpperBound( unsigned index ) const;
    void setStoreSymbolicBounds( bool storeSymbolicBounds );

    void setWorkingMemory( double *work1SymbolicLb,
                           double *work1SymbolicUb,
                           double *work2SymbolicLb,
                           double *work2SymbolicUb,
                           double *workSymbolicLowerBias,
                           double *workSymbolicUpperBias );

    void
    setOutputLayerSymbolicBoundsMemory( Map<unsigned, double *> *outputLayerSymbolicLb,
                                        Map<unsigned, double *> *outputLayerSymbolicUb,
                                        Map<unsigned, double *> *outputLayerSymbolicLowerBias,
                                        Map<unsigned, double *> *outputLayerSymbolicUpperBias );

    void storeWorkSymbolicBounds( unsigned sourceLayerSize,
                                  double *work1SymbolicLb,
                                  double *work1SymbolicUb,
                                  double *workSymbolicLowerBias,
                                  double *workSymbolicUpperBias,
                                  Map<unsigned, double *> &residualLb,
                                  Map<unsigned, double *> &residualUb,
                                  Set<unsigned> &residualLayerIndices,
                                  const Map<unsigned, DeepPolyElement *> &deepPolyElementsBefore );

    double getLowerBoundFromLayer( unsigned index ) const;
    double getUpperBoundFromLayer( unsigned index ) const;

protected:
    Layer *_layer;
    unsigned _size;
    unsigned _layerIndex;
    bool _storeSymbolicBounds;

    /*
      Abstract element described in
      https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
      The symbolicBounds are in terms of the preceding layer.
    */
    double *_symbolicLb;
    double *_symbolicUb;
    double *_symbolicLowerBias;
    double *_symbolicUpperBias;
    double *_lb;
    double *_ub;

    double *_work1SymbolicLb;
    double *_work1SymbolicUb;
    double *_work2SymbolicLb;
    double *_work2SymbolicUb;
    double *_workSymbolicLowerBias;
    double *_workSymbolicUpperBias;

    Map<unsigned, double *> *_outputLayerSymbolicLb;
    Map<unsigned, double *> *_outputLayerSymbolicUb;
    Map<unsigned, double *> *_outputLayerSymbolicLowerBias;
    Map<unsigned, double *> *_outputLayerSymbolicUpperBias;

    void allocateMemory();
    void freeMemoryIfNeeded();

    // Obtain the concrete bounds stored in _layer.
    void getConcreteBounds();
};

} // namespace NLR

#endif // __DeepPolyElement_h__
