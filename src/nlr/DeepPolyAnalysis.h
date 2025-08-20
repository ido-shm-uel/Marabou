/*********************                                                        */
/*! \file DeepPolyAnalysis.h
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

#ifndef __DeepPolyAnalysis_h__
#define __DeepPolyAnalysis_h__

#include "DeepPolyElement.h"
#include "Layer.h"
#include "LayerOwner.h"
#include "Map.h"

#include <climits>

namespace NLR {

class DeepPolyAnalysis
{
public:
    DeepPolyAnalysis( LayerOwner *layerOwner,
                      bool storeOutputLayerSymbolicBounds = false,
                      bool storeSymbolicBoundsInTermsOfPredecessor = false,
                      bool useParameterisedSBT = false,
                      Map<unsigned, Vector<double>> *layerIndicesToParameters = NULL,
                      Map<unsigned, Vector<double>> *outputLayerSymbolicLb = NULL,
                      Map<unsigned, Vector<double>> *outputLayerSymbolicUb = NULL,
                      Map<unsigned, Vector<double>> *outputLayerSymbolicLowerBias = NULL,
                      Map<unsigned, Vector<double>> *outputLayerSymbolicUpperBias = NULL,
                      Map<unsigned, Vector<double>> *symbolicLbInTermsOfPredecessor = NULL,
                      Map<unsigned, Vector<double>> *symbolicUbInTermsOfPredecessor = NULL,
                      Map<unsigned, Vector<double>> *symbolicLowerBiasInTermsOfPredecessor = NULL,
                      Map<unsigned, Vector<double>> *symbolicUpperBiasInTermsOfPredecessor = NULL );
    ~DeepPolyAnalysis();

    void run();

private:
    LayerOwner *_layerOwner;
    bool _storeOutputLayerSymbolicBounds;
    bool _storeSymbolicBoundsInTermsOfPredecessor;
    bool _useParameterisedSBT;
    Map<unsigned, Vector<double>> *_layerIndicesToParameters;

    /*
      Maps layer index to the abstract element
    */
    Map<unsigned, DeepPolyElement *> _deepPolyElements;

    /*
      Working memory for the abstract elements to execute
    */
    double *_work1SymbolicLb;
    double *_work1SymbolicUb;
    double *_work2SymbolicLb;
    double *_work2SymbolicUb;
    double *_workSymbolicLowerBias;
    double *_workSymbolicUpperBias;

    Map<unsigned, Vector<double>> *_outputLayerSymbolicLb;
    Map<unsigned, Vector<double>> *_outputLayerSymbolicUb;
    Map<unsigned, Vector<double>> *_outputLayerSymbolicLowerBias;
    Map<unsigned, Vector<double>> *_outputLayerSymbolicUpperBias;

    Map<unsigned, Vector<double>> *_symbolicLbInTermsOfPredecessor;
    Map<unsigned, Vector<double>> *_symbolicUbInTermsOfPredecessor;
    Map<unsigned, Vector<double>> *_symbolicLowerBiasInTermsOfPredecessor;
    Map<unsigned, Vector<double>> *_symbolicUpperBiasInTermsOfPredecessor;

    unsigned _maxLayerSize;

    void allocateMemory();
    void freeMemoryIfNeeded();

    DeepPolyElement *createDeepPolyElement( Layer *layer );

    void log( const String &message );
};

} // namespace NLR

#endif // __DeepPolyAnalysis_h__
