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
                      bool storeSymbolicBounds = false,
                      Map<unsigned, double *> *outputLayerSymbolicLb = NULL,
                      Map<unsigned, double *> *outputLayerSymbolicUb = NULL,
                      Map<unsigned, double *> *outputLayerSymbolicLowerBias = NULL,
                      Map<unsigned, double *> *outputLayerSymbolicUpperBias = NULL );
    ~DeepPolyAnalysis();

    void run();

private:
    LayerOwner *_layerOwner;
    bool _storeSymbolicBounds;

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

    Map<unsigned, double *> *_outputLayerSymbolicLb;
    Map<unsigned, double *> *_outputLayerSymbolicUb;
    Map<unsigned, double *> *_outputLayerSymbolicLowerBias;
    Map<unsigned, double *> *_outputLayerSymbolicUpperBias;

    unsigned _maxLayerSize;

    void allocateMemory();
    void freeMemoryIfNeeded();

    DeepPolyElement *createDeepPolyElement( Layer *layer );

    void log( const String &message );
};

} // namespace NLR

#endif // __DeepPolyAnalysis_h__
