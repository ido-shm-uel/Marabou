/*********************                                                        */
/*! \file Test_NetworkLevelReasoner2.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Andrew Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/

#include "../../engine/tests/MockTableau.h" // TODO: fix this
#include "FloatUtils.h"
#include "Layer.h"
#include "NetworkLevelReasoner.h"
#include "Options.h"
#include "Query.h"
#include "Tightening.h"
#include "Vector.h"

#include <cxxtest/TestSuite.h>

class MockForNetworkLevelReasoner
{
public:
};

class NetworkLevelReasonerTestSuite : public CxxTest::TestSuite
{
public:
    MockForNetworkLevelReasoner *mock;

    void setUp()
    {
        TS_ASSERT( mock = new MockForNetworkLevelReasoner );
    }

    void tearDown()
    {
        TS_ASSERT_THROWS_NOTHING( delete mock );
    }

    void populateNetworkSBTRelu( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*
              2      R       1
          x0 --- x2 ---> x4 --- x6
            \    /              /
           1 \  /              /
              \/           -1 /
              /\             /
           3 /  \           /
            /    \   R     /
          x1 --- x3 ---> x5
              1
        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::RELU, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the ReLU sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 7 );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
    }

    void populateNetworkSBTReluResidual1( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*
                     -1
             __________________
            /                  \
           /  1      R       -1  1    R    3  1
          x0 --- x1 ---> x2 --- x3 ---> x4 --- x5
                  \                            /
                   \            3             /
                    \________________________/

        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 1 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 1 );
        nlr.addLayer( 2, NLR::Layer::RELU, 1 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );
        nlr.addLayer( 4, NLR::Layer::RELU, 1 );
        nlr.addLayer( 5, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 5; ++i )
            nlr.addLayerDependency( i - 1, i );
        nlr.addLayerDependency( 0, 3 );
        nlr.addLayerDependency( 1, 5 );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 2, 0, 3, 0, -1 );
        nlr.setWeight( 4, 0, 5, 0, 3 );
        nlr.setWeight( 0, 0, 3, 0, -1 );
        nlr.setWeight( 1, 0, 5, 0, 3 );


        nlr.setBias( 3, 0, 1 );
        nlr.setBias( 5, 0, 1 );

        // Mark the ReLU sources
        nlr.addActivationSource( 1, 0, 2, 0 );

        nlr.addActivationSource( 3, 0, 4, 0 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 1 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 3 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 5, 0 ), 5 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 6 );
        tableau.setLowerBound( 1, -large );
        tableau.setUpperBound( 1, large );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
    }

    void populateNetworkSBTReluResidual2( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*
                     -1
             __________________
            /                  \
           /  1      R       -1  1    R     3  1   1
          x0 --- x1 ---> x2 --- x3 ---> x4 --- x5 --- x6
           \                                   /
            \                1                /
             \_______________________________/

        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 1 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 1 );
        nlr.addLayer( 2, NLR::Layer::RELU, 1 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );
        nlr.addLayer( 4, NLR::Layer::RELU, 1 );
        nlr.addLayer( 5, NLR::Layer::WEIGHTED_SUM, 1 );
        nlr.addLayer( 6, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 6; ++i )
            nlr.addLayerDependency( i - 1, i );
        nlr.addLayerDependency( 0, 3 );
        nlr.addLayerDependency( 0, 5 );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 2, 0, 3, 0, -1 );
        nlr.setWeight( 4, 0, 5, 0, 3 );
        nlr.setWeight( 0, 0, 3, 0, -1 );
        nlr.setWeight( 0, 0, 5, 0, 1 );
        nlr.setWeight( 5, 0, 6, 0, 1 );

        nlr.setBias( 3, 0, 1 );
        nlr.setBias( 5, 0, 1 );

        // Mark the ReLU sources
        nlr.addActivationSource( 1, 0, 2, 0 );

        nlr.addActivationSource( 3, 0, 4, 0 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 1 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 3 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 5, 0 ), 5 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 6, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 7 );
        tableau.setLowerBound( 1, -large );
        tableau.setUpperBound( 1, large );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
    }

    void populateNetworkSBTReluReindex( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*

              1             1            1   1
          x0 --- x2    x5 --- x6     x9 --- x10
            \    /\    /\    /  \    / \    /
           1 \  / R\  /-1\  /  R \  / 1 \  /
              \/    \/    \/      \/     \/
              /\    /\    /\      /\     /\
           1 /  \ R/  \ 1/  \  R /  \ 1 /  \
            /    \/    \/    \  /    \ / 0  \
          x1 --- x3    x4 --- x7     x8 --- x11
              -1           1

          The example described in Fig. 3 of
          https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::RELU, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 4, NLR::Layer::RELU, 2 );
        nlr.addLayer( 5, NLR::Layer::WEIGHTED_SUM, 2 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 5; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 1 );
        nlr.setWeight( 0, 1, 1, 1, -1 );

        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 0, 3, 1, -1 );
        nlr.setWeight( 2, 1, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 1, 1 );

        nlr.setWeight( 4, 0, 5, 0, 1 );
        nlr.setWeight( 4, 0, 5, 1, 1 );
        nlr.setWeight( 4, 1, 5, 0, 1 );
        nlr.setWeight( 4, 1, 5, 1, 0 );

        nlr.setBias( 5, 0, 1 );

        // Mark the ReLU sources
        nlr.addActivationSource( 1, 0, 2, 1 );
        nlr.addActivationSource( 1, 1, 2, 0 );

        nlr.addActivationSource( 3, 0, 4, 1 );
        nlr.addActivationSource( 3, 1, 4, 0 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 1 ), 7 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 0 ), 9 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 1 ), 8 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 5, 0 ), 10 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 5, 1 ), 11 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 12 );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
        tableau.setLowerBound( 7, -large );
        tableau.setUpperBound( 7, large );
        tableau.setLowerBound( 8, -large );
        tableau.setUpperBound( 8, large );
        tableau.setLowerBound( 9, -large );
        tableau.setUpperBound( 9, large );
        tableau.setLowerBound( 10, -large );
        tableau.setUpperBound( 10, large );
        tableau.setLowerBound( 11, -large );
        tableau.setUpperBound( 11, large );
    }

    void populateNetworkSBTLeakyReLU( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*

              1      LR      1     LR      1   1
          x0 --- x2 ---> x4 --- x6 ---> x8 --- x10
            \    /        \    /          \    /
           1 \  /        1 \  /          0 \  /
              \/            \/              \/
              /\            /\              /\
           1 /  \        1 /  \          1 /  \
            /    \   LR   /    \    LR    / 1  \
          x1 --- x3 ---> x5 --- x7 ---> x9 --- x11
              -1            -1

          The example described in Fig. 3 of
          https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
          using LeakyReLU activation instead of ReLU
        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::LEAKY_RELU, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 4, NLR::Layer::LEAKY_RELU, 2 );
        nlr.addLayer( 5, NLR::Layer::WEIGHTED_SUM, 2 );

        nlr.getLayer( 2 )->setAlpha( 0.2 );
        nlr.getLayer( 4 )->setAlpha( 0.2 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 5; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 1 );
        nlr.setWeight( 0, 1, 1, 1, -1 );

        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 0, 3, 1, 1 );
        nlr.setWeight( 2, 1, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 1, -1 );

        nlr.setWeight( 4, 0, 5, 0, 1 );
        nlr.setWeight( 4, 0, 5, 1, 0 );
        nlr.setWeight( 4, 1, 5, 0, 1 );
        nlr.setWeight( 4, 1, 5, 1, 1 );

        nlr.setBias( 5, 0, 1 );

        // Mark the LeakyReLU sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        nlr.addActivationSource( 3, 0, 4, 0 );
        nlr.addActivationSource( 3, 1, 4, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 1 ), 7 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 0 ), 8 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 1 ), 9 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 5, 0 ), 10 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 5, 1 ), 11 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 12 );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
        tableau.setLowerBound( 7, -large );
        tableau.setUpperBound( 7, large );
        tableau.setLowerBound( 8, -large );
        tableau.setUpperBound( 8, large );
        tableau.setLowerBound( 9, -large );
        tableau.setUpperBound( 9, large );
        tableau.setLowerBound( 10, -large );
        tableau.setUpperBound( 10, large );
        tableau.setLowerBound( 11, -large );
        tableau.setUpperBound( 11, large );
    }

    void populateNetworkSBTSigmoidsAndRound( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*

              1      S       1     Rd
          x0 --- x2 ---> x4 --- x6 --- x8
            \    /        \    /
           1 \  /        1 \  /
              \/            \/
              /\            /\
           1 /  \        1 /  \
            /    \   S    /    \   Rd
          x1 --- x3 ---> x5 --- x7 --- x9
              -1            -1

        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::SIGMOID, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 4, NLR::Layer::ROUND, 2 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 4; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 1 );
        nlr.setWeight( 0, 1, 1, 1, -1 );

        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 0, 3, 1, 1 );
        nlr.setWeight( 2, 1, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 1, -1 );

        // Mark the Sigmoid sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );
        nlr.addActivationSource( 3, 0, 4, 0 );
        nlr.addActivationSource( 3, 1, 4, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 1 ), 7 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 0 ), 8 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 1 ), 9 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 10 );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
        tableau.setLowerBound( 7, -large );
        tableau.setUpperBound( 7, large );
        tableau.setLowerBound( 8, -large );
        tableau.setUpperBound( 8, large );
        tableau.setLowerBound( 9, -large );
        tableau.setUpperBound( 9, large );
    }

    void populateNetworkSBTMax( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*

              1      R          Max  2
          x0 --- x2 ---> x4 --- x6  ---> x7
           \    /               /
          1 \  /               /
             \/               /
             /\              /
          1 /  \            /
           /    \    R     /
          x1 --- x3 ---> x5
             -1

        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::RELU, 2 );
        nlr.addLayer( 3, NLR::Layer::MAX, 1 );
        nlr.addLayer( 4, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 4; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 1 );
        nlr.setWeight( 0, 1, 1, 1, -1 );
        nlr.setWeight( 3, 0, 4, 0, 2 );

        // Mark the ReLU sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Mark the Max sources
        nlr.addActivationSource( 2, 0, 3, 0 );
        nlr.addActivationSource( 2, 1, 3, 0 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 4, 0 ), 7 );
        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 8 );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
        tableau.setLowerBound( 7, -large );
        tableau.setUpperBound( 7, large );
    }

    void populateNetworkSBTSoftmax( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*


          x0      x3  S  x6

          x1      x4  S  x7

          x2      x5  S  x8

          x3 = x0 - x1 + x2 + 1
          x4 = -x0 + x1 + x2 + 2
          x5 = -x0 - x1 - x2 + 3

          x6 x7 x8 = softmax(x3, x4, x5)

          x9 = x6 + x7 + x8
          x10 = - x6 - x7 - x8

        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 3 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 3 );
        nlr.addLayer( 2, NLR::Layer::SOFTMAX, 3 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 2 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, -1 );
        nlr.setWeight( 0, 0, 1, 2, -1 );
        nlr.setWeight( 0, 1, 1, 0, -1 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 2, -1 );
        nlr.setWeight( 0, 2, 1, 0, 1 );
        nlr.setWeight( 0, 2, 1, 1, 1 );
        nlr.setWeight( 0, 2, 1, 2, -1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, 1 );
        nlr.setWeight( 2, 2, 3, 0, 1 );
        nlr.setWeight( 2, 0, 3, 1, -1 );
        nlr.setWeight( 2, 1, 3, 1, -1 );
        nlr.setWeight( 2, 2, 3, 1, -1 );

        nlr.setBias( 1, 0, 1 );
        nlr.setBias( 1, 1, 2 );
        nlr.setBias( 1, 2, 3 );

        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 0 );
        nlr.addActivationSource( 1, 2, 2, 0 );
        nlr.addActivationSource( 1, 0, 2, 1 );
        nlr.addActivationSource( 1, 1, 2, 1 );
        nlr.addActivationSource( 1, 2, 2, 1 );
        nlr.addActivationSource( 1, 0, 2, 2 );
        nlr.addActivationSource( 1, 1, 2, 2 );
        nlr.addActivationSource( 1, 2, 2, 2 );


        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 2 ), 2 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 3 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 2 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 6 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 7 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 2 ), 8 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 9 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 1 ), 10 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 11 );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
        tableau.setLowerBound( 7, -large );
        tableau.setUpperBound( 7, large );
        tableau.setLowerBound( 8, -large );
        tableau.setUpperBound( 8, large );
        tableau.setLowerBound( 9, -large );
        tableau.setUpperBound( 9, large );
        tableau.setLowerBound( 10, -large );
        tableau.setUpperBound( 10, large );
    }

    void populateNetworkSBTSoftmax2( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*


          x0      x3  S  x8

          x1      x4  S  x9

          x2      x5  S  x10

                  x6  S  x11

                  x7  S  x12

          x3 = x0 - x1 + x2 + 1
          x4 = -x0 + x1 + x2 + 2
          x5 = -x0 - x1 - x2 + 3
          x6 = -x0 - x1 - x2 + 2
          x7 = -x0 - x1 - x2 + 1

          x8 x10 x12 = softmax(x3, x5, x7)

          x9 x11 = softmax(x4, x6)

          x13 = x8 + x10 + x12
          x14 = -x8 - x10 - x12
          x15 = x9 + x11
          x16 = -x9 - x11

        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 3 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 5 );
        nlr.addLayer( 2, NLR::Layer::SOFTMAX, 5 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 4 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, -1 );
        nlr.setWeight( 0, 0, 1, 2, -1 );
        nlr.setWeight( 0, 0, 1, 3, -1 );
        nlr.setWeight( 0, 0, 1, 4, -1 );
        nlr.setWeight( 0, 1, 1, 0, -1 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 2, -1 );
        nlr.setWeight( 0, 1, 1, 3, -1 );
        nlr.setWeight( 0, 1, 1, 4, -1 );
        nlr.setWeight( 0, 2, 1, 0, 1 );
        nlr.setWeight( 0, 2, 1, 1, 1 );
        nlr.setWeight( 0, 2, 1, 2, -1 );
        nlr.setWeight( 0, 2, 1, 3, -1 );
        nlr.setWeight( 0, 2, 1, 4, -1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 2, 3, 0, 1 );
        nlr.setWeight( 2, 4, 3, 0, 1 );
        nlr.setWeight( 2, 0, 3, 1, -1 );
        nlr.setWeight( 2, 2, 3, 1, -1 );
        nlr.setWeight( 2, 4, 3, 1, -1 );
        nlr.setWeight( 2, 1, 3, 2, 1 );
        nlr.setWeight( 2, 3, 3, 2, 1 );
        nlr.setWeight( 2, 1, 3, 3, -1 );
        nlr.setWeight( 2, 3, 3, 3, -1 );

        nlr.setBias( 1, 0, 1 );
        nlr.setBias( 1, 1, 2 );
        nlr.setBias( 1, 2, 3 );
        nlr.setBias( 1, 3, 2 );
        nlr.setBias( 1, 4, 1 );

        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 2, 2, 0 );
        nlr.addActivationSource( 1, 4, 2, 0 );
        nlr.addActivationSource( 1, 0, 2, 2 );
        nlr.addActivationSource( 1, 2, 2, 2 );
        nlr.addActivationSource( 1, 4, 2, 2 );
        nlr.addActivationSource( 1, 0, 2, 4 );
        nlr.addActivationSource( 1, 2, 2, 4 );
        nlr.addActivationSource( 1, 4, 2, 4 );
        nlr.addActivationSource( 1, 1, 2, 1 );
        nlr.addActivationSource( 1, 3, 2, 1 );
        nlr.addActivationSource( 1, 1, 2, 3 );
        nlr.addActivationSource( 1, 3, 2, 3 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 2 ), 2 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 3 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 2 ), 5 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 3 ), 6 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 4 ), 7 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 8 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 9 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 2 ), 10 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 3 ), 11 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 4 ), 12 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 13 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 1 ), 14 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 2 ), 15 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 3 ), 16 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 17 );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );
        tableau.setLowerBound( 7, -large );
        tableau.setUpperBound( 7, large );
        tableau.setLowerBound( 8, -large );
        tableau.setUpperBound( 8, large );
        tableau.setLowerBound( 9, -large );
        tableau.setUpperBound( 9, large );
        tableau.setLowerBound( 10, -large );
        tableau.setUpperBound( 10, large );
        tableau.setLowerBound( 11, -large );
        tableau.setUpperBound( 11, large );
        tableau.setLowerBound( 12, -large );
        tableau.setUpperBound( 12, large );
        tableau.setLowerBound( 13, -large );
        tableau.setUpperBound( 13, large );
        tableau.setLowerBound( 14, -large );
        tableau.setUpperBound( 14, large );
        tableau.setLowerBound( 15, -large );
        tableau.setUpperBound( 15, large );
        tableau.setLowerBound( 16, -large );
        tableau.setUpperBound( 16, large );
    }

    void populateNetworkSBTBilinear( NLR::NetworkLevelReasoner &nlr, MockTableau &tableau )
    {
        /*


          x0    x2
                    x  x4 -- x5
          x1    x3

          x2 = x0 - 2 * x1
          x3 = x0 + x1
          x4 = -x5

          x4 = x2 * x3
        */

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::BILINEAR, 1 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Set the weights and biases for the weighted sum layers
        nlr.setWeight( 0, 0, 1, 0, 1 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, -2 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, -1 );

        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 0 );


        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 5 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.getBoundManager().initialize( 6 );
        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
    }

    void test_symbolic_bound_maps_relus_all_active()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1
          x2.lb = 2x0 + 3x1   : [11, 27]
          x2.ub = 2x0 + 3x1   : [11, 27]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          Both ReLUs active, bound survive through activations:

          x2 <= x4 <= x2
          x4.lb = 2x0 + 3x1   : [11, 27]
          x4.ub = 2x0 + 3x1   : [11, 27]

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          => x2 - x3 <= x6 <= x2 - x3
          x6.lb =  x0 + 2x1   : [6, 16]
          x6.ub =  x0 + 2x1   : [6, 16]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, 11, Tightening::LB ),
            Tightening( 2, 27, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 11, Tightening::LB ),
            Tightening( 4, 27, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, 6, Tightening::LB ),
            Tightening( 6, 16, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          x2 <= x4 <= x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          x2 - x3 <= x6 <= x2 - x3

          Layer 0:
          Using x2 = 2x0 + 3x1, x3 = x0 + x1:
          x0 + 2x1 <= x6 <= x0 + 2x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        {
            Options::get()->setString( Options::MILP_SOLVER_BOUND_TIGHTENING_TYPE,
                                       "backward-pmnr-random" );
            comparePMNRScores( nlr, Map<NLR::NeuronIndex, double>( {} ) );
        }
        {
            Options::get()->setString( Options::MILP_SOLVER_BOUND_TIGHTENING_TYPE,
                                       "backward-pmnr-gradient" );
            comparePMNRScores( nlr, Map<NLR::NeuronIndex, double>( {} ) );
        }
        {
            Options::get()->setString( Options::MILP_SOLVER_BOUND_TIGHTENING_TYPE,
                                       "backward-pmnr-bbps" );
            comparePMNRScores( nlr, Map<NLR::NeuronIndex, double>( {} ) );
        }
    }

    void test_symbolic_bound_maps_relus_active_and_inactive()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -30 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1 - 30
          x2.lb = 2x0 + 3x1 - 30   : [-19, -3]
          x2.ub = 2x0 + 3x1 - 30   : [-19, -3]

          x3 = x0 + x1
          x3.lb = x0 + x1   : [5, 11]
          x3.ub = x0 + x1   : [5, 11]

          First ReLU is inactive, bounds get zeroed
          Second ReLU is active, bounds surive the activation

          0 <= x4 <= 0
          x4.lb = 0
          x4.ub = 0

          x3 <= x5 <= x3
          x5.lb = x0 + x1   : [5, 11]
          x5.ub = x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          ==> -x3 <= x6 <= -x3
          x6.lb = -x0 - x1  : [-11, -5]
          x6.ub = -x0 - x1  : [-11, -5]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -19, Tightening::LB ),
            Tightening( 2, -3, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 0, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -11, Tightening::LB ),
            Tightening( 6, -5, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          -x3 <= x6 <= -x3

          Layer 0:
          Using x3 = x0 + x1:
          -x0 - x1 <= x6 <= -x0 - x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_symbolic_bound_maps_relus_active_and_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -15 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1 - 15
          x2.lb = 2x0 + 3x1 - 15   : [-4, 12]
          x2.ub = 2x0 + 3x1 - 15   : [-4, 12]

          x3 = x0 + x1
          x3.lb = x0 + x1   : [5, 11]
          x3.ub = x0 + x1   : [5, 11]

          First ReLU is undecided, bound is concretized. 12 = ub > -lb = 4, using ReLU lower
          coefficient of 1. Upper coefficient: 12/(12--4) = 12/16 = 0.75
          Second ReLU is active, bounds surive the activation

          x4 range: [-4, 12]
          x2 <= x4 <= 0.75 x2 + 3
          x4.lb = 2x0 + 3x1 - 15
          x4.ub = 0.75( 2x0 + 3x1 ) - 0.75 * 15 + 3  = 1.5x0 + 2.25x1 - 8.25

          x3 <= x5 <= x3
          x5.lb = x0 + x1   : [5, 11]
          x5.ub = x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          ==> x2 - x3 <= x6 <= 0.75x2 - x3 + 3
          x6.lb = x0 + 2x1 - 15
          x6.ub = 0.5x0 + 1.25x1 - 8.25

          x6 range: [4 + 2 - 15 = -9, 3 + 6.25 - 8.25 = 1] = [-9, 1]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -4, Tightening::LB ),
            Tightening( 2, 12, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, -4, Tightening::LB ),
            Tightening( 4, 12, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -9, Tightening::LB ),
            Tightening( 6, 1, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          x2 <= x4 <= 0.75x2 + 3
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          x2 - x3 <= x6 <= 0.75x2 - x3 + 3

          Layer 0:
          Using x2 = 2x0 + 3x1, x3 = x0 + x1:
          x0 + 2x1 - 15 <= x6 <= 0.5x0 + 1.25x1 - 8.25
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0.75, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 3, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0.75, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 3 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 0.5, 1.25 } ),
                                     Vector<double>( { -15 } ),
                                     Vector<double>( { -8.25 } ) );
    }

    void test_symbolic_bound_maps_relus_active_and_externally_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0). Should make the node unfixed.
        nlr.setBias( 1, 0, -15 );

        // However, one of the ReLU's variables has been eliminated
        nlr.eliminateVariable( 2, -3 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = -3
          x2 is eliminated, everything set to -3

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First ReLU is inactive (set externally), bounds get zeroed
          Second ReLU is active, bounds surive the activation

          0 <= x4 <= 0
          x4.lb = 0
          x4.ub = 0

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          ==> -x3 <= x6 <= -x3
          x6.lb =  - x0 - x1  : [-11, -5]
          x6.ub =  - x0 - x1  : [-11, -5]
        */

        List<Tightening> expectedBounds( {
            // x2 does not appear, because it has been eliminated

            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 0, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -11, Tightening::LB ),
            Tightening( 6, -5, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          -x3 <= x6 <= -x3

          Layer 0:
          Using x3 = x0 + x1:
          -x0 - x1 <= x6 <= -x0 - x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_symbolic_bound_maps_relu_residual1()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTReluResidual1( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [-1, 1]

          Layers 1. 2:

          x1 = x0
          x1.lb = x0   : [-1, 1]
          x1.ub = x0   : [-1, 1]

          ReLU is undecided, bound is concretized. 1 = ub <= -lb = 1, using ReLU lower
          coefficient of 0. Upper coefficient: 1/( 1--1 ) = 1/2 = 0.5

          0 <= x2 <= 0.5x1 + 0.5
          x2.lb = 0
          x2.ub = 0.5x0 + 0.5
          x2 range: [0, 1]

          Layers 3, 4 (with residual from x0):

          x3 = - x2 - x0 + 1
          x3.lb = -1( 0.5x0 + 0.5 ) -x0 + 1 = -1.5x0 + 0.5 : [-1, 2]
          x3.ub = -1( 0 ) -1x0 + 1 = -x0 + 1 : [0, 2]
          x3 range: [-1, 2]

          ReLU is undecided, bound is concretized. 2 = ub > -lb = 1, using ReLU lower
          coefficient of 1. Upper coefficient: 2/( 2--1 ) = 2/3.

          x3 <= x4 <= 2/3 x3 + 2/3
          x4.lb = -1.5x0 + 0.5
          x4.ub = 2/3 ( -x0 + 1 ) + 2/3 = -2/3 x0 + 4/3 : [1, 2]
          x4 range: [-1, 2]

          Layer 5 (with residual from x1):

          x5 = 3x4 + 3x1 + 1
          x5.lb =  3 ( -1.5x0 + 0.5 ) + 3 ( x0 ) + 1 = -1.5x0 + 2.5 : [1, 4]
          x5.ub =  3 ( -2/3 x0 + 4/3 ) + 3 ( x0 ) + 1 = x0 + 5 : [4, 6]
          x5 range: [1, 6]
        */

        List<Tightening> expectedBounds( {
            Tightening( 1, -1, Tightening::LB ),
            Tightening( 1, 1, Tightening::UB ),
            Tightening( 2, 0, Tightening::LB ),
            Tightening( 2, 1, Tightening::UB ),
            Tightening( 3, -1, Tightening::LB ),
            Tightening( 3, 2, Tightening::UB ),
            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, 2, Tightening::UB ),
            Tightening( 5, 1, Tightening::LB ),
            Tightening( 5, 6, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x2 <= 0.5x1 + 0.5

          Layer 4 (RELU):
          x3 <= x4 <= 2/3 x3 + 2/3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 5:
          x5 <= x5 <= x5

          Layer 4:
          Using x5 = 3x4 + 3x1 + 1:
          3x4 + 3x1 + 1 <= x5 <= 3x4 + 3x1 + 1
          Concretizing residual using x1 : [-1, 1]: 3x4 - 2 <= x5 <= 3x4 + 4

          Layer 3:
          Using x3 <= x4 <= 2/3 x3 + 2/3:
          3x3 + 3x1 + 1 <= x5 <= 2x3 + 3x1 + 3
          Concretizing residual using x1 : [-1, 1]: 3x3 - 2 <= x5 <= 2x3 + 6

          Layer 2:
          Using x3 = -x2 - x0 + 1:
          -3x2 + 3x1 - 3x0 + 4 <= x5 <= -2x2 + 3x1 - 2x0 + 5
          Concretizing residual using x0 : [-1, 1], x1 : [-1, 1]: -3x2 - 2 <= x5 <= -2x2 + 10

          Layer 1:
          Using 0 <= x2 <= 0.5x1 + 0.5:
          1.5x1 - 3x0 + 2.5 <= x5 <= 3x1 - 2x0 + 5
          Concretizing residual using x0 : [-1, 1]: 1.5x1 - 0.5 <= x5 <= 3x1 + 7

          Layer 0:
          Using x1 = x0:
          -1.5x0 + 2.5 <= x5 <= x0 + 5
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.5 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 1 } ),
                                          Vector<double>( { 0.6667 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.6667 } ) );

        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 4 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { -3 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 10 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1.5 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { -0.5 } ),
                                     Vector<double>( { 7 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1.5 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 2.5 } ),
                                     Vector<double>( { 5 } ) );
    }

    void test_symbolic_bound_maps_relu_residual2()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTReluResidual2( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [-1, 1]

          Layers 1, 2:

          x1 = x0
          x1.lb = x0   : [-1, 1]
          x1.ub = x0   : [-1, 1]

          ReLU is undecided, bound is concretized. 1 = ub <= -lb = 1, using ReLU lower
          coefficient of 0. Upper coefficient: 1/( 1--1 ) = 1/2 = 0.5

          0.5 x1 <= x2 <= 0.5x1 + 0.5
          x2.lb = 0
          x2.ub = 0.5x0 + 0.5
          x2 range: [0, 1]

          Layers 3, 4 (with residual from x0):

          x3 = - x2 - x0 + 1
          x3.lb = -1( 0.5x0 + 0.5 ) -x0 + 1 = -1.5x0 + 0.5 : [-1, 2]
          x3.ub = -1( 0 ) -1x0 + 1 = -x0 + 1 : [0, 2]
          x3 range: [-1, 2]

          ReLU is undecided, bound is concretized. 2 = ub > -lb = 1, using ReLU lower
          coefficient of 1. Upper coefficient: 2/( 2--1 ) = 2/3.

          x3 <= x4 <= 2/3 x3 + 2/3
          x4.lb = -1.5x0 + 0.5
          x4.ub = 2/3 ( -x0 + 1 ) + 2/3 = -2/3 x0 + 4/3 : [1, 2]
          x4 range: [-1, 2]

          Layer 5 (with residual from x0):

          x5 = 3x4 + x0 + 1
          x5.lb =  3 ( -1.5x0 + 0.5 ) + 1 ( x0 ) + 1 = -3.5x0 + 2.5 : [-1, 6]
          x5.ub =  3 ( -2/3 x0 + 4/3 ) + 1 ( x0 ) + 1 = -x0 + 5 : [4, 6]
          x5 range: [-1, 6]

          Layer 6:
          x6 = x5
          x6.lb = -3.5x0 + 2.5 : [-1, 6]
          x6.ub = -x0 + 5 : [4, 6]
          x6 range: [-1, 6]
        */

        List<Tightening> expectedBounds( {
            Tightening( 1, -1, Tightening::LB ),
            Tightening( 1, 1, Tightening::UB ),
            Tightening( 2, 0, Tightening::LB ),
            Tightening( 2, 1, Tightening::UB ),
            Tightening( 3, -1, Tightening::LB ),
            Tightening( 3, 2, Tightening::UB ),
            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, 2, Tightening::UB ),
            Tightening( 5, -1, Tightening::LB ),
            Tightening( 5, 6, Tightening::UB ),
            Tightening( 6, -1, Tightening::LB ),
            Tightening( 6, 6, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x2 <= 0.5x1 + 0.5

          Layer 4 (RELU):
          x3 <= x4 <= 2/3 x3 + 2/3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 6:
          x6 <= x6 <= x6

          Layer 5:
          Using x6 = x5:
          x5 <= x6 <= x5

          Layer 4:
          Using x5 = 3x4 + x0 + 1:
          3x4 + x0 + 1 <= x6 <= 3x4 + x0 + 1
          Concretizing residual using x0 : [-1, 1]: 3x4 <= x6 <= 3x4 + 2

          Layer 3:
          Using x3 <= x4 <= 2/3 x3 + 2/3:
          3x3 + x0 + 1 <= x6 <= 2x3 + x0 + 3
          Concretizing residual using x0 : [-1, 1]: 3x3 <= x6 <= 2x3 + 4

          Layer 2:
          Using x3 = -x2 - x0 + 1:
          -3x2 - 2x0 + 4 <= x6 <= -2x2 - x0 + 5
          Concretizing residual using x0 : [-1, 1]: -3x2 + 2 <= x6 <= -2x2 + 6

          Layer 1:
          Using 0 <= x2 <= 0.5x1 + 0.5:
          -1.5x1 - 2x0 + 2.5 <= x6 <= -x0 + 5
          Concretizing residual using x0 : [-1, 1]: -1.5x1 + 0.5 <= x6 <= 6

          Layer 0:
          Using x1 = x0:
          -3.5x0 + 2.5 <= x6 <= -x0 + 5
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.5 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 1 } ),
                                          Vector<double>( { 0.6667 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.6667 } ) );

        compareOutputSymbolicBounds( nlr,
                                     6,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 2 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 4 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { -3 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { -1.5 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0.5 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -3.5 } ),
                                     Vector<double>( { -1 } ),
                                     Vector<double>( { 2.5 } ),
                                     Vector<double>( { 5 } ) );
    }

    void test_symbolic_bound_maps_relu_reindex()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTReluReindex( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [-1, 1]
          x1: [-1, 1]

          Layers 1, 2:

          x2 = x0 + x1
          x2.lb = x0 + x1   : [-2, 2]
          x2.ub = x0 + x1   : [-2, 2]

          x3 = x0 - x1
          x3.lb = x0 - x1   : [-2, 2]
          x3.ub = x0 - x1   : [-2, 2]

          Both ReLUs are undecided, bounds are concretized. 2 = ub <= -lb = 2, using ReLU lower
          coefficient of 0. Upper coefficient: 2/( 2--2 ) = 2/4 = 0.5

          0 <= x4 <= 0.5x2 + 1
          x4.lb = 0
          x4.ub = 0.5 ( x0 + x1 ) + 1 = 0.5x0 + 0.5x1 + 1
          x4 range: [0, 2]

          0 <= x5 <= 0.5x3 + 1
          x5.lb = 0
          x5.ub = 0.5 ( x0 - x1 ) + 1 = 0.5x0 - 0.5x1 + 1
          x5 range: [0, 2]

          Layers 3, 4:

          x6 = x4 + x5
          x6.lb = 1 ( 0 ) + 1 ( 0 ) = 0   : [0, 0]
          x6.ub = 1 ( 0.5x0 + 0.5x1 + 1 ) + 1 ( 0.5x0 - 0.5x1 + 1 ) = x0 + 2   : [1, 3]
          x6 range: [0, 3]

          x7 = x4 - x5
          x7.lb = 1 ( 0 ) - 1 ( 0.5x0 - 0.5x1 + 1 ) = - 0.5x0 + 0.5x1 - 1   : [-2, 0]
          x7.ub = 1 ( 0.5x0 + 0.5x1 + 1 ) - 1 ( 0 ) = 0.5x0 + 0.5x1 + 1  : [0, 2]
          x7 range: [-2, 2]

          First ReLU is active, bounds surive the activation
          Second ReLUs is undecided, bound is concretized. 2 = ub <= -lb = 2, using ReLU lower
          coefficient of 0. Upper coefficient (second ReLU): 2/( 2--2 ) = 2/4 = 0.5

          x6 <= x8 <= x6
          x8.lb = 0
          x8.ub = x0 + 2
          x8 range: [0, 3]

          0 <= x9 <= 0.5 x7 + 1
          x9.lb = 0
          x9.ub = 0.5 ( 0.5x0 + 0.5x1 + 1 ) + 1 = 0.25x0 + 0.25x1 + 1.5
          x9 range: [0, 2]

          Layer 5:
          x10 = x8 + x9 + 1
          x10.lb =  1 ( 0 ) + 1 ( 0 ) + 1 = 1 : [1, 1]
          x10.ub = 1 ( x6 ) + 1 ( 0.5 x7 + 1 ) + 1 = 1 ( x4 + x5 ) + 1 ( 0.5 x4 - 0.5x5 + 1 ) + 1
          = 1.5x4 + 0.5x5 + 2 <= 0.75x2 + 0.25x3 + 4 = x0 + 0.5x1 + 4 : [2.5, 5.5]
          x10 range: [1, 5.5]

          x11 = x9
          x11.lb = 0
          x11.ub = 0.25x0 + 0.25x1 + 1.5
          x11 range: [0, 2]

        */

        List<Tightening> expectedBounds(
            { Tightening( 2, -2, Tightening::LB ), Tightening( 2, 2, Tightening::UB ),
              Tightening( 3, -2, Tightening::LB ), Tightening( 3, 2, Tightening::UB ),

              Tightening( 4, 0, Tightening::LB ),  Tightening( 4, 2, Tightening::UB ),
              Tightening( 5, 0, Tightening::LB ),  Tightening( 5, 2, Tightening::UB ),

              Tightening( 6, 0, Tightening::LB ),  Tightening( 6, 3, Tightening::UB ),
              Tightening( 7, -2, Tightening::LB ), Tightening( 7, 2, Tightening::UB ),

              Tightening( 8, 0, Tightening::LB ),  Tightening( 8, 3, Tightening::UB ),
              Tightening( 9, 0, Tightening::LB ),  Tightening( 9, 2, Tightening::UB ),

              Tightening( 10, 1, Tightening::LB ), Tightening( 10, 5.5, Tightening::UB ),
              Tightening( 11, 0, Tightening::LB ), Tightening( 11, 2, Tightening::UB )

            } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0.5x2 + 1
          0 <= x5 <= 0.5x3 + 1

          Layer 4 (RELU):
          x6 <= x8 <= x6
          0 <= x9 <= 0.5 x7 + 1

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 5:
          x10 <= x10 <= x10
          x11 <= x11 <= x11

          Layer 4:
          Using x10 = x8 + x9 + 1, x11 = x9:
          x8 + x9 + 1 <= x10 <= x8 + x9 + 1
          x9 <= x11 <= x9

          Layer 3:
          Using x6 <= x8 <= x6, 0 <= x9 <= 0.5 x7 + 1:
          x6 + 1 <= x10 <= x6 + 0.5 x7 + 2
          0 <= x11 <= 0.5 x7 + 1

          Layer 2:
          Using x6 = x4 + x5, x7 = x4 - x5:
          x4 + x5 + 1 <= x10 <= 1.5x4 + 0.5x5 + 2
          0 <= x11 <= 0.5x4 - 0.5x5 + 1

          Layer 1:
          Using 0 <= x4 <= 0.5x2 + 1, 0 <= x5 <= 0.5x3 + 1:
          1 <= x10 <= 0.75x2 + 0.25x3 + 4
          0 <= x11 <= 0.25x2 + 1.5

          Layer 0:
          Using x2 = x0 + x1, x3 = x0 - x1:
          1 <= x10 <= x0 + 0.5x1 + 4
          0 <= x11 <= 0.25x2 + 0.25x3 + 1.5
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 0 } ),
                                          Vector<double>( { 0, 0.5, 0.5, 0 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 1, 1 } ) );

        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 0, 1, 0, 0 } ),
                                          Vector<double>( { 0, 1, 0.5, 0 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 1, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1, 1, 1, 0 } ),
                                     Vector<double>( { 1, 1, 1, 0 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 1, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1, 0, 0, 0 } ),
                                     Vector<double>( { 1, 0, 0.5, 0.5 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 2, 1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, 0, 1, 0 } ),
                                     Vector<double>( { 0.5, -0.5, 1.5, 0.5 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 2, 1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, 0, 0, 0 } ),
                                     Vector<double>( { 0.75, 0.25, 0.25, 0 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 4, 1.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0, 0, 0, 0 } ),
                                     Vector<double>( { 1, 0.25, 0.5, 0.25 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 4, 1.5 } ) );
    }

    void test_symbolic_bound_maps_abs_all_positive()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1
          x2.lb = 2x0 + 3x1   : [11, 27]
          x2.ub = 2x0 + 3x1   : [11, 27]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          Both absolute values positive, bound survive through activations:

          x2 <= x4 <= x2
          x4.lb = 2x0 + 3x1   : [11, 27]
          x4.ub = 2x0 + 3x1   : [11, 27]

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:
          x5 = x4 - x5
          => x2 - x3 <= x5 <= x2 - x3
          x6.lb =  x0 + 2x1   : [6, 16]
          x6.ub =  x0 + 2x1   : [6, 16]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, 11, Tightening::LB ),
            Tightening( 2, 27, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 11, Tightening::LB ),
            Tightening( 4, 27, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, 6, Tightening::LB ),
            Tightening( 6, 16, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          x2 <= x4 <= x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          x2 - x3 <= x6 <= x2 - x3

          Layer 0:
          Using x2 = 2x0 + 3x1, x3 = x0 + x1:
          x0 + 2x1 <= x6 <= x0 + 2x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_symbolic_bound_maps_abs_positive_and_negative()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -30 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:
          x2 = 2x0 + 3x1 - 30
          x2.lb = 2x0 + 3x1 - 30   : [-19, -3]
          x2.ub = 2x0 + 3x1 - 30   : [-19, -3]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First absolute value is negative, bounds get flipped
          Second absolute value is positive, bounds surive the activation

          -x2 <= x4 <= -x2
          x4.lb = -2x0 -3x1 + 30   : [3, 19]
          x4.ub = -2x0 -3x1 + 30   : [3, 19]

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:
          x5 = x4 - x5
          => -x2 - x3 <= x5 <= -x2 - x3
          x6.lb =  - 3x0 - 4x1 + 30  : [-8, 14]
          x6.ub =  - 3x0 - 4x1 + 30  : [-8, 14]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -19, Tightening::LB ),
            Tightening( 2, -3, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 3, Tightening::LB ),
            Tightening( 4, 19, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -8, Tightening::LB ),
            Tightening( 6, 14, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          -x2 <= x4 <= -x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using -x2 <= x4 <= -x2, x3 <= x5 <= x3:
          -x2 - x3 <= x6 <= -x2 - x3

          Layer 0:
          Using x2 = 2x0 + 3x1 - 30, x3 = x0 + x1:
          -3x0 - 4x1 + 30 <= x6 <= -3x0 - 4x1 + 30
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -3, -4 } ),
                                     Vector<double>( { -3, -4 } ),
                                     Vector<double>( { 30 } ),
                                     Vector<double>( { 30 } ) );
    }

    void test_symbolic_bound_maps_absolute_values_positive_and_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -15 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:
          x2 = 2x0 + 3x1 - 15
          x2.lb = 2x0 + 3x1 - 15   : [-4, 12]
          x2.ub = 2x0 + 3x1 - 15   : [-4, 12]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First absolute value is undecided, bounds are concretized.
          Second absolute value is active, bounds surive the activation

          0 <= x4 <= 12
          x4 range: [0, 12]
          x4.lb = 0
          x4.ub = 12

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          => -x3 <= x6 <= -x3 + 12
          x6.lb =  - x0 - x1       : [-11, -5]
          x6.ub =  - x0 - x1 + 12  : [  1,  7]

          x6 range: [-11, 7]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -4, Tightening::LB ),
            Tightening( 2, 12, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 12, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -11, Tightening::LB ),
            Tightening( 6, 7, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          0 <= x4 <= 12
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using 0 <= x4 <= 12, x3 <= x5 <= x3:
          -x3 <= x6 <= -x3 + 12

          Layer 0:
          Using x3 = x0 + x1:
          -x0 - x1 <= x6 <= -x0 - x1 + 12
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 12, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 12 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 12 } ) );
    }

    void test_symbolic_bound_maps_absolute_values_active_and_externally_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0). Should make the node unfixed.
        nlr.setBias( 1, 0, -15 );

        // However, the weighted sum variable has been eliminated
        nlr.eliminateVariable( 2, -3 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = -3
          x2 is eliminated, everything set to -3

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First absolute value is negative, bounds get flipped
          Second absolute value is positive, bounds surive the activation

          -x2 <= x4 <= -x2
          x4: all set to 3

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          => -x2 - x3 <= x6 <= -x2 - x3
          => -x3 + 3 <= x6 <= -x3 + 3
          x6.lb =  - x0 - x1 + 3  : [-8, -2]
          x6.ub =  - x0 - x1 + 3  : [-8, -2]
        */

        List<Tightening> expectedBounds( {
            // x2 does not appear, because it has been eliminated

            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 3, Tightening::LB ),
            Tightening( 4, 3, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -8, Tightening::LB ),
            Tightening( 6, -2, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          -x2 <= x4 <= -x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using -x2 <= x4 <= -x2, x3 <= x5 <= x3:
          -x2 - x3 <= x6 <= -x2 - x3
          x2 = -3 is eliminated.
          -x3 + 3 <= x6 <= -x3 + 3

          Layer 0:
          Using x3 = x0 + x1:
          - x0 - x1 + 3 <= x6 <= - x0 - x1 + 3
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ) );
    }

    void test_symbolic_bound_maps_signs_positive_and_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::SIGN, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Sign sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -15 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1 - 15
          x2.lb = 2x0 + 3x1 - 15   : [-4, 12]
          x2.ub = 2x0 + 3x1 - 15   : [-4, 12]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

         First sign is undecided, bounds are concretized.
          Second sign is active, bounds become constant 1
            Coefficient (first Sign, lower): 2/12 = 1/6.
            Coefficient (first Sign, upper): -2/-4 = 1/2.

          1/6 x2 - 1 <= x4 <= 1/2 x2 + 1
          x4.lb = 1/6 ( 2x0 + 3x1 - 15 ) - 1 = 2/6 x0 + 3/6 x1 - 21/6
          x4.ub = 1/2 ( 2x0 + 3x1 - 15 ) + 1 = x0 + 1.5x1 - 6.5
          x4 range: [-1, 1]

          1 <= x5 <= 1
          x5.lb = 1
          x5.ub = 1
          x5 range: [1, 1]

          Layer 3:

          x6 = x4 - x5 : [-2, 0]
          => 1/6 x2 - 2 <= x6 <= 1/2 x2 : [-8/3, 6]
            x6.lb =  1 ( 2/6 x0 + 3/6 x1 - 21/6 ) - 1 ( 1 ) = 1/3 x0 + 1/2 x1 - 4.5 : [-16/6, 0]
          x6.ub =  1 ( x0 + 1.5x1 - 6.5 ) - 1 ( 1 ) = x0 + 1.5x1 - 7.5 : [-2, 6]

          x6 range: [-2, 0]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -4, Tightening::LB ),
            Tightening( 2, 12, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, 1, Tightening::UB ),
            Tightening( 5, 1, Tightening::LB ),
            Tightening( 5, 1, Tightening::UB ),

            Tightening( 6, -2, Tightening::LB ),
            Tightening( 6, 0, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SIGN):
          1/6 x2 - 1 <= x4 <= 1/2 x2 + 1
          1 <= x5 <= 1

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using 1/6 x2 - 1 <= x4 <= 1/2 x2 + 1, 1 <= x5 <= 1:
          1/6 x2 - 2 <= x6 <= 1/2 x2

          Layer 0:
          Using x2 = 2x0 + 3x1 - 15:
          1/3 x0 + 1/2 x1 - 4.5 <= x6 <= x0 + 1.5x1 - 7.5
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.1667, 0, 0, 0 } ),
                                          Vector<double>( { 0.5, 0, 0, 0 } ),
                                          Vector<double>( { -1, 1 } ),
                                          Vector<double>( { 1, 1 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.1667, 0 } ),
                                     Vector<double>( { 0.5, 0 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.3333, 0.5 } ),
                                     Vector<double>( { 1, 1.5 } ),
                                     Vector<double>( { -4.5 } ),
                                     Vector<double>( { -7.5 } ) );
    }

    void test_symbolic_bound_maps_signs_active_and_externally_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::SIGN, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Sign sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0). Should make the node unfixed.
        nlr.setBias( 1, 0, -15 );

        // However, the weighted sum variable has been eliminated
        nlr.eliminateVariable( 2, -3 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = -3
          x2 is eliminated, everything set to -3

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First sign is negative, bounds become constant -1
          Second sign is positive, bounds become constant 1

          -1 <= x4 <= 1
          x4: all set to -1

          1 <= x5 <= 1
          x5: all set to 1

          Layer 3:

          x6 = x5 - x4
          x6.lb = 1 ( -1 ) - 1 ( 1 ) = -2
          x6.ub = 1 ( -1 ) - 1 ( 1 ) = -2
        */

        List<Tightening> expectedBounds( {
            // x2 does not appear, because it has been eliminated

            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, -1, Tightening::UB ),
            Tightening( 5, 1, Tightening::LB ),
            Tightening( 5, 1, Tightening::UB ),

            Tightening( 6, -2, Tightening::LB ),
            Tightening( 6, -2, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SIGN):
          -1 <= x4 <= -1
          1 <= x5 <= 1

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using -1 <= x4 <= -1, 1 <= x5 <= 1:
          -2 <= x6 <= -2

          Layer 0:
          -2 <= x6 <= -2
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 0 } ),
                                          Vector<double>( { 0, 0, 0, 0 } ),
                                          Vector<double>( { -1, 1 } ),
                                          Vector<double>( { -1, 1 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { -2 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { -2 } ) );
    }

    void test_symbolic_bound_maps_leaky_relu()
    {
        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTLeakyReLU( nlr, tableau ); // alpha = 0.2

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [-1, 1]
          x1: [-1, 1]

          Layers 1, 2:

          x2 = x0 + x1
          x2.lb = x0 + x1   : [-2, 2]
          x2.ub = x0 + x1   : [-2, 2]

          x3 = x0 - x1
          x3.lb = x0 - x1   : [-2, 2]
          x3.ub = x0 - x1   : [-2, 2]

          Both LeakyReLUs are undecided, bounds are concretized.
            Coefficient: ( 2 - 0.2*-2 )/( 2--2 ) = 2.4/4 = 0.6
            Bias: ( 0.2 - 1 ) * 2 * -2 / ( 2--2 ) = 0.8

          x2 <= x4 <= 0.6 x2 + 0.8
          x4.lb = x0 + x1
          x4.ub = 0.6 ( x0 + x1 ) + 0.8 = 0.6x0 + 0.6x1 + 0.8
          x4 range: [-2, 2]

          x3 <= x5 <= 0.6 x3 + 0.8
          x5.lb = x0 - x1
          x5.ub = 0.6 ( x0 - x1 ) + 0.8 = 0.6x0 - 0.6x1 + 0.8
          x5 range: [-2, 2]

          Layers 3, 4:

          x6 = x4 + x5
          => x2 + x3 <= x6 <= 0.6 x2 + 0.6 x3 + 1.6
          x6.lb = 1 ( x0 + x1 ) + 1 ( x0 - x1 ) = 2x0   : [-2, 2]
          x6.ub = 1 ( 0.6x0 + 0.6x1 + 0.8 ) + 1 ( 0.6x0 - 0.6x1 + 0.8 ) = 1.2x0 + 1.6   : [0.4, 2.8]
          x6 range: [-2, 2.8]

          x7 = x4 - x5
          => x2 - 0.6x3 - 0.8 <= x6 <= 0.6 x2 - x3 + 0.8
          x7.lb = 1 ( x0 + x1 ) - 1 ( 0.6x0 - 0.6x1 + 0.8 ) = 0.4x0 + 1.6x1 - 0.8   : [-2.8, 1.2]
          x7.ub = 1 ( 0.6x0 + 0.6x1 + 0.8 ) - 1 ( x0 - x1 ) = -0.4x0 + 1.6x1 + 0.8  : [-1.2, 2.8]
          x7 range: [-2.8, 2.8]

          Both LeakyReLUs are undecided, bounds are concretized.
            Coefficient (first LeakyReLU): ( 2.8 - 0.2*-2 )/( 2.8--2 ) = 3.2/4.8 = 10/15 = 2/3
            Bias (first LeakyReLU): ( 0.2 - 1 ) * 2.8 * -2 / ( 2.8--2 ) = 14/15

            Coefficient (second LeakyReLU): ( 2.8 - 0.2*-2.8 )/( 2.8--2.8 ) = 3.36/5.6 = 0.6
            Bias (second LeakyReLU): ( 0.2 - 1 ) * 2.8 * -2.8 / ( 2.8--2.8 ) = 1.12

          x6 <= x8 <= 10/15 x6 + 14/15
          x8.lb = 2x0
          x8.ub = 10/15 ( 1.2x0 + 1.6 ) + 14/15 = 0.8x0 + 2
          x8 range: [-2, 2.8]

          x7 <= x9 <= 0.6x7 + 1.12
          x9.lb = 0.4x0 + 1.6x1 - 0.8
          x9.ub = 0.6 ( -0.4x0 + 1.6x1 + 0.8 ) + 1.12 = -0.24 x0 + 0.96 x1 + 1.6
          x9 range: [-0.56, 2.8]

          Layer 5:

          x10 = x8 + x9 + 1
          => x6 + x7 + 1 <= x10 <= 2/3 x6 + 0.6 x7 + 229/75
          => 2x4 + 1 <= x10 <= 19/15 x4 + 1/15 x5 + 229/75
          => 2x2 + 1 <= x10 <= 0.76 x2 + 0.04 x3 + 4.12
          x10.lb = 2x0 + 2x1 + 1 : [-3, 5]
          x10.ub = 0.8 x0 + 0.72 x1 + 4.12 : [2.6, 5.64]
          x10 range: [-3, 5.64]

          x11 = x9
          => x7 <= x11 <= 0.6x7 + 1.12
          => x4 - x5 <= x11 <= 0.6x4 - 0.6x5 + 1.12
          => x2 - 0.6x3 - 0.8 <= x11 <= 0.36 x2 - 0.6 x3 + 1.6
          x11.lb = 0.4x0 + 1.6x1 - 0.8  : [-2.8, 1.2]
          x11.ub = -0.24 x0 + 0.96 x1 + 1.6 : [0.4, 2.8]
          x11 range: [-2.8, 2.8]
        */

        List<Tightening> expectedBounds(
            { Tightening( 2, -2, Tightening::LB ),    Tightening( 2, 2, Tightening::UB ),
              Tightening( 3, -2, Tightening::LB ),    Tightening( 3, 2, Tightening::UB ),

              Tightening( 4, -2, Tightening::LB ),    Tightening( 4, 2, Tightening::UB ),
              Tightening( 5, -2, Tightening::LB ),    Tightening( 5, 2, Tightening::UB ),

              Tightening( 6, -2, Tightening::LB ),    Tightening( 6, 2.8, Tightening::UB ),
              Tightening( 7, -2.8, Tightening::LB ),  Tightening( 7, 2.8, Tightening::UB ),

              Tightening( 8, -2, Tightening::LB ),    Tightening( 8, 2.8, Tightening::UB ),
              Tightening( 9, -2.8, Tightening::LB ),  Tightening( 9, 2.8, Tightening::UB ),

              Tightening( 10, -3, Tightening::LB ),   Tightening( 10, 5.64, Tightening::UB ),
              Tightening( 11, -2.8, Tightening::LB ), Tightening( 11, 2.8, Tightening::UB )

            } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (LEAKY_RELU):
          x2 <= x4 <= 0.6 x2 + 0.8
          x3 <= x5 <= 0.6 x3 + 0.8

          Layer 4 (LEAKY_RELU):
          x6 <= x8 <= 2/3 x6 + 14/15
          x7 <= x9 <= 0.6x7 + 1.12

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 5:
          x10 <= x10 <= x10
          x11 <= x11 <= x11

          Layer 4:
          Using x10 = x8 + x9 + 1, x11 = x9:
          x8 + x9 + 1 <= x10 <= x8 + x9 + 1
          x9 <= x11 <= x9

          Layer 3:
          Using x6 <= x8 <= 2/3 x6 + 14/15, x7 <= x9 <= 0.6x7 + 1.12:
          x6 + x7 + 1 <= x10 <= 2/3 x6 + 0.6 x7 + 229/75
          x7 <= x11 <= 0.6x7 + 1.12

          Layer 2:
          Using x6 = x4 + x5, x7 = x4 - x5:
          2x4 + 1 <= x10 <= 19/15 x4 + 1/15 x5 + 229/75
          x4 - x5 <= x11 <= 0.6x4 - 0.6x5 + 1.12

          Layer 1:
          Using x2 <= x4 <= 0.6 x2 + 0.8, x3 <= x5 <= 0.6 x3 + 0.8:
          2x2 + 1 <= x10 <= 0.76 x2 + 0.04 x3 + 4.12
          x2 - 0.6x3 - 0.8 <= x11 <= 0.36 x2 - 0.6 x3 + 1.6

          Layer 0:
          Using x2 = x0 + x1, x3 = x0 - x1:
          2x0 + 2x1 + 1 <= x10 <= 0.8 x0 + 0.72 x1 + 4.12
          0.4x0 + 1.6x1 - 0.8 <= x11 <= -0.24 x0 + 0.96 x1 + 1.6
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0.6, 0, 0, 0.6 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0.8, 0.8 } ) );

        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0.6667, 0, 0, 0.6 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0.9333, 1.12 } ) );

        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1, 0, 1, 1 } ),
                                     Vector<double>( { 1, 0, 1, 1 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 1, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1, 0, 1, 1 } ),
                                     Vector<double>( { 0.6667, 0, 0.6, 0.6 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 3.0533, 1.12 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 2, 1, 0, -1 } ),
                                     Vector<double>( { 1.2667, 0.6, 0.0667, -0.6 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 3.0533, 1.12 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 2, 1, 0, -0.6 } ),
                                     Vector<double>( { 0.76, 0.36, 0.04, -0.6 } ),
                                     Vector<double>( { 1, -0.8 } ),
                                     Vector<double>( { 4.12, 1.6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 2, 0.4, 2, 1.6 } ),
                                     Vector<double>( { 0.8, -0.24, 0.72, 0.96 } ),
                                     Vector<double>( { 1, -0.8 } ),
                                     Vector<double>( { 4.12, 1.6 } ) );
    }

    void test_symbolic_bound_maps_sigmoids_and_round()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTSigmoidsAndRound( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );

        // Layer 1
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getLb( 0 ), -2, 0.00001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getUb( 0 ), 2, 0.00001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getLb( 1 ), -2, 0.00001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getUb( 1 ), 2, 0.00001 ) );

        // Layer 2
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getLb( 0 ), 0.1192, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getUb( 0 ), 0.8807, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getLb( 1 ), 0.1192, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getUb( 1 ), 0.8807, 0.0001 ) );

        // Layer 3
        /*
         Double-check with Python
            ---
            from math import exp as e
            def g(x):
                return 1 / (1 + e(-x))

            def g_prime(x):
                return g(x) * (1 - g(x))

            def lam(l, u):
                return (g(u) - g(l)) / (u - l)

            def lam_prime(l, u):
                return min(g_prime(l), g_prime(u))

            l3 = l4 = -2
            u3 = u4 = 2
            l5 = l6 = g(-2)
            u5 = u6 = g(2)
            lambda7 = lam(l3, u3)
            lambda7_prime = lam_prime(l3, u3)
            lambda8 = lam(l4, u4)
            lambda8_prime = lam_prime(l4, u4)
            x7_l = lambda7_prime * (-2) + g(-2) + g(-2) - lambda7_prime * (-2 + -2)
            x7_u = lambda7_prime * (2) + g(2) + g(2) -lambda7_prime * (2 + 2)
            x8_l = lambda8_prime * (-2) + g(-2) - g(2) - lambda8_prime * (-2 - 2)
            x8_u = lambda8_prime * (2) + g(2) - g(-2) -lambda8_prime * (2 - -2)
            print(x7_l)
            print(x7_u)
            print(x8_l)
            print(x8_u)

            '''
            Sigmoid linear relaxation ( Layer 2 ):
            x4 >= lambda7_prime * x2 + ( g(l3) - lambda7_prime * l3 )
            x4 <= lambda7_prime * x2 + ( g(u3) - lambda7_prime * u3 )
            x5 >= lambda8_prime * x3 + ( g(l4) - lambda8_prime * l4 )
            x5 <= lambda8_prime * x3 + ( g(u4) - lambda7_prime * u4 )
            '''
            print('------------------')
            print(lambda7_prime)
            print(lambda8_prime)
            print(g(l3) - lambda7_prime * l3)
            print(g(u3) - lambda7_prime * u3)
            print(g(l4) - lambda8_prime * l4)
            print(g(u4) - lambda8_prime * u4)


            ---
            [output]:
            0.4483930148512481
            1.5516069851487517
            -0.5516069851487517
            0.5516069851487517
            ------------------
            0.1049935854035065
            0.1049935854035065
            0.3291900928291306
            0.6708099071708693
            0.3291900928291306
            0.6708099071708693
        */
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getLb( 0 ), 0.4483, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getUb( 0 ), 1.5516, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getLb( 1 ), -0.5516, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getUb( 1 ), 0.5516, 0.0001 ) );

        // Layer 4
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getLb( 0 ), 0 );
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getUb( 0 ), 2 );
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getLb( 1 ), -1 );
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getUb( 1 ), 1 );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SIGMOID):
          0.1050 x2 + 0.3292 <= x4 <= 0.1050 x2 + 0.6708
          0.1050 x3 + 0.3292 <= x5 <= 0.1050 x3 + 0.6708

          Layer 4 (ROUND):
          x6 - 0.5 <= x8 <= x6 + 0.5
          x7 - 0.5 <= x9 <= x7 + 0.5

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 4:
          x8 <= x8 <= x8
          x9 <= x9 <= x9

          Layer 3:
          Using x6 - 0.5 <= x8 <= x6 + 0.5, x7 - 0.5 <= x9 <= x7 + 0.5:
          x6 - 0.5 <= x8 <= x6 + 0.5
          x7 - 0.5 <= x9 <= x7 + 0.5

          Layer 2:
          Using x6 = x4 + x5, x7 = x4 - x5:
          x4 + x5 - 0.5 <= x8 <= x4 + x5 + 0.5
          x4 - x5 - 0.5 <= x9 <= x4 - x5 + 0.5

          Layer 1:
          Using
          0.1050 x2 + 0.3292 <= x4 <= 0.1050 x2 + 0.6708,
          0.1050 x3 + 0.3292 <= x5 <= 0.1050 x3 + 0.6708:
          0.1050 x2 + 0.1050 x3 + 0.1584 <= x8 <= 0.1050 x2 + 0.1050 x3 + 1.8416
          0.1050 x2 - 0.1050 x3 - 0.8416 <= x9 <= 0.1050 x2 - 0.1050 x3 + 0.8516

          Layer 0:
          Using x2 = x0 + x1, x3 = x0 - x1:
            0.2100 x0 + 0.1584 <= x8 <= 0.2100 x0 + 1.8416
            0.2100 x1 - 0.8416 <= x9 <= 0.2100 x1 + 0.8516
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.1050, 0, 0, 0.1050 } ),
                                          Vector<double>( { 0.1050, 0, 0, 0.1050 } ),
                                          Vector<double>( { 0.3292, 0.3292 } ),
                                          Vector<double>( { 0.6708, 0.6708 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { -0.5, -0.5 } ),
                                          Vector<double>( { 0.5, 0.5 } ) );

        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { -0.5, -0.5 } ),
                                     Vector<double>( { 0.5, 0.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, 1, 1, -1 } ),
                                     Vector<double>( { 1, 1, 1, -1 } ),
                                     Vector<double>( { -0.5, -0.5 } ),
                                     Vector<double>( { 0.5, 0.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.1050, 0.1050, 0.1050, -0.1050 } ),
                                     Vector<double>( { 0.1050, 0.1050, 0.1050, -0.1050 } ),
                                     Vector<double>( { 0.1584, -0.8416 } ),
                                     Vector<double>( { 1.8416, 0.8416 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.2100, 0, 0, 0.2100 } ),
                                     Vector<double>( { 0.2100, 0, 0, 0.2100 } ),
                                     Vector<double>( { 0.1584, -0.8416 } ),
                                     Vector<double>( { 1.8416, 0.8416 } ) );
    }

    void test_symbolic_bound_maps_max_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTMax( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 2 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [-1, 1]
          x1: [-1, 2]

          Layers 1, 2, 3:

          x2 = x0 + x1
          x2.lb =  x0 + x1   : [-2, 3]
          x2.ub =  x0 + x1   : [-2, 3]

          x3 = x0 - x1
          x3.lb =  x0 - x1   : [-3, 2]
          x3.ub =  x0 - x1   : [-3, 2]

          Both ReLUs are undecided, bounds are concretized.
          First ReLU: 3 = ub > -lb = 2, using lower ReLU coefficient of 1.
          Upper coefficient (first ReLU): 3/( 3--2 ) = 3/5 = 0.6.
          First ReLU: 2 = ub <= -lb = 3, using lower ReLU coefficient of 0.
          Upper coefficient (second ReLU): 2/( 2--3 ) = 2/5 = 0.4

          x2 <= x4 <= 0.6 x2 + 1.2
          x4.lb = x0 + x1
          x4.ub = 0.6 ( x0 + x1 ) + 1.2 = 0.6x0 + 0.6x1 + 1.2
          x4 range: [-2, 3]

          0 <= x5 <= 0.4 x3 + 1.2
          x5.lb =  0
          x5.ub =  0.4 ( x0 - x1 ) + 1.2 = 0.4x0 + 0.4x1 + 1.2
          x5 range: [0, 2]

          Max is not fixed because x5.lb <= x4.ub and x4.lb <= x5.ub
          Max inherits lower bound from x5, and its upper bound is constant 3.

          x5 <= x6 <= 3
          x6.lb =  0  : [0, 0]
          x6.ub =  3   : [3, 3]
          x6 range: [0, 3]

          Layer 4:

          x7 = 2x6
          => 2x5 <= x7 <= 6
          x7.lb = 2 ( 0 ) = 0   : [0, 0]
          x7.ub = 2 ( 3 ) = 6   : [6, 6]
          x7 range: [0, 6]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -2, Tightening::LB ),
            Tightening( 2, 3, Tightening::UB ),
            Tightening( 3, -3, Tightening::LB ),
            Tightening( 3, 2, Tightening::UB ),
            Tightening( 4, -2, Tightening::LB ),
            Tightening( 4, 3, Tightening::UB ),
            Tightening( 5, 0, Tightening::LB ),
            Tightening( 5, 2, Tightening::UB ),
            Tightening( 6, 0, Tightening::LB ),
            Tightening( 6, 3, Tightening::UB ),
            Tightening( 7, 0, Tightening::LB ),
            Tightening( 7, 6, Tightening::UB ),

        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          x2 <= x4 <= 0.6 x2 + 1.2
          0 <= x5 <= 0.4 x3 + 1.2

          Layer 3 (MAX):
          x5 <= x6 <= 6

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 4:
          x7 <= x7 <= x7

          Layer 3:
          Using x7 = 2x6:
          2x6 <= x7 <= 2x6

          Layer 2:
          Using x5 <= x6 <= 3:
          2x5 <= x7 <= 6

          Layer 1:
          Using 0 <= x5 <= 0.4 x3 + 1.2:
          0 <= x7 <= 6

          Layer 0:
          0 <= x7 <= 6
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 0 } ),
                                          Vector<double>( { 0.6, 0, 0, 0.4 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 1.2, 1.2 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          3,
                                          Vector<double>( { 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 3 } ) );

        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 6 } ) );
    }

    void test_symbolic_bound_maps_max_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTMax( nlr, tableau );

        tableau.setLowerBound( 0, 1 );
        tableau.setUpperBound( 0, 2 );
        tableau.setLowerBound( 1, -3 );
        tableau.setUpperBound( 1, -2 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [1, 2]
          x1: [-3, -2]

          Layer 1:

          x2 = x0 + x1
          x2.lb =  x0 + x1   : [-2, 0]
          x2.ub =  x0 + x1   : [-2, 0]

          x3 = x0 - x1
          x3.lb =  x0 - x1   : [3, 5]
          x3.ub =  x0 - x1   : [3, 5]

          First ReLU is negative, bounds become constant 0
          Second ReLU is positive, bounds survive the activation

          0 <= x4 <= 0
          x4: all set to 0

          x3 <= x5 <= x3
          x5.lb =  x0 - x1   : [3, 5]
          x5.ub =  x0 - x1   : [3, 5]

          Max is fixed because x5.lb > x4.ub, it inherits x5's bounds

          x5 <= x6 <= x5
          => x3 <= x6 <= x5
          x6.lb =  x0 - x1   : [3, 5]
          x6.ub =  x0 - x1   : [3, 5]

          Layer 3:

          x7 = 2x6
          => x7 = 2x5 = 2x3 = 2x0 - 2x1
          x7.lb = 2 ( x0 - x1 ) = 2x0 - 2x1   : [6, 10]
          x7.ub = 2 ( x0 - x1 ) = 2x0 - 2x1   : [6, 10]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -2, Tightening::LB ),
            Tightening( 2, 0, Tightening::UB ),
            Tightening( 3, 3, Tightening::LB ),
            Tightening( 3, 5, Tightening::UB ),
            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 0, Tightening::UB ),
            Tightening( 5, 3, Tightening::LB ),
            Tightening( 5, 5, Tightening::UB ),
            Tightening( 6, 3, Tightening::LB ),
            Tightening( 6, 5, Tightening::UB ),
            Tightening( 7, 6, Tightening::LB ),
            Tightening( 7, 10, Tightening::UB ),

        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0
          x3 <= x5 <= x3

          Layer 3 (MAX):
          x5 <= x6 <= x5

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 4:
          x7 <= x7 <= x7

          Layer 3:
          Using x7 = 2x6:
          2x6 <= x7 <= 2x6

          Layer 2:
          Using x5 <= x6 <= x5:
          2x5 <= x7 <= 2x5

          Layer 1:
          Using x3 <= x5 <= x3:
          2x3 <= x7 <= 2x3

          Layer 0:
          Using x3 = x0 - x1
          2x0 - 2x1 <= x7 <= 2x0 - 2x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          3,
                                          Vector<double>( { 0, 1 } ),
                                          Vector<double>( { 0, 1 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 2, -2 } ),
                                     Vector<double>( { 2, -2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_symbolic_bound_maps_softmax1()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTSoftmax( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );
        tableau.setLowerBound( 2, -1 );
        tableau.setUpperBound( 2, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );
    }

    void test_symbolic_bound_maps_softmax2()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        {
            Options::get()->setString( Options::SOFTMAX_BOUND_TYPE, "lse" );
            NLR::NetworkLevelReasoner nlr;
            MockTableau tableau;
            nlr.setTableau( &tableau );
            populateNetworkSBTSoftmax( nlr, tableau );

            tableau.setLowerBound( 0, 1 );
            tableau.setUpperBound( 0, 1.000001 );
            tableau.setLowerBound( 1, 1 );
            tableau.setUpperBound( 1, 1.000001 );
            tableau.setLowerBound( 2, 1 );
            tableau.setUpperBound( 2, 1.000001 );

            // Invoke initializeSymbolicBoundsMaps
            TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
            TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

            /*
              Input ranges:

              x0: [1, 1.0001]
              x1: [1, 1.0001]
              x2: [1, 1.0001]

              Layer 1:

              x3 = x0 - x1 + x2 + 1
              x3.lb = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3.ub = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3 range: [ 1.999999, 2.000002 ]

              x4 = -x0 + x1 + x2 + 2
              x4.lb = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4.ub = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4 range: [ 2.999999, 3.000002 ]

              x5 = -x0 - x1 - x2 + 3
              x5.lb = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5.ub = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5 range: [ -0.000003, 0 ]
            */

            unsigned size = nlr.getLayer( 2 )->getSize();
            Vector<double> sourceLbs = { 1.999899, 2.999899, -0.000003 };
            Vector<double> sourceUbs = { 2.000102, 3.000102, 0.0001 };
            Vector<double> sourceMids = { 2.0000005, 3.0000005, -0.0000015 };
            Vector<double> targetLbs( size, 0 );
            Vector<double> targetUbs( size, 0 );
            Vector<double> symbolicLb( size * size, 0 );
            Vector<double> symbolicUb( size * size, 0 );
            Vector<double> symbolicLowerBias( size, 0 );
            Vector<double> symbolicUpperBias( size, 0 );
            for ( unsigned i = 0; i < size; ++i )
            {
                targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
                targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
            }
            for ( unsigned i = 0; i < size; ++i )
            {
                symbolicLowerBias[i] =
                    NLR::Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, i ); // Using lse2
                symbolicUpperBias[i] =
                    NLR::Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, i );
                for ( unsigned j = 0; j < size; ++j )
                {
                    symbolicLb[size * j + i] =
                        NLR::Layer::dLSELowerBound2( sourceMids, sourceLbs, sourceUbs, i, j );
                    symbolicUb[size * j + i] =
                        NLR::Layer::dLSEUpperbound( sourceMids, targetLbs, targetUbs, i, j );
                    symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                    symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
                }
            }
            TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( symbolicLb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT( compareVectors( symbolicUb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicLowerBias, Vector<double>( { 0.4243, 0.4481, 0.1277 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicUpperBias, Vector<double>( { 0.4243, 0.4480, 0.1277 } ) ) );

            /*
                Layer 2:

0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
               x6.lb = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6.ub = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6 range: [ 0.2595, 0.2595 ]

-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4480 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
               x7.lb = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6062
               x7.ub = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6063
               x7 range: [ 0.7054, 0.7054 ]

-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277
               x8.lb = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8.ub = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8 range: [ 0.0351, 0.0351 ]

                Layer 3:

                x9 = x6 + x7 + x8
                => x9 = ( 0.1922 - 0.1830 - 0.0091 ) x3 + ( -0.1830 + 0.2078 - 0.0248 ) x4 + (
               -0.0091 - 0.0248 + 0.0339 ) x5 + ( 0.4243 + 0.4481 + 0.1277 )

                => x9 = 0.0001 x3 + 0 x4 + 0 x5 + 1.0001
                => ( Up to rounding ) 1 <= x9 <= 1.
                x9.lb = 1
                x9.ub = 1
                x9 range: [ 1, 1 ]

                x10 = - x6 - x7 - x8
                => x10 = - ( 0.1922 - 0.1830 - 0.0091 ) x3 - ( -0.1830 + 0.2078 - 0.0248 ) x4 - (
               -0.0091 - 0.0248 + 0.0339 ) x5 - ( 0.4243 + 0.4481 + 0.1277 )

                => x10 = - 0.0001 x3 - 0.0000 x4 - 0.0000 x5 - 1.0001
                => ( Up to rounding ) 1 <= x10 <= 1.
                x10.lb = 1
                x10.ub = 1
                x10 range: [ -1, -1 ]
            */

            List<Tightening> expectedBounds( { Tightening( 3, 2, Tightening::LB ),
                                               Tightening( 3, 2, Tightening::UB ),
                                               Tightening( 4, 3, Tightening::LB ),
                                               Tightening( 4, 3, Tightening::UB ),
                                               Tightening( 5, 0, Tightening::LB ),
                                               Tightening( 5, 0, Tightening::UB ),
                                               Tightening( 6, 0.2595, Tightening::LB ),
                                               Tightening( 6, 0.2595, Tightening::UB ),
                                               Tightening( 7, 0.7054, Tightening::LB ),
                                               Tightening( 7, 0.7054, Tightening::UB ),
                                               Tightening( 8, 0.0351, Tightening::LB ),
                                               Tightening( 8, 0.0351, Tightening::UB ),
                                               Tightening( 9, 1, Tightening::LB ),
                                               Tightening( 9, 1, Tightening::UB ),
                                               Tightening( 10, -1, Tightening::LB ),
                                               Tightening( 10, -1, Tightening::UB )

            } );

            List<Tightening> bounds;
            TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
            TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

            /*
              Symbolic bounds of every activation layer in terms of predecessor:

              Layer 2 (SOFTMAX):
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277

              Symbolic bounds of output layer in terms of every layer (backsubstitution):

              Layer 3:
              x9 <= x9 <= x9
              x10 <= x10 <= x10

              Layer 2:
              Using x9 = x6 + x7 + x8, x10 = -x6 - x7 - x8:
              x6 + x7 + x8 <= x9 <= x6 + x7 + x8
              -x6 - x7 - x8 <= x10 <= -x6 - x7 - x8

              Layer 1:
              Using
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243.
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481.
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277:
              1 <= x9 <= 1
              -1 <= x10 <= -1

              Layer 0:
              1 <= x9 <= 1
              -1 <= x10 <= -1
            */
            comparePredecessorSymbolicBounds( nlr,
                                              2,
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.4243, 0.4481, 0.1277 } ),
                                              Vector<double>( { 0.4243, 0.4480, 0.1277 } ) );

            compareOutputSymbolicBounds( nlr,
                                         3,
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         2,
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         1,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
            compareOutputSymbolicBounds( nlr,
                                         0,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
        }
        {
            Options::get()->setString( Options::SOFTMAX_BOUND_TYPE, "er" );
            NLR::NetworkLevelReasoner nlr;
            MockTableau tableau;
            nlr.setTableau( &tableau );
            populateNetworkSBTSoftmax( nlr, tableau );

            tableau.setLowerBound( 0, 1 );
            tableau.setUpperBound( 0, 1.000001 );
            tableau.setLowerBound( 1, 1 );
            tableau.setUpperBound( 1, 1.000001 );
            tableau.setLowerBound( 2, 1 );
            tableau.setUpperBound( 2, 1.000001 );

            // Invoke initializeSymbolicBoundsMaps
            TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
            TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

            /*
              Input ranges:

              x0: [1, 1.0001]
              x1: [1, 1.0001]
              x2: [1, 1.0001]

              Layer 1:

              x3 = x0 - x1 + x2 + 1
              x3.lb = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3.ub = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3 range: [ 1.999999, 2.000002 ]

              x4 = -x0 + x1 + x2 + 2
              x4.lb = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4.ub = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4 range: [ 2.999999, 3.000002 ]

              x5 = -x0 - x1 - x2 + 3
              x5.lb = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5.ub = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5 range: [ -0.000003, 0 ]
            */

            unsigned size = nlr.getLayer( 2 )->getSize();
            Vector<double> sourceLbs = { 1.999899, 2.999899, -0.000003 };
            Vector<double> sourceUbs = { 2.000102, 3.000102, 0.0001 };
            Vector<double> sourceMids = { 2.0000005, 3.0000005, -0.0000015 };
            Vector<double> targetLbs( size, 0 );
            Vector<double> targetUbs( size, 0 );
            Vector<double> symbolicLb( size * size, 0 );
            Vector<double> symbolicUb( size * size, 0 );
            Vector<double> symbolicLowerBias( size, 0 );
            Vector<double> symbolicUpperBias( size, 0 );
            for ( unsigned i = 0; i < size; ++i )
            {
                targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
                targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
            }
            for ( unsigned i = 0; i < size; ++i )
            {
                symbolicLowerBias[i] =
                    NLR::Layer::ERLowerBound( sourceMids, sourceLbs, sourceUbs, i ); // Using er
                symbolicUpperBias[i] =
                    NLR::Layer::ERUpperBound( sourceMids, targetLbs, targetUbs, i );
                for ( unsigned j = 0; j < size; ++j )
                {
                    symbolicLb[size * j + i] =
                        NLR::Layer::dERLowerBound( sourceMids, sourceLbs, sourceUbs, i, j );
                    symbolicUb[size * j + i] =
                        NLR::Layer::dERUpperBound( sourceMids, targetLbs, targetUbs, i, j );
                    symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                    symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
                }
            }
            TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( symbolicLb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT( compareVectors( symbolicUb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicLowerBias, Vector<double>( { 0.4243, 0.4481, 0.1277 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicUpperBias, Vector<double>( { 0.4243, 0.4480, 0.1277 } ) ) );

            /*
                Layer 2:

0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
               x6.lb = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6.ub = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6 range: [ 0.2595, 0.2595 ]

-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4480 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
               x7.lb = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6062
               x7.ub = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6063
               x7 range: [ 0.7054, 0.7054 ]

-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277
               x8.lb = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8.ub = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8 range: [ 0.0351, 0.0351 ]

                Layer 3:

                x9 = x6 + x7 + x8
                => x9 = ( 0.1922 - 0.1830 - 0.0091 ) x3 + ( -0.1830 + 0.2078 - 0.0248 ) x4 + (
               -0.0091 - 0.0248 + 0.0339 ) x5 + ( 0.4243 + 0.4481 + 0.1277 )

                => x9 = 0.0001 x3 + 0 x4 + 0 x5 + 1.0001
                => ( Up to rounding ) 1 <= x9 <= 1.
                x9.lb = 1
                x9.ub = 1
                x9 range: [ 1, 1 ]

                x10 = - x6 - x7 - x8
                => x10 = - ( 0.1922 - 0.1830 - 0.0091 ) x3 - ( -0.1830 + 0.2078 - 0.0248 ) x4 - (
               -0.0091 - 0.0248 + 0.0339 ) x5 - ( 0.4243 + 0.4481 + 0.1277 )

                => x10 = - 0.0001 x3 - 0.0000 x4 - 0.0000 x5 - 1.0001
                => ( Up to rounding ) 1 <= x10 <= 1.
                x10.lb = 1
                x10.ub = 1
                x10 range: [ -1, -1 ]
            */
            List<Tightening> expectedBounds( { Tightening( 3, 2, Tightening::LB ),
                                               Tightening( 3, 2, Tightening::UB ),
                                               Tightening( 4, 3, Tightening::LB ),
                                               Tightening( 4, 3, Tightening::UB ),
                                               Tightening( 5, 0, Tightening::LB ),
                                               Tightening( 5, 0, Tightening::UB ),
                                               Tightening( 6, 0.2595, Tightening::LB ),
                                               Tightening( 6, 0.2595, Tightening::UB ),
                                               Tightening( 7, 0.7054, Tightening::LB ),
                                               Tightening( 7, 0.7054, Tightening::UB ),
                                               Tightening( 8, 0.0351, Tightening::LB ),
                                               Tightening( 8, 0.0351, Tightening::UB ),
                                               Tightening( 9, 1, Tightening::LB ),
                                               Tightening( 9, 1, Tightening::UB ),
                                               Tightening( 10, -1, Tightening::LB ),
                                               Tightening( 10, -1, Tightening::UB )

            } );

            List<Tightening> bounds;
            TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
            TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

            /*
              Symbolic bounds of every activation layer in terms of predecessor:

              Layer 2 (SOFTMAX):
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277

              Symbolic bounds of output layer in terms of every layer (backsubstitution):

              Layer 3:
              x9 <= x9 <= x9
              x10 <= x10 <= x10

              Layer 2:
              Using x9 = x6 + x7 + x8, x10 = -x6 - x7 - x8:
              x6 + x7 + x8 <= x9 <= x6 + x7 + x8
              -x6 - x7 - x8 <= x10 <= -x6 - x7 - x8

              Layer 1:
              Using
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243.
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481.
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277:
              1 <= x9 <= 1
              -1 <= x10 <= -1

              Layer 0:
              1 <= x9 <= 1
              -1 <= x10 <= -1
            */
            comparePredecessorSymbolicBounds( nlr,
                                              2,
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.4243, 0.4481, 0.1277 } ),
                                              Vector<double>( { 0.4243, 0.4480, 0.1277 } ) );

            compareOutputSymbolicBounds( nlr,
                                         3,
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         2,
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         1,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
            compareOutputSymbolicBounds( nlr,
                                         0,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
        }
    }

    void test_symbolic_bound_maps_softmax3()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );
        Options::get()->setString( Options::SOFTMAX_BOUND_TYPE, "lse" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTSoftmax2( nlr, tableau );

        tableau.setLowerBound( 0, 1 );
        tableau.setUpperBound( 0, 1.00001 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 1.00001 );
        tableau.setLowerBound( 2, 1 );
        tableau.setUpperBound( 2, 1.00001 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
              Input ranges:

              x0: [1, 1.0001]
              x1: [1, 1.0001]
              x2: [1, 1.0001]

              Layer 1:

              x3 = x0 - x1 + x2 + 1
              x3.lb = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3.ub = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3 range: [ 1.999999, 2.000002 ]

              x4 = -x0 + x1 + x2 + 2
              x4.lb = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4.ub = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4 range: [ 2.999999, 3.000002 ]

              x5 = -x0 - x1 - x2 + 3
              x5.lb = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5.ub = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5 range: [ -0.000003, 0 ]

              x6 = -x0 - x1 - x2 + 2
              x6.lb = -x0 - x1 - x2 + 2    : [ -1.000003, -1 ]
              x6.ub = -x0 - x1 - x2 + 2    : [ -1.000003, -1 ]
              x6 range: [ -1.000003, -1 ]

              x7 = -x0 - x1 - x2 + 1
              x7.lb = -x0 - x1 - x2 + 1    : [ -2.000003, -2 ]
              x7.ub = -x0 - x1 - x2 + 1    : [ -2.000003, -2 ]
              x7 range: [ -2.000003, -2 ]
            */

        // First Sigmoid: x8 x10 x12 = softmax( x3, x5, x7 ).
        unsigned size = nlr.getLayer( 2 )->getActivationSources( 0 ).size();
        Vector<double> sourceLbs = { 1.999899, -0.000003, -2.000103 };
        Vector<double> sourceUbs = { 2.000102, 0.0001, -1.999 };
        Vector<double> sourceMids = { 2.0000005, -0.0000015, -2.0000015 };
        Vector<double> targetLbs( size, 0 );
        Vector<double> targetUbs( size, 0 );
        Vector<double> symbolicLb( size * size, 0 );
        Vector<double> symbolicUb( size * size, 0 );
        Vector<double> symbolicLowerBias( size, 0 );
        Vector<double> symbolicUpperBias( size, 0 );
        for ( unsigned i = 0; i < size; ++i )
        {
            targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
            targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
        }
        for ( unsigned i = 0; i < size; ++i )
        {
            symbolicLowerBias[i] =
                NLR::Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, i ); // Using lse2
            symbolicUpperBias[i] = NLR::Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, i );
            for ( unsigned j = 0; j < size; ++j )
            {
                symbolicLb[size * j + i] =
                    NLR::Layer::dLSELowerBound2( sourceMids, sourceLbs, sourceUbs, i, j );
                symbolicUb[size * j + i] =
                    NLR::Layer::dLSEUpperbound( sourceMids, targetLbs, targetUbs, i, j );
                symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
            }
        }
        TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.8668, 0.1173, 0.0159 } ) ) );
        TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.8668, 0.1173, 0.0159 } ) ) );
        TS_ASSERT( compareVectors( symbolicLb,
                                   Vector<double>( { 0.1155,
                                                     -0.1017,
                                                     -0.0138,
                                                     -0.1017,
                                                     0.1035,
                                                     -0.0019,
                                                     -0.0138,
                                                     -0.0019,
                                                     0.0156 } ) ) );
        TS_ASSERT( compareVectors( symbolicUb,
                                   Vector<double>( { 0.1154,
                                                     -0.1017,
                                                     -0.0138,
                                                     -0.1017,
                                                     0.1036,
                                                     -0.0019,
                                                     -0.0138,
                                                     -0.0019,
                                                     0.0156 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicLowerBias, Vector<double>( { 0.6084, 0.3170, 0.0747 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicUpperBias, Vector<double>( { 0.6084, 0.3170, 0.0747 } ) ) );

        // Second Sigmoid: x9 x11 = softmax( x4, x6 ).
        size = nlr.getLayer( 2 )->getActivationSources( 1 ).size();
        sourceLbs = Vector<double>( { 2.999899, -1.000103 } );
        sourceUbs = Vector<double>( { 3.000102, -0.9999 } );
        sourceMids = Vector<double>( { 3.0000005, -1.0000015 } );
        targetLbs = Vector<double>( size, 0 );
        targetUbs = Vector<double>( size, 0 );
        symbolicLb = Vector<double>( size * size, 0 );
        symbolicUb = Vector<double>( size * size, 0 );
        symbolicLowerBias = Vector<double>( size, 0 );
        symbolicUpperBias = Vector<double>( size, 0 );
        for ( unsigned i = 0; i < size; ++i )
        {
            targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
            targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
        }
        for ( unsigned i = 0; i < size; ++i )
        {
            symbolicLowerBias[i] =
                NLR::Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, i ); // Using lse2
            symbolicUpperBias[i] = NLR::Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, i );
            for ( unsigned j = 0; j < size; ++j )
            {
                symbolicLb[size * j + i] =
                    NLR::Layer::dLSELowerBound2( sourceMids, sourceLbs, sourceUbs, i, j );
                symbolicUb[size * j + i] =
                    NLR::Layer::dLSEUpperbound( sourceMids, targetLbs, targetUbs, i, j );
                symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
            }
        }
        TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.9820, 0.0180 } ) ) );
        TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.9820, 0.0180 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicLb, Vector<double>( { 0.0177, -0.0177, -0.0177, 0.0177 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicUb, Vector<double>( { 0.0177, -0.0177, -0.0177, 0.0177 } ) ) );
        TS_ASSERT( compareVectors( symbolicLowerBias, Vector<double>( { 0.9114, 0.0886 } ) ) );
        TS_ASSERT( compareVectors( symbolicUpperBias, Vector<double>( { 0.9114, 0.0886 } ) ) );

        /*
            Layer 2:

            First Sigmoid: x8 x10 x12 = softmax( x3, x5, x7 ).
0.1155 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084 <= x8 <= 0.1154 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084
           x8.lb = 0.2310 x0 + 0.0001 x1 + 0.2310 x2 + 0.4051
           x8.ub = 0.2310 x0 + 0.0000 x1 + 0.2310 x2 + 0.4050
           x8 range: [ 0.8668, 0.8668 ]

-0.1017 x3 + 0.1035 x5 - 0.0019 x7 + 0.3170 <= x10 <= -0.1017 x3 + 0.1036 x5 - 0.0019 x7 + 0.3170
           x10.lb = -0.2033 x0 + 0.0001 x1 - 0.2033 x2 + 0.5239
           x10.ub = -0.2033 x0 + 0.0000 x1 - 0.2033 x2 + 0.5241
           x10 range: [ 0.1173, 0.1173 ]

-0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747 <= x12 <= -0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747
           x12.lb = -0.0275 x0 + 0.0001 x1 - 0.0275 x2 + 0.0708
           x12.ub = -0.0275 x0 + 0.0001 x1 - 0.0275 x2 + 0.0708
           x12 range: [ 0.0159, 0.0159 ]

           Second Sigmoid: x9 x11 = softmax( x4, x6 ).
0.0177 x4 - 0.0177 x6 + 0.9114 <= x9 <= 0.0177 x4 - 0.0177 x6 + 0.9114
           x9.lb = 0 x0 + 0.0354 x1 + 0.0354 x2 + 0.9114
           x9.ub = 0 x0 + 0.0354 x1 + 0.0354 x2 + 0.9114
           x9 range: [ 0.9820, 0.0180 ]

-0.0177 x4 + 0.0177 x6 + 0.0886 <= x11 <= -0.0177 x4 + 0.0177 x6 + 0.0886
           x11.lb = 0 x0 - 0.0354 x1 - 0.0354 x2 + 0.0886
           x11.ub = 0 x0 - 0.0354 x1 - 0.0354 x2 + 0.0886
           x11 range: [ 0.9820, 0.0180 ]

            Layer 3:

            x13 = x8 + x10 + x12
            => x13 = ( 0.1155 - 0.1017 - 0.0138 ) x3 + ( -0.1017 + 0.1035 - 0.0019 ) x5
            + ( -0.0138 - 0.0019 + 0.0156 ) x7 + ( 0.6084 + 0.3170 + 0.0747 )

            => x13 = 0 x3 - 0.0001 x5 - 0.0001 x7 + 1.0001
            => ( Up to rounding ) 1 <= x13 <= 1.
            x13.lb = 1
            x13.ub = 1
            x13 range: [ 1, 1 ]

            x14 = - x8 - x10 - x12
            => x14 = - ( 0.1155 - 0.1017 - 0.0138 ) x3 - ( -0.1017 + 0.1035 - 0.0019 ) x5
            - ( -0.0138 - 0.0019 + 0.0156 ) x7 - ( 0.6084 + 0.3170 + 0.0747 )

            => x14 = 0 x3 + 0.0001 x5 + 0.0001 x7 - 1.0001
            => ( Up to rounding ) -1 <= x14 <= -1.
            x14.lb = -1
            x14.ub = -1
            x14 range: [ -1, -1 ]

            x15 = x9 + x11
            => x15 = ( 0.0177 - 0.0177 ) x4 + ( -0.0177 + 0.0177 ) x6 + ( 0.9114 + 0.0886 )

            => x15 = 0 x4 + 0 x6 + 1
            => ( Up to rounding ) 1 <= x15 <= 1.
            x15.lb = 1
            x15.ub = 1
            x15 range: [ 1, 1 ]

            x16 = - x9 - x11
            => x16 = - ( 0.0177 - 0.0177 ) x4 - ( -0.0177 + 0.0177 ) x6 - ( 0.9114 + 0.0886 )

            => x16 = 0 x4 + 0 x6 - 1
            => ( Up to rounding ) -1 <= x16 <= -1.
            x16.lb = -1
            x16.ub = -1
            x16 range: [ -1, -1 ]
        */

        List<Tightening> expectedBounds( {
            Tightening( 3, 2, Tightening::LB ),         Tightening( 3, 2, Tightening::UB ),
            Tightening( 4, 3, Tightening::LB ),         Tightening( 4, 3, Tightening::UB ),
            Tightening( 5, 0, Tightening::LB ),         Tightening( 5, 0, Tightening::UB ),
            Tightening( 6, -1, Tightening::LB ),        Tightening( 6, -1, Tightening::UB ),
            Tightening( 7, -2, Tightening::LB ),        Tightening( 7, -2, Tightening::UB ),
            Tightening( 8, 0.86681, Tightening::LB ),   Tightening( 8, 0.86682, Tightening::UB ),
            Tightening( 9, 0.98201, Tightening::LB ),   Tightening( 9, 0.98201, Tightening::UB ),
            Tightening( 10, 0.11731, Tightening::LB ),  Tightening( 10, 0.11731, Tightening::UB ),
            Tightening( 11, 0.017985, Tightening::LB ), Tightening( 11, 0.017986, Tightening::UB ),
            Tightening( 12, 0.015875, Tightening::LB ), Tightening( 12, 0.015876, Tightening::UB ),
            Tightening( 13, 1, Tightening::LB ),        Tightening( 13, 1, Tightening::UB ),
            Tightening( 14, -1, Tightening::LB ),       Tightening( 14, -1, Tightening::UB ),
            Tightening( 15, 1, Tightening::LB ),        Tightening( 15, 1, Tightening::UB ),
            Tightening( 16, -1, Tightening::LB ),       Tightening( 16, -1, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SOFTMAX):
0.1155 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084 <= x8 <= 0.1154 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084
0.0177 x4 - 0.0177 x6 + 0.9114 <= x9 <= 0.0177 x4 - 0.0177 x6 + 0.9114
-0.1017 x3 + 0.1035 x5 - 0.0019 x7 + 0.3170 <= x10 <= -0.1017 x3 + 0.1036 x5 - 0.0019 x7 + 0.3170
-0.0177 x4 + 0.0177 x6 + 0.0886 <= x11 <= -0.0177 x4 + 0.0177 x6 + 0.0886
-0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747 <= x12 <= -0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x13 <= x13 <= x13
          x14 <= x14 <= x14
          x15 <= x15 <= x15
          x16 <= x16 <= x16

          Layer 2:
          Using x13 = x8 + x10 + x12, x14 = -x8 - x10 - x12, x15 = x9 + x11, x16 = -x9 - x11:
          x8 + x10 + x12 <= x13 <= x8 + x10 + x12
          -x8 - x10 - x12 <= x14 <= -x8 - x10 - x12
          x9 + x11 <= x15 <= x9 + x11
          -x9 - x11 <= x16 <= -x9 - x11

          Layer 1:
          Using
0.1155 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084 <= x8 <= 0.1154 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084
0.0177 x4 - 0.0177 x6 + 0.9114 <= x9 <= 0.0177 x4 - 0.0177 x6 + 0.9114
-0.1017 x3 + 0.1035 x5 - 0.0019 x7 + 0.3170 <= x10 <= -0.1017 x3 + 0.1036 x5 - 0.0019 x7 + 0.3170
-0.0177 x4 + 0.0177 x6 + 0.0886 <= x11 <= -0.0177 x4 + 0.0177 x6 + 0.0886
-0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747 <= x12 <= -0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747
          1 <= x13 <= 1
          -1 <= x14 <= -1
          1 <= x15 <= 1
          -1 <= x16 <= -1

          Layer 0:
          1 <= x13 <= 1
          -1 <= x14 <= -1
          1 <= x15 <= 1
          -1 <= x16 <= -1
        */
        comparePredecessorSymbolicBounds(
            nlr,
            2,
            Vector<double>( { 0.1155,  0.0000,  -0.1017, 0.0000,  -0.0138, 0.0000, 0.0177,
                              0.0000,  -0.0177, 0.0000,  -0.1017, 0.0000,  0.1035, 0.0000,
                              -0.0019, 0.0000,  -0.0177, 0.0000,  0.0177,  0.0000, -0.0138,
                              0.0000,  -0.0019, 0.0000,  0.0156 } ),
            Vector<double>( { 0.1155,  0.0000,  -0.1017, 0.0000,  -0.0138, 0.0000, 0.0177,
                              0.0000,  -0.0177, 0.0000,  -0.1017, 0.0000,  0.1035, 0.0000,
                              -0.0019, 0.0000,  -0.0177, 0.0000,  0.0177,  0.0000, -0.0138,
                              0.0000,  -0.0019, 0.0000,  0.0156 } ),
            Vector<double>( { 0.6084, 0.9114, 0.3170, 0.0886, 0.0747 } ),
            Vector<double>( { 0.6084, 0.9114, 0.3170, 0.0886, 0.0747 } ) );

        compareOutputSymbolicBounds(
            nlr,
            3,
            Vector<double>( { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 } ),
            Vector<double>( { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 } ),
            Vector<double>( { 0, 0, 0, 0 } ),
            Vector<double>( { 0, 0, 0, 0 } ) );
        compareOutputSymbolicBounds(
            nlr,
            2,
            Vector<double>( { 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0 } ),
            Vector<double>( { 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0 } ),
            Vector<double>( { 0, 0, 0, 0 } ),
            Vector<double>( { 0, 0, 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( 20, 0 ),
                                     Vector<double>( 20, 0 ),
                                     Vector<double>( { 1, -1, 1, -1 } ),
                                     Vector<double>( { 1, -1, 1, -1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( 12, 0 ),
                                     Vector<double>( 12, 0 ),
                                     Vector<double>( { 1, -1, 1, -1 } ),
                                     Vector<double>( { 1, -1, 1, -1 } ) );
    }

    void test_symbolic_bound_maps_bilinear()
    {
        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTBilinear( nlr, tableau );

        tableau.setLowerBound( 0, 1 );
        tableau.setUpperBound( 0, 2 );
        tableau.setLowerBound( 1, -2 );
        tableau.setUpperBound( 1, 1 );

        // Invoke initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps() );

        /*
          Input ranges:

          x0: [1, 2]
          x1: [-2, 1]

          Layers 1, 2:

          x2 = x0 - 2x1
          x2.lb = x0 - 2x1   : [-1, 6]
          x2.ub = x0 - 2x1   : [-1, 6]

          x3 = x0 + x1
          x3.lb = x0 + x1   : [-1, 3]
          x3.ub = x0 + x1   : [-1, 3]

          Coefficients for bilinear layer:
          Lower bound:
              alpha_l = x3.lb = -1
              beta = x2.lb = -1
              gamma_l = -x2.lb x3.lb = --1 * -1 = -1

          Upper bound:
              alpha_u = x3.ub = 3
              beta = x2.lb = -1
              gamma_u = -x2.lb x3.ub = --1 * 3 = 3

          -x2 - x3 - 1 <= x4 <= 3x2 - x3 + 3
          x4.lb = -1 ( x0 - 2x1 ) + -1 ( x0 + x1 ) + -1 = -2x0 + x1 - 1     : [-7, -2]
          x4.ub = 3 ( x0 - 2x1 ) + -1 ( x0 + x1 ) + 3 = 2x0 - 7x1 + 3    : [0, 21]
          x4 range: [-6, 18]

          Layer 3:

          x5 = -x4
          => -3x2 + x3 - 3 <= x4 <= x2 + x3 + 1
          x5.lb = -1 ( 2x0 - 5x1 + 3 ) = -2x0 + 7x1 - 3   : [-21, 0]
          x5.ub = -1 ( -2x0 + x1 - 1 ) = 2x0 - x1 + 1   : [2, 7]
          x5 range: [-18, 6]
        */

        List<Tightening> expectedBounds( { Tightening( 2, -1, Tightening::LB ),
                                           Tightening( 2, 6, Tightening::UB ),
                                           Tightening( 3, -1, Tightening::LB ),
                                           Tightening( 3, 3, Tightening::UB ),
                                           Tightening( 4, -6, Tightening::LB ),
                                           Tightening( 4, 18, Tightening::UB ),
                                           Tightening( 5, -18, Tightening::LB ),
                                           Tightening( 5, 6, Tightening::UB ) } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (BILINEAR):
          -x2 - x3 - 1 <= x5 <= 3x2 - x3 + 3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x4 <= x5 <= x4

          Layer 2:
          Using x5 = -x4:
          -x3 <= x4 <= -x4

          Layer 1:
          Using -x2 - x3 - 1 <= x4 <= 3x2 - x3 + 3:
          -3x2 + x3 - 3 <= x5 <= x2 + x3 + 1

          Layer 0:
          Using x2 = x0 - 2x1, x3 = x0 + x1:
          -2x0 + 7x1 - 3 <= x5 <= 2x0 - x1 + 1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { -1, -1 } ),
                                          Vector<double>( { 3, -1 } ),
                                          Vector<double>( { -1 } ),
                                          Vector<double>( { 3 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { -1 } ),
                                     Vector<double>( { -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { -3, 1 } ),
                                     Vector<double>( { 1, 1 } ),
                                     Vector<double>( { -3 } ),
                                     Vector<double>( { 1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -2, 7 } ),
                                     Vector<double>( { 2, -1 } ),
                                     Vector<double>( { -3 } ),
                                     Vector<double>( { 1 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relus_all_active()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1
          x2.lb = 2x0 + 3x1   : [11, 27]
          x2.ub = 2x0 + 3x1   : [11, 27]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          Both ReLUs active, bound survive through activations:

          x2 <= x4 <= x2
          x4.lb = 2x0 + 3x1   : [11, 27]
          x4.ub = 2x0 + 3x1   : [11, 27]

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          => x2 - x3 <= x6 <= x2 - x3
          x6.lb =  x0 + 2x1   : [6, 16]
          x6.ub =  x0 + 2x1   : [6, 16]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, 11, Tightening::LB ),
            Tightening( 2, 27, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 11, Tightening::LB ),
            Tightening( 4, 27, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, 6, Tightening::LB ),
            Tightening( 6, 16, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          x2 <= x4 <= x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          x2 - x3 <= x6 <= x2 - x3

          Layer 0:
          Using x2 = 2x0 + 3x1, x3 = x0 + x1:
          x0 + 2x1 <= x6 <= x0 + 2x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relus_active_and_inactive()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -30 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1 - 30
          x2.lb = 2x0 + 3x1 - 30   : [-19, -3]
          x2.ub = 2x0 + 3x1 - 30   : [-19, -3]

          x3 = x0 + x1
          x3.lb = x0 + x1   : [5, 11]
          x3.ub = x0 + x1   : [5, 11]

          First ReLU is inactive, bounds get zeroed
          Second ReLU is active, bounds surive the activation

          0 <= x4 <= 0
          x4.lb = 0
          x4.ub = 0

          x3 <= x5 <= x3
          x5.lb = x0 + x1   : [5, 11]
          x5.ub = x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          ==> -x3 <= x6 <= -x3
          x6.lb = -x0 - x1  : [-11, -5]
          x6.ub = -x0 - x1  : [-11, -5]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -19, Tightening::LB ),
            Tightening( 2, -3, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 0, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -11, Tightening::LB ),
            Tightening( 6, -5, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          -x3 <= x6 <= -x3

          Layer 0:
          Using x3 = x0 + x1:
          -x0 - x1 <= x6 <= -x0 - x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relus_active_and_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -15 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1 - 15
          x2.lb = 2x0 + 3x1 - 15   : [-4, 12]
          x2.ub = 2x0 + 3x1 - 15   : [-4, 12]

          x3 = x0 + x1
          x3.lb = x0 + x1   : [5, 11]
          x3.ub = x0 + x1   : [5, 11]

          First ReLU is undecided, bound is concretized. Using custom ReLU lower
          coefficient of 0.5. Upper coefficient: 12/(12--4) = 12/16 = 0.75
          Second ReLU is active, bounds surive the activation

          x4 range: [-2, 12]
          0.5 x2 <= x4 <= 0.75 x2 + 3
          x4.lb = 0.5 ( 2x0 + 3x1 - 15 ) = x0 + 1.5 x1 - 7.5
          x4.ub = 0.75( 2x0 + 3x1 ) - 0.75 * 15 + 3  = 1.5x0 + 2.25x1 - 8.25

          x3 <= x5 <= x3
          x5.lb = x0 + x1   : [5, 11]
          x5.ub = x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          ==> 0.5 x2 - x3 <= x6 <= 0.75x2 - x3 + 3
          x6.lb = 0.5 x1 - 7.5
          x6.ub = 0.5x0 + 1.25x1 - 8.25

          x6 range: [0.5 - 7.5 = -7, 3 + 6.25 - 8.25 = 1] = [-7, 1]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -4, Tightening::LB ),
            Tightening( 2, 12, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, -2, Tightening::LB ),
            Tightening( 4, 12, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -7, Tightening::LB ),
            Tightening( 6, 1, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0.5 x2 <= x4 <= 0.75 x2 + 3
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          0.5 x2 - x3 <= x6 <= 0.75x2 - x3 + 3

          Layer 0:
          Using x2 = 2x0 + 3x1, x3 = x0 + x1:
          0.5 x1 - 7.5 <= x6 <= 0.5x0 + 1.25x1 - 8.25
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.5, 0, 0, 1 } ),
                                          Vector<double>( { 0.75, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 3, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.5, -1 } ),
                                     Vector<double>( { 0.75, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 3 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0, 0.5 } ),
                                     Vector<double>( { 0.5, 1.25 } ),
                                     Vector<double>( { -7.5 } ),
                                     Vector<double>( { -8.25 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relus_active_and_externally_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTRelu( nlr, tableau );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0). Should make the node unfixed.
        nlr.setBias( 1, 0, -15 );

        // However, one of the ReLU's variables has been eliminated
        nlr.eliminateVariable( 2, -3 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = -3
          x2 is eliminated, everything set to -3

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First ReLU is inactive (set externally), bounds get zeroed
          Second ReLU is active, bounds surive the activation

          0 <= x4 <= 0
          x4.lb = 0
          x4.ub = 0

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          ==> -x3 <= x6 <= -x3
          x6.lb =  - x0 - x1  : [-11, -5]
          x6.ub =  - x0 - x1  : [-11, -5]
        */

        List<Tightening> expectedBounds( {
            // x2 does not appear, because it has been eliminated

            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 0, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -11, Tightening::LB ),
            Tightening( 6, -5, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          -x3 <= x6 <= -x3

          Layer 0:
          Using x3 = x0 + x1:
          -x0 - x1 <= x6 <= -x0 - x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relu_residual1()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTReluResidual1( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [-1, 1]

          Layers 1. 2:

          x1 = x0
          x1.lb = x0   : [-1, 1]
          x1.ub = x0   : [-1, 1]

          ReLU is undecided, bound is concretized. Using custom ReLU lower
          coefficient of 0.5. Upper coefficient: 1/( 1--1 ) = 1/2 = 0.5

          0.5 x1 <= x2 <= 0.5x1 + 0.5
          x2.lb = 0.5 x0
          x2.ub = 0.5 x0 + 0.5
          x2 range: [-0.5, 1]

          Layers 3, 4 (with residual from x0):

          x3 = - x2 - x0 + 1
          x3.lb = -1( 0.5x0 + 0.5 ) -x0 + 1 = -1.5 x0 + 0.5 : [-1, 2]
          x3.ub = -1( 0.5 x0 ) -1x0 + 1 = -1.5 x0 + 1 : [-0.5, 2.5]
          x3 range: [-1, 2.5]

          ReLU is undecided, bound is concretized. Using custom ReLU lower
          coefficient of 0.5. Upper coefficient: 2.5/( 2.5--1 ) = 2.5/3.5 = 5/7.

          0.5 x3 <= x4 <= 5/7 x3 + 5/7
          x4.lb = 0.5 ( -1.5 x0 + 0.5 ) = -0.75 x0 + 0.25 : [-0.5, 1]
          x4.ub = 5/7 ( -1.5 x0 + 1 ) + 5/7 = -15/14 x0 + 20/14 : [1, 35/14 = 2.5]
          x4 range: [-0.5, 2.5]

          Layer 5 (with residual from x1):

          x5 = 3x4 + 3x1 + 1
          x5.lb =  3 ( -0.75 x0 + 0.25 ) + 3 ( x0 ) + 1 = 0.75x0 + 1.75 : [1, 2.5]
          x5.ub =  3 ( -15/14 x0 + 20/14 ) + 3 ( x0 ) + 1 = -3/14 x0 + 74/14 : [71/14, 77/14 = 5.5]
          x5 range: [1, 5.5]
        */

        List<Tightening> expectedBounds( {
            Tightening( 1, -1, Tightening::LB ),
            Tightening( 1, 1, Tightening::UB ),
            Tightening( 2, -0.5, Tightening::LB ),
            Tightening( 2, 1, Tightening::UB ),
            Tightening( 3, -1, Tightening::LB ),
            Tightening( 3, 2.5, Tightening::UB ),
            Tightening( 4, -0.5, Tightening::LB ),
            Tightening( 4, 2.5, Tightening::UB ),
            Tightening( 5, 1, Tightening::LB ),
            Tightening( 5, 5.5, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0.5 x1 <= x2 <= 0.5x1 + 0.5

          Layer 4 (RELU):
          0.5 x3 <= x4 <= 5/7 x3 + 5/7

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 5:
          x5 <= x5 <= x5

          Layer 4:
          Using x5 = 3x4 + 3x1 + 1:
          3x4 + 3x1 + 1 <= x5 <= 3x4 + 3x1 + 1
          Concretizing residual using x1 : [-1, 1]: 3x4 - 2 <= x5 <= 3x4 + 4

          Layer 3:
          Using 0.5 x3 <= x4 <= 5/7 x3 + 5/7:
          1.5 x3 + 3x1 + 1 <= x5 <= 15/7 x3 + 3x1 + 22/7
          Concretizing residual using x1 : [-1, 1]: 1.5 x3 - 2 <= x5 <= 15/7 x3 + 43/7

          Layer 2:
          Using x3 = -x2 - x0 + 1:
          -1.5 x2 + 3x1 - 1.5 x0 + 2.5 <= x5 <= -15/7 x2 + 3x1 - 15/7 x0 + 37/7
          Concretizing residual using x0 : [-1, 1], x1 : [-1, 1]: -1.5 x2 - 2 <= x5 <= -15/7 x2 +
          73/7

          Layer 1:
          Using 0.5 x1 <= x2 <= 0.5x1 + 0.5:
          2.25 x1 - 1.5 x0 + 1.75 <= x5 <= 27/14 x1 - 15/7 x0 + 37/7
          Concretizing residual using x0 : [-1, 1]: 2.25x1 + 0.25 <= x5 <= 27/14 x1 + 52/7

          Layer 0:
          Using x1 = x0:
          0.75 x0 + 1.75 <= x5 <= -3/14 x0 + 37/7
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.5 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0.7143 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.7143 } ) );

        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 4 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1.5 } ),
                                     Vector<double>( { 2.1429 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 6.1429 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { -1.5 } ),
                                     Vector<double>( { -2.1429 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 10.4286 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 2.25 } ),
                                     Vector<double>( { 1.9286 } ),
                                     Vector<double>( { 0.25 } ),
                                     Vector<double>( { 7.4286 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.75 } ),
                                     Vector<double>( { -0.2143 } ),
                                     Vector<double>( { 1.75 } ),
                                     Vector<double>( { 5.2857 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relu_residual2()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTReluResidual2( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [-1, 1]

          Layers 1, 2:

          x1 = x0
          x1.lb = x0   : [-1, 1]
          x1.ub = x0   : [-1, 1]

          ReLU is undecided, bound is concretized. Using custom ReLU lower
          coefficient of 0.5. Upper cCoefficient: 1/( 1--1 ) = 1/2 = 0.5

          0.5 x1 <= x2 <= 0.5x1 + 0.5
          x2.lb = 0.5x0
          x2.ub = 0.5x0 + 0.5
          x2 range: [-0.5, 1]

          Layers 3, 4 (with residual from x0):

          x3 = - x2 - x0 + 1
          x3.lb = -1( 0.5x0 + 0.5 ) -x0 + 1 = -1.5x0 + 0.5 : [-1, 2]
          x3.ub = -1( 0.5 x0 ) -1x0 + 1 = -1.5 x0 + 1 : [-0.5, 2.5]
          x3 range: [-1, 2.5]

          ReLU is undecided, bound is concretized. Using custom ReLU lower
          coefficient of 0.5. Upper coefficient: 2.5/( 2.5--1 ) = 2.5/3.5 = 5/7.

          0.5 x3 <= x4 <= 5/7 x3 + 5/7
          x4.lb = 0.5 ( -1.5 x0 + 0.5 ) = -0.75 x0 + 0.25 : [-0.5, 1]
          x4.ub = 5/7 ( -1.5 x0 + 1 ) + 5/7 = -15/14 x0 + 20/14 : [1, 35/14 = 2.5]
          x4 range: [-0.5, 2.5]

          Layer 5 (with residual from x0):

          x5 = 3x4 + x0 + 1
          x5.lb =  3 ( -0.75 x0 + 0.25 ) + ( x0 ) + 1 = -1.25x0 + 1.75 : [0.5, 3]
          x5.ub =  3 ( -15/14 x0 + 20/14 ) + ( x0 ) + 1 = -31/14 x0 + 74/14 : [43/14, 105/14 = 7.5]
          x5 range: [0.5, 7.5]

          Layer 6:
          x6 = x5
          x6.lb = -1.25x0 + 1.75 : [0.5, 3]
          x6.ub = -31/14 x0 + 74/14 : [43/14, 7.5]
          x6 range: [0.5, 7.5]
        */

        List<Tightening> expectedBounds( {
            Tightening( 1, -1, Tightening::LB ),
            Tightening( 1, 1, Tightening::UB ),
            Tightening( 2, -0.5, Tightening::LB ),
            Tightening( 2, 1, Tightening::UB ),
            Tightening( 3, -1, Tightening::LB ),
            Tightening( 3, 2.5, Tightening::UB ),
            Tightening( 4, -0.5, Tightening::LB ),
            Tightening( 4, 2.5, Tightening::UB ),
            Tightening( 5, 0.5, Tightening::LB ),
            Tightening( 5, 7.5, Tightening::UB ),
            Tightening( 6, 0.5, Tightening::LB ),
            Tightening( 6, 7.5, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0.5 x1 <= x2 <= 0.5x1 + 0.5

          Layer 4 (RELU):
          0.5 x3 <= x4 <= 5/7 x3 + 5/7

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 6:
          x6 <= x6 <= x6

          Layer 5:
          Using x6 = x5:
          x5 <= x6 <= x5

          Layer 4:
          Using x5 = 3x4 + x0 + 1:
          3x4 + x0 + 1 <= x6 <= 3x4 + x0 + 1
          Concretizing residual using x0 : [-1, 1]: 3x4 <= x6 <= 3x4 + 2

          Layer 3:
          Using 0.5 x3 <= x4 <= 5/7 x3 + 5/7:
          1.5 x3 + x0 + 1 <= x6 <= 15/7 x3 + x0 + 22/7
          Concretizing residual using x0 : [-1, 1]: 1.5 x3 <= x6 <= 15/7 x3 + 29/7

          Layer 2:
          Using x3 = -x2 - x0 + 1:
          -1.5 x2 - 0.5 x0 + 2.5 <= x6 <= -15/7 x2 - 8/7 x0 + 37/7
          Concretizing residual using x0 : [-1, 1]: -1.5 x2 + 2 <= x6 <= -15/7 x2 + 45/7

          Layer 1:
          Using 0.5 x1 <= x2 <= 0.5x1 + 0.5:
          -0.75x1 - 0.5 x0 + 1.75 <= x6 <= -15/14 x1 - 8/7 x0 + 37/7
          Concretizing residual using x0 : [-1, 1]: -0.75x1 + 1.25 <= x6 <= -15/14 x1 + 45/7

          Layer 0:
          Using x1 = x0:
          -1.25 x0 + 1.75 <= x6 <= -31/14 x0 + 37/7
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.5 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 0.5 } ),
                                          Vector<double>( { 0.7143 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0.7143 } ) );

        compareOutputSymbolicBounds( nlr,
                                     6,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 2 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1.5 } ),
                                     Vector<double>( { 2.1429 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 4.1429 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { -1.5 } ),
                                     Vector<double>( { -2.1429 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 6.4286 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { -0.75 } ),
                                     Vector<double>( { -1.0714 } ),
                                     Vector<double>( { 1.25 } ),
                                     Vector<double>( { 6.4286 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1.25 } ),
                                     Vector<double>( { -2.2143 } ),
                                     Vector<double>( { 1.75 } ),
                                     Vector<double>( { 5.2857 } ) );
    }

    void test_parameterised_symbolic_bound_maps_relu_reindex()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTReluReindex( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [-1, 1]
          x1: [-1, 1]

          Layers 1, 2:

          x2 = x0 + x1
          x2.lb = x0 + x1   : [-2, 2]
          x2.ub = x0 + x1   : [-2, 2]

          x3 = x0 - x1
          x3.lb = x0 - x1   : [-2, 2]
          x3.ub = x0 - x1   : [-2, 2]

          Both ReLUs are undecided, bounds are concretized. Using custom ReLU lower
          coefficient of 0.5. Upper coefficient: 2/( 2--2 ) = 2/4 = 0.5

          0.5 x2 <= x4 <= 0.5x2 + 1
          x4.lb = 0.5 ( x0 + x1 ) = 0.5x0 + 0.5x1
          x4.ub = 0.5 ( x0 + x1 ) + 1 = 0.5x0 + 0.5x1 + 1
          x4 range: [-1, 2]

          0.5 x3 <= x5 <= 0.5x3 + 1
          x5.lb = 0.5 ( x0 - x1 ) = 0.5x0 - 0.5x1
          x5.ub = 0.5 ( x0 - x1 ) + 1 = 0.5x0 - 0.5x1 + 1
          x5 range: [-1, 2]

          Layers 3, 4:

          x6 = x4 + x5
          x6.lb = 1 ( 0.5x0 + 0.5x1 ) + 1 ( 0.5x0 - 0.5x1 ) = x0   : [-1, 1]
          x6.ub = 1 ( 0.5x0 + 0.5x1 + 1 ) + 1 ( 0.5x0 - 0.5x1 + 1 ) = x0 + 2   : [1, 3]
          x6 range: [-1, 3]

          x7 = x4 - x5
          x7.lb = 1 ( 0.5x0 + 0.5x1 ) - 1 ( 0.5x0 - 0.5x1 + 1 ) = x1 - 1   : [-2, 0]
          x7.ub = 1 ( 0.5x0 + 0.5x1 + 1 ) - 1 ( 0.5x0 - 0.5x1 ) = x1 + 1  : [0, 2]
          x7 range: [-2, 2]

          Both ReLUs are undecided, bounds are concretized. Using custom ReLU lower
          coefficient of 0.5.
            Upper coefficient (first ReLU): 3/( 3--1 ) = 3/4 = 0.75
            Upper coefficient (second ReLU): 2/( 2--2 ) = 2/4 = 0.5

          0.5 x6 <= x8 <= 0.75 x6 + 0.75
          x8.lb = 0.5 ( x0 ) = 0.5 x0
          x8.ub = 0.75 ( x0 + 2 ) + 0.75 = 0.75 x0 + 2.25
          x8 range: [-0.5, 3]

          0.5 x7 <= x9 <= 0.5 x7 + 1
          x9.lb = 0.5 ( x1 - 1 ) = 0.5 x1 - 0.5
          x9.ub = 0.5 ( x1 + 1 ) + 1 = 0.5x1 + 1.5
          x9 range: [-1, 2]

          Layer 5:
          x10 = x8 + x9 + 1
          x10.lb = 1 ( 0.5 x6 ) + 1 ( 0.5 x7 ) + 1 = ( 0.5 x4 + 0.5x5 ) + 1 ( 0.5 x4 - 0.5x5 ) + 1
          = x4 + 1 >= 0.5 x2 + 1 = 0.5 x0 + 0.5x1 + 1 : [0, 2]
          x10.ub = 1 ( 0.75 x6 + 0.75 ) + 1 ( 0.5 x7 + 1 ) + 1
          = ( 0.75 x4 + 0.75 x5 + 0.75 ) + 1 ( 0.5 x4 - 0.5x5 + 1 ) + 1
          = 1.25 x4 + 0.25 x5 + 2.75 <= 0.625 x4 + 0.125 x5 + 4.25
          = 0.75 x0 + 0.5 x1 + 4.25 : [2.5, 5.5]
          x10 range: [0, 5.5]

          x11 = x9
          x11.lb = 0.5 x1 - 0.5 : [-1, 0]
          x11.ub = 0.5x1 + 1.5 : [1, 2]
          x11 range: [-1, 2]

        */

        List<Tightening> expectedBounds(
            { Tightening( 2, -2, Tightening::LB ),   Tightening( 2, 2, Tightening::UB ),
              Tightening( 3, -2, Tightening::LB ),   Tightening( 3, 2, Tightening::UB ),

              Tightening( 4, -1, Tightening::LB ),   Tightening( 4, 2, Tightening::UB ),
              Tightening( 5, -1, Tightening::LB ),   Tightening( 5, 2, Tightening::UB ),

              Tightening( 6, -1, Tightening::LB ),   Tightening( 6, 3, Tightening::UB ),
              Tightening( 7, -2, Tightening::LB ),   Tightening( 7, 2, Tightening::UB ),

              Tightening( 8, -0.5, Tightening::LB ), Tightening( 8, 3, Tightening::UB ),
              Tightening( 9, -1, Tightening::LB ),   Tightening( 9, 2, Tightening::UB ),

              Tightening( 10, 0, Tightening::LB ),   Tightening( 10, 5.5, Tightening::UB ),
              Tightening( 11, -1, Tightening::LB ),  Tightening( 11, 2, Tightening::UB )

            } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0.5 x2 <= x4 <= 0.5x2 + 1
          0.5 x3 <= x5 <= 0.5x3 + 1

          Layer 4 (RELU):
          0.5 x6 <= x8 <= 0.75 x6 + 0.75
          0.5 x7 <= x9 <= 0.5 x7 + 1

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 5:
          x10 <= x10 <= x10
          x11 <= x11 <= x11

          Layer 4:
          Using x10 = x8 + x9 + 1, x11 = x9:
          x8 + x9 + 1 <= x10 <= x8 + x9 + 1
          x9 <= x11 <= x9

          Layer 3:
          Using 0.5 x6 <= x8 <= x6, 0.5 x7 <= x9 <= 0.5 x7 + 1:
          0.5 x6 + 0.5 x7 + 1 <= x10 <= 0.75 x6 + 0.5 x7 + 2.75
          0.5 x7 <= x11 <= 0.5 x7 + 1

          Layer 2:
          Using x6 = x4 + x5, x7 = x4 - x5:
          x4 + 1 <= x10 <= 1.25 x4 + 0.25 x5 + 2.75
          0.5 x4 - 0.5 x5 <= x11 <= 0.5 x4 - 0.5 x5 + 1

          Layer 1:
          Using 0.5 x2 <= x4 <= 0.5x2 + 1, 0.5 x3 <= x5 <= 0.5x3 + 1:
          0.5 x2 + 1 <= x10 <= 0.625 x2 + 0.125 x3 + 4.25
          0.25 x2 - 0.25 x3 - 0.5 <= x11 <= 0.25 x2 - 0.25 x3 + 1.5

          Layer 0:
          Using x2 = x0 + x1, x3 = x0 - x1:
          0.5 x0 + 0.5 x1 + 1 <= x10 <= 0.75 x0 + 0.5 x1 + 4.25
          0.5 x1 - 0.5 <= x11 <= 0.5 x1 + 1.5
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0.5, 0.5, 0 } ),
                                          Vector<double>( { 0, 0.5, 0.5, 0 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 1, 1 } ) );

        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 0, 0.5, 0.5, 0 } ),
                                          Vector<double>( { 0, 0.75, 0.5, 0 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 1, 0.75 } ) );

        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1, 1, 1, 0 } ),
                                     Vector<double>( { 1, 1, 1, 0 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 1, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 0.5, 0, 0.5, 0.5 } ),
                                     Vector<double>( { 0.75, 0, 0.5, 0.5 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 2.75, 1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 0, -0.5, 1, 0.5 } ),
                                     Vector<double>( { 0.25, -0.5, 1.25, 0.5 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 2.75, 1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.5, 0.25, 0, -0.25 } ),
                                     Vector<double>( { 0.625, 0.25, 0.125, -0.25 } ),
                                     Vector<double>( { 1, -0.5 } ),
                                     Vector<double>( { 4.25, 1.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.5, 0, 0.5, 0.5 } ),
                                     Vector<double>( { 0.75, 0, 0.5, 0.5 } ),
                                     Vector<double>( { 1, -0.5 } ),
                                     Vector<double>( { 4.25, 1.5 } ) );
    }

    void test_parameterised_symbolic_bound_maps_abs_all_positive()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1
          x2.lb = 2x0 + 3x1   : [11, 27]
          x2.ub = 2x0 + 3x1   : [11, 27]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          Both absolute values positive, bound survive through activations:

          x2 <= x4 <= x2
          x4.lb = 2x0 + 3x1   : [11, 27]
          x4.ub = 2x0 + 3x1   : [11, 27]

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:
          x5 = x4 - x5
          => x2 - x3 <= x5 <= x2 - x3
          x6.lb =  x0 + 2x1   : [6, 16]
          x6.ub =  x0 + 2x1   : [6, 16]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, 11, Tightening::LB ),
            Tightening( 2, 27, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 11, Tightening::LB ),
            Tightening( 4, 27, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, 6, Tightening::LB ),
            Tightening( 6, 16, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          x2 <= x4 <= x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using x2 <= x4 <= x2, x3 <= x5 <= x3:
          x2 - x3 <= x6 <= x2 - x3

          Layer 0:
          Using x2 = 2x0 + 3x1, x3 = x0 + x1:
          x0 + 2x1 <= x6 <= x0 + 2x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 1, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_parameterised_symbolic_bound_maps_abs_positive_and_negative()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -30 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:
          x2 = 2x0 + 3x1 - 30
          x2.lb = 2x0 + 3x1 - 30   : [-19, -3]
          x2.ub = 2x0 + 3x1 - 30   : [-19, -3]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First absolute value is negative, bounds get flipped
          Second absolute value is positive, bounds surive the activation

          -x2 <= x4 <= -x2
          x4.lb = -2x0 -3x1 + 30   : [3, 19]
          x4.ub = -2x0 -3x1 + 30   : [3, 19]

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:
          x5 = x4 - x5
          => -x2 - x3 <= x5 <= -x2 - x3
          x6.lb =  - 3x0 - 4x1 + 30  : [-8, 14]
          x6.ub =  - 3x0 - 4x1 + 30  : [-8, 14]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -19, Tightening::LB ),
            Tightening( 2, -3, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 3, Tightening::LB ),
            Tightening( 4, 19, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -8, Tightening::LB ),
            Tightening( 6, 14, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          -x2 <= x4 <= -x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using -x2 <= x4 <= -x2, x3 <= x5 <= x3:
          -x2 - x3 <= x6 <= -x2 - x3

          Layer 0:
          Using x2 = 2x0 + 3x1 - 30, x3 = x0 + x1:
          -3x0 - 4x1 + 30 <= x6 <= -3x0 - 4x1 + 30
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -3, -4 } ),
                                     Vector<double>( { -3, -4 } ),
                                     Vector<double>( { 30 } ),
                                     Vector<double>( { 30 } ) );
    }

    void test_parameterised_symbolic_bound_maps_absolute_values_positive_and_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -15 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:
          x2 = 2x0 + 3x1 - 15
          x2.lb = 2x0 + 3x1 - 15   : [-4, 12]
          x2.ub = 2x0 + 3x1 - 15   : [-4, 12]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First absolute value is undecided, bounds are concretized.
          Second absolute value is active, bounds surive the activation

          0 <= x4 <= 12
          x4 range: [0, 12]
          x4.lb = 0
          x4.ub = 12

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          => -x3 <= x6 <= -x3 + 12
          x6.lb =  - x0 - x1       : [-11, -5]
          x6.ub =  - x0 - x1 + 12  : [  1,  7]

          x6 range: [-11, 7]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -4, Tightening::LB ),
            Tightening( 2, 12, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 12, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -11, Tightening::LB ),
            Tightening( 6, 7, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          0 <= x4 <= 12
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using 0 <= x4 <= 12, x3 <= x5 <= x3:
          -x3 <= x6 <= -x3 + 12

          Layer 0:
          Using x3 = x0 + x1:
          -x0 - x1 <= x6 <= -x0 - x1 + 12
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 12, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 12 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 12 } ) );
    }

    void test_parameterised_symbolic_bound_maps_absolute_values_active_and_externally_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::ABSOLUTE_VALUE, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Abs sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0). Should make the node unfixed.
        nlr.setBias( 1, 0, -15 );

        // However, the weighted sum variable has been eliminated
        nlr.eliminateVariable( 2, -3 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = -3
          x2 is eliminated, everything set to -3

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First absolute value is negative, bounds get flipped
          Second absolute value is positive, bounds surive the activation

          -x2 <= x4 <= -x2
          x4: all set to 3

          x3 <= x5 <= x3
          x5.lb =  x0 + x1   : [5, 11]
          x5.ub =  x0 + x1   : [5, 11]

          Layer 3:

          x6 = x4 - x5
          => -x2 - x3 <= x6 <= -x2 - x3
          => -x3 + 3 <= x6 <= -x3 + 3
          x6.lb =  - x0 - x1 + 3  : [-8, -2]
          x6.ub =  - x0 - x1 + 3  : [-8, -2]
        */

        List<Tightening> expectedBounds( {
            // x2 does not appear, because it has been eliminated

            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, 3, Tightening::LB ),
            Tightening( 4, 3, Tightening::UB ),
            Tightening( 5, 5, Tightening::LB ),
            Tightening( 5, 11, Tightening::UB ),

            Tightening( 6, -8, Tightening::LB ),
            Tightening( 6, -2, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (ABSOLUTE_VALUE):
          -x2 <= x4 <= -x2
          x3 <= x5 <= x3

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using -x2 <= x4 <= -x2, x3 <= x5 <= x3:
          -x2 - x3 <= x6 <= -x2 - x3
          x2 = -3 is eliminated.
          -x3 + 3 <= x6 <= -x3 + 3

          Layer 0:
          Using x3 = x0 + x1:
          - x0 - x1 + 3 <= x6 <= - x0 - x1 + 3
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { -1, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 0, -1 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { -1, -1 } ),
                                     Vector<double>( { 3 } ),
                                     Vector<double>( { 3 } ) );
    }

    void test_parameterised_symbolic_bound_maps_signs_positive_and_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::SIGN, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Sign sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0)
        nlr.setBias( 1, 0, -15 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = 2x0 + 3x1 - 15
          x2.lb = 2x0 + 3x1 - 15   : [-4, 12]
          x2.ub = 2x0 + 3x1 - 15   : [-4, 12]

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

         First sign is undecided, bounds are concretized.
          Second sign is active, bounds become constant 1
          Using custom coefficients with alpha = { 0.5, 0.5 }.
            Coefficient (first Sign, lower): 2/12 * 0.5 = 1/12.
            Coefficient (first Sign, upper): -2/-4 * 0.5 = 1/4.

          1/12 x2 - 1 <= x4 <= 1/4 x2 + 1
          x4.lb = 1/12 ( 2x0 + 3x1 - 15 ) - 1 = 2/12 x0 + 3/12 x1 - 27/12
          x4.ub = 1/4 ( 2x0 + 3x1 - 15 ) + 1 = 0.5 x0 + 0.75x1 - 2.75
          x4 range: [-1, 1]

          1 <= x5 <= 1
          x5.lb = 1
          x5.ub = 1
          x5 range: [1, 1]

          Layer 3:

          x6 = x4 - x5 : [-2, 0]
          => 1/12 x2 - 2 <= x6 <= 1/4 x2 : [-8/3, 6]
          x6.lb =  1 ( 2/12 x0 + 3/12 x1 - 27/12 ) - 1 ( 1 ) = 2/12 x0 + 3/12 x1 - 39/12 :
          [-28/12 = -7/3, -1]
          x6.ub =  1 ( 0.5 x0 + 0.75x1 - 2.75 ) - 1 ( 1 ) = 0.5 x0 + 0.75x1 - 3.75 : [-1, 3]

          x6 range: [-2, 0]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -4, Tightening::LB ),
            Tightening( 2, 12, Tightening::UB ),
            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, 1, Tightening::UB ),
            Tightening( 5, 1, Tightening::LB ),
            Tightening( 5, 1, Tightening::UB ),

            Tightening( 6, -2, Tightening::LB ),
            Tightening( 6, 0, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SIGN):
          1/12 x2 - 1 <= x4 <= 1/4 x2 + 1
          1 <= x5 <= 1

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using 1/12 x2 - 1 <= x4 <= 1/4 x2 + 1, 1 <= x5 <= 1:
          1/12 x2 - 2 <= x6 <= 1/4 x2

          Layer 0:
          Using x2 = 2x0 + 3x1 - 15:
          1/6 x0 + 1/4 x1 - 3.25 <= x6 <= 0.5 x0 + 0.75x1 - 3.75
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.0833, 0, 0, 0 } ),
                                          Vector<double>( { 0.25, 0, 0, 0 } ),
                                          Vector<double>( { -1, 1 } ),
                                          Vector<double>( { 1, 1 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.0833, 0 } ),
                                     Vector<double>( { 0.25, 0 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.1667, 0.25 } ),
                                     Vector<double>( { 0.5, 0.75 } ),
                                     Vector<double>( { -3.25 } ),
                                     Vector<double>( { -3.75 } ) );
    }

    void test_parameterised_symbolic_bound_maps_signs_active_and_externally_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        tableau.getBoundManager().initialize( 7 );
        nlr.setTableau( &tableau );

        // Create the layers
        nlr.addLayer( 0, NLR::Layer::INPUT, 2 );
        nlr.addLayer( 1, NLR::Layer::WEIGHTED_SUM, 2 );
        nlr.addLayer( 2, NLR::Layer::SIGN, 2 );
        nlr.addLayer( 3, NLR::Layer::WEIGHTED_SUM, 1 );

        // Mark layer dependencies
        for ( unsigned i = 1; i <= 3; ++i )
            nlr.addLayerDependency( i - 1, i );

        // Weights
        nlr.setWeight( 0, 0, 1, 0, 2 );
        nlr.setWeight( 0, 0, 1, 1, 1 );
        nlr.setWeight( 0, 1, 1, 0, 3 );
        nlr.setWeight( 0, 1, 1, 1, 1 );
        nlr.setWeight( 2, 0, 3, 0, 1 );
        nlr.setWeight( 2, 1, 3, 0, -1 );

        // Mark the Sign sources
        nlr.addActivationSource( 1, 0, 2, 0 );
        nlr.addActivationSource( 1, 1, 2, 1 );

        // Variable indexing
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 0 ), 0 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 0, 1 ), 1 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 0 ), 2 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 1, 1 ), 3 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 0 ), 4 );
        nlr.setNeuronVariable( NLR::NeuronIndex( 2, 1 ), 5 );

        nlr.setNeuronVariable( NLR::NeuronIndex( 3, 0 ), 6 );

        // Very loose bounds for neurons except inputs
        double large = 1000000;

        tableau.setLowerBound( 2, -large );
        tableau.setUpperBound( 2, large );
        tableau.setLowerBound( 3, -large );
        tableau.setUpperBound( 3, large );
        tableau.setLowerBound( 4, -large );
        tableau.setUpperBound( 4, large );
        tableau.setLowerBound( 5, -large );
        tableau.setUpperBound( 5, large );
        tableau.setLowerBound( 6, -large );
        tableau.setUpperBound( 6, large );

        tableau.setLowerBound( 0, 4 );
        tableau.setUpperBound( 0, 6 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 5 );

        // Strong negative bias for x2, which is node (1,0). Should make the node unfixed.
        nlr.setBias( 1, 0, -15 );

        // However, the weighted sum variable has been eliminated
        nlr.eliminateVariable( 2, -3 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [4, 6]
          x1: [1, 5]

          Layers 1, 2:

          x2 = -3
          x2 is eliminated, everything set to -3

          x3 = x0 + x1
          x3.lb =  x0 + x1   : [5, 11]
          x3.ub =  x0 + x1   : [5, 11]

          First sign is negative, bounds become constant -1
          Second sign is positive, bounds become constant 1

          -1 <= x4 <= 1
          x4: all set to -1

          1 <= x5 <= 1
          x5: all set to 1

          Layer 3:

          x6 = x5 - x4
          x6.lb = 1 ( -1 ) - 1 ( 1 ) = -2
          x6.ub = 1 ( -1 ) - 1 ( 1 ) = -2
        */

        List<Tightening> expectedBounds( {
            // x2 does not appear, because it has been eliminated

            Tightening( 3, 5, Tightening::LB ),
            Tightening( 3, 11, Tightening::UB ),

            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, -1, Tightening::UB ),
            Tightening( 5, 1, Tightening::LB ),
            Tightening( 5, 1, Tightening::UB ),

            Tightening( 6, -2, Tightening::LB ),
            Tightening( 6, -2, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SIGN):
          -1 <= x4 <= -1
          1 <= x5 <= 1

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x6 <= x6 <= x6

          Layer 2:
          Using x6 = x5 - x4:
          x4 - x5 <= x6 <= x4 - x5

          Layer 1:
          Using -1 <= x4 <= -1, 1 <= x5 <= 1:
          -2 <= x6 <= -2

          Layer 0:
          -2 <= x6 <= -2
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 0 } ),
                                          Vector<double>( { 0, 0, 0, 0 } ),
                                          Vector<double>( { -1, 1 } ),
                                          Vector<double>( { -1, 1 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 1, -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { -2 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { -2 } ),
                                     Vector<double>( { -2 } ) );
    }

    void test_parameterised_symbolic_bound_maps_leaky_relu()
    {
        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTLeakyReLU( nlr, tableau ); // alpha = 0.2

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [-1, 1]
          x1: [-1, 1]

          Layer 1:
          x2 = x0 + x1
          x2.lb = x0 + x1 : [-2, 2]
          x2.ub = x0 + x1 : [-2, 2]

          x3 = x0 - x1
          x3.lb = x0 - x1 : [-2, 2]
          x3.ub = x0 - x1 : [-2, 2]

          Both LeakyReLUs are undecided, bounds are concretized. Using custom lower coefficient with
          alpha = { 0.5 }.
            Lower Coefficient: ( 1 - 0.2 ) * 0.5 + 0.2 = 0.6
            Lower Bias: 0
            Upper Coefficient: ( 2 - 0.2*-2 )/( 2--2 ) = 2.4/4 = 0.6
            Upper Bias: ( 0.2 - 1 ) * 2 * -2 /( 2--2 ) = 0.8

          0.6 x2 <= x4 <= 0.6 x2 + 0.8
          x4.lb = 0.6 ( x0 + x1 ) = 0.6 x0 + 0.6x1
          x4.ub = 0.6 ( x0 + x1 ) + 0.8 = 0.6 x0 + 0.6 x1 + 0.8
          x4 range: [-1.2, 2]

          0.6 x3 <= x5 <= 0.6 x3 + 0.8
          x5.lb = 0.6 ( x0 - x1 ) = 0.6 x0 - 0.6 x1
          x5.ub = 0.6 ( x0 - x1 ) + 0.8 = 0.6 x0 - 0.6 x1 + 0.8
          x5 range: [-1.2, 2]

          Layer 2:

          x6 = x4 + x5
          x6.lb = 1 ( 0.6x0 + 0.6x1 ) + 1 ( 0.6x0 - 0.6x1 ) = 1.2 x0 : [-1.2, 1.2]
          x6.ub = 1 ( 0.6x0 + 0.6x1 + 0.8 ) + 1 ( 0.6x0 - 0.6x1 + 0.8 ) = 1.2 x0 + 1.6 :
          [0.4, 2.8] x6 range: [-1.2, 2.8]

          x7 = x4 - x5
          x7.lb = 1 ( 0.6x0 + 0.6x1 ) - 1 ( 0.6x0 - 0.6x1 + 0.8 ) = 1.2 x1 - 0.8 : [-2, 0.4]
          x7.ub = 1 ( 0.6x0 + 0.6x1 + 0.8 ) - 1 ( 0.6x0 - 0.6x1 ) = 1.2 x1 + 0.8 : [-0.4, 2]
          x7 range: [-2, 2]

          Both LeakyReLUs are undecided, bounds are concretized. Using custom lower coefficient with
          alpha = { 0.5 }.
            Lower Coefficient (first LeakyReLU): ( 1 - 0.2 ) * 0.5 + 0.2 = 0.6
            Lower Bias (first LeakyReLU): 0
            Upper Coefficient (first LeakyReLU): ( 2.8 - 0.2*-1.2 )/( 2.8--1.2 ) = 3.04/4 = 0.76
            Upper Bias (first LeakyReLU): ( 0.2 - 1 ) * 2.8 * -1.2 / ( 2.8--1.2 ) = 0.672

            Lower Coefficient (second LeakyReLU): ( 1 - 0.2 ) * 0.5 + 0.2 = 0.6
            Lower Bias (second LeakyReLU): 0
            Upper Coefficient (second LeakyReLU): ( 2 - 0.2*-2 )/( 2--2 ) = 2.4/4 = 0.6
            Upper Bias (second LeakyReLU): ( 0.2 - 1 ) * 2 * -2 / ( 2--2 ) = 0.8

          0.6 x6 <= x8 <= 0.76 x6 + 0.672
          x8.lb = 0.6 ( 1.2x0 ) = 0.72 x0
          x8.ub = 0.76 ( 1.2x0 + 1.6 ) + 0.672 = 0.912 x0 + 1.888
          x8 range: [-0.72, 2.8]

          0.6 x7 <= x9 <= 0.6 x7 + 0.8
          x9.lb = 0.6 ( 1.2x1 - 0.8 ) = 0.72 x0 - 0.48
          x9.ub = 0.6 ( 1.2x1 + 0.8 ) + 0.8 = 0.72 x1 + 1.28
          x9 range: [-1.2, 2]

          Layer 3:

          x10 = x8 + x9 + 1
          x10.lb = 0.6 x6 + 0.6 x7 + 1 >= 0.6 ( x4 + x5 ) + 0.6 ( x4 - x5 ) + 1 =
          1.2 x4 + 1 >= 1.2 ( 0.6 x2 ) + 1 = 0.72 x2 + 1
          = 0.72 x0 + 0.72 x1 + 1 : [-0.44, 2.44]
          x10.lb = ( 0.76 x6 + 0.672 ) + ( 0.6 x7 + 0.8 ) + 1 = 0.76 x6 + 0.6 x7 + 2.472
          >= 0.76 ( x4 + x5 ) + 0.6 ( x4 - x5 ) + 2.472 = 1.36 x4 + 0.16 x5 + 2.472
          >= 1.36 ( 0.6 x2 + 0.8 ) + 0.16 ( 0.6 x3 + 0.8 ) + 2.472
          = 0.816 x2 + 0.096 x3 + 3.688 = 0.912 x0 + 0.72 x1 + 3.688 : [2.056, 5.32]
          x10 range: [-0.44, 5.32]

          x11.lb = 0.72 x0 - 0.48 : [-1.2, 0.24]
          x11.ub = 0.72 x1 + 1.28 : [-0.56, 2]
          x11 range: [-1.2, 2]

        */

        List<Tightening> expectedBounds(
            { Tightening( 2, -2, Tightening::LB ),     Tightening( 2, 2, Tightening::UB ),
              Tightening( 3, -2, Tightening::LB ),     Tightening( 3, 2, Tightening::UB ),

              Tightening( 4, -1.2, Tightening::LB ),   Tightening( 4, 2, Tightening::UB ),
              Tightening( 5, -1.2, Tightening::LB ),   Tightening( 5, 2, Tightening::UB ),

              Tightening( 6, -1.2, Tightening::LB ),   Tightening( 6, 2.8, Tightening::UB ),
              Tightening( 7, -2, Tightening::LB ),     Tightening( 7, 2, Tightening::UB ),

              Tightening( 8, -0.72, Tightening::LB ),  Tightening( 8, 2.8, Tightening::UB ),
              Tightening( 9, -1.2, Tightening::LB ),   Tightening( 9, 2, Tightening::UB ),

              Tightening( 10, -0.44, Tightening::LB ), Tightening( 10, 5.32, Tightening::UB ),
              Tightening( 11, -1.2, Tightening::LB ),  Tightening( 11, 2, Tightening::UB )

            } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (LEAKY_RELU):
          0.6 x2 <= x4 <= 0.6 x2 + 0.8
          0.6 x3 <= x5 <= 0.6 x3 + 0.8

          Layer 4 (LEAKY_RELU):
          0.6 x6 <= x8 <= 0.76 x6 + 0.672
          0.6 x7 <= x9 <= 0.6 x7 + 0.8

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 5:
          x10 <= x10 <= x10
          x11 <= x11 <= x11

          x10 = x8 + x9 + 1
          x10.lb =  >= 0.6 ( x4 + x5 ) + 0.6 ( x4 - x5 ) + 1 =
          1.2 x4 + 1 >= 1.2 ( 0.6 x2 ) + 1 = 0.72 x2 + 1
          = 0.72 x0 - 0.72 x1 + 1 : [-0.44, 2.44]
          x10.lb = ( 0.76 x6 + 0.672 ) + ( 0.6 x7 + 0.8 ) + 1 = 0.76 x6 + 0.6 x7 + 2.472
          >= 0.76 ( x4 + x5 ) + 0.6 ( x4 - x5 ) + 2.472 = 1.36 x4 + 0.16 x5 + 2.472
          >= 1.36 ( 0.6 x2 + 0.8 ) + 0.16 ( 0.6 x3 + 0.8 ) + 2.472
          = 0.816 x2 + 0.096 x3 + 3.688 = 0.912 x0 - 0.72 x1 + 3.688 : [2.056, 5.32]
          x10 range: [-0.44, 5.32]

          Layer 4:
          Using x10 = x8 + x9 + 1, x11 = x9:
          x8 + x9 + 1 <= x10 <= x8 + x9 + 1
          x9 <= x11 <= x9

          Layer 3:
          Using 0.6 x6 <= x8 <= 0.76 x6 + 0.672, 0.6 x7 <= x9 <= 0.6 x7 + 0.8:
          0.6 x6 + 0.6 x7 + 1 <= x10 <= 0.76 x6 + 0.6 x7 + 2.472
          0.6 x7 <= x11 <= 0.6 x7 + 0.8

          Layer 2:
          Using x6 = x4 + x5, x7 = x4 - x5:
          1.2 x4 + 1 <= x10 <= 1.36 x4 + 0.16 x5 + 2.472
          0.6 x4 - 0.6 x5 <= x11 <= 0.6 x4 - 0.6 x5 + 0.8

          Layer 1:
          Using 0.6 x2 <= x4 <= 0.6 x2 + 0.8, 0.6 x3 <= x5 <= 0.6 x3 + 0.8:
          0.72 x2 + 1 <= x10 <= 0.816 x2 + 0.096 x3 + 3.688
          0.36 x2 - 0.36 x3 - 0.48 <= x11 <= 0.36 x2 - 0.36 x3 + 1.28

          Layer 0:
          Using x2 = x0 + x1, x3 = x0 - x1:
          0.72 x0 + 0.72 x1 + 1 <= x10 <= 0.912 x0 + 0.72 x1 + 3.688
          0.72 x1 - 0.48 <= x11 <= 0.72 x1 + 1.28
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.6, 0, 0, 0.6 } ),
                                          Vector<double>( { 0.6, 0, 0, 0.6 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0.8, 0.8 } ) );

        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 0.6, 0, 0, 0.6 } ),
                                          Vector<double>( { 0.76, 0, 0, 0.6 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0.672, 0.8 } ) );

        compareOutputSymbolicBounds( nlr,
                                     5,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1, 0, 1, 1 } ),
                                     Vector<double>( { 1, 0, 1, 1 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 1, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 0.6, 0, 0.6, 0.6 } ),
                                     Vector<double>( { 0.76, 0, 0.6, 0.6 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 2.472, 0.8 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1.2, 0.6, 0, -0.6 } ),
                                     Vector<double>( { 1.36, 0.6, 0.16, -0.6 } ),
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 2.472, 0.8 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.72, 0.36, 0, -0.36 } ),
                                     Vector<double>( { 0.816, 0.36, 0.096, -0.36 } ),
                                     Vector<double>( { 1, -0.48 } ),
                                     Vector<double>( { 3.688, 1.28 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.72, 0, 0.72, 0.72 } ),
                                     Vector<double>( { 0.912, 0, 0.72, 0.72 } ),
                                     Vector<double>( { 1, -0.48 } ),
                                     Vector<double>( { 3.688, 1.28 } ) );
    }

    void test_parameterised_symbolic_bound_maps_sigmoids_and_round()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTSigmoidsAndRound( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );

        // Layer 1
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getLb( 0 ), -2, 0.00001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getUb( 0 ), 2, 0.00001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getLb( 1 ), -2, 0.00001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 1 )->getUb( 1 ), 2, 0.00001 ) );

        // Layer 2
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getLb( 0 ), 0.1192, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getUb( 0 ), 0.8807, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getLb( 1 ), 0.1192, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 2 )->getUb( 1 ), 0.8807, 0.0001 ) );

        // Layer 3
        /*
         Double-check with Python
            ---
            from math import exp as e
            def g(x):
                return 1 / (1 + e(-x))

            def g_prime(x):
                return g(x) * (1 - g(x))

            def lam(l, u):
                return (g(u) - g(l)) / (u - l)

            def lam_prime(l, u):
                return min(g_prime(l), g_prime(u))

            l3 = l4 = -2
            u3 = u4 = 2
            l5 = l6 = g(-2)
            u5 = u6 = g(2)
            lambda7 = lam(l3, u3)
            lambda7_prime = lam_prime(l3, u3)
            lambda8 = lam(l4, u4)
            lambda8_prime = lam_prime(l4, u4)
            x7_l = lambda7_prime * (-2) + g(-2) + g(-2) - lambda7_prime * (-2 + -2)
            x7_u = lambda7_prime * (2) + g(2) + g(2) -lambda7_prime * (2 + 2)
            x8_l = lambda8_prime * (-2) + g(-2) - g(2) - lambda8_prime * (-2 - 2)
            x8_u = lambda8_prime * (2) + g(2) - g(-2) -lambda8_prime * (2 - -2)
            print(x7_l)
            print(x7_u)
            print(x8_l)
            print(x8_u)

            '''
            Sigmoid linear relaxation ( Layer 2 ):
            x4 >= lambda7_prime * x2 + ( g(l3) - lambda7_prime * l3 )
            x4 <= lambda7_prime * x2 + ( g(u3) - lambda7_prime * u3 )
            x5 >= lambda8_prime * x3 + ( g(l4) - lambda8_prime * l4 )
            x5 <= lambda8_prime * x3 + ( g(u4) - lambda7_prime * u4 )
            '''
            print('------------------')
            print(lambda7_prime)
            print(lambda8_prime)
            print(g(l3) - lambda7_prime * l3)
            print(g(u3) - lambda7_prime * u3)
            print(g(l4) - lambda8_prime * l4)
            print(g(u4) - lambda8_prime * u4)


            ---
            [output]:
            0.4483930148512481
            1.5516069851487517
            -0.5516069851487517
            0.5516069851487517
            ------------------
            0.1049935854035065
            0.1049935854035065
            0.3291900928291306
            0.6708099071708693
            0.3291900928291306
            0.6708099071708693
        */
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getLb( 0 ), 0.4483, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getUb( 0 ), 1.5516, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getLb( 1 ), -0.5516, 0.0001 ) );
        TS_ASSERT( FloatUtils::areEqual( nlr.getLayer( 3 )->getUb( 1 ), 0.5516, 0.0001 ) );

        // Layer 4
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getLb( 0 ), 0 );
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getUb( 0 ), 2 );
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getLb( 1 ), -1 );
        TS_ASSERT_EQUALS( nlr.getLayer( 4 )->getUb( 1 ), 1 );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SIGMOID):
          0.1050 x2 + 0.3292 <= x4 <= 0.1050 x2 + 0.6708
          0.1050 x3 + 0.3292 <= x5 <= 0.1050 x3 + 0.6708

          Layer 4 (ROUND):
          x6 - 0.5 <= x8 <= x6 + 0.5
          x7 - 0.5 <= x9 <= x7 + 0.5

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 4:
          x8 <= x8 <= x8
          x9 <= x9 <= x9

          Layer 3:
          Using x6 - 0.5 <= x8 <= x6 + 0.5, x7 - 0.5 <= x9 <= x7 + 0.5:
          x6 - 0.5 <= x8 <= x6 + 0.5
          x7 - 0.5 <= x9 <= x7 + 0.5

          Layer 2:
          Using x6 = x4 + x5, x7 = x4 - x5:
          x4 + x5 - 0.5 <= x8 <= x4 + x5 + 0.5
          x4 - x5 - 0.5 <= x9 <= x4 - x5 + 0.5

          Layer 1:
          Using
          0.1050 x2 + 0.3292 <= x4 <= 0.1050 x2 + 0.6708,
          0.1050 x3 + 0.3292 <= x5 <= 0.1050 x3 + 0.6708:
          0.1050 x2 + 0.1050 x3 + 0.1584 <= x8 <= 0.1050 x2 + 0.1050 x3 + 1.8416
          0.1050 x2 - 0.1050 x3 - 0.8416 <= x9 <= 0.1050 x2 - 0.1050 x3 + 0.8516

          Layer 0:
          Using x2 = x0 + x1, x3 = x0 - x1:
            0.2100 x0 + 0.1584 <= x8 <= 0.2100 x0 + 1.8416
            0.2100 x1 - 0.8416 <= x9 <= 0.2100 x1 + 0.8516
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.1050, 0, 0, 0.1050 } ),
                                          Vector<double>( { 0.1050, 0, 0, 0.1050 } ),
                                          Vector<double>( { 0.3292, 0.3292 } ),
                                          Vector<double>( { 0.6708, 0.6708 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          4,
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { 1, 0, 0, 1 } ),
                                          Vector<double>( { -0.5, -0.5 } ),
                                          Vector<double>( { 0.5, 0.5 } ) );

        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { 1, 0, 0, 1 } ),
                                     Vector<double>( { -0.5, -0.5 } ),
                                     Vector<double>( { 0.5, 0.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 1, 1, 1, -1 } ),
                                     Vector<double>( { 1, 1, 1, -1 } ),
                                     Vector<double>( { -0.5, -0.5 } ),
                                     Vector<double>( { 0.5, 0.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0.1050, 0.1050, 0.1050, -0.1050 } ),
                                     Vector<double>( { 0.1050, 0.1050, 0.1050, -0.1050 } ),
                                     Vector<double>( { 0.1584, -0.8416 } ),
                                     Vector<double>( { 1.8416, 0.8416 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 0.2100, 0, 0, 0.2100 } ),
                                     Vector<double>( { 0.2100, 0, 0, 0.2100 } ),
                                     Vector<double>( { 0.1584, -0.8416 } ),
                                     Vector<double>( { 1.8416, 0.8416 } ) );
    }

    void test_parameterised_symbolic_bound_maps_max_not_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTMax( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 2 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [-1, 1]
          x1: [-1, 2]

          Layers 1, 2, 3:

          x2 = x0 + x1
          x2.lb =  x0 + x1   : [-2, 3]
          x2.ub =  x0 + x1   : [-2, 3]

          x3 = x0 - x1
          x3.lb =  x0 - x1   : [-3, 2]
          x3.ub =  x0 - x1   : [-3, 2]

          Both ReLUs are undecided, bounds are concretized. Using custom ReLU lower
          coefficient of 0.5.\
            Upper coefficient (first ReLU): 3/( 3--2 ) = 3/5 = 0.6.
            Upper coefficient (second ReLU): 2/( 2--3 ) = 2/5 = 0.4

          0.5 x2 <= x4 <= 0.6 x2 + 1.2
          x4.lb = 0.5 ( x0 + x1 ) = 0.5 x0 + 0.5 x1
          x4.ub = 0.6 ( x0 + x1 ) + 1.2 = 0.6x0 + 0.6x1 + 1.2
          x4 range: [-1, 3]

          0.5 x3 <= x5 <= 0.4 x3 + 1.2
          x5.lb =  0.5 ( x0 - x1 ) = 0.5 x0 - 0.5 x1
          x5.ub =  0.4 ( x0 - x1 ) + 1.2 = 0.4x0 + 0.4x1 + 1.2
          x5 range: [-1.5, 2]

          Max is not fixed because x5.lb <= x4.ub and x4.lb <= x5.ub
          Max inherits lower bound from x4, and its upper bound is constant 3.

          x4 <= x6 <= 3
          x6.lb =  0.5 x0 + 0.5 x1  : [-1, 1.5]
          x6.ub =  3   : [3, 3]
          x6 range: [-1, 3]

          Layer 4:

          x7 = 2x6
          => 2x4 <= x7 <= 6
          x7.lb = 2 ( 0.5 x0 + 0.5 x1 ) = x0 + x1   : [-2, 3]
          x7.ub = 2 ( 3 ) = 6   : [6, 6]
          x7 range: [-2, 6]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -2, Tightening::LB ),
            Tightening( 2, 3, Tightening::UB ),
            Tightening( 3, -3, Tightening::LB ),
            Tightening( 3, 2, Tightening::UB ),
            Tightening( 4, -1, Tightening::LB ),
            Tightening( 4, 3, Tightening::UB ),
            Tightening( 5, -1.5, Tightening::LB ),
            Tightening( 5, 2, Tightening::UB ),
            Tightening( 6, -1, Tightening::LB ),
            Tightening( 6, 3, Tightening::UB ),
            Tightening( 7, -2, Tightening::LB ),
            Tightening( 7, 6, Tightening::UB ),

        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0.5 x2 <= x4 <= 0.6 x2 + 1.2
          0.5 x3 <= x5 <= 0.4 x3 + 1.2

          Layer 3 (MAX):
          x4 <= x6 <= 6

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 4:
          x7 <= x7 <= x7

          Layer 3:
          Using x7 = 2x6:
          2x6 <= x7 <= 2x6

          Layer 2:
          Using x5 <= x6 <= 3:
          2x4 <= x7 <= 6

          Layer 1:
          Using 0.5 x2 <= x4 <= 0.6 x2 + 1.2:
          x2 <= x7 <= 6

          Layer 0:
          Using x2 = x0 + x1:
          x0 + x1 <= x7 <= 6
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0.5, 0, 0, 0.5 } ),
                                          Vector<double>( { 0.6, 0, 0, 0.4 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 1.2, 1.2 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          3,
                                          Vector<double>( { 1, 0 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 3 } ) );

        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 2, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 1, 0 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 6 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 1, 1 } ),
                                     Vector<double>( { 0, 0 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 6 } ) );
    }

    void test_parameterised_symbolic_bound_maps_max_fixed()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTMax( nlr, tableau );

        tableau.setLowerBound( 0, 1 );
        tableau.setUpperBound( 0, 2 );
        tableau.setLowerBound( 1, -3 );
        tableau.setUpperBound( 1, -2 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [1, 2]
          x1: [-3, -2]

          Layer 1:

          x2 = x0 + x1
          x2.lb =  x0 + x1   : [-2, 0]
          x2.ub =  x0 + x1   : [-2, 0]

          x3 = x0 - x1
          x3.lb =  x0 - x1   : [3, 5]
          x3.ub =  x0 - x1   : [3, 5]

          First ReLU is negative, bounds become constant 0
          Second ReLU is positive, bounds survive the activation

          0 <= x4 <= 0
          x4: all set to 0

          x3 <= x5 <= x3
          x5.lb =  x0 - x1   : [3, 5]
          x5.ub =  x0 - x1   : [3, 5]

          Max is fixed because x5.lb > x4.ub, it inherits x5's bounds

          x5 <= x6 <= x5
          => x3 <= x6 <= x5
          x6.lb =  x0 - x1   : [3, 5]
          x6.ub =  x0 - x1   : [3, 5]

          Layer 3:

          x7 = 2x6
          => x7 = 2x5 = 2x3 = 2x0 - 2x1
          x7.lb = 2 ( x0 - x1 ) = 2x0 - 2x1   : [6, 10]
          x7.ub = 2 ( x0 - x1 ) = 2x0 - 2x1   : [6, 10]
        */

        List<Tightening> expectedBounds( {
            Tightening( 2, -2, Tightening::LB ),
            Tightening( 2, 0, Tightening::UB ),
            Tightening( 3, 3, Tightening::LB ),
            Tightening( 3, 5, Tightening::UB ),
            Tightening( 4, 0, Tightening::LB ),
            Tightening( 4, 0, Tightening::UB ),
            Tightening( 5, 3, Tightening::LB ),
            Tightening( 5, 5, Tightening::UB ),
            Tightening( 6, 3, Tightening::LB ),
            Tightening( 6, 5, Tightening::UB ),
            Tightening( 7, 6, Tightening::LB ),
            Tightening( 7, 10, Tightening::UB ),

        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (RELU):
          0 <= x4 <= 0
          x3 <= x5 <= x3

          Layer 3 (MAX):
          x5 <= x6 <= x5

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 4:
          x7 <= x7 <= x7

          Layer 3:
          Using x7 = 2x6:
          2x6 <= x7 <= 2x6

          Layer 2:
          Using x5 <= x6 <= x5:
          2x5 <= x7 <= 2x5

          Layer 1:
          Using x3 <= x5 <= x3:
          2x3 <= x7 <= 2x3

          Layer 0:
          Using x3 = x0 - x1
          2x0 - 2x1 <= x7 <= 2x0 - 2x1
        */
        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0, 0, 1 } ),
                                          Vector<double>( { 0, 0 } ),
                                          Vector<double>( { 0, 0 } ) );
        comparePredecessorSymbolicBounds( nlr,
                                          3,
                                          Vector<double>( { 0, 1 } ),
                                          Vector<double>( { 0, 1 } ),
                                          Vector<double>( { 0 } ),
                                          Vector<double>( { 0 } ) );

        compareOutputSymbolicBounds( nlr,
                                     4,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0, 2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { 2, -2 } ),
                                     Vector<double>( { 2, -2 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
    }

    void test_parameterised_symbolic_bound_maps_softmax1()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTSoftmax( nlr, tableau );

        tableau.setLowerBound( 0, -1 );
        tableau.setUpperBound( 0, 1 );
        tableau.setLowerBound( 1, -1 );
        tableau.setUpperBound( 1, 1 );
        tableau.setLowerBound( 2, -1 );
        tableau.setUpperBound( 2, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );
    }

    void test_parameterised_symbolic_bound_maps_softmax2()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );

        {
            Options::get()->setString( Options::SOFTMAX_BOUND_TYPE, "lse" );
            NLR::NetworkLevelReasoner nlr;
            MockTableau tableau;
            nlr.setTableau( &tableau );
            populateNetworkSBTSoftmax( nlr, tableau );

            tableau.setLowerBound( 0, 1 );
            tableau.setUpperBound( 0, 1.000001 );
            tableau.setLowerBound( 1, 1 );
            tableau.setUpperBound( 1, 1.000001 );
            tableau.setLowerBound( 2, 1 );
            tableau.setUpperBound( 2, 1.000001 );

            unsigned paramCount = nlr.getNumberOfParameters();
            Vector<double> coeffs( paramCount, 0.5 );

            // Invoke parameterised initializeSymbolicBoundsMaps
            TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
            TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

            /*
              Input ranges:

              x0: [1, 1.0001]
              x1: [1, 1.0001]
              x2: [1, 1.0001]

              Layer 1:

              x3 = x0 - x1 + x2 + 1
              x3.lb = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3.ub = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3 range: [ 1.999999, 2.000002 ]

              x4 = -x0 + x1 + x2 + 2
              x4.lb = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4.ub = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4 range: [ 2.999999, 3.000002 ]

              x5 = -x0 - x1 - x2 + 3
              x5.lb = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5.ub = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5 range: [ -0.000003, 0 ]
            */

            unsigned size = nlr.getLayer( 2 )->getSize();
            Vector<double> sourceLbs = { 1.999899, 2.999899, -0.000003 };
            Vector<double> sourceUbs = { 2.000102, 3.000102, 0.0001 };
            Vector<double> sourceMids = { 2.0000005, 3.0000005, -0.0000015 };
            Vector<double> targetLbs( size, 0 );
            Vector<double> targetUbs( size, 0 );
            Vector<double> symbolicLb( size * size, 0 );
            Vector<double> symbolicUb( size * size, 0 );
            Vector<double> symbolicLowerBias( size, 0 );
            Vector<double> symbolicUpperBias( size, 0 );
            for ( unsigned i = 0; i < size; ++i )
            {
                targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
                targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
            }
            for ( unsigned i = 0; i < size; ++i )
            {
                symbolicLowerBias[i] =
                    NLR::Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, i ); // Using lse2
                symbolicUpperBias[i] =
                    NLR::Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, i );
                for ( unsigned j = 0; j < size; ++j )
                {
                    symbolicLb[size * j + i] =
                        NLR::Layer::dLSELowerBound2( sourceMids, sourceLbs, sourceUbs, i, j );
                    symbolicUb[size * j + i] =
                        NLR::Layer::dLSEUpperbound( sourceMids, targetLbs, targetUbs, i, j );
                    symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                    symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
                }
            }
            TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( symbolicLb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT( compareVectors( symbolicUb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicLowerBias, Vector<double>( { 0.4243, 0.4481, 0.1277 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicUpperBias, Vector<double>( { 0.4243, 0.4480, 0.1277 } ) ) );

            /*
                Layer 2:

0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
               x6.lb = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6.ub = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6 range: [ 0.2595, 0.2595 ]

-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4480 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
               x7.lb = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6062
               x7.ub = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6063
               x7 range: [ 0.7054, 0.7054 ]

-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277
               x8.lb = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8.ub = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8 range: [ 0.0351, 0.0351 ]

                Layer 3:

                x9 = x6 + x7 + x8
                => x9 = ( 0.1922 - 0.1830 - 0.0091 ) x3 + ( -0.1830 + 0.2078 - 0.0248 ) x4 + (
               -0.0091 - 0.0248 + 0.0339 ) x5 + ( 0.4243 + 0.4481 + 0.1277 )

                => x9 = 0.0001 x3 + 0 x4 + 0 x5 + 1.0001
                => ( Up to rounding ) 1 <= x9 <= 1.
                x9.lb = 1
                x9.ub = 1
                x9 range: [ 1, 1 ]

                x10 = - x6 - x7 - x8
                => x10 = - ( 0.1922 - 0.1830 - 0.0091 ) x3 - ( -0.1830 + 0.2078 - 0.0248 ) x4 - (
               -0.0091 - 0.0248 + 0.0339 ) x5 - ( 0.4243 + 0.4481 + 0.1277 )

                => x10 = - 0.0001 x3 - 0.0000 x4 - 0.0000 x5 - 1.0001
                => ( Up to rounding ) 1 <= x10 <= 1.
                x10.lb = 1
                x10.ub = 1
                x10 range: [ -1, -1 ]
            */

            List<Tightening> expectedBounds( { Tightening( 3, 2, Tightening::LB ),
                                               Tightening( 3, 2, Tightening::UB ),
                                               Tightening( 4, 3, Tightening::LB ),
                                               Tightening( 4, 3, Tightening::UB ),
                                               Tightening( 5, 0, Tightening::LB ),
                                               Tightening( 5, 0, Tightening::UB ),
                                               Tightening( 6, 0.2595, Tightening::LB ),
                                               Tightening( 6, 0.2595, Tightening::UB ),
                                               Tightening( 7, 0.7054, Tightening::LB ),
                                               Tightening( 7, 0.7054, Tightening::UB ),
                                               Tightening( 8, 0.0351, Tightening::LB ),
                                               Tightening( 8, 0.0351, Tightening::UB ),
                                               Tightening( 9, 1, Tightening::LB ),
                                               Tightening( 9, 1, Tightening::UB ),
                                               Tightening( 10, -1, Tightening::LB ),
                                               Tightening( 10, -1, Tightening::UB )

            } );

            List<Tightening> bounds;
            TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
            TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

            /*
              Symbolic bounds of every activation layer in terms of predecessor:

              Layer 2 (SOFTMAX):
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277

              Symbolic bounds of output layer in terms of every layer (backsubstitution):

              Layer 3:
              x9 <= x9 <= x9
              x10 <= x10 <= x10

              Layer 2:
              Using x9 = x6 + x7 + x8, x10 = -x6 - x7 - x8:
              x6 + x7 + x8 <= x9 <= x6 + x7 + x8
              -x6 - x7 - x8 <= x10 <= -x6 - x7 - x8

              Layer 1:
              Using
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243.
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481.
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277:
              1 <= x9 <= 1
              -1 <= x10 <= -1

              Layer 0:
              1 <= x9 <= 1
              -1 <= x10 <= -1
            */
            comparePredecessorSymbolicBounds( nlr,
                                              2,
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.4243, 0.4481, 0.1277 } ),
                                              Vector<double>( { 0.4243, 0.4480, 0.1277 } ) );

            compareOutputSymbolicBounds( nlr,
                                         3,
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         2,
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         1,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
            compareOutputSymbolicBounds( nlr,
                                         0,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
        }
        {
            Options::get()->setString( Options::SOFTMAX_BOUND_TYPE, "er" );
            NLR::NetworkLevelReasoner nlr;
            MockTableau tableau;
            nlr.setTableau( &tableau );
            populateNetworkSBTSoftmax( nlr, tableau );

            tableau.setLowerBound( 0, 1 );
            tableau.setUpperBound( 0, 1.000001 );
            tableau.setLowerBound( 1, 1 );
            tableau.setUpperBound( 1, 1.000001 );
            tableau.setLowerBound( 2, 1 );
            tableau.setUpperBound( 2, 1.000001 );

            unsigned paramCount = nlr.getNumberOfParameters();
            Vector<double> coeffs( paramCount, 0.5 );

            // Invoke parameterised initializeSymbolicBoundsMaps
            TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
            TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

            /*
              Input ranges:

              x0: [1, 1.0001]
              x1: [1, 1.0001]
              x2: [1, 1.0001]

              Layer 1:

              x3 = x0 - x1 + x2 + 1
              x3.lb = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3.ub = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3 range: [ 1.999999, 2.000002 ]

              x4 = -x0 + x1 + x2 + 2
              x4.lb = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4.ub = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4 range: [ 2.999999, 3.000002 ]

              x5 = -x0 - x1 - x2 + 3
              x5.lb = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5.ub = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5 range: [ -0.000003, 0 ]
            */

            unsigned size = nlr.getLayer( 2 )->getSize();
            Vector<double> sourceLbs = { 1.999899, 2.999899, -0.000003 };
            Vector<double> sourceUbs = { 2.000102, 3.000102, 0.0001 };
            Vector<double> sourceMids = { 2.0000005, 3.0000005, -0.0000015 };
            Vector<double> targetLbs( size, 0 );
            Vector<double> targetUbs( size, 0 );
            Vector<double> symbolicLb( size * size, 0 );
            Vector<double> symbolicUb( size * size, 0 );
            Vector<double> symbolicLowerBias( size, 0 );
            Vector<double> symbolicUpperBias( size, 0 );
            for ( unsigned i = 0; i < size; ++i )
            {
                targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
                targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
            }
            for ( unsigned i = 0; i < size; ++i )
            {
                symbolicLowerBias[i] =
                    NLR::Layer::ERLowerBound( sourceMids, sourceLbs, sourceUbs, i ); // Using er
                symbolicUpperBias[i] =
                    NLR::Layer::ERUpperBound( sourceMids, targetLbs, targetUbs, i );
                for ( unsigned j = 0; j < size; ++j )
                {
                    symbolicLb[size * j + i] =
                        NLR::Layer::dERLowerBound( sourceMids, sourceLbs, sourceUbs, i, j );
                    symbolicUb[size * j + i] =
                        NLR::Layer::dERUpperBound( sourceMids, targetLbs, targetUbs, i, j );
                    symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                    symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
                }
            }
            TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.2595, 0.7054, 0.0351 } ) ) );
            TS_ASSERT( compareVectors( symbolicLb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT( compareVectors( symbolicUb,
                                       Vector<double>( { 0.1922,
                                                         -0.1830,
                                                         -0.0091,
                                                         -0.1830,
                                                         0.2078,
                                                         -0.0248,
                                                         -0.0091,
                                                         -0.0248,
                                                         0.0339 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicLowerBias, Vector<double>( { 0.4243, 0.4481, 0.1277 } ) ) );
            TS_ASSERT(
                compareVectors( symbolicUpperBias, Vector<double>( { 0.4243, 0.4480, 0.1277 } ) ) );

            /*
                Layer 2:

0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
               x6.lb = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6.ub = 0.3843 x0 - 0.3661 x1 + 0.0183 x2 + 0.2232
               x6 range: [ 0.2595, 0.2595 ]

-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4480 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
               x7.lb = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6062
               x7.ub = -0.3660 x0 - 0.4156 x1 + 0.0496 x2 + 0.6063
               x7 range: [ 0.7054, 0.7054 ]

-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277
               x8.lb = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8.ub = -0.0182 x0 - 0.0496 x1 - 0.0678 x2 + 0.1707
               x8 range: [ 0.0351, 0.0351 ]

                Layer 3:

                x9 = x6 + x7 + x8
                => x9 = ( 0.1922 - 0.1830 - 0.0091 ) x3 + ( -0.1830 + 0.2078 - 0.0248 ) x4 + (
               -0.0091 - 0.0248 + 0.0339 ) x5 + ( 0.4243 + 0.4481 + 0.1277 )

                => x9 = 0.0001 x3 + 0 x4 + 0 x5 + 1.0001
                => ( Up to rounding ) 1 <= x9 <= 1.
                x9.lb = 1
                x9.ub = 1
                x9 range: [ 1, 1 ]

                x10 = - x6 - x7 - x8
                => x10 = - ( 0.1922 - 0.1830 - 0.0091 ) x3 - ( -0.1830 + 0.2078 - 0.0248 ) x4 - (
               -0.0091 - 0.0248 + 0.0339 ) x5 - ( 0.4243 + 0.4481 + 0.1277 )

                => x10 = - 0.0001 x3 - 0.0000 x4 - 0.0000 x5 - 1.0001
                => ( Up to rounding ) 1 <= x10 <= 1.
                x10.lb = 1
                x10.ub = 1
                x10 range: [ -1, -1 ]
            */
            List<Tightening> expectedBounds( { Tightening( 3, 2, Tightening::LB ),
                                               Tightening( 3, 2, Tightening::UB ),
                                               Tightening( 4, 3, Tightening::LB ),
                                               Tightening( 4, 3, Tightening::UB ),
                                               Tightening( 5, 0, Tightening::LB ),
                                               Tightening( 5, 0, Tightening::UB ),
                                               Tightening( 6, 0.2595, Tightening::LB ),
                                               Tightening( 6, 0.2595, Tightening::UB ),
                                               Tightening( 7, 0.7054, Tightening::LB ),
                                               Tightening( 7, 0.7054, Tightening::UB ),
                                               Tightening( 8, 0.0351, Tightening::LB ),
                                               Tightening( 8, 0.0351, Tightening::UB ),
                                               Tightening( 9, 1, Tightening::LB ),
                                               Tightening( 9, 1, Tightening::UB ),
                                               Tightening( 10, -1, Tightening::LB ),
                                               Tightening( 10, -1, Tightening::UB )

            } );

            List<Tightening> bounds;
            TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
            TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

            /*
              Symbolic bounds of every activation layer in terms of predecessor:

              Layer 2 (SOFTMAX):
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277

              Symbolic bounds of output layer in terms of every layer (backsubstitution):

              Layer 3:
              x9 <= x9 <= x9
              x10 <= x10 <= x10

              Layer 2:
              Using x9 = x6 + x7 + x8, x10 = -x6 - x7 - x8:
              x6 + x7 + x8 <= x9 <= x6 + x7 + x8
              -x6 - x7 - x8 <= x10 <= -x6 - x7 - x8

              Layer 1:
              Using
0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243 <= x6 <= 0.1922 x3 - 0.1830 x4 - 0.0091 x5 + 0.4243.
-0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481 <= x7 <= -0.1830 x3 + 0.2078 x4 - 0.0248 x5 + 0.4481.
-0.0091 x3 - 0.0248 x4 + 0.0339 x5 + 0.1277 <= x8 <= 0.1922 x3 -0.0248 x4 + 0.0339 x5 + 0.1277:
              1 <= x9 <= 1
              -1 <= x10 <= -1

              Layer 0:
              1 <= x9 <= 1
              -1 <= x10 <= -1
            */
            comparePredecessorSymbolicBounds( nlr,
                                              2,
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.1922,
                                                                -0.1830,
                                                                -0.0091,
                                                                -0.1830,
                                                                0.2078,
                                                                -0.0248,
                                                                -0.0091,
                                                                -0.0248,
                                                                0.0339 } ),
                                              Vector<double>( { 0.4243, 0.4481, 0.1277 } ),
                                              Vector<double>( { 0.4243, 0.4480, 0.1277 } ) );

            compareOutputSymbolicBounds( nlr,
                                         3,
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 1, 0, 0, 1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         2,
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 1, -1, 1, -1, 1, -1 } ),
                                         Vector<double>( { 0, 0 } ),
                                         Vector<double>( { 0, 0 } ) );
            compareOutputSymbolicBounds( nlr,
                                         1,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
            compareOutputSymbolicBounds( nlr,
                                         0,
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 0, 0, 0, 0, 0, 0 } ),
                                         Vector<double>( { 1, -1 } ),
                                         Vector<double>( { 1, -1 } ) );
        }
    }

    void test_parameterised_symbolic_bound_maps_softmax3()
    {
        Options::get()->setString( Options::SYMBOLIC_BOUND_TIGHTENING_TYPE, "sbt" );
        Options::get()->setString( Options::SOFTMAX_BOUND_TYPE, "lse" );

        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTSoftmax2( nlr, tableau );

        tableau.setLowerBound( 0, 1 );
        tableau.setUpperBound( 0, 1.00001 );
        tableau.setLowerBound( 1, 1 );
        tableau.setUpperBound( 1, 1.00001 );
        tableau.setLowerBound( 2, 1 );
        tableau.setUpperBound( 2, 1.00001 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
              Input ranges:

              x0: [1, 1.0001]
              x1: [1, 1.0001]
              x2: [1, 1.0001]

              Layer 1:

              x3 = x0 - x1 + x2 + 1
              x3.lb = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3.ub = x0 - x1 + x2 + 1    : [ 1.999999, 2.000002 ]
              x3 range: [ 1.999999, 2.000002 ]

              x4 = -x0 + x1 + x2 + 2
              x4.lb = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4.ub = -x0 + x1 + x2 + 2    : [ 2.999999, 3.000002 ]
              x4 range: [ 2.999999, 3.000002 ]

              x5 = -x0 - x1 - x2 + 3
              x5.lb = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5.ub = -x0 - x1 - x2 + 3    : [ -0.000003, 0 ]
              x5 range: [ -0.000003, 0 ]

              x6 = -x0 - x1 - x2 + 2
              x6.lb = -x0 - x1 - x2 + 2    : [ -1.000003, -1 ]
              x6.ub = -x0 - x1 - x2 + 2    : [ -1.000003, -1 ]
              x6 range: [ -1.000003, -1 ]

              x7 = -x0 - x1 - x2 + 1
              x7.lb = -x0 - x1 - x2 + 1    : [ -2.000003, -2 ]
              x7.ub = -x0 - x1 - x2 + 1    : [ -2.000003, -2 ]
              x7 range: [ -2.000003, -2 ]
            */

        // First Sigmoid: x8 x10 x12 = softmax( x3, x5, x7 ).
        unsigned size = nlr.getLayer( 2 )->getActivationSources( 0 ).size();
        Vector<double> sourceLbs = { 1.999899, -0.000003, -2.000103 };
        Vector<double> sourceUbs = { 2.000102, 0.0001, -1.999 };
        Vector<double> sourceMids = { 2.0000005, -0.0000015, -2.0000015 };
        Vector<double> targetLbs( size, 0 );
        Vector<double> targetUbs( size, 0 );
        Vector<double> symbolicLb( size * size, 0 );
        Vector<double> symbolicUb( size * size, 0 );
        Vector<double> symbolicLowerBias( size, 0 );
        Vector<double> symbolicUpperBias( size, 0 );
        for ( unsigned i = 0; i < size; ++i )
        {
            targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
            targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
        }
        for ( unsigned i = 0; i < size; ++i )
        {
            symbolicLowerBias[i] =
                NLR::Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, i ); // Using lse2
            symbolicUpperBias[i] = NLR::Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, i );
            for ( unsigned j = 0; j < size; ++j )
            {
                symbolicLb[size * j + i] =
                    NLR::Layer::dLSELowerBound2( sourceMids, sourceLbs, sourceUbs, i, j );
                symbolicUb[size * j + i] =
                    NLR::Layer::dLSEUpperbound( sourceMids, targetLbs, targetUbs, i, j );
                symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
            }
        }
        TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.8668, 0.1173, 0.0159 } ) ) );
        TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.8668, 0.1173, 0.0159 } ) ) );
        TS_ASSERT( compareVectors( symbolicLb,
                                   Vector<double>( { 0.1155,
                                                     -0.1017,
                                                     -0.0138,
                                                     -0.1017,
                                                     0.1035,
                                                     -0.0019,
                                                     -0.0138,
                                                     -0.0019,
                                                     0.0156 } ) ) );
        TS_ASSERT( compareVectors( symbolicUb,
                                   Vector<double>( { 0.1154,
                                                     -0.1017,
                                                     -0.0138,
                                                     -0.1017,
                                                     0.1036,
                                                     -0.0019,
                                                     -0.0138,
                                                     -0.0019,
                                                     0.0156 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicLowerBias, Vector<double>( { 0.6084, 0.3170, 0.0747 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicUpperBias, Vector<double>( { 0.6084, 0.3170, 0.0747 } ) ) );

        // Second Sigmoid: x9 x11 = softmax( x4, x6 ).
        size = nlr.getLayer( 2 )->getActivationSources( 1 ).size();
        sourceLbs = Vector<double>( { 2.999899, -1.000103 } );
        sourceUbs = Vector<double>( { 3.000102, -0.9999 } );
        sourceMids = Vector<double>( { 3.0000005, -1.0000015 } );
        targetLbs = Vector<double>( size, 0 );
        targetUbs = Vector<double>( size, 0 );
        symbolicLb = Vector<double>( size * size, 0 );
        symbolicUb = Vector<double>( size * size, 0 );
        symbolicLowerBias = Vector<double>( size, 0 );
        symbolicUpperBias = Vector<double>( size, 0 );
        for ( unsigned i = 0; i < size; ++i )
        {
            targetLbs[i] = NLR::Layer::linearLowerBound( sourceLbs, sourceUbs, i );
            targetUbs[i] = NLR::Layer::linearUpperBound( sourceLbs, sourceUbs, i );
        }
        for ( unsigned i = 0; i < size; ++i )
        {
            symbolicLowerBias[i] =
                NLR::Layer::LSELowerBound2( sourceMids, sourceLbs, sourceUbs, i ); // Using lse2
            symbolicUpperBias[i] = NLR::Layer::LSEUpperBound( sourceMids, targetLbs, targetUbs, i );
            for ( unsigned j = 0; j < size; ++j )
            {
                symbolicLb[size * j + i] =
                    NLR::Layer::dLSELowerBound2( sourceMids, sourceLbs, sourceUbs, i, j );
                symbolicUb[size * j + i] =
                    NLR::Layer::dLSEUpperbound( sourceMids, targetLbs, targetUbs, i, j );
                symbolicLowerBias[i] -= symbolicLb[size * j + i] * sourceMids[j];
                symbolicUpperBias[i] -= symbolicUb[size * j + i] * sourceMids[j];
            }
        }
        TS_ASSERT( compareVectors( targetLbs, Vector<double>( { 0.9820, 0.0180 } ) ) );
        TS_ASSERT( compareVectors( targetUbs, Vector<double>( { 0.9820, 0.0180 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicLb, Vector<double>( { 0.0177, -0.0177, -0.0177, 0.0177 } ) ) );
        TS_ASSERT(
            compareVectors( symbolicUb, Vector<double>( { 0.0177, -0.0177, -0.0177, 0.0177 } ) ) );
        TS_ASSERT( compareVectors( symbolicLowerBias, Vector<double>( { 0.9114, 0.0886 } ) ) );
        TS_ASSERT( compareVectors( symbolicUpperBias, Vector<double>( { 0.9114, 0.0886 } ) ) );

        /*
            Layer 2:

            First Sigmoid: x8 x10 x12 = softmax( x3, x5, x7 ).
0.1155 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084 <= x8 <= 0.1154 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084
           x8.lb = 0.2310 x0 + 0.0001 x1 + 0.2310 x2 + 0.4051
           x8.ub = 0.2310 x0 + 0.0000 x1 + 0.2310 x2 + 0.4050
           x8 range: [ 0.8668, 0.8668 ]

-0.1017 x3 + 0.1035 x5 - 0.0019 x7 + 0.3170 <= x10 <= -0.1017 x3 + 0.1036 x5 - 0.0019 x7 + 0.3170
           x10.lb = -0.2033 x0 + 0.0001 x1 - 0.2033 x2 + 0.5239
           x10.ub = -0.2033 x0 + 0.0000 x1 - 0.2033 x2 + 0.5241
           x10 range: [ 0.1173, 0.1173 ]

-0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747 <= x12 <= -0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747
           x12.lb = -0.0275 x0 + 0.0001 x1 - 0.0275 x2 + 0.0708
           x12.ub = -0.0275 x0 + 0.0001 x1 - 0.0275 x2 + 0.0708
           x12 range: [ 0.0159, 0.0159 ]

           Second Sigmoid: x9 x11 = softmax( x4, x6 ).
0.0177 x4 - 0.0177 x6 + 0.9114 <= x9 <= 0.0177 x4 - 0.0177 x6 + 0.9114
           x9.lb = 0 x0 + 0.0354 x1 + 0.0354 x2 + 0.9114
           x9.ub = 0 x0 + 0.0354 x1 + 0.0354 x2 + 0.9114
           x9 range: [ 0.9820, 0.0180 ]

-0.0177 x4 + 0.0177 x6 + 0.0886 <= x11 <= -0.0177 x4 + 0.0177 x6 + 0.0886
           x11.lb = 0 x0 - 0.0354 x1 - 0.0354 x2 + 0.0886
           x11.ub = 0 x0 - 0.0354 x1 - 0.0354 x2 + 0.0886
           x11 range: [ 0.9820, 0.0180 ]

            Layer 3:

            x13 = x8 + x10 + x12
            => x13 = ( 0.1155 - 0.1017 - 0.0138 ) x3 + ( -0.1017 + 0.1035 - 0.0019 ) x5
            + ( -0.0138 - 0.0019 + 0.0156 ) x7 + ( 0.6084 + 0.3170 + 0.0747 )

            => x13 = 0 x3 - 0.0001 x5 - 0.0001 x7 + 1.0001
            => ( Up to rounding ) 1 <= x13 <= 1.
            x13.lb = 1
            x13.ub = 1
            x13 range: [ 1, 1 ]

            x14 = - x8 - x10 - x12
            => x14 = - ( 0.1155 - 0.1017 - 0.0138 ) x3 - ( -0.1017 + 0.1035 - 0.0019 ) x5
            - ( -0.0138 - 0.0019 + 0.0156 ) x7 - ( 0.6084 + 0.3170 + 0.0747 )

            => x14 = 0 x3 + 0.0001 x5 + 0.0001 x7 - 1.0001
            => ( Up to rounding ) -1 <= x14 <= -1.
            x14.lb = -1
            x14.ub = -1
            x14 range: [ -1, -1 ]

            x15 = x9 + x11
            => x15 = ( 0.0177 - 0.0177 ) x4 + ( -0.0177 + 0.0177 ) x6 + ( 0.9114 + 0.0886 )

            => x15 = 0 x4 + 0 x6 + 1
            => ( Up to rounding ) 1 <= x15 <= 1.
            x15.lb = 1
            x15.ub = 1
            x15 range: [ 1, 1 ]

            x16 = - x9 - x11
            => x16 = - ( 0.0177 - 0.0177 ) x4 - ( -0.0177 + 0.0177 ) x6 - ( 0.9114 + 0.0886 )

            => x16 = 0 x4 + 0 x6 - 1
            => ( Up to rounding ) -1 <= x16 <= -1.
            x16.lb = -1
            x16.ub = -1
            x16 range: [ -1, -1 ]
        */

        List<Tightening> expectedBounds( {
            Tightening( 3, 2, Tightening::LB ),         Tightening( 3, 2, Tightening::UB ),
            Tightening( 4, 3, Tightening::LB ),         Tightening( 4, 3, Tightening::UB ),
            Tightening( 5, 0, Tightening::LB ),         Tightening( 5, 0, Tightening::UB ),
            Tightening( 6, -1, Tightening::LB ),        Tightening( 6, -1, Tightening::UB ),
            Tightening( 7, -2, Tightening::LB ),        Tightening( 7, -2, Tightening::UB ),
            Tightening( 8, 0.86681, Tightening::LB ),   Tightening( 8, 0.86682, Tightening::UB ),
            Tightening( 9, 0.98201, Tightening::LB ),   Tightening( 9, 0.98201, Tightening::UB ),
            Tightening( 10, 0.11731, Tightening::LB ),  Tightening( 10, 0.11731, Tightening::UB ),
            Tightening( 11, 0.017985, Tightening::LB ), Tightening( 11, 0.017986, Tightening::UB ),
            Tightening( 12, 0.015875, Tightening::LB ), Tightening( 12, 0.015876, Tightening::UB ),
            Tightening( 13, 1, Tightening::LB ),        Tightening( 13, 1, Tightening::UB ),
            Tightening( 14, -1, Tightening::LB ),       Tightening( 14, -1, Tightening::UB ),
            Tightening( 15, 1, Tightening::LB ),        Tightening( 15, 1, Tightening::UB ),
            Tightening( 16, -1, Tightening::LB ),       Tightening( 16, -1, Tightening::UB ),
        } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (SOFTMAX):
0.1155 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084 <= x8 <= 0.1154 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084
0.0177 x4 - 0.0177 x6 + 0.9114 <= x9 <= 0.0177 x4 - 0.0177 x6 + 0.9114
-0.1017 x3 + 0.1035 x5 - 0.0019 x7 + 0.3170 <= x10 <= -0.1017 x3 + 0.1036 x5 - 0.0019 x7 + 0.3170
-0.0177 x4 + 0.0177 x6 + 0.0886 <= x11 <= -0.0177 x4 + 0.0177 x6 + 0.0886
-0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747 <= x12 <= -0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x13 <= x13 <= x13
          x14 <= x14 <= x14
          x15 <= x15 <= x15
          x16 <= x16 <= x16

          Layer 2:
          Using x13 = x8 + x10 + x12, x14 = -x8 - x10 - x12, x15 = x9 + x11, x16 = -x9 - x11:
          x8 + x10 + x12 <= x13 <= x8 + x10 + x12
          -x8 - x10 - x12 <= x14 <= -x8 - x10 - x12
          x9 + x11 <= x15 <= x9 + x11
          -x9 - x11 <= x16 <= -x9 - x11

          Layer 1:
          Using
0.1155 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084 <= x8 <= 0.1154 x3 - 0.1017 x5 - 0.0138 x7 + 0.6084
0.0177 x4 - 0.0177 x6 + 0.9114 <= x9 <= 0.0177 x4 - 0.0177 x6 + 0.9114
-0.1017 x3 + 0.1035 x5 - 0.0019 x7 + 0.3170 <= x10 <= -0.1017 x3 + 0.1036 x5 - 0.0019 x7 + 0.3170
-0.0177 x4 + 0.0177 x6 + 0.0886 <= x11 <= -0.0177 x4 + 0.0177 x6 + 0.0886
-0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747 <= x12 <= -0.0138 x3 - 0.0019 x5 + 0.0156 x7 + 0.0747
          1 <= x13 <= 1
          -1 <= x14 <= -1
          1 <= x15 <= 1
          -1 <= x16 <= -1

          Layer 0:
          1 <= x13 <= 1
          -1 <= x14 <= -1
          1 <= x15 <= 1
          -1 <= x16 <= -1
        */
        comparePredecessorSymbolicBounds(
            nlr,
            2,
            Vector<double>( { 0.1155,  0.0000,  -0.1017, 0.0000,  -0.0138, 0.0000, 0.0177,
                              0.0000,  -0.0177, 0.0000,  -0.1017, 0.0000,  0.1035, 0.0000,
                              -0.0019, 0.0000,  -0.0177, 0.0000,  0.0177,  0.0000, -0.0138,
                              0.0000,  -0.0019, 0.0000,  0.0156 } ),
            Vector<double>( { 0.1155,  0.0000,  -0.1017, 0.0000,  -0.0138, 0.0000, 0.0177,
                              0.0000,  -0.0177, 0.0000,  -0.1017, 0.0000,  0.1035, 0.0000,
                              -0.0019, 0.0000,  -0.0177, 0.0000,  0.0177,  0.0000, -0.0138,
                              0.0000,  -0.0019, 0.0000,  0.0156 } ),
            Vector<double>( { 0.6084, 0.9114, 0.3170, 0.0886, 0.0747 } ),
            Vector<double>( { 0.6084, 0.9114, 0.3170, 0.0886, 0.0747 } ) );

        compareOutputSymbolicBounds(
            nlr,
            3,
            Vector<double>( { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 } ),
            Vector<double>( { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 } ),
            Vector<double>( { 0, 0, 0, 0 } ),
            Vector<double>( { 0, 0, 0, 0 } ) );
        compareOutputSymbolicBounds(
            nlr,
            2,
            Vector<double>( { 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0 } ),
            Vector<double>( { 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0 } ),
            Vector<double>( { 0, 0, 0, 0 } ),
            Vector<double>( { 0, 0, 0, 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( 20, 0 ),
                                     Vector<double>( 20, 0 ),
                                     Vector<double>( { 1, -1, 1, -1 } ),
                                     Vector<double>( { 1, -1, 1, -1 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( 12, 0 ),
                                     Vector<double>( 12, 0 ),
                                     Vector<double>( { 1, -1, 1, -1 } ),
                                     Vector<double>( { 1, -1, 1, -1 } ) );
    }

    void test_parameterised_symbolic_bound_maps_bilinear()
    {
        NLR::NetworkLevelReasoner nlr;
        MockTableau tableau;
        nlr.setTableau( &tableau );
        populateNetworkSBTBilinear( nlr, tableau );

        tableau.setLowerBound( 0, 1 );
        tableau.setUpperBound( 0, 2 );
        tableau.setLowerBound( 1, -2 );
        tableau.setUpperBound( 1, 1 );

        unsigned paramCount = nlr.getNumberOfParameters();
        Vector<double> coeffs( paramCount, 0.5 );

        // Invoke parameterised initializeSymbolicBoundsMaps
        TS_ASSERT_THROWS_NOTHING( nlr.obtainCurrentBounds() );
        TS_ASSERT_THROWS_NOTHING( nlr.initializeSymbolicBoundsMaps( coeffs ) );

        /*
          Input ranges:

          x0: [1, 2]
          x1: [-2, 1]

          Layers 1, 2:

          x2 = x0 - 2x1
          x2.lb = x0 - 2x1   : [-1, 6]
          x2.ub = x0 - 2x1   : [-1, 6]

          x3 = x0 + x1
          x3.lb = x0 + x1   : [-1, 3]
          x3.ub = x0 + x1   : [-1, 3]

          Using custom coefficients with alpha = { 0.5, 0.5 }.
          Coefficients for bilinear layer:
          Lower bound:
              alpha_l = 0.5 x3.lb + ( 1 - 0.5 ) x3.ub = 0.5 * -1 + 0.5 * 3 = 1
              beta_l = 0.5 x2.lb + ( 1 - 0.5 ) x2.ub = 0.5 * -1 + 0.5 * 6 = 2.5
              gamma_l = -0.5 x2.lb x3.lb - ( 1 - 0.5 ) x2.ub x3.ub = -0.5 * -1 * -1 - 0.5 * 6 * 3 =
          -9.5.

          Upper bound:
              alpha_l = 0.5 x3.ub + ( 1 - 0.5 ) x3.lb = 0.5 * -1 + 0.5 * 3 = 1
              beta_l = 0.5 x2.lb + ( 1 - 0.5 ) x2.ub = 0.5 * -1 + 0.5 * 6 = 2.5
              gamma_l = -0.5 x2.lb x3.ub - ( 1 - 0.5 ) x2.ub x3.lb = -0.5 * -1 * 6 - 0.5 * -1 * 3
          = 4.5.

          S = { x2.lb x3.lb, x2.ub x3.lb, x2.lb x3.ub, x2.ub x3.ub } = { 1, -3, -6, 18 }
          -6 <= min S <= x4 <= max S = 18
          x2 + 2.5 x3 - 9.5 <= x4 <= x2 + 2.5 x3 + 4.5
          x4.lb = 1 ( x0 - 2x1 ) + 2.5 ( x0 + x1 ) - 9.5 = 3.5 x0 + 0.5 x1 - 9.5     : [-7, -2]
          x4.ub = 1 ( x0 - 2x1 ) + 2.5 ( x0 + x1 ) + 4.5 = 3.5 x0 + 0.5 x1 + 4.5    : [7, 12]
          x4 range: [-6, 18]

          Layer 3:

          x5 = -x4 : [-18, 6]
          => -x2 - 2.5 x3 - 4.5 <= x4 <= -x2 - 2.5 x3 + 9.5
          x5.lb = -1 ( 3.5 x0 + 0.5 x1 + 4.5 ) = -3.5 x0 - 0.5 x1 - 4.5   : [-12, 0]
          x5.ub = -1 ( 3.5 x0 + 0.5 x1 - 9.5 ) = -3.5 x0 - 0.5 x1 + 9.5   : [2, 7]
          x5 range: [-12, 6]
        */

        List<Tightening> expectedBounds( { Tightening( 2, -1, Tightening::LB ),
                                           Tightening( 2, 6, Tightening::UB ),
                                           Tightening( 3, -1, Tightening::LB ),
                                           Tightening( 3, 3, Tightening::UB ),
                                           Tightening( 4, -6, Tightening::LB ),
                                           Tightening( 4, 18, Tightening::UB ),
                                           Tightening( 5, -12, Tightening::LB ),
                                           Tightening( 5, 6, Tightening::UB ) } );

        List<Tightening> bounds;
        TS_ASSERT_THROWS_NOTHING( nlr.getConstraintTightenings( bounds ) );
        TS_ASSERT( boundsEqual( bounds, expectedBounds ) );

        /*
          Symbolic bounds of every activation layer in terms of predecessor:

          Layer 2 (BILINEAR):
          x2 + 2.5 x3 - 9.5 <= x4 <= x2 + 2.5 x3 + 4.5

          Symbolic bounds of output layer in terms of every layer (backsubstitution):

          Layer 3:
          x4 <= x5 <= x4

          Layer 2:
          Using x5 = -x4:
          -x4 <= x5 <= -x4

          Layer 1:
          Using x2 + 2.5 x3 - 9.5 <= x4 <= x2 + 2.5 x3 + 4.5:
          -x2 - 2.5 x3 - 4.5 <= x5 <= -x2 - 2.5 x3 + 9.5

          Layer 0:
          Using x2 = x0 - 2x1, x3 = x0 + x1:
          -3.5 x0 - 0.5 x1 - 4.5 <= x5 <= -3.5 x0 - 0.5 x1 + 9.5
        */

        comparePredecessorSymbolicBounds( nlr,
                                          2,
                                          Vector<double>( { 1, 2.5 } ),
                                          Vector<double>( { 1, 2.5 } ),
                                          Vector<double>( { -9.5 } ),
                                          Vector<double>( { 4.5 } ) );

        compareOutputSymbolicBounds( nlr,
                                     3,
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     2,
                                     Vector<double>( { -1 } ),
                                     Vector<double>( { -1 } ),
                                     Vector<double>( { 0 } ),
                                     Vector<double>( { 0 } ) );
        compareOutputSymbolicBounds( nlr,
                                     1,
                                     Vector<double>( { -1, -2.5 } ),
                                     Vector<double>( { -1, -2.5 } ),
                                     Vector<double>( { -4.5 } ),
                                     Vector<double>( { 9.5 } ) );
        compareOutputSymbolicBounds( nlr,
                                     0,
                                     Vector<double>( { -3.5, -0.5 } ),
                                     Vector<double>( { -3.5, -0.5 } ),
                                     Vector<double>( { -4.5 } ),
                                     Vector<double>( { 9.5 } ) );
    }

    bool boundsEqual( const List<Tightening> &bounds, const List<Tightening> &expectedBounds )
    {
        if ( bounds.size() != expectedBounds.size() )
            return false;

        bool allFound = true;
        for ( const auto &bound : bounds )
        {
            bool currentFound = false;
            for ( const auto &expectedBound : expectedBounds )
            {
                currentFound |=
                    ( bound._type == expectedBound._type &&
                      bound._variable == expectedBound._variable &&
                      FloatUtils::areEqual( bound._value, expectedBound._value, 0.0001 ) );
            }
            allFound &= currentFound;
        }
        return allFound;
    }

    void updateTableau( MockTableau &tableau, List<Tightening> &tightenings )
    {
        for ( const auto &tightening : tightenings )
        {
            if ( tightening._type == Tightening::LB )
            {
                tableau.setLowerBound( tightening._variable, tightening._value );
            }

            if ( tightening._type == Tightening::UB )
            {
                tableau.setUpperBound( tightening._variable, tightening._value );
            }
        }
    }

    void compareOutputSymbolicBounds( NLR::NetworkLevelReasoner &nlr,
                                      unsigned layerIndex,
                                      const Vector<double> &symbolicLb,
                                      const Vector<double> &symbolicUb,
                                      const Vector<double> &symbolicLowerBias,
                                      const Vector<double> &symbolicUpperBias )
    {
        Vector<double> outputSymbolicLb;
        Vector<double> outputSymbolicUb;
        Vector<double> outputSymbolicLowerBias;
        Vector<double> outputSymbolicUpperBias;
        TS_ASSERT_THROWS_NOTHING( outputSymbolicLb = nlr.getOutputSymbolicLb( layerIndex ) );
        TS_ASSERT_THROWS_NOTHING( outputSymbolicUb = nlr.getOutputSymbolicUb( layerIndex ) );
        TS_ASSERT_THROWS_NOTHING( outputSymbolicLowerBias =
                                      nlr.getOutputSymbolicLowerBias( layerIndex ) );
        TS_ASSERT_THROWS_NOTHING( outputSymbolicUpperBias =
                                      nlr.getOutputSymbolicUpperBias( layerIndex ) );
        TS_ASSERT( compareVectors( outputSymbolicLb, symbolicLb ) );
        TS_ASSERT( compareVectors( outputSymbolicUb, symbolicUb ) );
        TS_ASSERT( compareVectors( outputSymbolicLowerBias, symbolicLowerBias ) );
        TS_ASSERT( compareVectors( outputSymbolicUpperBias, symbolicUpperBias ) );
    }

    void comparePredecessorSymbolicBounds( NLR::NetworkLevelReasoner &nlr,
                                           unsigned layerIndex,
                                           const Vector<double> &symbolicLb,
                                           const Vector<double> &symbolicUb,
                                           const Vector<double> &symbolicLowerBias,
                                           const Vector<double> &symbolicUpperBias )
    {
        Vector<double> predecessorSymbolicLb;
        Vector<double> predecessorSymbolicUb;
        Vector<double> predecessorSymbolicLowerBias;
        Vector<double> predecessorSymbolicUpperBias;
        TS_ASSERT_THROWS_NOTHING( predecessorSymbolicLb =
                                      nlr.getPredecessorSymbolicLb( layerIndex ) );
        TS_ASSERT_THROWS_NOTHING( predecessorSymbolicUb =
                                      nlr.getPredecessorSymbolicUb( layerIndex ) );
        TS_ASSERT_THROWS_NOTHING( predecessorSymbolicLowerBias =
                                      nlr.getPredecessorSymbolicLowerBias( layerIndex ) );
        TS_ASSERT_THROWS_NOTHING( predecessorSymbolicUpperBias =
                                      nlr.getPredecessorSymbolicUpperBias( layerIndex ) );
        TS_ASSERT( compareVectors( predecessorSymbolicLb, symbolicLb ) );
        TS_ASSERT( compareVectors( predecessorSymbolicUb, symbolicUb ) );
        TS_ASSERT( compareVectors( predecessorSymbolicLowerBias, symbolicLowerBias ) );
        TS_ASSERT( compareVectors( predecessorSymbolicUpperBias, symbolicUpperBias ) );
    }

    void comparePMNRScores( NLR::NetworkLevelReasoner &nlr,
                            const Map<NLR::NeuronIndex, double> &neuronScores )
    {
        double PMNRScore;
        for ( const auto &pair : neuronScores )
        {
            TS_ASSERT_THROWS_NOTHING( PMNRScore = nlr.getPMNRScore( pair.first ) );
            TS_ASSERT( FloatUtils::areEqual( PMNRScore, pair.second, 0.0001 ) );
        }
    }

    bool compareVectors( const Vector<double> &vectorA, const Vector<double> &vectorB )
    {
        if ( vectorA.size() != vectorB.size() )
            return false;

        for ( unsigned i = 0; i < vectorA.size(); ++i )
        {
            if ( !FloatUtils::areEqual( vectorA[i], vectorB[i], 0.0001 ) )
                return false;
        }

        return true;
    }
};
