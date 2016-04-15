package edu.stanford.nlp.parser.lexparser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.MapFactory;
import edu.stanford.nlp.util.MutableDouble;
import edu.stanford.nlp.util.MutableInteger;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.ThreeDimensionalMap;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.TwoDimensionalMap;


import java.io.*;

/**
 * This class is a reimplementation of Berkeley's state splitting
 * grammar.  This work is experimental and still in progress.  There
 * are several extremely important pieces to implement:
 * <ol>
 * <li> this code should use log probabilities throughout instead of 
 *      multiplying tiny numbers
 * <li> time efficiency of the training code is fawful
 * <li> states should be recombined by a user-specified fraction
 * <li> there are better ways to extract parses using this grammar than
 *      the method in ExhaustivePCFGParser
 * <li> we should also implement cascading parsers that let us
 *      shortcircuit low quality parses earlier (which could possibly
 *      benefit non-split parsers as well)
 * <li> when looping, we should short circuit if we go too many loops
 * <li> ought to smooth as per page 436
 * <li> the end symbol should not be split
 * </ol>
 *
 * @author John Bauer
 */
public class SplittingGrammarExtractor {
  Options op;
  /**
   * These objects are created and filled in here.  The caller can get
   * the data from the extractor once it is finished.
   */
  Index<String> stateIndex;
  Index<String> wordIndex;
  Index<String> tagIndex;
  /**
   * This is a list gotten from the list of startSymbols in op.langpack()
   */
  List<String> startSymbols;

  /**
   * A combined list of all the trees in the training set.
   */
  List<Tree> trees = new ArrayList<Tree>();

  /**
   * All of the weights associated with the trees in the training set.
   * In general, this is just the weight of the original treebank.
   * Note that this uses an identity hash map to map from tree pointer
   * to weight.
   */
  Counter<Tree> treeWeights = new ClassicCounter<Tree>(MapFactory.<Tree,MutableDouble>identityHashMapFactory());

  /**
   * How many total weighted trees we have
   */
  double trainSize;

  /**
   * The original states in the trees
   */
  Set<String> originalStates = new HashSet<String>();

  /**
   * The current number of times a particular state has been split
   */
  IntCounter<String> stateSplitCounts = new IntCounter<String>();

  /**
   * The binary betas are weights to go from Ax to By, Cz.  This maps
   * from (A, B, C) to (x, y, z) to beta(Ax, By, Cz).
   */
  ThreeDimensionalMap<String, String, String, double[][][]> binaryBetas = new ThreeDimensionalMap<String, String, String, double[][][]>();
  /**
   * The unary betas are weights to go from Ax to By.  This maps 
   * from (A, B) to (x, y) to beta(Ax, By).
   */
  TwoDimensionalMap<String, String, double[][]> unaryBetas = new TwoDimensionalMap<String, String, double[][]>();

  /**
   * The latest lexicon we trained.  At the end of the process, this
   * is the lexicon for the parser.
   */
  Lexicon lex;

  transient Index<String> tempWordIndex;
  transient Index<String> tempTagIndex;
  
  /**
   * The lexicon we are in the process of building in each iteration.
   */
  transient Lexicon tempLex;

  /**
   * The latest pair of unary and binary grammars we trained.
   */
  Pair<UnaryGrammar, BinaryGrammar> bgug;

  Random random = new Random(87543875943265L);

  static final double LEX_SMOOTH = 0.01;
  static final double STATE_SMOOTH = 0.01;

  public SplittingGrammarExtractor(Options op) {
    this.op = op;
    startSymbols = Arrays.asList(op.langpack().startSymbols());
  }

  public void outputBetas() {
    System.out.println("UNARY:");
    for (String parent : unaryBetas.firstKeySet()) { 
      for (String child : unaryBetas.get(parent).keySet()) {
        System.out.println("  " + parent + "->" + child);
        double[][] betas = unaryBetas.get(parent).get(child);
        int parentStates = betas.length;
        int childStates = betas[0].length;
        for (int i = 0; i < parentStates; ++i) {
          for (int j = 0; j < childStates; ++j) {
            System.out.println("    " + i + "->" + j + " " + betas[i][j]);
          }
        }
      }
    }
    System.out.println("BINARY:");
    for (String parent : binaryBetas.firstKeySet()) { 
      for (String left : binaryBetas.get(parent).firstKeySet()) {
        for (String right : binaryBetas.get(parent).get(left).keySet()) {
          System.out.println("  " + parent + "->" + left + "," + right);
          double[][][] betas = binaryBetas.get(parent).get(left).get(right);
          int parentStates = betas.length;
          int leftStates = betas[0].length;
          int rightStates = betas[0][0].length;
          for (int i = 0; i < parentStates; ++i) {
            for (int j = 0; j < leftStates; ++j) {
              for (int k = 0; k < rightStates; ++k) {
                System.out.println("    " + i + "->" + j + "," + k + " " + betas[i][j][k]);
              }
            }
          }
        }
      }
    }
  }

  public String state(String tag, int i) {
    if (startSymbols.contains(tag)) {
      return tag;
    }
    return tag + "^" + i;
  }

  public int getStateSplitCount(Tree tree) {
    return stateSplitCounts.getIntCount(tree.label().value());
  }

  public int getStateSplitCount(String label) {
    return stateSplitCounts.getIntCount(label);
  }


  /**
   * Count all the internal labels in all the trees, and set their
   * initial state counts to 1.
   */
  public void countOriginalStates() {
    originalStates.clear();
    for (Tree tree : trees) {
      countOriginalStates(tree);
    }

    for (String state : originalStates) {
      stateSplitCounts.incrementCount(state, 1);
    }
  }

  /**
   * Counts the labels in the tree, but not the words themselves.
   */
  private void countOriginalStates(Tree tree) {
    if (tree.isLeaf()) {
      return;
    }

    originalStates.add(tree.label().value());
    for (Tree child : tree.children()) {
      if (child.isLeaf())
        continue;
      countOriginalStates(child);
    }
  }

  private void initialBetasAndLexicon() {
    wordIndex = new HashIndex<String>();
    tagIndex = new HashIndex<String>();
    lex = op.tlpParams.lex(op, wordIndex, tagIndex);
    lex.initializeTraining(trainSize);

    for (Tree tree : trees) {
      double weight = treeWeights.getCount(tree);
      lex.incrementTreesRead(weight);
      initialBetasAndLexicon(tree, 0, weight);
    }

    lex.finishTraining();
  }

  private int initialBetasAndLexicon(Tree tree, int position, double weight) {
    if (tree.isLeaf()) {
      // should never get here, unless a training tree is just one leaf
      return position;
    }

    if (tree.isPreTerminal()) {
      // fill in initial lexicon here
      String tag = tree.label().value();
      String word = tree.children()[0].label().value();
      TaggedWord tw = new TaggedWord(word, state(tag, 0));
      lex.train(tw, position, weight);
      return (position + 1);
    } 

    if (tree.children().length == 2) {
      String label = tree.label().value();
      String leftLabel = tree.getChild(0).label().value();
      String rightLabel = tree.getChild(1).label().value();
      if (!binaryBetas.contains(label, leftLabel, rightLabel)) {
        double[][][] map = new double[1][1][1];
        map[0][0][0] = 1.0;
        binaryBetas.put(label, leftLabel, rightLabel, map);
      }
    } else if (tree.children().length == 1) {
      String label = tree.label().value();
      String childLabel = tree.getChild(0).label().value();
      if (!unaryBetas.contains(label, childLabel)) {
        double[][] map = new double[1][1];
        map[0][0] = 1.0;
        unaryBetas.put(label, childLabel, map);
      }
    } else {
      // should have been binarized
      throw new RuntimeException("Trees should have been binarized, expected 1 or 2 children");
    }

    for (Tree child : tree.children()) {
      position = initialBetasAndLexicon(child, position, weight);
    }
    return position;
  }


  /**
   * Given a tree and a map from subtree to its transitions, this
   * method splits each of the transitions in two.  This is called to
   * get the transition probabilities for trees on the first iteration
   * of a new split state.
   */
  public void splitTransitions(Tree tree,
                               IdentityHashMap<Tree, double[][]> oldUnaryTransitions,
                               IdentityHashMap<Tree, double[][][]> oldBinaryTransitions,
                               IdentityHashMap<Tree, double[][]> newUnaryTransitions,
                               IdentityHashMap<Tree, double[][][]> newBinaryTransitions) {
    splitTransitions(tree, false, oldUnaryTransitions, oldBinaryTransitions, newUnaryTransitions, newBinaryTransitions);
  }

  /**
   * Given a tree and a map from subtree to its transitions, this
   * method splits each of the transitions in two.  This is called to
   * get the transition probabilities for trees on the first iteration
   * of a new split state.
   * <br>
   * The splitParent parameter controls whether or not to treat the
   * parent as a state that needs to be split.  This is only false for
   * the root node, which should be a root production that does not
   * need to be split.
   */
  public void splitTransitions(Tree tree, boolean splitParent,
                               IdentityHashMap<Tree, double[][]> oldUnaryTransitions,
                               IdentityHashMap<Tree, double[][][]> oldBinaryTransitions,
                               IdentityHashMap<Tree, double[][]> newUnaryTransitions,
                               IdentityHashMap<Tree, double[][][]> newBinaryTransitions) {
    if (tree.isLeaf() || tree.isPreTerminal()) {
      return;
    }

    if (tree.children().length == 2) {
      double[][][] oldMap = oldBinaryTransitions.get(tree);
      double[][][] newMap;

      int parentSplits = getStateSplitCount(tree);
      int leftSplits = getStateSplitCount(tree.children()[0]);
      int rightSplits = getStateSplitCount(tree.children()[1]);
      if (splitParent) {
        newMap = new double[parentSplits][leftSplits / 2][rightSplits / 2];
        for (int i = 0; i < parentSplits / 2; ++i) {
          for (int j = 0; j < leftSplits / 2; ++j) {
            for (int k = 0; k < rightSplits / 2; ++k) {
              newMap[i * 2][j][k] = oldMap[i][j][k];
              newMap[i * 2 + 1][j][k] = oldMap[i][j][k];
            }
          }
        }
        oldMap = newMap;
      }
      newMap = new double[parentSplits][leftSplits][rightSplits];
      for (int i = 0; i < parentSplits; ++i) {
        for (int j = 0; j < leftSplits / 2; ++j) {
          double leftWeight = 0.45 + random.nextDouble() * 0.1;
          for (int k = 0; k < rightSplits / 2; ++k) {
            double rightWeight = 0.45 + random.nextDouble() * 0.1;
            double oldWeight = oldMap[i][j][k];
            newMap[i][j * 2][k * 2] = oldWeight * leftWeight * rightWeight;
            newMap[i][j * 2][k * 2 + 1] = oldWeight * leftWeight * (1.0 - rightWeight);
            newMap[i][j * 2 + 1][k * 2] = oldWeight * (1.0 - leftWeight) * rightWeight;
            newMap[i][j * 2 + 1][k * 2 + 1] = oldWeight * (1.0 - leftWeight) * (1.0 - rightWeight);
          }
        }
      }
      newBinaryTransitions.put(tree, newMap);
    } else { // length == 1
      double[][] oldMap = oldUnaryTransitions.get(tree);
      double[][] newMap;

      int parentSplits = getStateSplitCount(tree);
      int childSplits = getStateSplitCount(tree.children()[0]);
      if (splitParent) {
        newMap = new double[parentSplits][childSplits / 2];
        for (int i = 0; i < parentSplits / 2; ++i) {
          for (int j = 0; j < childSplits / 2; ++j) {
            newMap[i * 2][j] = oldMap[i][j];
            newMap[i * 2 + 1][j] = oldMap[i][j];
          }
        }
        oldMap = newMap;
      }
      newMap = new double[parentSplits][childSplits];
      for (int i = 0; i < parentSplits; ++i) {
        for (int j = 0; j < childSplits / 2; ++j) {
          double childWeight = 0.45 + random.nextDouble() * 0.1;
          double oldWeight = oldMap[i][j];
          newMap[i][j * 2] = oldWeight * childWeight;
          newMap[i][j * 2 + 1] = oldWeight * (1.0 - childWeight);
        }
      }
      newUnaryTransitions.put(tree, newMap);
    }

    for (Tree child : tree.children()) {
      splitTransitions(child, true, oldUnaryTransitions, oldBinaryTransitions, newUnaryTransitions, newBinaryTransitions);
    }
  }

  /**
   * Splits the state counts.  Root states do not get their counts
   * increased, and all others are doubled.  Betas and transition
   * weights are handled later.
   */
  private void splitStateCounts() {
    // double the count of states...
    IntCounter<String> newStateSplitCounts = new IntCounter<String>();
    newStateSplitCounts.addAll(stateSplitCounts);
    newStateSplitCounts.addAll(stateSplitCounts);

    // root states should only have 1
    for (String root : startSymbols) {
      newStateSplitCounts.setCount(root, 1);
    }

    // TODO: terminal .$. states? end symbol should not be split

    stateSplitCounts = newStateSplitCounts;
  }
  

  static final double EPSILON = 0.0001;


  /**
   * Recalculates the betas for all known transitions.  The current
   * betas are used to produce probabilities, which then are used to
   * compute new betas.  If splitStates is true, then the
   * probabilities produced are as if the states were split again from
   * the last time betas were calculated.
   * <br>
   * The return value is whether or not the betas have mostly
   * converged from the last time this method was called.  Obviously
   * if splitStates was true, the betas will be entirely different, so
   * this is false.  Otherwise, the new betas are compared against the
   * old values, and convergence means they differ by less than
   * EPSILON.
   */
  public boolean recalculateBetas(boolean splitStates) {
    TwoDimensionalMap<String, String, double[][]> tempUnaryBetas = new TwoDimensionalMap<String, String, double[][]>();
    ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas = new ThreeDimensionalMap<String, String, String, double[][][]>();

    recalculateTemporaryBetas(splitStates, null, tempUnaryBetas, tempBinaryBetas);
    return useNewBetas(!splitStates, tempUnaryBetas, tempBinaryBetas);
  }

  public boolean useNewBetas(boolean testConverged,
                             TwoDimensionalMap<String, String, double[][]> tempUnaryBetas,
                             ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas) {
    rescaleTemporaryBetas(tempUnaryBetas, tempBinaryBetas);

    // if we just split states, we have obviously not converged
    boolean converged = testConverged && testConvergence(tempUnaryBetas, tempBinaryBetas);

    unaryBetas = tempUnaryBetas;
    binaryBetas = tempBinaryBetas;

    wordIndex = tempWordIndex;
    tagIndex = tempTagIndex;
    lex = tempLex;
    tempWordIndex = null;
    tempTagIndex = null;
    tempLex = null;

    return converged;
  }

  /**
   * Creates temporary beta data structures and fills them in by
   * iterating over the trees.
   */
  public void recalculateTemporaryBetas(boolean splitStates, Map<String, double[]> totalStateMass,
                                        TwoDimensionalMap<String, String, double[][]> tempUnaryBetas,
                                        ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas) {
    tempWordIndex = new HashIndex<String>();
    tempTagIndex = new HashIndex<String>();
    tempLex = op.tlpParams.lex(op, tempWordIndex, tempTagIndex);
    tempLex.initializeTraining(trainSize);

    for (Tree tree : trees) {
      double weight = treeWeights.getCount(tree);
      tempLex.incrementTreesRead(weight);
      recalculateTemporaryBetas(tree, splitStates, totalStateMass, tempUnaryBetas, tempBinaryBetas);
    }

    tempLex.finishTraining();
  }

  public boolean testConvergence(TwoDimensionalMap<String, String, double[][]> tempUnaryBetas,
                                 ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas) {

    // now, we check each of the new betas to see if it's close to the
    // old value for the same transition.  if not, we have not yet
    // converged.  if all of them are, we have converged.
    for (String parentLabel : unaryBetas.firstKeySet()) {
      for (String childLabel : unaryBetas.get(parentLabel).keySet()) {
        double[][] betas = unaryBetas.get(parentLabel, childLabel);
        double[][] newBetas = tempUnaryBetas.get(parentLabel, childLabel);
        int parentStates = betas.length;
        int childStates = betas[0].length;
        for (int i = 0; i < parentStates; ++i) {
          for (int j = 0; j < childStates; ++j) {
            double oldValue = betas[i][j];
            double newValue = newBetas[i][j];
            if (Math.abs(newValue - oldValue) > EPSILON) {
              return false;
            }
          }
        }
      }
    }
    for (String parentLabel : binaryBetas.firstKeySet()) {
      for (String leftLabel : binaryBetas.get(parentLabel).firstKeySet()) {
        for (String rightLabel : binaryBetas.get(parentLabel).get(leftLabel).keySet()) {
          double[][][] betas = binaryBetas.get(parentLabel, leftLabel, rightLabel);
          double[][][] newBetas = tempBinaryBetas.get(parentLabel, leftLabel, rightLabel);
          int parentStates = betas.length;
          int leftStates = betas[0].length;
          int rightStates = betas[0][0].length;
          for (int i = 0; i < parentStates; ++i) {
            for (int j = 0; j < leftStates; ++j) {
              for (int k = 0; k < rightStates; ++k) {
                double oldValue = betas[i][j][k];
                double newValue = newBetas[i][j][k];
                if (Math.abs(newValue - oldValue) > EPSILON) {
                  return false;
                }
              }
            }
          }
        }
      }
    }

    return true;
  }

  public void recalculateTemporaryBetas(Tree tree, boolean splitStates,
                                        Map<String, double[]> totalStateMass,
                                        TwoDimensionalMap<String, String, double[][]> tempUnaryBetas,
                                        ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas) {
    double[] stateWeights = { treeWeights.getCount(tree) };

    IdentityHashMap<Tree, double[][]> unaryTransitions = new IdentityHashMap<Tree, double[][]>();
    IdentityHashMap<Tree, double[][][]> binaryTransitions = new IdentityHashMap<Tree, double[][][]>();
    recountTree(tree, unaryTransitions, binaryTransitions);

    if (splitStates) {
      IdentityHashMap<Tree, double[][]> oldUnaryTransitions = unaryTransitions;
      IdentityHashMap<Tree, double[][][]> oldBinaryTransitions = binaryTransitions;
      unaryTransitions = new IdentityHashMap<Tree, double[][]>();
      binaryTransitions = new IdentityHashMap<Tree, double[][][]>();
      splitTransitions(tree, oldUnaryTransitions, oldBinaryTransitions, unaryTransitions, binaryTransitions);
    }

    recalculateTemporaryBetas(tree, stateWeights, 0, unaryTransitions, binaryTransitions, 
                              totalStateMass, tempUnaryBetas, tempBinaryBetas);
  }

  public int recalculateTemporaryBetas(Tree tree, double[] stateWeights, int position,
                                       IdentityHashMap<Tree, double[][]> unaryTransitions,
                                       IdentityHashMap<Tree, double[][][]> binaryTransitions,
                                       Map<String, double[]> totalStateMass,
                                       TwoDimensionalMap<String, String, double[][]> tempUnaryBetas,
                                       ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas) {
    if (tree.isLeaf()) {
      // possible to get here if we have a tree with no structure
      return position;
    }

    if (totalStateMass != null) {
      double[] stateTotal = totalStateMass.get(tree.label().value());
      if (stateTotal == null) {
        stateTotal = new double[stateWeights.length];
        totalStateMass.put(tree.label().value(), stateTotal);
      }
      for (int i = 0; i < stateWeights.length; ++i) {
        stateTotal[i] += stateWeights[i];
      }
    }

    if (tree.isPreTerminal()) {
      // fill in our new lexicon here
      String tag = tree.label().value();
      String word = tree.children()[0].label().value();
      for (int state = 0; state < stateWeights.length; ++state) {
        TaggedWord tw = new TaggedWord(word, state(tag, state));
        tempLex.train(tw, position, stateWeights[state]);
      }
      return position + 1;
    }

    if (tree.children().length == 1) {
      String parentLabel = tree.label().value();
      String childLabel = tree.children()[0].label().value();
      double[][] transitions = unaryTransitions.get(tree);
      int parentStates = transitions.length;
      int childStates = transitions[0].length;
      double[][] betas = tempUnaryBetas.get(parentLabel, childLabel);
      if (betas == null) {
        betas = new double[parentStates][childStates];
        // for (int i = 0; i < parentStates; ++i) {
        //   for (int j = 0; j < childStates; ++j) {
        //     betas[i][j] = 0.0;
        //   }
        // }
        tempUnaryBetas.put(parentLabel, childLabel, betas);
      }
      double[] childWeights = new double[childStates];
      for (int i = 0; i < parentStates; ++i) {
        for (int j = 0; j < childStates; ++j) {
          double weight = transitions[i][j];
          //System.out.println(parentLabel + " -> " + childLabel +
          //                   ": " + i + "," + j + 
          //                   " + " + (weight * stateWeights[i]));
          betas[i][j] += weight * stateWeights[i];
          childWeights[j] += weight * stateWeights[i];
        }
      }
      position = recalculateTemporaryBetas(tree.children()[0], childWeights, position, unaryTransitions, binaryTransitions, totalStateMass, tempUnaryBetas, tempBinaryBetas);
    } else { // length == 2
      String parentLabel = tree.label().value();
      String leftLabel = tree.children()[0].label().value();
      String rightLabel = tree.children()[1].label().value();
      double[][][] transitions = binaryTransitions.get(tree);
      int parentStates = transitions.length;
      int leftStates = transitions[0].length;
      int rightStates = transitions[0][0].length;

      double[][][] betas = tempBinaryBetas.get(parentLabel, leftLabel, rightLabel);
      if (betas == null) {
        betas = new double[parentStates][leftStates][rightStates];
        // for (int i = 0; i < parentStates; ++i) {
        //   for (int j = 0; j < leftStates; ++j) {
        //     for (int k = 0; k < rightStates; ++k) {
        //       betas[i][j][k] = 0.0;
        //     }
        //   }
        // }
        tempBinaryBetas.put(parentLabel, leftLabel, rightLabel, betas);
      }
      double[] leftWeights = new double[leftStates];
      double[] rightWeights = new double[rightStates];
      for (int i = 0; i < parentStates; ++i) {
        for (int j = 0; j < leftStates; ++j) {
          for (int k = 0; k < rightStates; ++k) {
            double weight = transitions[i][j][k];
            //System.out.println(parentLabel + " -> " + leftLabel + "," +
            //                   rightLabel + ": " + i + "," + j + "," + 
            //                   k + " + " + weight * stateWeights[i]);
            betas[i][j][k] += weight * stateWeights[i];
            leftWeights[j] += weight * stateWeights[i];
            rightWeights[k] += weight * stateWeights[i];
          }
        }
      }
      position = recalculateTemporaryBetas(tree.children()[0], leftWeights, position, unaryTransitions, binaryTransitions, totalStateMass, tempUnaryBetas, tempBinaryBetas);
      position = recalculateTemporaryBetas(tree.children()[1], rightWeights, position, unaryTransitions, binaryTransitions, totalStateMass, tempUnaryBetas, tempBinaryBetas);
    }
    return position;
  }

  public void rescaleTemporaryBetas(TwoDimensionalMap<String, String, double[][]> tempUnaryBetas,
                                    ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas) {
    for (String parent : tempUnaryBetas.firstKeySet()) {
      for (String child : tempUnaryBetas.get(parent).keySet()) {
        double[][] betas = tempUnaryBetas.get(parent).get(child);
        int parentStates = betas.length;
        int childStates = betas[0].length;
        for (int i = 0; i < parentStates; ++i) {
          double sum = 0.0;
          for (int j = 0; j < childStates; ++j) {
            sum += betas[i][j];
          }
          for (int j = 0; j < childStates; ++j) {
            betas[i][j] /= sum;
          }
        }
      }
    }

    for (String parent : tempBinaryBetas.firstKeySet()) {
      for (String left : tempBinaryBetas.get(parent).firstKeySet()) {
        for (String right : tempBinaryBetas.get(parent).get(left).keySet()) {
          double[][][] betas = tempBinaryBetas.get(parent).get(left).get(right);
          int parentStates = betas.length;
          int leftStates = betas[0].length;
          int rightStates = betas[0][0].length;
          for (int i = 0; i < parentStates; ++i) {
            double sum = 0.0;
            for (int j = 0; j < leftStates; ++j) {
              for (int k = 0; k < rightStates; ++k) {
                sum += betas[i][j][k];
              }
            }
            for (int j = 0; j < leftStates; ++j) {
              for (int k = 0; k < rightStates; ++k) {
                betas[i][j][k] /= sum;
              }
            }
          }
        }
      }
    }
  }

  public void recountTree(Tree tree,
                          IdentityHashMap<Tree, double[][]> unaryTransitions,
                          IdentityHashMap<Tree, double[][][]> binaryTransitions) {
    IdentityHashMap<Tree, double[]> probIn = new IdentityHashMap<Tree, double[]>();
    IdentityHashMap<Tree, double[]> probOut = new IdentityHashMap<Tree, double[]>();
    recountTree(tree, probIn, probOut, unaryTransitions, binaryTransitions);
  }

  public void recountTree(Tree tree,
                          IdentityHashMap<Tree, double[]> probIn,
                          IdentityHashMap<Tree, double[]> probOut,
                          IdentityHashMap<Tree, double[][]> unaryTransitions,
                          IdentityHashMap<Tree, double[][][]> binaryTransitions) {
    recountInside(tree, 0, probIn);
    recountOutside(tree, probIn, probOut);
    recountWeights(tree, probIn, probOut, unaryTransitions, binaryTransitions);
  }

  public void recountWeights(Tree tree, 
                             IdentityHashMap<Tree, double[]> probIn,
                             IdentityHashMap<Tree, double[]> probOut,
                             IdentityHashMap<Tree, double[][]> unaryTransitions,
                             IdentityHashMap<Tree, double[][][]> binaryTransitions) {
    if (tree.isLeaf() || tree.isPreTerminal()) {
      return;
    }
    if (tree.children().length == 1) {
      Tree child = tree.children()[0];
      String parentLabel = tree.label().value();
      String childLabel = child.label().value();
      double[][] betas = unaryBetas.get(parentLabel, childLabel);
      double[] childInside = probIn.get(child);
      double[] parentOutside = probOut.get(tree);
      int parentStates = betas.length;
      int childStates = betas[0].length;
      double[][] transitions = new double[parentStates][childStates];
      unaryTransitions.put(tree, transitions);
      for (int i = 0; i < parentStates; ++i) {
        for (int j = 0; j < childStates; ++j) {
          transitions[i][j] = parentOutside[i] * childInside[j] * betas[i][j];
        }
      }
      // renormalize
      for (int i = 0; i < parentStates; ++i) {
        double total = 0.0;
        for (int j = 0; j < childStates; ++j) {
          total += transitions[i][j];
        }
        double scale = 1.0 / (1.0 + STATE_SMOOTH * childStates) / total;
        for (int j = 0; j < childStates; ++j) {
          transitions[i][j] = (transitions[i][j] + total * STATE_SMOOTH) * scale;
        }
      }
      recountWeights(child, probIn, probOut, unaryTransitions, binaryTransitions);
    } else { // length == 2
      Tree left = tree.children()[0];
      Tree right = tree.children()[1];
      String parentLabel = tree.label().value();
      String leftLabel = left.label().value();
      String rightLabel = right.label().value();
      double[][][] betas = binaryBetas.get(parentLabel, leftLabel, rightLabel);
      double[] leftInside = probIn.get(left);
      double[] rightInside = probIn.get(right);
      double[] parentOutside = probOut.get(tree);
      int parentStates = betas.length;
      int leftStates = betas[0].length;
      int rightStates = betas[0][0].length;
      double[][][] transitions = new double[parentStates][leftStates][rightStates];
      binaryTransitions.put(tree, transitions);
      for (int i = 0; i < parentStates; ++i) {
        for (int j = 0; j < leftStates; ++j) {
          for (int k = 0; k < rightStates; ++k) {
            transitions[i][j][k] = parentOutside[i] * leftInside[j] * rightInside[k] * betas[i][j][k];
          }
        }
      }
      // renormalize
      for (int i = 0; i < parentStates; ++i) {
        double total = 0.0;
        for (int j = 0; j < leftStates; ++j) {
          for (int k = 0; k < rightStates; ++k) {
            total += transitions[i][j][k];
          }
        }
        double scale = 1.0 / (1.0 + STATE_SMOOTH * leftStates * rightStates) / total;
        for (int j = 0; j < leftStates; ++j) {
          for (int k = 0; k < rightStates; ++k) {
            transitions[i][j][k] = (transitions[i][j][k] + STATE_SMOOTH * total) * scale;
          }
        }
      }
      recountWeights(left, probIn, probOut, unaryTransitions, binaryTransitions);
      recountWeights(right, probIn, probOut, unaryTransitions, binaryTransitions);
    }
  }

  public void recountOutside(Tree tree, 
                             IdentityHashMap<Tree, double[]> probIn,
                             IdentityHashMap<Tree, double[]> probOut) {
    double[] rootScores = { 1.0 };
    probOut.put(tree, rootScores);
    recurseOutside(tree, probIn, probOut);
  }

  public void recurseOutside(Tree tree,
                             IdentityHashMap<Tree, double[]> probIn,
                             IdentityHashMap<Tree, double[]> probOut) {
    if (tree.isLeaf() || tree.isPreTerminal()) {
      return;
    }
    if (tree.children().length == 1) {
      recountOutside(tree.children()[0], tree, probIn, probOut);
    } else { // length == 2
      recountOutside(tree.children()[0], tree.children()[1], tree, 
                     probIn, probOut);
    }
  }

  public void recountOutside(Tree child, Tree parent,
                             IdentityHashMap<Tree, double[]> probIn,
                             IdentityHashMap<Tree, double[]> probOut) {
    String parentLabel = parent.label().value();
    String childLabel = child.label().value();
    double[] parentScores = probOut.get(parent);
    double[][] betas = unaryBetas.get(parentLabel, childLabel);
    int parentStates = betas.length;
    int childStates = betas[0].length;

    double[] scores = new double[childStates];
    probOut.put(child, scores);

    for (int i = 0; i < parentStates; ++i) {
      for (int j = 0; j < childStates; ++j) {
        // TODO: no inside scores here, right?
        scores[j] += betas[i][j] * parentScores[i];
      }
    }

    recurseOutside(child, probIn, probOut);
  }

  public void recountOutside(Tree left, Tree right, Tree parent,
                             IdentityHashMap<Tree, double[]> probIn,
                             IdentityHashMap<Tree, double[]> probOut) {
    String parentLabel = parent.label().value();
    String leftLabel = left.label().value();
    String rightLabel = right.label().value();
    double[] leftInsideScores = probIn.get(left);
    double[] rightInsideScores = probIn.get(right);
    double[] parentScores = probOut.get(parent);
    double[][][] betas = binaryBetas.get(parentLabel, leftLabel, rightLabel);
    int parentStates = betas.length;
    int leftStates = betas[0].length;
    int rightStates = betas[0][0].length;

    double[] leftScores = new double[leftStates];
    probOut.put(left, leftScores);
    double[] rightScores = new double[rightStates];
    probOut.put(right, rightScores);

    for (int i = 0; i < parentStates; ++i) {
      for (int j = 0; j < leftStates; ++j) {
        for (int k = 0; k < rightStates; ++k) {
          leftScores[j] += betas[i][j][k] * parentScores[i] * rightInsideScores[k];
          rightScores[k] += betas[i][j][k] * parentScores[i] * leftInsideScores[j];
        }
      }
    }

    recurseOutside(left, probIn, probOut);
    recurseOutside(right, probIn, probOut);
  }

  public int recountInside(Tree tree, int loc,
                           IdentityHashMap<Tree, double[]> probIn) {
    if (tree.isLeaf()) {
      throw new RuntimeException();
    } else if (tree.isPreTerminal()) {
      int stateCount = getStateSplitCount(tree);
      String word = tree.children()[0].label().value();
      String tag = tree.label().value();

      double[] scores = new double[stateCount];
      probIn.put(tree, scores);

      for (int i = 0; i < stateCount; ++i) {
        IntTaggedWord tw = new IntTaggedWord(word, state(tag, i),
                                             wordIndex, tagIndex);
        scores[i] += Math.exp(lex.score(tw, loc, word, null));
      }
      loc = loc + 1;
    } else if (tree.children().length == 1) {
      loc = recountInside(tree.children()[0], loc, probIn);
      double[] childScores = probIn.get(tree.children()[0]);
      String parentLabel = tree.label().value();
      String childLabel = tree.children()[0].label().value();
      double[][] betas = unaryBetas.get(parentLabel, childLabel);
      int parentStates = betas.length; // size of the first key
      int childStates = betas[0].length;

      double[] scores = new double[parentStates];
      probIn.put(tree, scores);

      for (int i = 0; i < parentStates; ++i) {
        for (int j = 0; j < childStates; ++j) {
          scores[i] += childScores[j] * betas[i][j];
        }
      }
    } else { // length == 2
      loc = recountInside(tree.children()[0], loc, probIn);
      loc = recountInside(tree.children()[1], loc, probIn);
      double[] leftScores = probIn.get(tree.children()[0]);
      double[] rightScores = probIn.get(tree.children()[1]);
      String parentLabel = tree.label().value();
      String leftLabel = tree.children()[0].label().value();
      String rightLabel = tree.children()[1].label().value();
      double[][][] betas = binaryBetas.get(parentLabel, leftLabel, rightLabel);
      int parentStates = betas.length;
      int leftStates = betas[0].length;
      int rightStates = betas[0][0].length;

      double[] scores = new double[parentStates];
      probIn.put(tree, scores);

      for (int i = 0; i < parentStates; ++i) {
        for (int j = 0; j < leftStates; ++j) {
          for (int k = 0; k < rightStates; ++k) {
            scores[i] += leftScores[j] * rightScores[k] * betas[i][j][k];
          }
        }
      }
    }
    return loc;
  }

  public void mergeStates() {
    if (op.trainOptions.splitRecombineRate <= 0.0) {
      return;
    }

    // we go through the machinery to sum up the temporary betas,
    // counting the total mass
    TwoDimensionalMap<String, String, double[][]> tempUnaryBetas = new TwoDimensionalMap<String, String, double[][]>();
    ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas = new ThreeDimensionalMap<String, String, String, double[][][]>();
    Map<String, double[]> totalStateMass = new HashMap<String, double[]>();
    recalculateTemporaryBetas(false, totalStateMass, tempUnaryBetas, tempBinaryBetas);

    // Next, for each tree we count the effect of merging its
    // annotations.  We only consider the most recently split
    // annotations as candidates for merging.
    Map<String, double[]> deltaAnnotations = new HashMap<String, double[]>();
    for (Tree tree : trees) {
      countMergeEffects(tree, totalStateMass, deltaAnnotations);
    }

    // Now we have a map of the (approximate) likelihood loss from
    // merging each state.  We merge the ones that provide the least
    // benefit, up to the splitRecombineRate
    List<Triple<String, Integer, Double>> sortedDeltas = 
      new ArrayList<Triple<String, Integer, Double>>();
    for (String state : deltaAnnotations.keySet()) {
      double[] scores = deltaAnnotations.get(state);
      for (int i = 0; i < scores.length; ++i) {
        sortedDeltas.add(new Triple<String, Integer, Double>(state, i * 2, scores[i]));
      }
    }
    Collections.sort(sortedDeltas, new Comparator<Triple<String, Integer, Double>>() {
        public int compare(Triple<String, Integer, Double> first,
                           Triple<String, Integer, Double> second) {
          // The most useful splits will have a large loss in
          // likelihood if they are merged.  Thus, we want those at
          // the end of the list.  This means we make the comparison
          // "backwards", sorting from high to low.
          return Double.compare(second.third(), first.third());
        }
        public boolean equals(Object o) { return o == this; }
      });

    // for (Triple<String, Integer, Double> delta : sortedDeltas) {
    //   System.out.println(delta.first() + "-" + delta.second() + ": " + delta.third());
    // }
    // System.out.println("-------------");

    // Only merge a fraction of the splits based on what the user
    // originally asked for
    int splitsToMerge = (int) (sortedDeltas.size() * op.trainOptions.splitRecombineRate);
    splitsToMerge = Math.max(0, splitsToMerge);
    splitsToMerge = Math.min(sortedDeltas.size() - 1, splitsToMerge);
    sortedDeltas = sortedDeltas.subList(0, splitsToMerge);

    System.out.println();
    System.out.println(sortedDeltas);

    Map<String, int[]> mergeCorrespondence = buildMergeCorrespondence(sortedDeltas);

    recalculateMergedBetas(mergeCorrespondence);

    for (Triple<String, Integer, Double> delta : sortedDeltas) {
      stateSplitCounts.decrementCount(delta.first(), 1);
    }
  }

  public void recalculateMergedBetas(Map<String, int[]> mergeCorrespondence) {
    TwoDimensionalMap<String, String, double[][]> tempUnaryBetas = new TwoDimensionalMap<String, String, double[][]>();
    ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas = new ThreeDimensionalMap<String, String, String, double[][][]>();

    tempWordIndex = new HashIndex<String>();
    tempTagIndex = new HashIndex<String>();
    tempLex = op.tlpParams.lex(op, tempWordIndex, tempTagIndex);
    tempLex.initializeTraining(trainSize);

    for (Tree tree : trees) {
      double[] stateWeights = { treeWeights.getCount(tree) };
      tempLex.incrementTreesRead(stateWeights[0]);

      IdentityHashMap<Tree, double[][]> oldUnaryTransitions = new IdentityHashMap<Tree, double[][]>();
      IdentityHashMap<Tree, double[][][]> oldBinaryTransitions = new IdentityHashMap<Tree, double[][][]>();
      recountTree(tree, oldUnaryTransitions, oldBinaryTransitions);

      IdentityHashMap<Tree, double[][]> unaryTransitions = new IdentityHashMap<Tree, double[][]>();
      IdentityHashMap<Tree, double[][][]> binaryTransitions = new IdentityHashMap<Tree, double[][][]>();
      mergeTransitions(tree, oldUnaryTransitions, oldBinaryTransitions, unaryTransitions, binaryTransitions, stateWeights, mergeCorrespondence);

      recalculateTemporaryBetas(tree, stateWeights, 0, unaryTransitions, binaryTransitions, 
                                null, tempUnaryBetas, tempBinaryBetas);
    }

    tempLex.finishTraining();
    useNewBetas(false, tempUnaryBetas, tempBinaryBetas);
  }

  /**
   * Given a tree and the original set of transition probabilities
   * from one state to the next in the tree, along with a list of the
   * weights in the tree and a count of the mass in each substate at
   * the current node, this method merges the probabilities as
   * necessary.  The results go into newUnaryTransitions and
   * newBinaryTransitions.
   */
  public void mergeTransitions(Tree parent, 
                               IdentityHashMap<Tree, double[][]> oldUnaryTransitions,
                               IdentityHashMap<Tree, double[][][]> oldBinaryTransitions,
                               IdentityHashMap<Tree, double[][]> newUnaryTransitions,
                               IdentityHashMap<Tree, double[][][]> newBinaryTransitions,
                               double[] stateWeights, 
                               Map<String, int[]> mergeCorrespondence) {
    if (parent.isPreTerminal() || parent.isLeaf()) {
      return;
    }
    if (parent.children().length == 1) {
      double[][] oldTransitions = oldUnaryTransitions.get(parent);

      String parentLabel = parent.label().value();
      int[] parentCorrespondence = mergeCorrespondence.get(parentLabel);
      int parentStates = parentCorrespondence[parentCorrespondence.length - 1] + 1;

      String childLabel = parent.children()[0].label().value();
      int[] childCorrespondence = mergeCorrespondence.get(childLabel);
      int childStates = childCorrespondence[childCorrespondence.length - 1] + 1;

      // System.out.println("P: " + parentLabel + " " + parentStates + 
      //                    " C: " + childLabel + " " + childStates);


      // Add up the probabilities of transitioning to each state,
      // scaled by the probability of being in a given state to begin
      // with.  This accounts for when two states in the parent are
      // collapsed into one state.
      double[][] newTransitions = new double[parentStates][childStates];
      newUnaryTransitions.put(parent, newTransitions);
      for (int i = 0; i < oldTransitions.length; ++i) {
        int ti = parentCorrespondence[i];
        for (int j = 0; j < oldTransitions[0].length; ++j) {
          int tj = childCorrespondence[j];
          // System.out.println(i + " " + ti + " " + j + " " + tj);
          newTransitions[ti][tj] += oldTransitions[i][j] * stateWeights[i];
        }
      }

      for (int i = 0; i < parentStates; ++i) {
        double total = 0.0;
        for (int j = 0; j < childStates; ++j) {
          total += newTransitions[i][j];
        }
        if (total <= EPSILON) {
          for (int j = 0; j < childStates; ++j) {
            newTransitions[i][j] = 0.0;
          }
        } else {
          for (int j = 0; j < childStates; ++j) {
            newTransitions[i][j] /= total;
          }
        }
      }

      double[] childWeights = new double[oldTransitions[0].length];
      for (int i = 0; i < oldTransitions.length; ++i) {
        for (int j = 0; j < oldTransitions[0].length; ++j) {
          double weight = oldTransitions[i][j];
          childWeights[j] += weight * stateWeights[i];
        }
      }

      mergeTransitions(parent.children()[0], oldUnaryTransitions, oldBinaryTransitions, newUnaryTransitions, newBinaryTransitions, childWeights, mergeCorrespondence);
    } else {
      double[][][] oldTransitions = oldBinaryTransitions.get(parent);

      String parentLabel = parent.label().value();
      int[] parentCorrespondence = mergeCorrespondence.get(parentLabel);
      int parentStates = parentCorrespondence[parentCorrespondence.length - 1] + 1;

      String leftLabel = parent.children()[0].label().value();
      int[] leftCorrespondence = mergeCorrespondence.get(leftLabel);
      int leftStates = leftCorrespondence[leftCorrespondence.length - 1] + 1;

      String rightLabel = parent.children()[1].label().value();
      int[] rightCorrespondence = mergeCorrespondence.get(rightLabel);
      int rightStates = rightCorrespondence[rightCorrespondence.length - 1] + 1;

      // System.out.println("P: " + parentLabel + " " + parentStates + 
      //                    " L: " + leftLabel + " " + leftStates +
      //                    " R: " + rightLabel + " " + rightStates);

      double[][][] newTransitions = new double[parentStates][leftStates][rightStates];
      newBinaryTransitions.put(parent, newTransitions);
      for (int i = 0; i < oldTransitions.length; ++i) {
        int ti = parentCorrespondence[i];
        for (int j = 0; j < oldTransitions[0].length; ++j) {
          int tj = leftCorrespondence[j];
          for (int k = 0; k < oldTransitions[0][0].length; ++k) {
            int tk = rightCorrespondence[k];
            // System.out.println(i + " " + ti + " " + j + " " + tj + " " + k + " " + tk);
            newTransitions[ti][tj][tk] += oldTransitions[i][j][k] * stateWeights[i];
          }
        }
      }


      for (int i = 0; i < parentStates; ++i) {
        double total = 0.0;
        for (int j = 0; j < leftStates; ++j) {
          for (int k = 0; k < rightStates; ++k) {
            total += newTransitions[i][j][k];
          }
        }
        if (total <= EPSILON) {
          for (int j = 0; j < leftStates; ++j) {
            for (int k = 0; k < rightStates; ++k) {
              newTransitions[i][j][k] = 0.0;
            }
          }
        } else {
          for (int j = 0; j < leftStates; ++j) {
            for (int k = 0; k < rightStates; ++k) {
              newTransitions[i][j][k] /= total;
            }
          }
        }
      }

      double[] leftWeights = new double[oldTransitions[0].length];
      double[] rightWeights = new double[oldTransitions[0][0].length];
      for (int i = 0; i < oldTransitions.length; ++i) {
        for (int j = 0; j < oldTransitions[0].length; ++j) {
          for (int k = 0; k < oldTransitions[0][0].length; ++k) {
            double weight = oldTransitions[i][j][k];
            leftWeights[j] += weight * stateWeights[i];
            rightWeights[k] += weight * stateWeights[i];
          }
        }
      }

      mergeTransitions(parent.children()[0], oldUnaryTransitions, oldBinaryTransitions, newUnaryTransitions, newBinaryTransitions, leftWeights, mergeCorrespondence);
      mergeTransitions(parent.children()[1], oldUnaryTransitions, oldBinaryTransitions, newUnaryTransitions, newBinaryTransitions, rightWeights, mergeCorrespondence);
    }
  }

  Map<String, int[]> buildMergeCorrespondence(List<Triple<String, Integer, Double>> deltas) {
    Map<String, int[]> mergeCorrespondence = new HashMap<String, int[]>();
    for (String state : originalStates) {
      int states = getStateSplitCount(state);
      int[] correspondence = new int[states];
      for (int i = 0; i < states; ++i) {
        correspondence[i] = i;
      }
      mergeCorrespondence.put(state, correspondence);
    }
    for (Triple<String, Integer, Double> merge : deltas) {
      int states = getStateSplitCount(merge.first());
      int split = merge.second();
      int[] correspondence = mergeCorrespondence.get(merge.first());
      for (int i = split + 1; i < states; ++i) {
        correspondence[i] = correspondence[i] - 1;
      }
    }
    return mergeCorrespondence;
  }

  public void countMergeEffects(Tree tree, Map<String, double[]> totalStateMass, 
                                Map<String, double[]> deltaAnnotations) {
    IdentityHashMap<Tree, double[]> probIn = new IdentityHashMap<Tree, double[]>();
    IdentityHashMap<Tree, double[]> probOut = new IdentityHashMap<Tree, double[]>();
    IdentityHashMap<Tree, double[][]> unaryTransitions = new IdentityHashMap<Tree, double[][]>();
    IdentityHashMap<Tree, double[][][]> binaryTransitions = new IdentityHashMap<Tree, double[][][]>();
    recountTree(tree, probIn, probOut, unaryTransitions, binaryTransitions);

    // no need to count the root
    for (Tree child : tree.children()) {
      countMergeEffects(child, totalStateMass, deltaAnnotations, probIn, probOut);
    }
  }

  public void countMergeEffects(Tree tree, Map<String, double[]> totalStateMass, 
                                Map<String, double[]> deltaAnnotations,
                                IdentityHashMap<Tree, double[]> probIn,
                                IdentityHashMap<Tree, double[]> probOut) {
    if (tree.isLeaf()) {
      return;
    }

    String label = tree.label().value();
    double totalMass = 0.0;
    double[] stateMass = totalStateMass.get(label);
    for (double mass : stateMass) {
      totalMass += mass;
    }
    
    double[] nodeProbIn = probIn.get(tree);
    double[] nodeProbOut = probOut.get(tree);

    double[] nodeDelta = deltaAnnotations.get(label);
    if (nodeDelta == null) {
      nodeDelta = new double[nodeProbIn.length / 2];
      deltaAnnotations.put(label, nodeDelta);
    }

    for (int i = 0; i < nodeProbIn.length / 2; ++i) {
      double probInMerged = ((stateMass[i * 2] / totalMass) * nodeProbIn[i * 2] +
                             (stateMass[i * 2 + 1] / totalMass) * nodeProbIn[i * 2 + 1]);
      double probOutMerged = nodeProbOut[i * 2] + nodeProbOut[i * 2 + 1];
      double probMerged = probInMerged * probOutMerged;
      double probUnmerged = (nodeProbIn[i * 2] * nodeProbOut[i * 2] + 
                             nodeProbIn[i * 2 + 1] * nodeProbOut[i * 2 + 1]);
      nodeDelta[i] += Math.log(probMerged / probUnmerged);
    }

    if (tree.isPreTerminal()) {
      return;
    }
    for (Tree child : tree.children()) {
      countMergeEffects(child, totalStateMass, deltaAnnotations, probIn, probOut);
    }
  }

  public void buildStateIndex() {
    stateIndex = new HashIndex<String>();
    for (String key : stateSplitCounts.keySet()) {
      for (int i = 0; i < stateSplitCounts.getIntCount(key); ++i) {
        stateIndex.indexOf(state(key, i), true);
      }
    }
  }

  public void buildGrammars() {
    // In order to build the grammars, we first need to fill in the
    // temp betas with the sums of the transitions from Ax to By or Ax
    // to By,Cz.  We also need the sum total of the mass in each state
    // Ax over all the trees.

    // we go through the machinery to sum up the temporary betas,
    // counting the total mass...
    TwoDimensionalMap<String, String, double[][]> tempUnaryBetas = new TwoDimensionalMap<String, String, double[][]>();
    ThreeDimensionalMap<String, String, String, double[][][]> tempBinaryBetas = new ThreeDimensionalMap<String, String, String, double[][][]>();
    Map<String, double[]> totalStateMass = new HashMap<String, double[]>();
    recalculateTemporaryBetas(false, totalStateMass, tempUnaryBetas, tempBinaryBetas);

    // ... but note we don't actually rescale the betas.
    // instead we use the temporary betas and the total mass in each
    // state to calculate the grammars

    // First build up a BinaryGrammar.
    // The score for each rule will be the Beta scores found earlier,
    // scaled by the total weight of a transition between unsplit states
    BinaryGrammar bg = new BinaryGrammar(stateIndex);
    for (String parent : tempBinaryBetas.firstKeySet()) {
      int parentStates = getStateSplitCount(parent);
      double[] stateTotal = totalStateMass.get(parent);
      for (String left : tempBinaryBetas.get(parent).firstKeySet()) {
        int leftStates = getStateSplitCount(left);
        for (String right : tempBinaryBetas.get(parent).get(left).keySet()) {
          int rightStates = getStateSplitCount(right);
          double[][][] betas = tempBinaryBetas.get(parent, left, right);
          for (int i = 0; i < parentStates; ++i) {
            if (stateTotal[i] < EPSILON) {
              continue;
            }
            for (int j = 0; j < leftStates; ++j) {
              for (int k = 0; k < rightStates; ++k) {
                int parentIndex = stateIndex.indexOf(state(parent, i));
                int leftIndex = stateIndex.indexOf(state(left, j));
                int rightIndex = stateIndex.indexOf(state(right, k));
                double score = Math.log(betas[i][j][k] / stateTotal[i]);
                BinaryRule br = new BinaryRule(parentIndex, leftIndex, rightIndex, score);
                bg.addRule(br);
              }
            }
          }
        }
      }
    }

    // Now build up a UnaryGrammar
    UnaryGrammar ug = new UnaryGrammar(stateIndex);
    for (String parent : tempUnaryBetas.firstKeySet()) {
      int parentStates = getStateSplitCount(parent);
      double[] stateTotal = totalStateMass.get(parent);
      for (String child : tempUnaryBetas.get(parent).keySet()) {
        int childStates = getStateSplitCount(child);
        double[][] betas = tempUnaryBetas.get(parent, child);
        for (int i = 0; i < parentStates; ++i) {
          if (stateTotal[i] < EPSILON) {
            continue;
          }
          for (int j = 0; j < childStates; ++j) {
            int parentIndex = stateIndex.indexOf(state(parent, i));
            int childIndex = stateIndex.indexOf(state(child, j));
            double score = Math.log(betas[i][j] / stateTotal[i]);
            UnaryRule ur = new UnaryRule(parentIndex, childIndex, score);
            ug.addRule(ur);
          }
        }
      }
    }


    bgug = new Pair<UnaryGrammar, BinaryGrammar>(ug, bg);
  }

  public void saveTrees(Collection<Tree> trees1, double weight1,
                        Collection<Tree> trees2, double weight2) {
    trainSize = 0.0;
    int treeCount = 0;
    trees.clear();
    treeWeights.clear();
    for (Tree tree : trees1) {
      trees.add(tree);
      treeWeights.incrementCount(tree, weight1);
      trainSize += weight1;
    }
    treeCount += trees1.size();
    if (trees2 != null && weight2 >= 0.0) {
      for (Tree tree : trees2) {
        trees.add(tree);
        treeWeights.incrementCount(tree, weight2);
        trainSize += weight2;
      }
      treeCount += trees2.size();
    }
    System.err.println("Found " + treeCount + 
                       " trees with total weight " + trainSize);
  }

  public void extract(Collection<Tree> treeList) {
    extract(treeList, 1.0, null, 0.0);
  }

  /**
   * First, we do a few setup steps.  We read in all the trees, which
   * is necessary because we continually reprocess them and use the
   * object pointers as hash keys rather than hashing the trees
   * themselves.  We then count the initial states in the treebank.
   * <br>
   * Having done that, we then assign initial probabilities to the
   * trees.  At first, each state has 1.0 of the probability mass for
   * each Ax-ByCz and Ax-By transition.  We then split the number of
   * states and the probabilities on each tree.
   * <br>
   * We then repeatedly recalculate the betas and reannotate the
   * weights, going until we converge, which is defined as no betas
   * move more then epsilon.
   *
   * java -mx4g edu.stanford.nlp.parser.lexparser.LexicalizedParser  -PCFG -saveToSerializedFile englishSplit.ser.gz -saveToTextFile englishSplit.txt -maxLength 40 -train ../data/wsj/wsjtwentytrees.mrg    -testTreebank ../data/wsj/wsjtwentytrees.mrg   -evals "factDA,tsv" -uwm 0  -hMarkov 0 -vMarkov 0 -simpleBinarizedLabels -predictSplits -splitTrainingThreads 1 -splitCount 1 -splitRecombineRate 0.5
   */
  public void extract(Collection<Tree> trees1, double weight1, 
                      Collection<Tree> trees2, double weight2) {
    saveTrees(trees1, weight1, trees2, weight2);

    countOriginalStates();

    // Initial betas will be 1 for all possible unary and binary
    // transitions in our treebank
    initialBetasAndLexicon();

    for (int cycle = 0; cycle < op.trainOptions.splitCount; ++cycle) {
      // All states except the root state get split into 2
      splitStateCounts();

      // first, recalculate the betas and the lexicon for having split
      // the transitions
      recalculateBetas(true);

      // now, loop until we converge while recalculating betas
      // TODO: add a loop counter, stop after X iterations
      int iteration = 0;
      boolean converged = false;
      while (!converged) {
        converged = recalculateBetas(false);
        ++iteration;
      }

      System.err.println("Converged for cycle " + cycle + 
                         " in " + iteration + " iterations");

      mergeStates();
    }

    // Build up the state index.  The BG & UG both expect a set count
    // of states.
    buildStateIndex();

    buildGrammars();
  }  
}
