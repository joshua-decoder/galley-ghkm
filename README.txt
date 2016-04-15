Stanford GHKM Rule Extractor - August 2012
--------------------------------------------

(c) 2009-2012  The Board of Trustees of The Leland Stanford Junior University.
All Rights Reserved. 

Rule extractor written by Michel Galley.
Support code written by Stanford JavaNLP members.
The system requires Java 6 (JDK1.6).

The GHKM rule extractor included in this distribution replicates almost
exactly the one described in (Galley et al.; 2004, 2006). Sentence pairs for
which the two extractors generate different outputs are rare (one such
sentence is shown in samples/costs.*). The new implementation generates rule
extracts but not derivation forests. The original implementation could
generate derivation forests, though (DeNeefe et al., 2007) indirectly suggests
that such derivations (needed to run EM) are not needed to get
state-of-the-art performance.


USAGE

Unix: 
> extract.sh <root> <memory> <joshua_format>

<root>: <root>.{f,ptb,a} should be files containing respectively
	source-language sentences, target-language parsed sentences (PTB format), 
	and word alignments (see ./samples/ for sample data files).
	<memory>: heap size for the JVM
	<joshua_format>: whether to print rules in Joshua format (true|false).
	If false, print rules in xrs format.

Sample usage 1: 
> extract.sh samples/astronauts 1g false

The above command prints rules to stdout in the following format:
xrs rule LHS -> xrs rule RHS ||| features
e.g.,
VP(VBG("coming") PP(IN("from") x0:NP)) -> "来自" x0 ||| 0.33333334 0.5

The two features printed by default are relative frequencies for:
p(rule | root)
p(rule | LHS)

In the example given above, this corresponds to:

p(rule | VP) = 0.33
p(rule | VP(VBG("coming") PP(IN("from") x0:NP))) = 0.6

Sample usage 2: 
> extract.sh samples/astronauts 1g true

In this case, the output is printed in a format that looks like Joshua's:
root ||| source-language yield ||| target-language yield ||| features
e.g.,
[VP] ||| 来自 [NP,1] ||| coming from [NP,1] ||| 0.33333334 0.5

Printing rules in Joshua format discards internal tree annotation.
To get a behavior strictly equivalent to GHKM as implemented in (Galley et
al., 2006), one should create auxiliary non-terminals (not implemented), 
though it is not clear whether this extra step would really be useful.


OPTIONS

The class edu.stanford.nlp.mt.syntax.RuleExtractor supports the following arguments:

 -joshuaFormat (true|false) [default: false]
  Determines whether to print rules in Joshua format.
 -maxLHS (N) [default: 15]
  Maximum number of nodes in LHS tree.
 -maxRHS (N) [default: 10]
  Maximum number of elements in RHS sequence.
 -maxCompositions (N) [default: 0]
  Maximum number of compositions of minimal rules (see (Galley et al., 2006) for 
	an explanation of the difference between minimal and composed rules).
 -startAtLine (N) [default: 0]
  Start extraction at the given line.
 -endAtLine (N) [default: -1]
  End extraction at the given line. 
 -reversedAlignment (true|false) [default: false]
  If RuleExtractor complains about your word alignment, try this.
 -fFilterCorpus (file) [default: nil]
	Option to filter rules on the fly against a specific dev or test file.  If
	the RHS (i.e., source language) side of the rule contains any word that does
	appear in the dev/test file, it is automatically discarded. This way of
	filtering rules is admittedly very crude, and a suffix-array implementation
	would be ideal to extract very large rule sets (one may want to reuse
	Joshua's implementation).
 - extractors <feature_extractor_1>:...:<feature_extractor_N>
  See "Feature API" section.
  
  
Parameters to maxCompositions, maxLHS, and maxRHS can have a dramatic impact on the
amount of memory needed for extraction.


TESTING

to make sure you are getting the same output as we do, please cd to samples and run:

  make sample

astronauts.grammar.joshua and astronauts.grammar.xrs should then be identical to 
astronauts.grammar.joshua.orig and astronauts.grammar.xrs.orig.


FEATURE API

The rule extractor currently only generates two features for each rule, though 
it is relatively easy to add new ones. The first step is to create a new class 
that implements edu.stanford.nlp.mt.syntax.train.FeatureExtractor. The second 
step is to enable its use by adding it to RuleExtractor's command line, e.g., 
if you just created NewExtractor:

> java edu.stanford.nlp.mt.syntax.train.RuleExtractor -extractors edu.stanford.nlp.mt.syntax.train.RelativeFrequencyFeatureExtractor:edu.stanford.nlp.mt.syntax.train.NewExtractor


TODO

This release lacks a decoder and a suffix array implementation. 
Components that could be added in future releases:

- interfacing with Joshua to exploit its support suffix arrays;
- multi-threading;
- additional features.


LICENSE

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

 For more information, bug reports, fixes, contact:
    Michel Galley
    Dept of Computer Science, Gates 2A
    Stanford CA 94305-9020
    USA
    mgalley@cs.stanford.edu


CHANGES

2012-08-09
  Bug fix contributed by Karl Moritz Hermann: 
	a very small subset of rules were missing due to bad hashing.

2010-03-08
	Added support for multi-threading.
	Reduced memory usage.

2010-02-21
	Initial release.


MORE INFORMATION

The details of this software can be found in these papers:

Michel Galley, Mark Hopkins, Kevin Knight, Daniel Marcu: What's in a
translation rule? HLT-NAACL 2004: 273-280.

Michel Galley, Jonathan Graehl, Kevin Knight, Daniel Marcu, Steve
DeNeefe, Wei Wang, Ignacio Thayer: Scalable Inference and Training of
Context-Rich Syntactic Translation Models. ACL 2006.

For more information, look in the included Javadoc, starting with the 
edu.stanford.nlp.mt.syntax.train.RuleExtractor class documentation.

Please send any questions or feedback to mgalley@cs.stanford.edu.
