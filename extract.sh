#!/bin/sh

#################################################
# GHKM rule extractor written by Michel Galley.
# Contact: mgalley@cs.stanford.edu
#################################################

usage() {
  echo "Usage: $0 <root> <memory> <joshua_format>" >&2
  echo "<root>: <root>.{f,ptb,a} identify files containing respectively source-language" >&2 
	echo "        sentences, parsed target-language sentences, and word alignments" >&2
	echo "<memory>: heap size" >&2
  echo "<joshua_format>: whether to print rules in Joshua format (true|false)." >&2
  echo "                 If false, print rules in xrs format." >&2
  echo "Example: $0 samples/astronauts 1g true" >&2
  exit
}

if [ $# -ne 3 ]; then
	usage
fi

ROOT=$1
MEM=$2
JOSHUA=$3
BASEDIR=`dirname $0`
GHKM_OPTS="-fCorpus $ROOT.f -eParsedCorpus $ROOT.ptb -align $ROOT.a -joshuaFormat $JOSHUA"
JVM_OPTS="-Xmx$MEM -Xms$MEM -cp $BASEDIR/ghkm.jar:$BASEDIR/lib/fastutil.jar -XX:+UseCompressedOops"

JAVACMD="java $JVM_OPTS edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor $GHKM_OPTS"

echo -n "Command: " >&2
echo $JAVACMD
echo "-------------------------------" >&2

$JAVACMD
