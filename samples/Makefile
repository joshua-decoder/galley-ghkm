.SECONDARY:

TARGETS=out/astronauts.out.diff out/gunman.out.diff out/livelihood.out.diff out/rescue.out.diff out/costs.out.diff
ATARGETS=out/astronauts.all.diff out/gunman.all.diff out/livelihood.all.diff out/rescue.all.diff 

J=java-mirror
#J=java -cp ../:../lib/fastutil.jar
#J=java -cp ../ghkm.jar:../lib/fastutil.jar
ISI_CMD=../../../xrs-extract/extract.sh

sample: astronauts.grammar.xrs astronauts.grammar.joshua
diff: $(TARGETS) $(ATARGETS)

#############################################
# Rule extraction (with features)
#############################################

%.grammar.xrs: %.f %.ptb %.a
	$J edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor -fCorpus $(*F).f -eParsedCorpus $(*F).ptb -align $(*F).a 2> logs/$@.log > $@

%.grammar.joshua: %.f %.ptb %.a
	$J edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor -fCorpus $(*F).f -eParsedCorpus $(*F).ptb -align $(*F).a -joshuaFormat true 2> logs/$@.log > $@

#############################################
# Comparison with ISI
#############################################

%.diff: %.isi %.su
	diff $+ > $@ || true

#############################################
# Rule extraction (without features)
#############################################

# Stanford system:
out/%.out.su: %.f %.ptb %.a
	$J -DSynCatRHS=false -DDebugGHKM=true -ea edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor -maxLHS 10000 -maxLHS 10000 -fCorpus $(*F).f -eParsedCorpus $(*F).ptb -align $(*F).a -maxCompositions 0 2> logs/$@.log | sed 's/ ||| .*//' | tr 'A-Z' 'a-z' | LC_ALL=C sort -u > $@

out/%.all.su: %.f %.ptb %.a
	$J -Xmx7000m -Xms7000m -DSynCatRHS=false -DDebugGHKM=false -ea edu.stanford.nlp.mt.syntax.ghkm.RuleExtractor -maxLHS 10000 -maxLHS 10000 -fCorpus $(*F).f -eParsedCorpus $(*F).ptb -align $(*F).a -maxCompositions 100 2> logs/$@.log | sed 's/ ||| .*//' | tr 'A-Z' 'a-z' | LC_ALL=C sort -u > $@

# ISI system:
out/%.out.isi: %.f %.e-parse %.a
	$(ISI_CMD) -r $(*F) -O -l 0:0 -x - 2> logs/$@.log | sed -n '4,$$p' | sed 's/ ### .*//' | tr 'A-Z' 'a-z' | LC_ALL=C sort -u > $@

out/%.all.isi: %.f %.e-parse %.a
	$(ISI_CMD) -r $(*F) -O -l 100000:100000 -x - 2> logs/$@.log | sed -n '4,$$p' | sed 's/ ### .*//' | tr 'A-Z' 'a-z' | LC_ALL=C sort -u > $@

#############################################
# Tree transformation
#############################################

%.e-parse: %.ptb
	../../../xrs-extract/scripts/ptb2isi < $< > $@
