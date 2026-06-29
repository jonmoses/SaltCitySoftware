# Composable annotation pipeline — the artifact dependency DAG.
#
# This Makefile *is* the orchestration (it replaces the old va-train run()).
# Each target builds one named artifact in $(WORK) by invoking one single-purpose
# tool. `make` rebuilds only what is out of date. See the plan for the full design.
#
# Override on the command line, e.g.:
#   make labels WORK=work OBO=data/go-basic.obo
#
# Status: the pure spine (propagate, correct) is wired and tested. fetch, cluster,
# split, embed, vocab, fit, predict, score are added as their tools land.

PY      ?= .venv/bin/python
WORK    ?= work
OBO     ?= data/go-basic.obo
NS      ?= mf

$(WORK):
	mkdir -p $(WORK)

# --- pure spine (implemented) ----------------------------------------------

# labels.tsv: true-path-closed labels from raw annotations.
$(WORK)/labels.tsv: $(WORK)/annotations.tsv $(OBO) | $(WORK)
	$(PY) -m tools.propagate --obo $(OBO) --in $< --out $@

# pred.corrected.tsv: DAG-consistent prediction scores.
$(WORK)/pred.corrected.tsv: $(WORK)/pred.tsv $(OBO) | $(WORK)
	$(PY) -m tools.correct --obo $(OBO) --in $< --out $@

.PHONY: labels correct
labels: $(WORK)/labels.tsv
correct: $(WORK)/pred.corrected.tsv

# --- remaining stages (TODO as tools land) ---------------------------------
# obo        : tools.obo_fetch          -> $(OBO)
# records    : tools.fetch --query ...  -> $(WORK)/records.jsonl, annotations.tsv
# clusters   : tools.cluster            -> $(WORK)/clusters.tsv
# split      : tools.split              -> $(WORK)/split.tsv
# embed      : tools.embed --model ...  -> $(WORK)/embeddings/<model>.npz
# vocab      : tools.vocab --namespace  -> $(WORK)/vocab.$(NS).txt
# fit        : tools.fit --namespace    -> $(WORK)/model.$(NS).pt
# predict    : tools.predict            -> $(WORK)/pred.$(NS).tsv
# score      : tools.score              -> $(WORK)/metrics.json

.PHONY: test clean
test:
	$(PY) -m pytest -q

clean:
	rm -rf $(WORK)
