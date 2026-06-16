# Walkthrough 1 — For a Newcomer (light CS background, no biology)

This is the gentlest of three walkthroughs. It assumes you can read a little code
and know what a classifier is, but it assumes **nothing** about biology. The other
two go deeper:

- **Walkthrough 2** — for someone who knows biology but not machine learning.
- **Walkthrough 3** — a full, file-by-file engineering deep dive.

---

## 1. The one-sentence version

> We take a virus protein, which is just a string of letters, and we automatically
> attach labels describing **what that protein does** — then we flag the labels that
> correspond to dangerous behavior.

If you've ever used a model that takes a sentence and predicts tags for it (spam/not
spam, topic labels, sentiment), you already understand the shape of this project.
Swap "sentence" for "protein" and "topic tags" for "biological function tags," and
you're 80% of the way there.

## 2. The vocabulary you actually need

Just four ideas. Everything else is built from these.

**A protein is a string.** Living things build molecular machines called proteins.
Each protein is written as a sequence of *amino acids*, and there are only 20 of
them, so a protein is literally a string over a 20-letter alphabet, e.g.
`MKTAYIAKQR...`. A typical one is a few hundred letters long. That string is the only
input we get.

**A "function label" is a tag from a fixed dictionary.** Biologists maintain a giant,
standardized dictionary of things a protein can *do* or *be involved in*, called the
**Gene Ontology** (GO). Each entry has an ID like `GO:0046718` and a human name like
"viral entry into host cell." Think of GO IDs as tags from a controlled vocabulary —
you can't make up your own; you pick from the dictionary.

**That dictionary is a graph, not a flat list.** The tags are arranged from general
to specific. "viral entry into host cell" is a *child* of the more general
"interaction with host." If a protein has a specific tag, it logically also has all
the more-general tags above it. This is a **DAG** (a directed acyclic graph — a tree
where a node is allowed to have more than one parent). This single fact drives a
surprising amount of the design, so hold onto it.

**There are three separate sub-dictionaries.** GO is split into three independent
namespaces, and we treat them as three separate labeling problems:
- **Molecular Function (MF)** — what the protein does mechanically ("binds RNA").
- **Biological Process (BP)** — the larger process it participates in ("viral entry").
- **Cellular Component (CC)** — where in the cell it operates ("host cell membrane").

So the whole task is: **given a protein string, predict which tags from this graph-
structured dictionary apply.** It's multi-label classification (a protein has many
tags at once), with a twist that the labels have a hierarchy.

## 3. The pipeline, end to end

Here is the entire system as a flow. Each arrow is a stage we'll unpack.

```
virus protein string
   │
   ▼  (1) ESM-2: turn the string into numbers          embeddings/esm.py
per-letter vectors  ──▶  one vector per protein
   │
   ▼  (2) classifier: vector → a score per tag         classifier/model.py
a probability for each GO tag
   │
   ▼  (3) hierarchy fix-up: make scores obey the DAG    ontology/go_dag.py
consistent GO annotations
   │
   ▼  (4) threat mapping: which tags are "dangerous"?   threat.py
a danger profile for the virus
```

### Stage 1 — Turn the string into numbers (embeddings)

A classifier can't consume raw letters; it needs numbers. The trick the whole field
relies on is a **protein language model**. It's the exact same idea as the language
models you know (BERT, GPT): a big neural network was pre-trained on hundreds of
millions of protein strings, learning the statistical "grammar" of proteins. We use
one called **ESM-2**.

We don't train it. We use it frozen, as a **feature extractor** — exactly like
grabbing the second-to-last layer of a pretrained image model to get a useful vector
for a new task. You feed it the protein string; it hands back one vector (a list of
~1280 numbers) for *each letter* in the sequence. These vectors are called
**embeddings**, and proteins that behave similarly end up with similar embeddings,
even if their letters differ.

One catch: a classifier wants **one** vector per protein, not one per letter. So we
**pool** the per-letter vectors into a single vector. The simplest pooling is just
the average ("mean pooling") — and that's the default. (Walkthroughs 2 and 3 explain
a smarter, learned pooling we use for one of the three sub-dictionaries.)

### Stage 2 — Vector to tag scores (the classifier)

Now it's an ordinary machine-learning problem: vector in, tags out. The model
(`classifier/model.py`) is deliberately tiny — basically a logistic regression with
one output per tag. Each output passes through a **sigmoid**, which squashes it to a
probability between 0 and 1. Crucially we use a sigmoid *per tag*, not a softmax over
all tags, because the tags aren't mutually exclusive — a protein can be "binds RNA"
**and** "enters host cell" at the same time. Each tag is an independent yes/no
question.

Where do the right answers come from to train this? From a public database called
**UniProt**, which stores known proteins along with GO tags that human curators (or
automated tools) have already attached. We download the virus proteins plus their
known tags (`data/labels.py`), and that's our labeled training set — the same way
you'd download a labeled dataset for any supervised-learning task.

### Stage 3 — Make the output obey the hierarchy

Remember the DAG. A raw classifier might output something nonsensical like:
"viral entry into host cell" = 0.9 but its parent "interaction with host" = 0.2. That
violates logic — the specific thing can't be more certain than the general thing it
implies. So we do a cleanup pass (`ontology/go_dag.py`) that lifts every parent's
score to be at least as high as its most confident child. This is called
**true-path correction**, and it guarantees the output is internally consistent.

(The same DAG fact is used in the *opposite* direction during training: if the
training data says a protein has a specific tag, we automatically add all the parent
tags too, before training. This is "propagation." Same rule, both directions.)

After this stage, we have the core deliverable: a clean set of GO annotations for the
protein, with a confidence for each.

### Stage 4 — From "what it does" to "is it dangerous?"

This is the payoff stage (`threat.py`, `data/danger_terms.py`). A human expert wrote
down a short list of **danger categories** — "host-cell entry & membrane fusion,"
"immune evasion," "host-cell killing," etc. — and pinned each one to a few high-level
GO tags. Because the dictionary is a DAG, we automatically expand each category to
*all* the specific tags underneath it.

Then, for a given virus, we annotate every one of its proteins (stages 1–3) and check
which predicted tags fall into a danger category. The output is a **triage signal**:
"this virus shows strong signals of immune evasion, and *this specific protein* is the
one driving it."

There's one subtlety worth flagging because it's clever. Almost every virus protein
scores high on generic terms like "enters host cell" — that's just what viruses do —
so a naive threshold would flag everything and tell you nothing. Instead we rank each
protein by its **lift over the background**: how much it stands out above the *average*
protein in that same virus. That's what surfaces the genuinely interesting protein
(the entry machinery, the immune-system saboteur) instead of the crowd.

## 4. How we know it actually works

Two ideas from any ML course apply directly.

**A baseline to beat.** The dumbest possible model just predicts each tag at its
overall frequency in the training data ("Naive"). If our model can't beat that, it
learned nothing. Every result is reported as model-score *minus* Naive-score, so you
can see the real lift. (Across the full set the model beats Naive by about +0.10 on
the headline metric — a real, if modest, gain.)

**Don't let the test set leak into training.** Here's a biology-specific trap: two
proteins can be near-copies of each other (evolution reuses parts). If a near-copy of
a test protein sits in the training set, the model can "win" by memorization, and the
score is a lie. We prevent this by **clustering** similar sequences and forcing an
entire cluster onto one side of the train/test split (`data/cluster.py`,
`data/split.py`) — like deduplicating near-identical examples before splitting.

We go one step further with a **zero-shot test**: we remove an *entire* virus family
(the coronavirus family) from training and ask the model to annotate it cold. That
simulates the real mission — a brand-new pathogen nobody has annotated yet. The
honest finding: the model recovers **what a protein does** (Molecular Function)
strikingly well on a never-seen family, but struggles to predict *where* it acts or
*which* larger process it joins. For threat triage, "what it does" is the part that
matters most.

## 5. How you'd actually run it

The project installs a few command-line tools:

- `va-train` — train the classifier from scratch (download data, embed, fit, score).
- `va-benchmark` — run a rigorous time-based comparison against standard methods.
- `va-threat --panel` — the headline demo: annotate a set of dangerous viruses and
  print their threat profiles.

That's the whole story at altitude: **a string goes in, a protein-language model
turns it into numbers, a small classifier turns those into hierarchical function
tags, and a curated danger map turns the tags into a threat triage signal.** When
you're ready for the biology behind *why* each choice was made, read Walkthrough 2.
