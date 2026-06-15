## Process
##### 1. Characterize taxonomy 
	Identify broadly what branch of pathogen family it belongs to

##### 2. Identify and annotate proteins of interest in the genomic data 
use ESM-2 as the embedding backbone
- predicting, for each of thousands of possible GO terms, a probability that this protein has that function

##### 3. Identify host-pathogen interactions and characterize the threat


### Training Plan (ANNOTATING)

## Step 1: Get a single fixed-length vector per protein

ESM doesn't give you one vector per protein by default — it gives you one vector per amino acid. If you run a protein of length L through ESM-2, you get back a matrix of shape [L × d], where d is the embedding dimension (1280 for the 650M model, 2560 for the 3B, larger for the 15B). That's the per-residue representation.

But your classifier needs a single fixed-length input regardless of protein length, so you have to collapse that [L × d] matrix into one [d]-dimensional vector. The standard move is **mean pooling**: average the residue vectors across the length dimension, giving one d-dimensional vector that summarizes the whole protein. Some pipelines instead use the representation of the special start-of-sequence token (analogous to BERT's CLS token) as the protein-level summary. Mean pooling is the more common and generally robust default. This pooled vector is your feature representation — the "usable" form the embedding gets converted into.

One practical note: you typically extract embeddings from one of the later transformer layers, not the final layer necessarily — the most informative layer varies, and it's worth treating "which layer to pool from" as a small hyperparameter to check.

## Step 2: The classifier itself

Now you map that d-dimensional vector to GO-term probabilities. "Logistic regression" in this multi-label context means, concretely:

A single linear layer with a weight matrix **W** of shape [N × d] and a bias vector of length N, where N is the number of GO terms you're predicting (this could be anywhere from a few hundred to several thousand depending on how you scope the term set). You compute `z = Wx + b`, giving N raw scores (logits), one per GO term. Then you apply a **sigmoid** to each logit independently, squashing each into a probability between 0 and 1.

The single most important architectural point: you use **sigmoid, not softmax**. Softmax would force the outputs to sum to 1, which encodes "exactly one of these classes is true" — that's single-label classification. But a protein has many functions simultaneously, so the labels are not mutually exclusive. Sigmoid treats each GO term as its own independent binary question ("does this protein have function k, yes or no?"), which is exactly the multi-label structure you want. Effectively you're training N independent binary classifiers that happen to share the same input features and be trained jointly.

That's the linear version, which is what NetGO found worked surprisingly well. If you want more capacity, you insert one or two hidden layers with a nonlinearity (ReLU) and dropout before the final sigmoid layer, turning it into a small MLP. The tradeoff is more capacity versus more overfitting risk and more data hunger — for a prototype I'd start linear, establish the baseline, then see whether an MLP buys you measurable Fmax improvement.

## Step 3: The training labels and the loss

Your training data is a set of proteins with known GO annotations — pulled from UniProt/Swiss-Prot. For each training protein you build a binary label vector of length N: a 1 in position k if the protein is annotated with GO term k, 0 otherwise.

A critical data-prep detail tied to the GO hierarchy: the **true-path rule**. Because GO is a DAG where specific terms imply their general ancestors, an annotation to "serine protease activity" logically implies "protease activity," "hydrolase activity," and so on up to the root. So before training, you **propagate** each protein's annotations upward through the DAG, setting all ancestor terms to 1. If you skip this, you're training the model on inconsistent labels (positive on the child, negative on the parent), which is incoherent.

The loss function is **binary cross-entropy**, computed per label and summed or averaged across all N terms and all proteins. Each output node contributes a standard BCE term comparing its predicted probability to the 0/1 truth.

The thing that will bite you here is **extreme class imbalance**. Any given protein has maybe a few dozen of the thousands of possible GO terms, so the label vectors are overwhelmingly zeros. A naive BCE will let the model get low loss by predicting "no" for everything. Standard mitigations: weight the positive examples more heavily in the loss, use focal loss (which down-weights easy negatives), or restrict your term set to GO terms that appear often enough in your training data to be learnable (very rare terms with a handful of examples can't be predicted reliably anyway).

## Step 4: Enforcing hierarchical consistency at prediction time

At inference, your raw sigmoid outputs can violate the hierarchy — the model might assign a child term higher probability than its parent, which is logically impossible under the true-path rule. There are two ways to handle this. The simpler is **post-hoc correction**: after predicting, propagate probabilities so that a parent's probability is at least the maximum of its children's (or apply one of several standard hierarchical-correction schemes). The more sophisticated is to bake the constraint into the architecture or loss. For a prototype, post-hoc correction is fine and is what gets you consistent, interpretable output.

## Step 5: From probabilities to discrete annotations — thresholding

The sigmoid gives you a probability per term, but a final annotation is a yes/no decision, so you threshold. You can use a single global threshold (everything above 0.5, say) or — better — per-term thresholds tuned on a validation set, since different GO terms have different base rates and calibration. You don't always _need_ to threshold: for many downstream uses (like the pathway-enrichment step in your pipeline, or ranking candidate functions for an analyst) the ranked probabilities themselves are more useful than a hard cutoff. The threshold mainly matters for reporting discrete predictions and for certain metrics.

## Step 6: Evaluation — SBIR requirements

How you measure performance is not an afterthought here; it's half of what your D2P2 feasibility documentation needs. The field's standard, from the CAFA competition, is **Fmax** — the maximum F1 score achieved as you sweep the decision threshold across its whole range, computed in a protein-centric way. Report this and you're speaking the field's language and can position against CAFA-benchmarked tools. Complementary metrics: **AUPR** (area under precision-recall, good for imbalanced data), **Smin** (a semantic-distance measure that accounts for how informative the terms are), and per-term AUROC.

The part that directly satisfies your "rigorous data separation" requirement is **how you split train from test**. The naive random split is a trap in protein work: if a near-identical homolog of a test protein is sitting in your training set, the model can "cheat" by effectively memorizing, and your reported numbers are inflated. The rigorous approach is **sequence-identity-based splitting** — cluster all your proteins at some identity threshold (commonly 30%) and assign whole clusters to either train or test, so no test protein has a close relative in training. This is the standard you'd cite to demonstrate the separation was real.

And for your **zero-shot validation** specifically — recovering known interactions/functions for a pathogen the model never trained on — you go further than sequence clustering: you hold out an _entire organism or viral family_. Train on everything except, say, one held-out virus, then show the model recovers that virus's known functional annotations despite never having seen it. That experimental design is the evidence the topic is asking for, and the embedding approach is what makes it plausible, because ESM's self-supervised pretraining gives meaningful vectors even for sequences with no annotated relatives in your training set.

## Putting it together

The end-to-end recipe: run each protein through ESM-2 → mean-pool the per-residue vectors into one d-dimensional vector → feed through a linear (or small MLP) layer with N sigmoid outputs → train with positive-weighted binary cross-entropy against hierarchy-propagated GO labels → apply hierarchical correction → optionally threshold → evaluate with Fmax/AUPR/Smin under identity-based splits, with whole-organism holdout for the zero-shot claim.

Two honest caveats. First, I've given you the standard, well-established methodology for this class of problem — it's the right mental model and a sound starting architecture, but the specific hyperparameters (which ESM layer, pooling choice, hidden-layer sizes, loss weighting, thresholds) are things you tune empirically against your own validation data, not values I can hand you. Second, for exactly reproducing NetGO 3.0's reported numbers you'd want their paper's methods and any released training code, since details like their precise feature construction and ranking step affect results; what I've described is how you'd build a clean, defensible equivalent from first principles rather than a line-by-line reimplementation.
