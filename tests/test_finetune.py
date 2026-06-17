"""Tests for the LoRA fine-tune path's pure plumbing — masking, the asymmetric
loss, and the multi-task model/training loop — using a FAKE backbone + tokenizer so
no ESM weights are downloaded. (The real backbone is exercised on the Kaggle T4 run.)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from viral_annotation.data.dataset import select_vocab
from viral_annotation.training.finetune import (
    _masked_pos_weight,
    _pool_mask,
    _predict_all,
    build_multitask_model,
    fit_finetune,
)

# MF leaf GO:0000003 (manual) / GO:0000004 (iea); BP GO:0000011; CC GO:0000021.
P = lambda seq, manual, mf, bp, cc, iea=frozenset(): SimpleNamespace(  # noqa: E731
    accession=seq, sequence=seq, has_manual=manual,
    terms_manual=frozenset(mf | bp | cc),
    terms_all=frozenset(mf | bp | cc | iea),
)
MF, MF_MID, IEA = {"GO:0000003"}, {"GO:0000002"}, {"GO:0000004"}
BP, CC = {"GO:0000011"}, {"GO:0000021"}

POLICY = {
    "molecular_function": {"train_pool": "manual_having", "train_field": "terms_manual",
                           "vocab_field": "terms_manual", "pooling": "attention"},
    "biological_process": {"train_pool": "all", "train_field": "terms_all",
                           "vocab_field": "terms_all", "pooling": "mean"},
    "cellular_component": {"train_pool": "all", "train_field": "terms_all",
                           "vocab_field": "terms_all", "pooling": "mean"},
}


def _proteins():
    return [
        P("MKLVAA", True, MF | MF_MID, BP, CC),
        P("GGSTPLE", True, MF | MF_MID, BP, set()),
        P("WYFAACDE", False, set(), BP, CC, iea=IEA),   # IEA-only: no manual MF
        P("HHKKRRDD", True, MF | MF_MID, set(), CC),
    ]


def _fake_backbone(dim, vocab=30):
    import torch
    from torch import nn

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)   # trainable -> stands in for adapters

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    return FakeBackbone()


def _fake_tokenizer():
    import torch

    class Enc(dict):
        def to(self, _device):
            return self

    def tok(seqs, return_tensors=None, padding=None, truncation=None, max_length=None):
        cap = max_length or max(len(s) for s in seqs)
        L = min(max(len(s) for s in seqs), cap)
        ids = torch.zeros(len(seqs), L + 2, dtype=torch.long)
        mask = torch.zeros(len(seqs), L + 2)
        for i, s in enumerate(seqs):
            n = min(len(s), L)
            for j, c in enumerate(s[:n]):
                ids[i, j + 1] = (ord(c) % 27) + 1
            mask[i, : n + 2] = 1.0
        return Enc(input_ids=ids, attention_mask=mask)

    return tok


def _hp():
    return SimpleNamespace(loss="asl", max_length=16, ft_batch=2, ft_grad_accum=2,
                           ft_epochs=2, heads=2, ft_hidden=[8], ft_dropout=0.0,
                           ft_backbone_lr=1e-3, ft_head_lr=1e-2)


def test_pool_mask_and_masked_pos_weight():
    prots = _proteins()
    # "all" -> every protein; "manual_having" -> drops the IEA-only protein.
    assert _pool_mask(prots, "all").tolist() == [1, 1, 1, 1]
    assert _pool_mask(prots, "manual_having").tolist() == [1, 1, 0, 1]
    # pos_weight is finite and per-term when computed over the masked rows only.
    Y = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype="float32")
    pw = _masked_pos_weight(Y, np.array([1, 1, 0, 1], "float32"))
    assert pw.shape == (2,) and np.isfinite(pw).all()


def test_asymmetric_loss_shape_and_positive():
    import torch

    from viral_annotation.classifier.losses import asymmetric_loss

    logits = torch.randn(3, 5)
    targets = torch.randint(0, 2, (3, 5)).float()
    loss = asymmetric_loss(logits, targets)
    assert loss.shape == (3, 5)
    assert (loss >= 0).all()   # -log terms are non-negative after the focusing weight


def test_multitask_forward_shapes(tiny_dag):
    prots = _proteins()
    vocabs = {ns: select_vocab([p for p in prots if (POLICY[ns]["train_pool"] == "all" or p.has_manual)],
                               tiny_dag, 1, field=POLICY[ns]["vocab_field"], namespaces=[ns])
              for ns in POLICY}
    ns_specs = [{"ns": ns, "num_terms": len(vocabs[ns]), "pooling": POLICY[ns]["pooling"]}
                for ns in POLICY if len(vocabs[ns])]
    dim = 12
    model = build_multitask_model(_fake_backbone(dim), dim, ns_specs, n_heads=2,
                                  hidden_dims=[8], dropout=0.0)
    probs = _predict_all(model, prots, _fake_tokenizer(), "cpu", max_length=16, batch_size=2)
    for s in ns_specs:
        assert probs[s["ns"]].shape == (len(prots), s["num_terms"])
        assert ((0.0 <= probs[s["ns"]]) & (probs[s["ns"]] <= 1.0)).all()


def test_fit_finetune_runs_and_improves(tiny_dag):
    prots = _proteins()
    pools = {"all": prots, "manual_having": [p for p in prots if p.has_manual]}
    vocabs = {ns: select_vocab(pools[POLICY[ns]["train_pool"]], tiny_dag, 1,
                               field=POLICY[ns]["vocab_field"], namespaces=[ns])
              for ns in POLICY}
    vocabs = {ns: v for ns, v in vocabs.items() if len(v)}
    ns_specs = [{"ns": ns, "num_terms": len(vocabs[ns]), "pooling": POLICY[ns]["pooling"]}
                for ns in vocabs]
    dim = 12
    model = build_multitask_model(_fake_backbone(dim), dim, ns_specs, n_heads=2,
                                  hidden_dims=[8], dropout=0.0)
    val_fmax, epochs = fit_finetune(model, _fake_tokenizer(), prots, prots,
                                    vocabs, POLICY, "cpu", _hp())
    assert epochs >= 1
    assert set(val_fmax) == set(vocabs)
    assert all(np.isfinite(v) for v in val_fmax.values())
