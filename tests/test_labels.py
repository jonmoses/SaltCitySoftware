"""Tests for UniProt entry parsing + tier-split propagation."""

from viral_annotation.data import labels


SAMPLE_ENTRY = {
    "primaryAccession": "P12345",
    "organism": {"scientificName": "Test virus", "lineage": ["Viruses", "Testviridae"]},
    "sequence": {"value": "MKTAYIAK"},
    "uniProtKBCrossReferences": [
        {"database": "GO", "id": "GO:0000003",
         "properties": [{"key": "GoTerm", "value": "F:leaf"},
                        {"key": "GoEvidenceType", "value": "IDA:UniProtKB"}]},
        {"database": "GO", "id": "GO:0000004",
         "properties": [{"key": "GoEvidenceType", "value": "IEA:InterPro"}]},
        {"database": "Pfam", "id": "PF00001", "properties": []},
    ],
}


def test_parse_entry_tags_evidence_tiers():
    rec = labels._parse_entry(SAMPLE_ENTRY)
    assert rec.accession == "P12345"
    assert rec.sequence == "MKTAYIAK"
    assert rec.organism == "Test virus"
    assert rec.lineage == ["Viruses", "Testviridae"]
    # Pfam ignored; GO split into manual (IDA) and iea (IEA).
    assert set(rec.annotations) == {("GO:0000003", "manual"), ("GO:0000004", "iea")}


def test_parse_next_link_handles_commas_in_url():
    header = (
        '<https://rest.uniprot.org/uniprotkb/search?fields=accession,sequence,go_id'
        '&cursor=abc&size=500>; rel="next"'
    )
    nxt = labels._parse_next_link(header)
    assert nxt.endswith("size=500")
    assert "fields=accession,sequence,go_id" in nxt


def test_parse_next_link_none_when_absent():
    assert labels._parse_next_link(None) is None
    assert labels._parse_next_link('<https://x>; rel="prev"') is None


def test_label_proteins_tier_split_propagation(tiny_dag):
    rec = labels._parse_entry(SAMPLE_ENTRY)
    [lab] = labels.label_proteins([rec], tiny_dag)
    # manual term 0003 propagates to 0002 + root; iea term 0004 adds itself + 0002.
    assert lab.terms_manual == {"GO:0000003", "GO:0000002", "GO:0003674"}
    assert lab.terms_all == {"GO:0000003", "GO:0000004", "GO:0000002", "GO:0003674"}
    assert lab.terms_manual <= lab.terms_all
    assert lab.n_manual == 1 and lab.n_iea == 1
    assert lab.has_manual is True
