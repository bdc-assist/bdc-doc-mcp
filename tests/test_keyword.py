import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bdc_doc_mcp.api import _keyword_rank

docs = [
    "PIC-SURE offers an API for querying data.",
    "The pic-sure user guide covers API basics and API auth.",
    "Unrelated text about dbGaP.",
]
metas = [{"i": 0}, {"i": 1}, {"i": 2}]

hits = _keyword_rank("PIC-SURE guide", docs, metas, k=5)
# doc 1 matches both terms (case-insensitively), doc 0 one, doc 2 none
assert [h["metadata"]["i"] for h in hits] == [1, 0], hits
assert hits[0]["score"] == 2.0, hits[0]

# punctuation in the query is stripped; k caps the results
hits = _keyword_rank('"API",', docs, metas, k=1)
assert len(hits) == 1 and hits[0]["metadata"]["i"] == 1, hits  # 2 occurrences beats 1

# fuzzy: punctuation-insensitive and typo-tolerant
hits = _keyword_rank("PICSURE", docs, metas, k=5)
assert len(hits) == 2, hits  # matches "PIC-SURE" / "pic-sure"
hits = _keyword_rank("picsur", docs, metas, k=5)  # typo, one letter short
assert len(hits) == 2, hits

assert _keyword_rank("nomatch", docs, metas, k=5) == []

print("keyword ranking self-check passed")
