"""Code-SFT extraction + record contract (torch-free)."""

from cfna.training.code_sft import extract_functions, functions_to_records, records_from_files
from cfna.training.mixed_sft import render_record

SAMPLE = '''
def add(a, b):
    """Return the sum of a and b."""
    return a + b


def no_doc(x):
    return x * 2


async def fetch(url):
    """Fetch a URL and return its text body."""
    resp = await get(url)
    return resp.text
'''


def test_extract_only_docstringed_functions_in_band():
    funcs = extract_functions(SAMPLE)
    names = {f["name"] for f in funcs}
    assert names == {"add", "fetch"}          # no_doc skipped (no docstring)
    add = next(f for f in funcs if f["name"] == "add")
    assert add["doc"] == "Return the sum of a and b."
    assert add["signature"] == "def add(a, b):"
    assert "return a + b" in add["full"]


def test_three_record_kinds_per_function():
    recs = list(functions_to_records(extract_functions(SAMPLE), source="t"))
    kinds = {r["category"] for r in recs}
    assert kinds == {"code_impl", "code_complete", "code_explain"}
    assert len(recs) == 6                      # 2 functions x 3 kinds


def test_records_render_canonical_with_response_mask():
    rec = next(functions_to_records(extract_functions(SAMPLE), source="t"))
    b, m = render_record(rec)                   # must plug into the SFT pipeline
    text = b.decode("utf-8")
    assert "<|user|>" in text and "<|assistant|>" in text
    masked = bytes(v for v, keep in zip(b, m) if keep).decode("utf-8")
    assert "return a + b" in masked            # code is the loss target
    assert "Write a Python function" not in masked  # instruction is not


def test_records_from_files_dedupes(tmp_path):
    f1 = tmp_path / "a.py"; f1.write_text(SAMPLE)
    f2 = tmp_path / "b.py"; f2.write_text(SAMPLE)   # identical -> no new pairs
    recs = list(records_from_files([str(f1), str(f2)], source="t"))
    assert len(recs) == 6
    assert len({r["id"] for r in recs}) == 6


def test_records_from_rows_filters_license_and_extracts():
    from cfna.training.code_sft import records_from_rows
    rows = [
        {"content": SAMPLE, "license": "mit"},          # kept
        {"content": SAMPLE, "license": "gpl-3.0"},      # dropped (copyleft)
        {"content": "not code at all", "license": "mit"},
    ]
    recs = list(records_from_rows(rows, source="stack"))
    assert len(recs) == 6                                # only the MIT row's 2 funcs x 3
    assert all(r["category"].startswith("code_") for r in recs)
