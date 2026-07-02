import hashlib, json
from pathlib import Path

from scripts.validate_public_domain_corpus import validate
from scripts.convert_public_domain_corpus import convert


def _sample(root: Path):
    d=root/"declaration-of-independence"; d.mkdir(parents=True)
    text=d/"text.txt"; text.write_text("Declaration of Independence\n\nWhen in the Course of human events...\n", encoding="utf-8")
    data=text.read_bytes(); rec={"id":"work_0001","slug":"declaration-of-independence","title":"Declaration of Independence","author":"Continental Congress","subject":"civic documents","path":str(text),"source_name":"U.S. National Archives","source_url":"https://www.archives.gov/founding-docs/declaration-transcript","source_identifier":"us-declaration-independence","edition":"official transcript","translator":None,"publication_year":1776,"rights_bucket":"A_PD_CC0","rights_basis":"U.S. federal/public-domain civic document","sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    (root/"manifest.jsonl").write_text(json.dumps(rec)+"\n", encoding="utf-8")


def test_validate_and_convert_public_domain_corpus(tmp_path):
    root=tmp_path/"corpus/raw/public_domain"; root.mkdir(parents=True); _sample(root)
    errors, summary=validate(root)
    assert errors == []
    assert summary["records"] == 1
    out=tmp_path/"data/processed/public_domain/documents.jsonl"
    result=convert(root, out)
    assert result["documents"] == 1
    row=json.loads(out.read_text(encoding="utf-8"))
    assert row["license"] == "public-domain-us"
    assert "When in the Course" in row["text"]
