#!/usr/bin/env python3
"""Build a multi-document CFNA corpus from repo books plus bounded HF streams."""

from __future__ import annotations

import argparse, html, json, re, shutil, zipfile
from collections import Counter
from datetime import date
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path

END = "\n\n<|END_DOCUMENT|>\n\n"
SUFFIXES = {".txt", ".tex", ".epub", ".html", ".htm", ".xhtml"}
DIRS = ("books", "corpus", "math", "psych", "enlgish grammar", "english grammar")
PERMISSIVE = {"mit","apache-2.0","bsd-2-clause","bsd-3-clause","isc","cc0-1.0","unlicense"}
GSTART = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.I | re.S)
GEND = re.compile(r"\*\*\*\s*END OF.*", re.I | re.S)

class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.out=[]; self.skip=0
    def handle_starttag(self, tag, attrs):
        if tag in {"script","style","svg"}: self.skip += 1
        elif not self.skip and tag in {"p","div","br","li","section","h1","h2","h3","pre"}: self.out.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script","style","svg"} and self.skip: self.skip -= 1
        elif not self.skip and tag in {"p","div","li","section","h1","h2","h3","pre"}: self.out.append("\n")
    def handle_data(self, data):
        if not self.skip: self.out.append(data)

def clean(raw):
    text = raw.replace("\r\n","\n").replace("\r","\n").replace("\x00","")
    if "START OF" in text.upper(): text = GSTART.split(text,1)[-1]
    if "END OF" in text.upper(): text = GEND.split(text,1)[0]
    text = "\n".join(re.sub(r"[\t\f\v ]+"," ",x).strip() for x in text.splitlines())
    return re.sub(r"\n{3,}","\n\n",text).strip()

def read_file(path):
    if path.suffix.lower() == ".epub":
        parts=[]
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if Path(name).suffix.lower() in {".html",".htm",".xhtml"}:
                    p=HTMLText(); p.feed(z.read(name).decode("utf-8","replace"))
                    if "".join(p.out).strip(): parts.append("".join(p.out))
        return "\n\n".join(parts)
    raw=path.read_text(encoding="utf-8",errors="replace")
    if path.suffix.lower() in {".html",".htm",".xhtml"}:
        p=HTMLText(); p.feed(raw); return html.unescape("".join(p.out))
    return raw

def slug(s):
    return re.sub(r"[^a-z0-9]+","_",s.lower()).strip("_")[:90] or "doc"

def discover(repo):
    found={p.resolve() for p in repo.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES}
    for d in DIRS:
        root=repo/d
        if root.is_dir():
            found |= {p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES}
    return sorted(found,key=lambda p:str(p.relative_to(repo)).lower())

def parse_job(spec):
    source, budget = spec.split("=",1)
    return source.strip(), int(budget)

def hf_text(entry,row):
    for field in entry.text_fields:
        value=row.get(field)
        if isinstance(value,str) and value.strip(): return value
    return None

def allowed_code(row):
    raw=row.get("license") or row.get("licenses") or row.get("repo_license") or ""
    values=[str(x).lower() for x in raw] if isinstance(raw,list) else [str(raw).lower()]
    return any(a==v.strip() or a in v for v in values for a in PERMISSIVE)

def record(doc_id,title,source,locator,license_id,n_bytes,split,path,role="base_pretraining",n_docs=1):
    return {
        "document_id":doc_id,"title":title,"author":"dataset/repository",
        "document_type":role,"source_collection":source,"source_locator":locator,
        "license":license_id,"license_id":license_id,
        "commercial_use":"noncommercial" not in license_id.lower(),
        "attribution_required":any(x in license_id.lower() for x in ("by","sharing","share")),
        "language":"en","publication_year":None,"retrieved_at":date.today().isoformat(),
        "content_hash":"sha256:"+sha256((Path(path).read_bytes() if Path(path).is_absolute() else b"")).hexdigest(),
        "quality_score":1.0,"n_bytes":n_bytes,"split":split,
        "bucket":"base_pretraining","path":str(path),"n_docs":n_docs,"phase":1,"role":role,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,default=Path.cwd())
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--include-hf",action="store_true")
    ap.add_argument("--hf-job",action="append",default=[])
    ap.add_argument("--hf-shard-bytes",type=int,default=4_000_000)
    ap.add_argument("--hf-val-every",type=int,default=10)
    ap.add_argument("--val-fraction",type=float,default=.10)
    ap.add_argument("--min-bytes",type=int,default=2_000)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    repo=args.repo_root.resolve(); out=args.out.resolve(); tmp=out.with_name(out.name+".incomplete")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    records=[]; seen=set(); local=[]; skipped=[]

    for path in discover(repo):
        try: text=clean(read_file(path))
        except Exception as exc:
            skipped.append({"path":str(path.relative_to(repo)),"reason":repr(exc)}); continue
        data=text.encode("utf-8")
        digest=sha256(data).hexdigest()
        if len(data)<args.min_bytes or digest in seen:
            skipped.append({"path":str(path.relative_to(repo)),"reason":"small_or_duplicate"}); continue
        seen.add(digest); local.append((path,text,digest))

    ranked=sorted(local,key=lambda x:sha256(f"{args.seed}:{x[0]}:{x[2]}".encode()).hexdigest())
    nval=max(1,min(len(ranked)-1,round(len(ranked)*args.val_fraction))) if len(ranked)>1 else 0
    val_paths={x[0] for x in ranked[:nval]}
    for path,text,digest in local:
        split="val" if path in val_paths else "train"
        rel=Path("documents")/split/f"repo_{slug(path.relative_to(repo).with_suffix('').as_posix())}_{digest[:10]}.txt"
        dest=tmp/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(text,encoding="utf-8")
        r=record(rel.stem,path.stem,"repository-vendored-public-domain",
                 f"repo://{path.relative_to(repo).as_posix()}","public-domain-us",
                 len(text.encode()),split,rel,"base_pretraining_books")
        r["content_hash"]="sha256:"+digest; records.append(r)

    hf_summary=[]
    if args.include_hf:
        from datasets import load_dataset
        from cfna.corpus.stack import get_entry
        for source_id,budget in map(parse_job,args.hf_job):
            entry=get_entry(source_id); accepted=0; shard=0; rows=0; buf=[]; buf_bytes=0; status="complete"; error=None
            try:
                kw={"path":entry.dataset_name,"split":entry.split,"streaming":True}
                if entry.dataset_config: kw["name"]=entry.dataset_config
                for row in load_dataset(**kw):
                    if source_id=="the_stack_smol" and not allowed_code(row): continue
                    text=hf_text(entry,row)
                    if not text: continue
                    text=clean(text); b=len(text.encode())
                    if b<200: continue
                    if accepted+b>budget: break
                    digest=sha256(text.encode()).hexdigest()
                    if digest in seen: continue
                    seen.add(digest); buf.append(text+END); buf_bytes+=b; accepted+=b; rows+=1
                    if buf_bytes>=args.hf_shard_bytes:
                        shard+=1; split="val" if shard%args.hf_val_every==0 else "train"
                        rel=Path("documents")/split/f"hf_{source_id}_{shard:04d}.txt"
                        dest=tmp/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text("".join(buf),encoding="utf-8")
                        r=record(rel.stem,f"{entry.name} shard {shard}",entry.name,entry.dataset_page,
                                 entry.license,dest.stat().st_size,split,rel,entry.role,len(buf))
                        r["content_hash"]="sha256:"+sha256(dest.read_bytes()).hexdigest(); records.append(r)
                        buf=[]; buf_bytes=0
                if buf:
                    shard+=1; split="val" if shard%args.hf_val_every==0 else "train"
                    rel=Path("documents")/split/f"hf_{source_id}_{shard:04d}.txt"
                    dest=tmp/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text("".join(buf),encoding="utf-8")
                    r=record(rel.stem,f"{entry.name} shard {shard}",entry.name,entry.dataset_page,
                             entry.license,dest.stat().st_size,split,rel,entry.role,len(buf))
                    r["content_hash"]="sha256:"+sha256(dest.read_bytes()).hexdigest(); records.append(r)
            except Exception as exc:
                status="failed"; error=repr(exc)
            hf_summary.append({"source_id":source_id,"status":status,"error":error,
                               "accepted_bytes":accepted,"rows":rows,"shards":shard})

    splits=Counter(r["split"] for r in records)
    if not splits["train"] or not splits["val"]: raise RuntimeError(f"bad splits: {dict(splits)}")
    with (tmp/"manifest.jsonl").open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    summary={"records":len(records),"splits":dict(splits),
             "bytes":{s:sum(r["n_bytes"] for r in records if r["split"]==s) for s in splits},
             "sources":dict(Counter(r["source_collection"] for r in records)),
             "local_files":len(local),"skipped":skipped,"hf":hf_summary}
    (tmp/"build_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if out.exists(): shutil.rmtree(out)
    tmp.rename(out)
    print(json.dumps(summary,indent=2))
    print("FULL CORPUS BUILD COMPLETE:",out)

if __name__=="__main__":
    main()
