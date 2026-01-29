#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from lxml import etree


def _safe_text(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _extract_year(article) -> Optional[int]:
    # Try common locations for year
    year_paths = [
        ".//PubDate/Year",
        ".//ArticleDate/Year",
        ".//DateCompleted/Year",
        ".//DateCreated/Year",
    ]
    for xp in year_paths:
        el = article.find(xp)
        if el is not None and el.text and el.text.strip().isdigit():
            y = int(el.text.strip())
            if 1800 <= y <= 2100:
                return y
    return None


def parse_pubmed_xml_gz(xml_gz_path: Path) -> List[Dict]:
    rows: List[Dict] = []

    with gzip.open(xml_gz_path, "rb") as f:
        # Stream parse for memory safety
        context = etree.iterparse(f, events=("end",), tag="PubmedArticle")

        for _, article in context:
            pmid = _safe_text(article.find(".//PMID"))
            title = _safe_text(article.find(".//ArticleTitle"))

            # Abstract can have multiple AbstractText sections
            abstract_parts = []
            for abs_el in article.findall(".//Abstract/AbstractText"):
                label = abs_el.get("Label") or abs_el.get("NlmCategory") or ""
                txt = _safe_text(abs_el)
                if not txt:
                    continue
                if label:
                    abstract_parts.append(f"{label}: {txt}")
                else:
                    abstract_parts.append(txt)

            abstract = "\n".join(abstract_parts).strip()
            if not pmid or not abstract:
                article.clear()
                continue

            journal = _safe_text(article.find(".//Journal/Title"))
            year = _extract_year(article)

            rows.append(
                {
                    "doc_id": f"pubmed:{pmid}",
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "source": "pubmed_baseline",
                    "file": xml_gz_path.name,
                }
            )

            # Free memory
            article.clear()
            while article.getprevious() is not None:
                del article.getparent()[0]

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PubMed baseline .xml.gz files to a single Parquet file.")
    ap.add_argument("--in_dir", required=True, help="Directory containing pubmed*.xml.gz files")
    ap.add_argument("--out_parquet", required=True, help="Output Parquet path")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_path = Path(args.out_parquet).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("pubmed*.xml.gz"))
    if not files:
        raise SystemExit(f"No pubmed*.xml.gz files found in {in_dir}")

    all_rows: List[Dict] = []
    for i, fp in enumerate(files, start=1):
        print(f"[parse] ({i}/{len(files)}) {fp.name}")
        rows = parse_pubmed_xml_gz(fp)
        print(f"[rows]  {len(rows)} extracted from {fp.name}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # Basic cleanup
    df["title"] = df["title"].fillna("")
    df["journal"] = df["journal"].fillna("")
    # Parquet write
    df.to_parquet(out_path, index=False)
    print(f"[done] wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
