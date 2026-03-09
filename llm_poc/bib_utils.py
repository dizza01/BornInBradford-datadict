"""
bib_utils.py
============
Shared utilities for the Born in Bradford data dictionary toolkit.

Sections
--------
1. Paths & constants
2. Metadata exploration    (from explore_metadata.py)
3. Variable validation     (from check_bib_vars.py)
4. Paper fetching          (from fetch_bib_papers.py)

Usage examples
--------------
    from bib_utils import explore_metadata, check_variables_in_html
    from bib_utils import search_pubmed, fetch_metadata, download_pdf, run_fetch
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

import xml.etree.ElementTree as ET


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR   = Path(__file__).parent
DATADICT_DIR = SCRIPT_DIR.parent
DOCS_DIR     = DATADICT_DIR / "docs"
CSV_DIR      = DOCS_DIR / "csv"
PDFS_DIR     = DATADICT_DIR / "papers"
META_JSON    = PDFS_DIR / "bib_papers_metadata.json"

PDFS_DIR.mkdir(exist_ok=True)

# NCBI / external API endpoints
ESEARCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_OA    = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EUPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2"

NCBI_API_KEY     = os.getenv("NCBI_API_KEY", "")
UNPAYWALL_EMAIL  = os.getenv("UNPAYWALL_EMAIL", "research@borninbradford.nhs.uk")
REQUEST_DELAY    = 0.11 if NCBI_API_KEY else 0.35

# PubMed search queries
PRIMARY_QUERY   = '"Born in Bradford"[Title/Abstract]'
SECONDARY_QUERY = (
    '("BiB cohort"[Title/Abstract] OR "BiB study"[Title/Abstract])'
    ' AND Bradford[Title/Abstract] AND cohort[Title/Abstract]'
)

EXCLUDE_PATTERNS = [
    r"\bbradford hill\b",
    r"\bbradford score\b",
    r"\bcity of bradford\b(?!.*born in bradford)",
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  METADATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

def explore_metadata(csv_path: str | Path | None = None) -> None:
    """
    Explore BiB metadata CSV files: table/variable counts, topics, interactive
    keyword search.

    Parameters
    ----------
    csv_path : path to the directory containing all_tables.csv and
               all_variables_meta.csv.  Defaults to DOCS_DIR/csv.
    """
    if not _PANDAS:
        print("❌ pandas is required for explore_metadata(). Install with: pip install pandas")
        return

    if csv_path is None:
        csv_path = CSV_DIR
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"❌ Path not found: {csv_path}")
        return

    print("=" * 70)
    print("BiB METADATA EXPLORER")
    print("=" * 70)

    try:
        tables_df    = pd.read_csv(csv_path / "all_tables.csv")
        variables_df = pd.read_csv(csv_path / "all_variables_meta.csv")
        print(f"\n✅ Successfully loaded metadata")
        print(f"   📊 Tables   : {len(tables_df)}")
        print(f"   📝 Variables: {len(variables_df)}")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return

    print("\n" + "=" * 70)
    print("TABLE OVERVIEW")
    print("=" * 70)
    print("\nProjects:")
    print(tables_df["project_name"].value_counts().head(10))
    print("\nTables with most variables:")
    top = tables_df.nlargest(10, "n_variables")[["table_name", "n_variables", "n_rows"]]
    print(top.to_string(index=False))

    print("\n" + "=" * 70)
    print("VARIABLE OVERVIEW")
    print("=" * 70)
    print("\nVariable types:")
    print(variables_df["value_type"].value_counts())
    print("\nVariables with topics:")
    print(variables_df["topic"].value_counts().head(15))

    print("\n" + "=" * 70)
    print("AGE OF WONDER EXAMPLE")
    print("=" * 70)
    aow = tables_df[tables_df["project_name"] == "BiB_AgeOfWonder"]
    print(f"\n{len(aow)} Age of Wonder tables:")
    print(aow[["table_name", "n_variables", "n_rows"]].to_string(index=False))

    print("\n\nRCADS Variables (Age of Wonder survey_mod02_dr23):")
    rcads = variables_df[
        (variables_df["table_id"] == "BiB_AgeOfWonder.survey_mod02_dr23")
        & variables_df["variable"].str.contains("rcads", case=False, na=False)
    ]
    if not rcads.empty:
        print(rcads[["variable", "label", "value_type"]].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("INTERACTIVE SEARCH")
    print("=" * 70)
    while True:
        keyword = input("\nSearch for variable (or 'quit' to exit): ").strip()
        if keyword.lower() in ("quit", "exit", "q", ""):
            break
        results = variables_df[
            variables_df["variable"].str.contains(keyword, case=False, na=False)
            | variables_df["label"].str.contains(keyword, case=False, na=False)
        ]
        if results.empty:
            print(f"   No variables found matching '{keyword}'")
        else:
            print(f"\n   Found {len(results)} variables:")
            print(results[["table_id", "variable", "label"]].head(20).to_string(index=False))
            if len(results) > 20:
                print(f"\n   ... and {len(results) - 20} more")

    print("\n✅ Metadata exploration complete!\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VARIABLE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def get_all_html_variables(docs_dir: Path | None = None) -> set[str]:
    """
    Parse every HTML file in docs_dir and return the set of all variable names
    found inside Reactable JSON blobs.
    """
    if docs_dir is None:
        docs_dir = DOCS_DIR

    all_vars: set[str] = set()
    for fn in Path(docs_dir).glob("*.html"):
        html = fn.read_text(encoding="utf-8", errors="ignore")
        for m in re.findall(r'"variable"\s*:\s*\[([^\]]+)\]', html):
            for v in re.findall(r'"([^"]+)"', m):
                all_vars.add(v)
    return all_vars


def check_variables_in_html(
    claimed: list[str],
    docs_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Cross-reference a list of variable names against every HTML file in
    docs_dir.  Returns (found, missing).

    Parameters
    ----------
    claimed   : list of variable names to check
    docs_dir  : path to the docs/ directory (defaults to DOCS_DIR)
    verbose   : print a summary table

    Returns
    -------
    found, missing : two lists of variable name strings
    """
    all_vars = get_all_html_variables(docs_dir)
    found   = [v for v in claimed if v in all_vars]
    missing = [v for v in claimed if v not in all_vars]

    if verbose:
        print(f"\n✅ FOUND in HTML ({len(found)}):")
        for v in found:
            print(f"   {v}")
        print(f"\n❌ NOT FOUND in HTML ({len(missing)}):")
        for v in missing:
            print(f"   {v}")

    return found, missing


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PAPER FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _api_params(extra: dict) -> dict:
    p = {"retmode": "json", **extra}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def _get(url: str, params: dict, retmode_xml: bool = False) -> Optional["requests.Response"]:
    """GET with retry and rate-limit back-off."""
    if not _REQUESTS:
        raise ImportError("requests is required. Run: pip install requests")
    if retmode_xml:
        params = {k: v for k, v in params.items() if k != "retmode"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"   ⏳ Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ⚠️  HTTP {r.status_code} for {url}")
                return None
        except requests.RequestException as e:
            print(f"   ⚠️  Request error: {e}")
            time.sleep(5)
    return None


def search_pubmed(query: str, max_results: int = 500) -> list[str]:
    """Return a list of PubMed IDs matching query."""
    print(f"\n🔍 Searching PubMed: {query[:80]}...")
    params = _api_params({"db": "pubmed", "term": query, "retmax": max_results})
    r = _get(ESEARCH, params)
    if not r:
        return []
    data  = r.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    total = data.get("esearchresult", {}).get("count", "?")
    print(f"   Found {total} total; fetching metadata for {len(pmids)}")
    time.sleep(REQUEST_DELAY)
    return pmids


def fetch_metadata(pmids: list[str]) -> list[dict]:
    """Fetch PubMed XML metadata for a list of PMIDs. Returns list of dicts."""
    if not pmids:
        return []
    print(f"\n📋 Fetching metadata for {len(pmids)} papers...")
    results    = []
    batch_size = 200

    for i in range(0, len(pmids), batch_size):
        batch  = pmids[i : i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "rettype": "xml", "retmax": batch_size}
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        r = _get(EFETCH, params, retmode_xml=True)
        if not r:
            continue
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            print(f"   ⚠️  XML parse error: {e}")
            continue

        for article in root.findall(".//PubmedArticle"):
            pmid_el  = article.find(".//PMID")
            pmid     = pmid_el.text if pmid_el is not None else ""
            title_el = article.find(".//ArticleTitle")
            title    = "".join(title_el.itertext()) if title_el is not None else ""
            abstract = " ".join(
                "".join(el.itertext()) for el in article.findall(".//AbstractText")
            )
            authors = []
            for author in article.findall(".//Author"):
                ln = author.findtext("LastName", "")
                fn = author.findtext("ForeName", "")
                authors.append(f"{ln} {fn}".strip())
            authors_str = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")

            pub_date = article.find(".//PubDate")
            year = ""
            if pub_date is not None:
                year = pub_date.findtext("Year", "") or pub_date.findtext("MedlineDate", "")[:4]

            journal = article.findtext(".//Title", "") or article.findtext(".//ISOAbbreviation", "")

            doi, pmc_id = "", ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text or ""
                if id_el.get("IdType") == "pmc":
                    pmc_id = id_el.text or ""

            results.append({
                "pmid": pmid, "pmc_id": pmc_id, "title": title.strip(),
                "abstract": abstract.strip(), "authors": authors_str,
                "year": year, "journal": journal, "doi": doi,
            })

        sys.stdout.write(f"\r   Parsed {min(i + batch_size, len(pmids))}/{len(pmids)}...")
        sys.stdout.flush()
        time.sleep(REQUEST_DELAY)

    print(f"\n   ✅ Metadata fetched for {len(results)} papers")
    return results


def _is_false_positive(title: str, abstract: str) -> bool:
    """Return True if the paper is clearly not about the BiB cohort study."""
    text = (title + " " + abstract).lower()
    return any(re.search(pat, text, re.IGNORECASE) for pat in EXCLUDE_PATTERNS)


def resolve_pmc_ids(papers: list[dict]) -> list[dict]:
    """Resolve missing PMC IDs via ELink. Updates each dict in-place."""
    need = [p for p in papers if not p.get("pmc_id") and p.get("pmid")]
    if not need:
        return papers

    print(f"\n🔗 Resolving PMC IDs for {len(need)} papers via ELink...")
    batch_size = 100
    resolved   = 0

    for i in range(0, len(need), batch_size):
        batch      = need[i : i + batch_size]
        pmids      = [p["pmid"] for p in batch]
        params_xml = {"dbfrom": "pubmed", "db": "pmc", "id": ",".join(pmids)}
        if NCBI_API_KEY:
            params_xml["api_key"] = NCBI_API_KEY

        rx = _get(ELINK, params_xml, retmode_xml=True)
        pmid_to_pmc: dict[str, str] = {}
        if rx:
            try:
                root = ET.fromstring(rx.content)
                for ls in root.findall(".//LinkSet"):
                    pmid_el = ls.find(".//IdList/Id")
                    if pmid_el is None:
                        continue
                    for link in ls.findall(".//LinkSetDb[DbTo='pmc']/Link/Id"):
                        pmid_to_pmc[pmid_el.text] = "PMC" + link.text
                        break
            except ET.ParseError:
                pass

        for p in batch:
            if p["pmid"] in pmid_to_pmc:
                p["pmc_id"] = pmid_to_pmc[p["pmid"]]
                resolved += 1

        sys.stdout.write(f"\r   Resolved {min(i + batch_size, len(need))}/{len(need)}...")
        sys.stdout.flush()
        time.sleep(REQUEST_DELAY)

    print(f"\n   ✅ Resolved {resolved} additional PMC IDs")
    return papers


def get_oa_pdf_url(pmc_id: str) -> Optional[str]:
    """Query the PMC Open Access API for a direct PDF URL."""
    clean_id = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
    params   = {"id": clean_id}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = _get(PMC_OA, params, retmode_xml=True)
    if not r:
        return None
    try:
        root = ET.fromstring(r.content)
        for link in root.findall(".//link[@format='pdf']"):
            href = link.get("href", "")
            if href:
                return href
    except ET.ParseError:
        pass
    return None


def get_europepmc_pdf_url(
    pmid: str = "", pmc_id: str = "", doi: str = ""
) -> Optional[str]:
    """Try Europe PMC's REST API for a full-text open-access PDF link."""
    if pmid:
        query = f"EXT_ID:{pmid} AND SRC:MED"
    elif pmc_id:
        clean = pmc_id.replace("PMC", "")
        query = f"PMC{clean}[PMCID]"
    elif doi:
        query = f'"{doi}"[DOI]'
    else:
        return None

    r = _get(f"{EUPMC_API}/search", {"query": query, "resultType": "core",
                                      "format": "json", "pageSize": 1})
    if not r:
        return None
    try:
        data    = r.json()
        results = data.get("resultList", {}).get("result", [])
        if not results or results[0].get("isOpenAccess", "N") != "Y":
            return None
        article = results[0]
        for entry in article.get("fullTextUrlList", {}).get("fullTextUrl", []):
            if (entry.get("documentStyle") == "pdf"
                    and "open" in entry.get("availability", "").lower()):
                return entry.get("url")
        for entry in article.get("fullTextUrlList", {}).get("fullTextUrl", []):
            url = entry.get("url", "")
            if ("open" in entry.get("availability", "").lower()
                    and "europepmc.org/articles" in url):
                return url.rstrip("/") + "?pdf=render"
    except Exception:
        pass
    return None


def get_unpaywall_pdf_url(doi: str) -> Optional[str]:
    """Query Unpaywall for a legal open-access PDF URL."""
    if not doi:
        return None
    doi_clean = doi.strip().lstrip("https://doi.org/").lstrip("http://dx.doi.org/")
    r = _get(f"{UNPAYWALL}/{doi_clean}", {"email": UNPAYWALL_EMAIL})
    if not r:
        return None
    try:
        data = r.json()
        if data.get("is_oa") is not True:
            return None

        def _pick(loc: dict) -> Optional[str]:
            u = loc.get("url_for_pdf")
            if u:
                return u
            u = loc.get("url", "")
            if u.lower().endswith(".pdf"):
                return u
            return None

        best = data.get("best_oa_location") or {}
        pdf_url = _pick(best)
        if pdf_url:
            return pdf_url
        for loc in data.get("oa_locations", []):
            pdf_url = _pick(loc)
            if pdf_url:
                return pdf_url
    except Exception:
        pass
    return None


def safe_filename(title: str, year: str, max_len: int = 120) -> str:
    """Convert a paper title + year to a safe filename stem."""
    clean = re.sub(r"[^\w\s\-]", "_", title)
    clean = re.sub(r"[\s_]+", "_", clean).strip("_")
    return f"{clean[:max_len]}_{year}" if year else clean[:max_len]


def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF to dest. Returns True on success. Validates magic bytes."""
    if url.startswith("ftp://"):
        url = "https://" + url[len("ftp://"):]
    headers = {"Accept": "application/pdf"} if "europepmc" in url else {}
    try:
        r = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return False
        buf = bytearray()
        for chunk in r.iter_content(8192):
            buf.extend(chunk)
        if bytes(buf[:4]) != b"%PDF":
            snippet = buf[:120].decode("utf-8", errors="replace").replace("\n", " ")
            print(f"   ⚠️  Not a PDF (got: {snippet[:80]!r})")
            return False
        dest.write_bytes(bytes(buf))
        return True
    except Exception as e:
        print(f"   ⚠️  Download failed: {e}")
        return False


def existing_stems() -> set[str]:
    """Return lowercase stems of PDFs already in PDFS_DIR."""
    return {p.stem.lower() for p in PDFS_DIR.glob("*.pdf")}


def load_existing_metadata() -> list[dict]:
    if META_JSON.exists():
        with open(META_JSON, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata(papers: list[dict]) -> None:
    seen: dict[str, dict] = {}
    for p in papers:
        key = p.get("pmid") or p.get("doi") or p.get("title", "")[:80]
        seen[key] = p
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(list(seen.values()), f, indent=2, ensure_ascii=False)
    print(f"\n💾 Metadata saved → {META_JSON}  ({len(seen)} papers total)")


def run_fetch(max_results: int = 500, download: bool = False) -> None:
    """
    Full BiB paper fetch pipeline:
    search → metadata → resolve PMC IDs → (optional) download PDFs.
    """
    pmids = list(dict.fromkeys(
        search_pubmed(PRIMARY_QUERY,   max_results)
        + search_pubmed(SECONDARY_QUERY, max_results)
    ))
    print(f"\n📊 Combined unique PMIDs: {len(pmids)}")

    papers = fetch_metadata(pmids)
    before = len(papers)
    papers = [p for p in papers if not _is_false_positive(p["title"], p["abstract"])]
    if before - len(papers):
        print(f"   🚫 Removed {before - len(papers)} false positives")

    papers   = resolve_pmc_ids(papers)
    has_pmc  = [p for p in papers if p.get("pmc_id")]
    no_pmc   = [p for p in papers if not p.get("pmc_id")]
    print(f"\n📑 {len(has_pmc)} with PMC ID  |  {len(no_pmc)} without")

    if not download:
        print("\n── DRY RUN (pass download=True to fetch PDFs) " + "─" * 30)
        print(f"{'PMID':<12} {'PMC ID':<12} {'Year':<6}  Title")
        print("─" * 80)
        for p in sorted(papers, key=lambda x: x.get("year", ""), reverse=True)[:40]:
            print(f"{p.get('pmid',''):<12} {p.get('pmc_id','—'):<12} {p.get('year',''):<6}  {p.get('title','')[:60]}")
        if len(papers) > 40:
            print(f"  ... and {len(papers) - 40} more")
        save_metadata(load_existing_metadata() + papers)
        print("\nRe-run with download=True to fetch open-access PDFs.")
        return

    stems      = existing_stems()
    downloaded = skipped = no_oa = 0
    candidates = has_pmc + [p for p in no_pmc if p.get("doi")]
    print(f"\n⬇️  Trying {len(candidates)} papers (PMC OA → Europe PMC → Unpaywall)...")

    for p in candidates:
        pmc_id = p.get("pmc_id", "")
        pmid   = p.get("pmid", "")
        doi    = p.get("doi", "")
        title  = p.get("title", "unknown")
        year   = p.get("year", "")
        stem   = safe_filename(title, year)

        if stem.lower() in stems or any(title.lower()[:40] in s for s in stems):
            print(f"   ⏭  Already on disk: {title[:60]}")
            skipped += 1
            p["pdf_file"] = stem + ".pdf"
            continue

        pdf_url = source = None
        if pmc_id:
            time.sleep(REQUEST_DELAY)
            pdf_url = get_oa_pdf_url(pmc_id)
            if pdf_url:
                source = "PMC-OA"
        if not pdf_url:
            time.sleep(REQUEST_DELAY)
            pdf_url = get_europepmc_pdf_url(pmid=pmid, pmc_id=pmc_id, doi=doi)
            if pdf_url:
                source = "EuropePMC"
        if not pdf_url and doi:
            time.sleep(REQUEST_DELAY)
            pdf_url = get_unpaywall_pdf_url(doi)
            if pdf_url:
                source = "Unpaywall"

        if not pdf_url:
            no_oa += 1
            p["pdf_available"] = False
            continue

        dest = PDFS_DIR / (stem + ".pdf")
        print(f"   ⬇️  [{source}] {title[:50]}...")
        if download_pdf(pdf_url, dest):
            downloaded += 1
            stems.add(stem.lower())
            p.update({"pdf_file": dest.name, "pdf_available": True, "pdf_source": source})
            print(f"        ✅ saved → {dest.name[:70]}")
        else:
            print(f"        ❌ failed ({pdf_url[:80]})")
            p["pdf_available"] = False
        time.sleep(REQUEST_DELAY)

    print(f"\n{'─'*60}")
    print(f"  ✅ Downloaded : {downloaded}")
    print(f"  ⏭  Skipped   : {skipped}  (already on disk)")
    print(f"  🔒 No OA found: {no_oa}")
    save_metadata(load_existing_metadata() + papers)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiB utilities CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("explore", help="Explore metadata CSVs interactively")

    p_check = sub.add_parser("check", help="Check variables exist in HTML docs")
    p_check.add_argument("variables", nargs="+", help="Variable names to check")

    p_fetch = sub.add_parser("fetch", help="Fetch BiB papers from PubMed")
    p_fetch.add_argument("--download", action="store_true", help="Download PDFs")
    p_fetch.add_argument("--max", type=int, default=500, help="Max results per query")

    args = parser.parse_args()

    if args.cmd == "explore":
        explore_metadata()
    elif args.cmd == "check":
        check_variables_in_html(args.variables)
    elif args.cmd == "fetch":
        if NCBI_API_KEY:
            print("🔑 NCBI API key active — higher rate limit")
        run_fetch(max_results=args.max, download=args.download)
    else:
        parser.print_help()
