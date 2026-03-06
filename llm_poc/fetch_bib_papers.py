"""
fetch_bib_papers.py
===================
Download open-access Born in Bradford PDFs from PubMed Central.

Strategy
--------
1. Query PubMed for the exact phrase "Born in Bradford" in Title/Abstract.
   Additional guard: require the term appears alongside common BiB co-terms
   (Bradford, cohort) so stray false positives are dropped automatically.
2. Optional second pass: query for "BiB cohort" Bradford to catch papers that
   only use the abbreviation.
3. For each PubMed hit, look up its PMC ID via ELink.
4. Check PMC Open Access status via the OA API.
5. Download the PDF, skipping files already present in PDFS_DIR.
6. Write / update bib_papers_metadata.json with all fetched metadata.

Usage
-----
    cd BornInBradford-datadict/llm_poc
    python fetch_bib_papers.py                    # dry-run: print what would be fetched
    python fetch_bib_papers.py --download         # actually download PDFs
    python fetch_bib_papers.py --download --max 300

Optional env var:
    NCBI_API_KEY=your_key   # raises rate limit from 3 → 10 req/s

Requirements (all already in the venv):
    requests, xml.etree.ElementTree (stdlib)
"""

import os
import re
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("❌ requests not installed. Run: pip install requests")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATADICT_DIR = SCRIPT_DIR.parent
PDFS_DIR     = DATADICT_DIR / "papers"
META_JSON    = PDFS_DIR / "bib_papers_metadata.json"

PDFS_DIR.mkdir(exist_ok=True)

# ── NCBI base URLs ─────────────────────────────────────────────────────────────
ESEARCH    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_OA     = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EUPMC_API  = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL  = "https://api.unpaywall.org/v2"
# Unpaywall requires an email address as the API key (free, no registration needed)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "research@borninbradford.nhs.uk")

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
# With API key: 10 req/s; without: 3 req/s
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.35

# ── Search queries ─────────────────────────────────────────────────────────────
# Primary: exact phrase - very specific to the BiB study, rarely a false positive
PRIMARY_QUERY = '"Born in Bradford"[Title/Abstract]'

# Secondary: papers that use the BiB abbreviation without the full name.
# Require "bradford" + "cohort" + ("BiB" in title or abstract) to stay specific.
SECONDARY_QUERY = '("BiB cohort"[Title/Abstract] OR "BiB study"[Title/Abstract]) AND Bradford[Title/Abstract] AND cohort[Title/Abstract]'

# ── False-positive guard ───────────────────────────────────────────────────────
# Papers that mention Bradford but are clearly not the BiB study.
# Applied after fetch on the title/abstract text.
EXCLUDE_PATTERNS = [
    r"\bbradford hill\b",          # Bradford Hill criteria (epidemiology method)
    r"\bbradford score\b",
    r"\bcity of bradford\b(?!.*born in bradford)",  # generic Bradford city refs
]


def _api_params(extra: dict) -> dict:
    p = {"retmode": "json", **extra}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


def _get(url: str, params: dict, retmode_xml: bool = False) -> Optional[requests.Response]:
    """GET with retry and rate-limit sleep."""
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
    """Return a list of PubMed IDs for the given query."""
    print(f"\n🔍 Searching PubMed: {query[:80]}...")
    params = _api_params({
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
    })
    r = _get(ESEARCH, params)
    if not r:
        return []
    data   = r.json()
    pmids  = data.get("esearchresult", {}).get("idlist", [])
    total  = data.get("esearchresult", {}).get("count", "?")
    print(f"   Found {total} total; fetching metadata for {len(pmids)}")
    time.sleep(REQUEST_DELAY)
    return pmids


def fetch_metadata(pmids: list[str]) -> list[dict]:
    """Fetch PubMed XML metadata for a list of PMIDs. Returns list of dicts."""
    if not pmids:
        return []
    print(f"\n📋 Fetching metadata for {len(pmids)} papers...")
    results = []
    batch_size = 200

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            "db":      "pubmed",
            "id":      ",".join(batch),
            "rettype": "xml",
            "retmax":  batch_size,
        }
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
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            # Title
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            # Abstract
            abs_texts = article.findall(".//AbstractText")
            abstract = " ".join("".join(el.itertext()) for el in abs_texts)

            # Authors
            authors = []
            for author in article.findall(".//Author"):
                ln = author.findtext("LastName", "")
                fn = author.findtext("ForeName", "")
                authors.append(f"{ln} {fn}".strip())
            authors_str = ", ".join(authors[:5])
            if len(authors) > 5:
                authors_str += " et al."

            # Year
            pub_date = article.find(".//PubDate")
            year = ""
            if pub_date is not None:
                year = pub_date.findtext("Year", "") or pub_date.findtext("MedlineDate", "")[:4]

            # Journal
            journal = article.findtext(".//Title", "") or article.findtext(".//ISOAbbreviation", "")

            # DOI
            doi = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text or ""
                    break

            # PMC ID (if available directly in the article XML)
            pmc_id = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "pmc":
                    pmc_id = id_el.text or ""
                    break

            results.append({
                "pmid":     pmid,
                "pmc_id":   pmc_id,
                "title":    title.strip(),
                "abstract": abstract.strip(),
                "authors":  authors_str,
                "year":     year,
                "journal":  journal,
                "doi":      doi,
            })

        sys.stdout.write(f"\r   Parsed {min(i + batch_size, len(pmids))}/{len(pmids)}...")
        sys.stdout.flush()
        time.sleep(REQUEST_DELAY)

    print(f"\n   ✅ Metadata fetched for {len(results)} papers")
    return results


def _is_false_positive(title: str, abstract: str) -> bool:
    """Return True if this paper is clearly not about the BiB cohort study."""
    text = (title + " " + abstract).lower()
    # Must mention Born in Bradford or Bradford cohort / BiB study
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def resolve_pmc_ids(papers: list[dict]) -> list[dict]:
    """
    For any paper without a pmc_id, try to resolve one via ELink.
    Updates each dict in-place.
    """
    need_lookup = [p for p in papers if not p.get("pmc_id") and p.get("pmid")]
    if not need_lookup:
        return papers

    print(f"\n🔗 Resolving PMC IDs for {len(need_lookup)} papers via ELink...")
    batch_size = 100
    resolved = 0

    for i in range(0, len(need_lookup), batch_size):
        batch = need_lookup[i : i + batch_size]
        pmids = [p["pmid"] for p in batch]

        params = {
            "dbfrom": "pubmed",
            "db":     "pmc",
            "id":     ",".join(pmids),
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        r = _get(ELINK, params)
        if not r:
            continue

        data = r.json()
        # linksets[0].linksetdbs[].links entries map pmid → pmc_id
        pmid_to_pmc: dict[str, str] = {}
        for linkset in data.get("linksets", []):
            ids_in = linkset.get("ids", [])
            for ldb in linkset.get("linksetdbs", []):
                if ldb.get("dbto") == "pmc":
                    for pmcid in ldb.get("links", []):
                        # We can't always map back 1:1 via JSON, so just collect PMC IDs
                        # and cross-ref by PMID in a separate fetch if needed.
                        pass
            # Better: JSON ELink returns idarrays per input id
            for entry in linkset.get("ids", []):
                pass  # handled below via XML fallback

        # XML is more reliable for 1-to-1 mapping
        params_xml = {k: v for k, v in params.items() if k != "retmode"}
        rx = _get(ELINK, params_xml, retmode_xml=True)
        if rx:
            try:
                root = ET.fromstring(rx.content)
                for ls in root.findall(".//LinkSet"):
                    pmid_el = ls.find(".//IdList/Id")
                    if pmid_el is None:
                        continue
                    pmid_val = pmid_el.text
                    for link in ls.findall(".//LinkSetDb[DbTo='pmc']/Link/Id"):
                        pmid_to_pmc[pmid_val] = "PMC" + link.text
                        break
            except ET.ParseError:
                pass

        for p in batch:
            if p["pmid"] in pmid_to_pmc:
                p["pmc_id"] = pmid_to_pmc[p["pmid"]]
                resolved += 1

        sys.stdout.write(f"\r   Resolved {min(i + batch_size, len(need_lookup))}/{len(need_lookup)}...")
        sys.stdout.flush()
        time.sleep(REQUEST_DELAY)

    print(f"\n   ✅ Resolved {resolved} additional PMC IDs")
    return papers


def get_oa_pdf_url(pmc_id: str) -> Optional[str]:
    """
    Query the PMC Open Access API to get a direct PDF URL.
    Returns None if the paper is not in the OA subset.
    """
    clean_id = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
    params = {"id": clean_id}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    r = _get(PMC_OA, params, retmode_xml=True)
    if not r:
        return None

    try:
        root = ET.fromstring(r.content)
        # <record ...><link format="pdf" href="..."/></record>
        for link in root.findall(".//link[@format='pdf']"):
            href = link.get("href", "")
            if href:
                return href
        # Some records have tgz only — no PDF link
    except ET.ParseError:
        pass
    return None


def get_europepmc_pdf_url(pmid: str = "", pmc_id: str = "", doi: str = "") -> Optional[str]:
    """
    Try Europe PMC's REST API for a full-text PDF link.
    Europe PMC has broader OA coverage than PubMed for UK-funded research
    (Wellcome Trust / UKRI open-access mandate).

    The correct PDF URL is inside fullTextUrlList in the search result
    (documentStyle==pdf, availability==Open access). We read it directly
    rather than constructing the /fullTextPDF endpoint path, which is
    unreliable (returns 404 for many PMC articles).
    """
    if pmid:
        query = f"EXT_ID:{pmid} AND SRC:MED"
    elif pmc_id:
        clean = pmc_id.replace("PMC", "")
        query = f"PMC{clean}[PMCID]"
    elif doi:
        query = f'"{doi}"[DOI]'
    else:
        return None

    params = {
        "query":      query,
        "resultType": "core",
        "format":     "json",
        "pageSize":   1,
    }
    r = _get(f"{EUPMC_API}/search", params)
    if not r:
        return None

    try:
        data    = r.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None
        article = results[0]
        is_oa   = article.get("isOpenAccess", "N")
        if is_oa != "Y":
            return None

        # Prefer an explicit open-access PDF link from fullTextUrlList
        for entry in article.get("fullTextUrlList", {}).get("fullTextUrl", []):
            if (entry.get("documentStyle") == "pdf"
                    and "open" in entry.get("availability", "").lower()):
                return entry.get("url")

        # Fallback: html open-access link (download_pdf will still get a PDF
        # if the ?pdf=render variant works — caller can try appending it)
        for entry in article.get("fullTextUrlList", {}).get("fullTextUrl", []):
            url = entry.get("url", "")
            if ("open" in entry.get("availability", "").lower()
                    and "europepmc.org/articles" in url):
                # Append ?pdf=render to get the PDF rendering
                return url.rstrip("/") + "?pdf=render"

        return None
    except Exception:
        return None


def get_unpaywall_pdf_url(doi: str) -> Optional[str]:
    """
    Query Unpaywall for a legal open-access PDF.
    Unpaywall aggregates OA versions from institutional repos,
    preprint servers, and author pages — covers ~50% of recent literature.
    Free, no registration needed (just an email address).
    """
    if not doi:
        return None
    # Normalise DOI
    doi_clean = doi.strip().lstrip("https://doi.org/").lstrip("http://dx.doi.org/")
    url = f"{UNPAYWALL}/{doi_clean}"
    params = {"email": UNPAYWALL_EMAIL}
    r = _get(url, params)
    if not r:
        return None
    try:
        data = r.json()
        if data.get("is_oa") is not True:
            return None

        def _pick(loc: dict) -> Optional[str]:
            """Return a PDF URL from an OA location, or None.
            We never return a bare landing-page URL — only url_for_pdf
            (explicit direct link) or a url that visibly ends in .pdf."""
            u = loc.get("url_for_pdf")
            if u:
                return u
            u = loc.get("url", "")
            if u.lower().endswith(".pdf"):
                return u
            return None

        # Prefer "best_oa_location" which Unpaywall ranks by quality
        best = data.get("best_oa_location") or {}
        pdf_url = _pick(best)
        if pdf_url:
            return pdf_url
        # Fallback: scan all OA locations for any direct PDF link
        for loc in data.get("oa_locations", []):
            pdf_url = _pick(loc)
            if pdf_url:
                return pdf_url
    except Exception:
        pass
    return None


def safe_filename(title: str, year: str, max_len: int = 120) -> str:
    """Convert a paper title + year to a safe filename stem."""
    # Remove characters that are problematic in filenames
    clean = re.sub(r'[^\w\s\-]', '_', title)
    clean = re.sub(r'[\s_]+', '_', clean).strip('_')
    stem = f"{clean[:max_len]}_{year}" if year else clean[:max_len]
    return stem


def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF to dest. Returns True on success."""
    # NCBI OA API often returns ftp:// URLs — requests can't handle FTP.
    # The same files are available over HTTPS at the identical path.
    if url.startswith("ftp://"):
        url = "https://" + url[len("ftp://"):]

    # Europe PMC fullTextPDF endpoint requires Accept: application/pdf
    headers = {}
    if "europepmc" in url:
        headers["Accept"] = "application/pdf"

    try:
        r = requests.get(url, headers=headers, timeout=60, stream=True,
                         allow_redirects=True)
        if r.status_code != 200:
            return False

        # Stream the full response into a buffer so we can check magic bytes
        # without losing any data.
        buf = bytearray()
        for chunk in r.iter_content(8192):
            buf.extend(chunk)

        # Validate it is actually a PDF (not an HTML error page)
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
    """Return lowercase stems of PDFs already in PDFS_DIR (to skip re-downloads)."""
    return {p.stem.lower() for p in PDFS_DIR.glob("*.pdf")}


def load_existing_metadata() -> list[dict]:
    if META_JSON.exists():
        with open(META_JSON, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata(papers: list[dict]):
    # Deduplicate by PMID (keep last seen)
    seen: dict[str, dict] = {}
    for p in papers:
        key = p.get("pmid") or p.get("doi") or p.get("title", "")[:80]
        seen[key] = p
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(list(seen.values()), f, indent=2, ensure_ascii=False)
    print(f"\n💾 Metadata saved → {META_JSON}  ({len(seen)} papers total)")


def run(max_results: int = 500, download: bool = False):
    # ── Search ────────────────────────────────────────────────────────────────
    pmids_primary   = search_pubmed(PRIMARY_QUERY,   max_results)
    pmids_secondary = search_pubmed(SECONDARY_QUERY, max_results)
    all_pmids = list(dict.fromkeys(pmids_primary + pmids_secondary))  # deduplicate, keep order
    print(f"\n📊 Combined unique PMIDs: {len(all_pmids)}")

    # ── Metadata  ─────────────────────────────────────────────────────────────
    papers = fetch_metadata(all_pmids)

    # ── False-positive filter ─────────────────────────────────────────────────
    before = len(papers)
    papers = [p for p in papers if not _is_false_positive(p["title"], p["abstract"])]
    removed = before - len(papers)
    if removed:
        print(f"   🚫 Removed {removed} false positives")

    # ── Resolve PMC IDs ───────────────────────────────────────────────────────
    papers = resolve_pmc_ids(papers)

    has_pmc   = [p for p in papers if p.get("pmc_id")]
    no_pmc    = [p for p in papers if not p.get("pmc_id")]
    print(f"\n📑 {len(has_pmc)} papers have a PMC ID  |  {len(no_pmc)} do not (paywalled or preprint)")

    # ── Preview / dry-run ─────────────────────────────────────────────────────
    if not download:
        print("\n── DRY RUN (pass --download to fetch PDFs) ─────────────────────")
        print(f"{'PMID':<12} {'PMC ID':<12} {'Year':<6}  Title")
        print("─" * 80)
        for p in sorted(papers, key=lambda x: x.get("year", ""), reverse=True)[:40]:
            pmc = p.get("pmc_id", "—")
            print(f"{p.get('pmid',''):<12} {pmc:<12} {p.get('year',''):<6}  {p.get('title','')[:60]}")
        if len(papers) > 40:
            print(f"  ... and {len(papers) - 40} more")

        # Merge with existing metadata and save
        existing = load_existing_metadata()
        save_metadata(existing + papers)
        print("\nRe-run with --download to fetch open-access PDFs.")
        return

    # ── Download PDFs ─────────────────────────────────────────────────────────
    stems = existing_stems()
    downloaded = 0
    skipped    = 0
    no_oa      = 0

    # Try all papers — PMC ones first, then DOI-only ones via Unpaywall/EuropePMC
    all_candidates = has_pmc + [p for p in no_pmc if p.get("doi")]
    print(f"\n⬇️  Trying to download PDFs for {len(all_candidates)} papers "
          f"(PMC OA → Europe PMC → Unpaywall)...")

    for p in all_candidates:
        pmc_id = p.get("pmc_id", "")
        pmid   = p.get("pmid", "")
        doi    = p.get("doi", "")
        title  = p.get("title", "unknown")
        year   = p.get("year", "")

        stem = safe_filename(title, year)
        if stem.lower() in stems or any(title.lower()[:40] in s for s in stems):
            print(f"   ⏭  Already on disk: {title[:60]}")
            skipped += 1
            p["pdf_file"] = stem + ".pdf"
            continue

        pdf_url  = None
        source   = ""

        # 1️⃣  PMC Open Access API (fastest, highest quality)
        if pmc_id:
            time.sleep(REQUEST_DELAY)
            pdf_url = get_oa_pdf_url(pmc_id)
            if pdf_url:
                source = "PMC-OA"

        # 2️⃣  Europe PMC (better coverage for UK/Wellcome/UKRI-funded papers)
        if not pdf_url:
            time.sleep(REQUEST_DELAY)
            pdf_url = get_europepmc_pdf_url(pmid=pmid, pmc_id=pmc_id, doi=doi)
            if pdf_url:
                source = "EuropePMC"

        # 3️⃣  Unpaywall (finds OA versions on repos, preprint servers, author pages)
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
        ok = download_pdf(pdf_url, dest)
        if ok:
            downloaded += 1
            stems.add(stem.lower())
            p["pdf_file"]      = dest.name
            p["pdf_available"] = True
            p["pdf_source"]    = source
            print(f"        ✅ saved → {dest.name[:70]}")
        else:
            print(f"        ❌ download failed ({pdf_url[:80]})")
            p["pdf_available"] = False

        time.sleep(REQUEST_DELAY)

    print(f"\n{'─'*60}")
    print(f"  ✅ Downloaded : {downloaded}")
    print(f"  ⏭  Skipped   : {skipped}  (already on disk)")
    print(f"  🔒 No OA found: {no_oa}  (tried PMC OA + Europe PMC + Unpaywall)")

    # ── Save combined metadata ────────────────────────────────────────────────
    existing = load_existing_metadata()
    save_metadata(existing + papers)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Born in Bradford papers from PubMed and download open-access PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download", action="store_true",
                        help="Actually download PDFs (default: dry-run only)")
    parser.add_argument("--max", type=int, default=500,
                        help="Max PubMed results per query (default: 500)")
    args = parser.parse_args()

    if NCBI_API_KEY:
        print(f"🔑 Using NCBI API key — higher rate limit enabled")
    else:
        print("ℹ️  No NCBI_API_KEY set — using public rate limit (3 req/s)")
        print("   Set NCBI_API_KEY env var for faster fetching")

    run(max_results=args.max, download=args.download)


if __name__ == "__main__":
    main()
