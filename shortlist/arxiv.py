"""
arXiv client.

- fetch_announced: today's announcement from the RSS feed (new papers and
  cross-lists in the given categories). One fast request; the main source.
- fetch_recent: the last few days from the search API, used when the RSS feed
  is empty (weekends and holidays). arXiv rate-limits this API (one request per
  3 seconds, and 429 responses when busy), so it is the fallback.
- fetch_by_ids: look up papers the user pasted.

In a hosted visit every request goes through the platform's proxy, and the
manifest declares the arXiv hosts (workshop-app.json, network.publicHttpsDomains).
"""

import base64
import html
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from http.client import HTTPSConnection
from typing import Any, Callable, Optional

from shortlist import config

RSS_URL = "https://rss.arxiv.org/rss/"
API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 500
MAX_PAGES = 6
REQUEST_GAP_SECONDS = 3.0
TIMEOUT_SECONDS = 60
USER_AGENT = "shortlist/0.2 (paper recommendation class project)"

_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

# Categories offered in the app, with display names
CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.IR": "Information Retrieval",
    "cs.HC": "Human-Computer Interaction",
    "cs.CR": "Cryptography and Security",
    "cs.DC": "Distributed Computing",
    "cs.DB": "Databases",
    "cs.DS": "Data Structures and Algorithms",
    "cs.GR": "Graphics",
    "cs.GT": "Game Theory",
    "cs.MA": "Multiagent Systems",
    "cs.PL": "Programming Languages",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SD": "Sound",
    "cs.AR": "Hardware Architecture",
    "cs.CE": "Computational Engineering",
    "cs.CY": "Computers and Society",
    "cs.IT": "Information Theory",
    "cs.NI": "Networking",
    "cs.MM": "Multimedia",
    "stat.ML": "Statistics: Machine Learning",
    "stat.ME": "Statistics: Methodology",
    "stat.AP": "Statistics: Applications",
    "eess.IV": "Image and Video Processing",
    "eess.AS": "Audio and Speech Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "q-bio.BM": "Biomolecules",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.CB": "Cell Behavior",
    "q-bio.TO": "Tissues and Organs",
    "physics.chem-ph": "Chemical Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.med-ph": "Medical Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.optics": "Optics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "quant-ph": "Quantum Physics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.CO": "Cosmology",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "hep-th": "High Energy Physics: Theory",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "math.OC": "Optimization and Control",
    "math.ST": "Statistics Theory",
    "math.PR": "Probability",
    "math.NA": "Numerical Analysis",
    "math.CO": "Combinatorics",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "q-fin.CP": "Computational Finance",
    "q-fin.ST": "Statistical Finance",
}


def parse_ids(text: str) -> list[str]:
    """arXiv IDs found in free text (bare IDs or abs/pdf links), versions stripped, in order."""
    return list(dict.fromkeys(match.group(1) for match in _ID_RE.finditer(text)))


class ArxivError(Exception):
    """arXiv could not be reached or is rate-limiting us."""


class _HTTPStatusError(Exception):
    def __init__(self, status: int, retry_after: Optional[str]):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _proxy() -> Optional[urllib.parse.SplitResult]:
    """The platform proxy in a hosted visit; None when running locally."""
    if not config.HOSTED:
        return None
    value = next((os.environ[k] for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy") if os.environ.get(k)), "")
    if not value:
        raise ArxivError("This hosted visit has no proxy for reaching arXiv.")
    return urllib.parse.urlsplit(value)


def _fetch(url: str, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """GET an https URL, following redirects. Hosted visits always tunnel through the proxy; NO_PROXY never bypasses it."""
    for _ in range(4):
        target = urllib.parse.urlsplit(url)
        proxy = _proxy()
        if proxy:
            connection = HTTPSConnection(proxy.hostname, proxy.port or 80, timeout=timeout)
            headers = {}
            if proxy.username:
                login = f"{urllib.parse.unquote(proxy.username)}:{urllib.parse.unquote(proxy.password or '')}"
                headers["Proxy-Authorization"] = "Basic " + base64.b64encode(login.encode()).decode()
            connection.set_tunnel(target.hostname, target.port or 443, headers=headers)
        else:
            connection = HTTPSConnection(target.hostname, target.port or 443, timeout=timeout)
        try:
            path = (target.path or "/") + (f"?{target.query}" if target.query else "")
            connection.request("GET", path, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
        if response.status in (301, 302, 303, 307, 308) and response.getheader("Location"):
            url = urllib.parse.urljoin(url, response.getheader("Location"))
            continue
        if response.status != 200:
            raise _HTTPStatusError(response.status, response.getheader("Retry-After"))
        return body
    raise ArxivError(f"Too many redirects from {url}")


def _get(url: str, attempts: int = 3) -> ET.Element:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return ET.fromstring(_fetch(url))
        except ArxivError:
            raise
        except _HTTPStatusError as exc:
            last_error = exc
            if exc.status == 429:
                if attempt + 1 < attempts:
                    time.sleep(min(float(exc.retry_after or 20), 60))
                continue
            time.sleep(REQUEST_GAP_SECONDS * (attempt + 1))
        except Exception as exc:  # timeouts and connection errors
            last_error = exc
            time.sleep(REQUEST_GAP_SECONDS * (attempt + 1))
    if isinstance(last_error, _HTTPStatusError) and last_error.status == 429:
        raise ArxivError("arXiv is rate-limiting requests right now. Try again in a few minutes.")
    raise ArxivError(f"Could not reach arXiv: {last_error}")


def _request(params: dict[str, Any], attempts: int = 3) -> ET.Element:
    return _get(f"{API_URL}?{urllib.parse.urlencode(params)}", attempts)


def _parse_entry(entry: ET.Element) -> Optional[dict[str, Any]]:
    id_text = entry.findtext("a:id", default="", namespaces=_NS)
    match = _ID_RE.search(id_text)
    title = " ".join(entry.findtext("a:title", default="", namespaces=_NS).split())
    if not match or not title or title == "Error":
        return None

    paper_id = match.group(1)
    authors = [
        " ".join(author.findtext("a:name", default="", namespaces=_NS).split())
        for author in entry.findall("a:author", _NS)
    ]
    primary = entry.find("arxiv:primary_category", _NS)
    categories = [c.get("term") for c in entry.findall("a:category", _NS) if c.get("term")]
    if primary is not None and primary.get("term") in categories:
        categories.remove(primary.get("term"))
        categories.insert(0, primary.get("term"))

    return {
        "id": paper_id,
        "title": title,
        "authors": ", ".join(a for a in authors if a),
        "abstract": " ".join(entry.findtext("a:summary", default="", namespaces=_NS).split()),
        "categories": " ".join(categories),
        "published": entry.findtext("a:published", default="", namespaces=_NS)[:10],
        "url": f"https://arxiv.org/abs/{paper_id}",
    }


def fetch_announced(categories: list[str]) -> list[dict[str, Any]]:
    """New papers and cross-lists in today's arXiv announcement for these categories."""
    if not categories:
        return []
    root = _get(RSS_URL + "+".join(categories))
    papers = []
    for item in root.findall("./channel/item"):
        if item.findtext("arxiv:announce_type", default="", namespaces=_NS) not in ("new", "cross"):
            continue
        match = _ID_RE.search(item.findtext("link", default=""))
        title = " ".join(item.findtext("title", default="").split())
        if not match or not title:
            continue
        description = item.findtext("description", default="")
        abstract = description.split("Abstract:", 1)[1] if "Abstract:" in description else ""
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", default="")).date().isoformat()
        except (TypeError, ValueError):
            published = ""
        papers.append({
            "id": match.group(1),
            "title": title,
            "authors": " ".join(item.findtext("dc:creator", default="", namespaces=_NS).split()),
            "abstract": " ".join(abstract.split()),
            "categories": " ".join(c.text for c in item.findall("category") if c.text),
            "published": published,
            "url": f"https://arxiv.org/abs/{match.group(1)}",
        })
    return papers


def fetch_recent(
    categories: list[str],
    since: str,
    on_page: Optional[Callable[[int], None]] = None,
) -> list[dict[str, Any]]:
    """Papers in any of `categories` published on or after `since` (YYYY-MM-DD)."""
    if not categories:
        return []

    query = " OR ".join(f"cat:{c}" for c in categories)
    papers: dict[str, dict[str, Any]] = {}
    for page in range(MAX_PAGES):
        if page:
            time.sleep(REQUEST_GAP_SECONDS)
        root = _request({
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": page * PAGE_SIZE,
            "max_results": PAGE_SIZE,
        })
        entries = [p for p in (_parse_entry(e) for e in root.findall("a:entry", _NS)) if p]
        reached_end = len(entries) < PAGE_SIZE
        for paper in entries:
            if paper["published"] >= since:
                papers[paper["id"]] = paper
        if on_page:
            on_page(len(papers))
        if reached_end or (entries and entries[-1]["published"] < since):
            break
    return list(papers.values())


def _fetch_abs_page(paper_id: str) -> Optional[dict[str, Any]]:
    """Title and abstract from a paper's arxiv.org page (served separately from the API)."""
    page = _fetch(f"https://arxiv.org/abs/{paper_id}", timeout=20).decode("utf-8", errors="replace")

    def meta(name: str) -> str:
        match = re.search(rf'<meta name="{name}" content="([^"]*)"', page)
        return " ".join(html.unescape(match.group(1)).split()) if match else ""

    title = meta("citation_title")
    return {"id": paper_id, "title": title, "abstract": meta("citation_abstract")} if title else None


def fetch_by_ids(ids: list[str], max_pages: int = 30) -> list[dict[str, Any]]:
    """Look up papers by arXiv ID.

    A few IDs are read from their arxiv.org pages (fast, not rate-limited like the
    API); many go through one API request, falling back to the pages if it's busy.
    """
    if len(ids) > 10:
        try:
            root = _request({"id_list": ",".join(ids[:100]), "max_results": min(len(ids), 100)}, attempts=1)
            return [p for p in (_parse_entry(e) for e in root.findall("a:entry", _NS)) if p]
        except ArxivError:
            pass

    papers = []
    for i, paper_id in enumerate(ids[:max_pages]):
        if i:
            time.sleep(1)
        try:
            paper = _fetch_abs_page(paper_id)
        except Exception:
            continue
        if paper:
            papers.append(paper)
    return papers
