"""Topic research from Wikipedia (free, public; text CC BY-SA 4.0). Facts are paraphrased in the script and every
one keeps its source URL + revision id, so each narrated claim can be traced and re-checked.

Wikimedia API etiquette (https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy): a User-Agent with contact
details, maxlag=5 with a pause when the servers are lagged, one request at a time, answers cached on disk.
"""
from __future__ import annotations

import hashlib, json, math, re, time, urllib.parse, urllib.request
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

API = "https://{lang}.wikipedia.org/w/api.php"
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those which who whom what when where why how than then there here also into over under such can could may might
will would not no do does did their they them his her he she we you your our has have had more most other some any
about after before during between both each few many much one two three first second""".split())


class Fact(BaseModel):
    id: str
    text: str
    source_url: str
    title: str
    revid: int | None = None


class FactSheet(BaseModel):
    topic: str
    facts: list[Fact] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)      # {title, url, revid, license}

    def by_id(self):
        return {f.id: f for f in self.facts}


class Wiki:
    def __init__(self, contact: str, cache_dir, lang: str = "en", log=print, min_interval=1.0, api: str | None = None):
        if not contact or "@" not in contact and "http" not in contact:
            raise ValueError("Wikimedia asks for contact details in the User-Agent: set WIKI_CONTACT to your e-mail "
                             "address or a web page (https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy)")
        self.ua = f"AIExplainerMaker/1.0 ({contact}) python-urllib"
        self.lang, self.cache, self.log, self.min_interval = lang, Path(cache_dir), log, min_interval
        self.api = api or API.format(lang=lang)             # e.g. https://commons.wikimedia.org/w/api.php
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0

    def get(self, **params):
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        url = self.api + "?" + urllib.parse.urlencode(sorted(params.items()))
        key = self.cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")
        if key.exists():
            return json.loads(key.read_text())
        for attempt in range(5):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)                                 # serial and paced: one request at a time
            self._last = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": self.ua})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    d = json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    pause = int(e.headers.get("Retry-After") or 5 * (attempt + 1))
                    self.log(f"wikipedia: HTTP {e.code}, waiting {pause}s"); time.sleep(pause); continue
                raise
            if d.get("error", {}).get("code") == "maxlag":
                self.log("wikipedia: servers lagged (maxlag), waiting 5s"); time.sleep(5 * (attempt + 1)); continue
            if "error" in d:
                raise RuntimeError(f"Wikipedia API error: {d['error']}")
            key.write_text(json.dumps(d))
            return d
        raise RuntimeError("Wikipedia API: still busy after 5 tries")

    def search(self, query, n=3):
        d = self.get(action="query", list="search", srsearch=query, srlimit=str(n), srnamespace="0")
        return [h["title"] for h in d["query"]["search"]]

    def article(self, title):
        # HTML, not the plain-text extract: plain text drops superscripts, so "2×10<sup>30</sup> kg" would become
        # "2×1030 kg" — a wrong number that a fact check would then happily "confirm"
        d = self.get(action="query", prop="extracts|info|revisions", titles=title, inprop="url", rvprop="ids",
                     redirects="1")
        pg = d["query"]["pages"][0]
        if pg.get("missing"):
            raise KeyError(title)
        return {"title": pg["title"], "url": pg["fullurl"], "revid": pg["revisions"][0]["revid"],
                "text": html_to_text(pg.get("extract", ""))}


def html_to_text(h):
    """Extract HTML → paragraphs of plain text, keeping exponents (10<sup>30</sup> → 10^30) and dropping
    citation marks, tables and headings."""
    import html as H
    h = re.sub(r"(?is)<(table|style|script|figure)\b.*?</\1>", " ", h)
    h = re.sub(r"(?is)<h\d\b.*?</h\d>", "\n", h)
    h = re.sub(r"(?is)<sup[^>]*>\s*\[[^<]*\]\s*</sup>", "", h)            # [1] citation markers
    h = re.sub(r"(?is)<sup[^>]*>(.*?)</sup>", lambda m: "^" + re.sub(r"<[^>]+>", "", m.group(1)).strip(), h)
    h = re.sub(r"(?is)<sub[^>]*>(.*?)</sub>", lambda m: re.sub(r"<[^>]+>", "", m.group(1)).strip(), h)
    h = re.sub(r"(?i)</p>|<br\s*/?>|</li>", "\n", h)
    t = H.unescape(re.sub(r"<[^>]+>", "", h))
    t = t.replace("\u2212", "-").replace("\xa0", " ")                     # Unicode minus, no-break space
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n"))


def sentences(text):
    """Plain article text → sentences (headings, lists and very short or very long lines dropped)."""
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < 40 or not para.endswith((".", "!", "?")):
            continue                                             # headings, captions, list fragments
        para = re.sub(r"\s*\((?:[^()\d]*|[^()]*(?:pronounced|listen|IPA)[^()]*)\)", "", para)  # pronunciations, alt. names
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para):
            s = s.strip()
            if 40 <= len(s) <= 300 and not re.search(r"\b(?:See also|References|Further reading)\b", s):
                out.append(s)
    return out


def _words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


def relevance(sentence, topic_words, df, n_docs, position, rank=0):
    """Topic-word overlap (idf-weighted: rare topic words such as "form" count more than "black"), a small bonus
    for early sentences (the lead summarises) and numbers, scaled down for lower-ranked search results."""
    ws = Counter(_words(sentence))
    score = sum(math.log(1 + n_docs / (1 + df[w])) for w in ws if w in topic_words)
    has_number = bool(re.search(r"\d", sentence))
    return (score + 0.2 * has_number + 0.5 / (1 + position / 10)) / (1 + 0.35 * rank)


def research(topic: str, wiki: Wiki, n_articles=3, max_facts=40, extra_titles=()):
    """Search Wikipedia for the topic, read the best articles, keep the most relevant sentences as numbered facts."""
    titles = list(dict.fromkeys(list(extra_titles) + wiki.search(topic, n_articles)))[:n_articles + len(extra_titles)]
    arts = []
    for t in titles:
        try:
            arts.append(wiki.article(t))
        except KeyError:
            wiki.log(f"wikipedia: '{t}' not found")
    sheet = FactSheet(topic=topic, sources=[{"title": a["title"], "url": a["url"], "revid": a["revid"],
                                             "license": "CC BY-SA 4.0 (text); facts paraphrased"} for a in arts])
    cands = []
    for rank, a in enumerate(arts):
        a["rank"] = rank
        for i, s in enumerate(sentences(a["text"])):
            cands.append((a, i, s))
    if not cands:
        return sheet
    df = Counter(w for _, _, s in cands for w in set(_words(s)))
    tw = set(_words(topic))                                     # the question's words (not every article title)
    scored = sorted(cands, key=lambda c: -relevance(c[2], tw, df, len(cands), c[1], c[0]["rank"]))
    seen = set()
    for a, i, s in scored:
        key = " ".join(sorted(set(_words(s))))[:120]
        if key in seen:
            continue
        seen.add(key)
        sheet.facts.append(Fact(id=f"F{len(sheet.facts) + 1}", text=s, source_url=a["url"], title=a["title"],
                                revid=a["revid"]))
        if len(sheet.facts) >= max_facts:
            break
    return sheet
