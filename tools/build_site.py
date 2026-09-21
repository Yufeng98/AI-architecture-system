#!/usr/bin/env python3
"""Build the static docs site from README.md.

Maps the three README heading levels onto the three panes of the site:

    ##   module      -> top navigation bar
    ###  section     -> left sidebar (one page each)
    #### subsection  -> right "On this page" pane

Body content written beneath a `###` or `####` heading is rendered too:
paragraphs, markdown tables (with an optional `Table: caption` line above them),
bullet lists, blockquotes, and figures written as `![Caption](figures/name.svg)`.
Figures resolve against tools/assets/figures/, which is copied into the build.
A standalone `---` is treated as a module separator and carries no content.

Citations are written `[@bibkey]` against tools/refs.bib. Each section page
numbers its own citations in order of first appearance and grows a reference
list at the bottom.

Stdlib only. Usage: python3 tools/build_site.py
"""
from __future__ import annotations

import html
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ASSET_SRC = Path(__file__).resolve().parent / "assets"
BIB_SRC = Path(__file__).resolve().parent / "refs.bib"
OUT = ROOT / "docs"

SITE_TITLE = "AI Architecture &amp; Systems"

# --------------------------------------------------------------------------- parsing

REF_DEF = re.compile(r'^\[([^\]]+)\]:\s*(\S+)(?:\s+"([^"]*)")?\s*$')
H1 = re.compile(r"^# (.+)$")
H2 = re.compile(r"^## (\d+)\.\s+(.+)$")
H3 = re.compile(r"^### (\d+\.\d+)\s+(.+)$")
H4 = re.compile(r"^#### (\d+\.\d+\.\d+)\s+(.+)$")
SUBNOTE = re.compile(r"^<sub>(.*)</sub>\s*$")
ANCHOR = re.compile(r'^<a id="([^"]+)"></a>\s*$')
QUOTE = re.compile(r"^>\s+(.+)$")


class BuildError(Exception):
    pass


@dataclass
class Sub:
    num: str
    raw: str
    body: list[str] = field(default_factory=list)


@dataclass
class Section:
    num: str
    raw: str
    sources: str | None = None
    subs: list[Sub] = field(default_factory=list)
    body: list[str] = field(default_factory=list)

    @property
    def page(self) -> str:
        return self.num.replace(".", "-") + ".html"


@dataclass
class Module:
    num: str
    raw: str
    slug: str
    note: str | None = None
    sections: list[Section] = field(default_factory=list)

    @property
    def en(self) -> str:
        return self.raw.split("|")[0].strip()

    @property
    def subtitle(self) -> str:
        """The clause after the pipe, where a module title carries one."""
        parts = self.raw.split("|", 1)
        return parts[1].strip() if len(parts) > 1 else ""


@dataclass
class Front:
    title: str = ""
    intro: list[str] = field(default_factory=list)
    table: list[tuple[str, str, str]] = field(default_factory=list)


def parse(md: str) -> tuple[Front, list[Module], dict[str, tuple[str, str]]]:
    lines = md.split("\n")

    refs: dict[str, tuple[str, str]] = {}
    for ln in lines:
        m = REF_DEF.match(ln)
        if m:
            refs[m.group(1)] = (m.group(2), m.group(3) or "")

    front = Front()
    modules: list[Module] = []
    pending_anchor: str | None = None
    mod: Module | None = None
    sec: Section | None = None
    body: list[str] | None = None  # where loose lines currently accumulate

    for ln in lines:
        if REF_DEF.match(ln) or ln.startswith("<!--") or ln.strip() == "---":
            continue

        m = ANCHOR.match(ln)
        if m:
            pending_anchor = m.group(1)
            continue

        m = H2.match(ln)
        if m:
            mod = Module(num=m.group(1), raw=m.group(2), slug=pending_anchor or f"module-{m.group(1)}")
            modules.append(mod)
            sec = None
            body = None
            pending_anchor = None
            continue

        if mod is None:  # front matter
            if not ln.strip():
                continue
            m = H1.match(ln)
            if m:
                front.title = m.group(1)
                continue
            if ln.startswith("|"):
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells) or cells[0] == "Module":
                    continue
                if len(cells) >= 3:
                    front.table.append((cells[0], cells[1], cells[2]))
                continue
            front.intro.append(ln)
            continue

        m = H3.match(ln)
        if m:
            sec = Section(num=m.group(1), raw=m.group(2))
            mod.sections.append(sec)
            body = sec.body
            continue

        m = H4.match(ln)
        if m:
            if sec is None:
                raise BuildError(f"subsection {m.group(1)} has no parent section")
            sub = Sub(num=m.group(1), raw=m.group(2))
            sec.subs.append(sub)
            body = sub.body
            continue

        m = SUBNOTE.match(ln)
        if m:
            if sec is not None:
                sec.sources = m.group(1)
            continue

        m = QUOTE.match(ln)
        if m and sec is None:
            mod.note = m.group(1)
            continue

        if body is not None:
            body.append(ln)

    validate(modules, refs)
    return front, modules, refs


def validate(modules: list[Module], refs: dict[str, tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for mod in modules:
        for sec in mod.sections:
            if sec.num in seen:
                raise BuildError(f"duplicate section number {sec.num}")
            seen[sec.num] = sec.raw
            if not sec.num.startswith(mod.num + "."):
                raise BuildError(f"section {sec.num} sits under module {mod.num}")
            for sub in sec.subs:
                if sub.num in seen:
                    raise BuildError(f"duplicate subsection number {sub.num}")
                seen[sub.num] = sub.raw
                if not sub.num.startswith(sec.num + "."):
                    raise BuildError(f"subsection {sub.num} sits under section {sec.num}")

    blobs: list[str] = []
    for mod in modules:
        blobs += [mod.raw, mod.note or ""]
        for sec in mod.sections:
            blobs += [sec.raw, sec.sources or ""] + sec.body
            for s in sec.subs:
                blobs += [s.raw] + s.body

    missing = set()
    for blob in blobs:
        for _, key in REFLINK.findall(blob):
            if key not in refs:
                missing.add(key)
    if missing:
        raise BuildError("unresolved reference keys: " + ", ".join(sorted(missing)))

    bad_cites = set()
    for blob in blobs:
        for keys in CITE.findall(blob):
            for key in _cite_keys(keys):
                if key not in BIB:
                    bad_cites.add(key)
    if bad_cites:
        raise BuildError("citation keys absent from refs.bib: " + ", ".join(sorted(bad_cites)))

    for blob in blobs:
        for _, src in FIGURE_INLINE.findall(blob):
            if src.startswith("figures/") and not (ASSET_SRC / src).is_file():
                raise BuildError(f"figure not found: tools/assets/{src}")


# --------------------------------------------------------------------------- bibliography


def _bib_split(raw: str) -> dict[str, str]:
    """Split a .bib file into {key: body}, where body is everything inside the braces."""
    entries: dict[str, str] = {}
    i = 0
    while True:
        at = raw.find("@", i)
        if at < 0:
            break
        brace = raw.find("{", at)
        if brace < 0:
            break
        depth, j = 0, brace
        while j < len(raw):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        inner = raw[brace + 1 : j]
        key = inner.split(",", 1)[0].strip()
        if key:
            entries[key] = inner
        i = j + 1
    return entries


FIELD_HEAD = re.compile(r"\s*([A-Za-z]+)\s*=\s*")


def _bib_fields(inner: str) -> dict[str, str]:
    _, _, rest = inner.partition(",")
    fields: dict[str, str] = {}
    i = 0
    while i < len(rest):
        m = FIELD_HEAD.match(rest, i)
        if not m:
            break
        name, i = m.group(1).lower(), m.end()
        if i < len(rest) and rest[i] == "{":
            depth, j = 0, i
            while j < len(rest):
                if rest[j] == "{":
                    depth += 1
                elif rest[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            val, i = rest[i + 1 : j], j + 1
        elif i < len(rest) and rest[i] == '"':
            j = rest.find('"', i + 1)
            val, i = rest[i + 1 : j], j + 1
        else:
            j = i
            while j < len(rest) and rest[j] != ",":
                j += 1
            val, i = rest[i:j], j
        fields[name] = val.strip()
        while i < len(rest) and rest[i] != ",":
            i += 1
        i += 1
    return fields


def _bib_clean(v: str) -> str:
    """LaTeX field value to plain text."""
    v = v.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    v = v.replace("--", "\u2013").replace("~", " ")
    v = re.sub(r"\\[a-zA-Z]+\s*", "", v)
    v = v.replace("{", "").replace("}", "")
    return " ".join(v.split())


def _bib_authors(v: str) -> str:
    people = [p.strip() for p in re.split(r"\s+and\s+", v) if p.strip()]
    out = []
    for p in people:
        p = _bib_clean(p)
        if "," in p:
            last, _, first = p.partition(",")
            p = f"{first.strip()} {last.strip()}".strip()
        out.append(p)
    if not out:
        return ""
    if len(out) > 3:
        return out[0] + " et al."
    if len(out) == 1:
        return out[0]
    return ", ".join(out[:-1]) + " and " + out[-1]


def load_bib(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return {k: _bib_fields(v) for k, v in _bib_split(path.read_text(encoding="utf-8")).items()}


BIB: dict[str, dict[str, str]] = load_bib(BIB_SRC)


def format_ref(key: str, num: int) -> str:
    e = BIB.get(key, {})
    bits = []
    authors = _bib_authors(e.get("author", ""))
    if authors:
        bits.append(esc(authors) + ("" if authors.endswith(".") else "."))
    title = _bib_clean(e.get("title", key))
    venue = _bib_clean(
        e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("howpublished") or ""
    )
    url = e.get("url", "").strip() or (
        f"https://doi.org/{e['doi'].strip()}" if e.get("doi") else ""
    )
    if url:
        bits.append(
            f'<a class="ext" href="{esc(url)}" target="_blank" rel="noopener">{esc(title)}</a>.'
        )
    else:
        bits.append(f"{esc(title)}.")
    tail = ", ".join(x for x in (venue, _bib_clean(e.get("year", ""))) if x)
    if tail:
        bits.append(f"{esc(tail)}.")
    note = _bib_clean(e.get("note", ""))
    if note:
        bits.append(f"{esc(note)}.")
    return (
        f'<li class="ref" id="ref-{num}">'
        f'<span class="ref-n">{num}.</span>'
        f'<span class="ref-t">{" ".join(bits)}</span></li>'
    )


class Cites:
    """Per-page citation numbering, in order of first appearance."""

    def __init__(self) -> None:
        self.order: list[str] = []

    def num(self, key: str) -> int:
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def render(self) -> str:
        if not self.order:
            return ""
        items = "".join(format_ref(k, i + 1) for i, k in enumerate(self.order))
        return f'<section class="refs"><h2 class="refs-h">References</h2><ol class="ref-list">{items}</ol></section>'


# --------------------------------------------------------------------------- inline rendering

CODE = re.compile(r"`([^`]+)`")
REFLINK = re.compile(r"\[([^\]]+)\]\[([^\]]+)\]")
URLLINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"\*([^*]+)\*")
CITE = re.compile(r"\[@([A-Za-z0-9_\-:.,@\s]+)\]")
FIGURE_INLINE = re.compile(r"!\[(.*)\]\(([^)\s]+)\)")


def _cite_keys(group: str) -> list[str]:
    return [k.strip().lstrip("@") for k in group.split(",") if k.strip().lstrip("@")]


INLINE_HTML = re.compile(r"</?(strong|em|code|b|i)>")
_HTML_TO_MD = {"strong": "**", "b": "**", "em": "*", "i": "*", "code": "`"}


def _demote_html(text: str) -> str:
    """Fold the handful of raw inline HTML tags in README.md back into markdown,
    so they survive escaping instead of showing up as literal &lt;strong&gt;."""
    return INLINE_HTML.sub(lambda m: _HTML_TO_MD[m.group(1)], text)


def inline(text: str, refs: dict[str, tuple[str, str]], cites: Cites | None = None) -> str:
    """Render README inline markdown to HTML. Escapes first, so link text is safe."""
    out = html.escape(_demote_html(text), quote=False)
    out = CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)

    def cite(m: re.Match[str]) -> str:
        keys = _cite_keys(m.group(1))
        if cites is None or not keys:
            return ""
        links = ", ".join(
            f'<a href="#ref-{n}">{n}</a>' for n in (cites.num(k) for k in keys)
        )
        return f'<sup class="cite">[{links}]</sup>'

    out = CITE.sub(cite, out)

    def ref(m: re.Match[str]) -> str:
        url, title = refs[m.group(2)]
        t = f' title="{html.escape(title, quote=True)}"' if title else ""
        return (
            f'<a class="ext" href="{html.escape(url, quote=True)}"{t}'
            f' target="_blank" rel="noopener">{m.group(1)}</a>'
        )

    out = REFLINK.sub(ref, out)
    out = URLLINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', out
    )
    out = BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    return out


def plain(text: str) -> str:
    """Markdown stripped to bare text, for nav labels, <title> and the search index."""
    out = CITE.sub("", _demote_html(text))
    out = REFLINK.sub(r"\1", out)
    out = URLLINK.sub(r"\1", out)
    out = out.replace("`", "").replace("**", "").replace("*", "")
    return out.strip()


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def anchor(num: str) -> str:
    return "s" + num.replace(".", "-")


# --------------------------------------------------------------------------- block rendering

FIGURE = re.compile(r"^!\[(.*)\]\(([^)\s]+)\)\s*$")
TABLE_ROW = re.compile(r"^\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\|[\s:|\-]+\|\s*$")
TABLE_CAP = re.compile(r"^Table:\s*(.+)$")
LIST_ITEM = re.compile(r"^[-*]\s+(.+)$")


def _list_items(group: list[str]) -> list[str] | None:
    """Bullet list, allowing an item to wrap onto indented continuation lines."""
    items: list[str] = []
    for x in group:
        m = LIST_ITEM.match(x)
        if m:
            items.append(m.group(1).strip())
        elif items and x[:1].isspace():
            items[-1] += " " + x.strip()
        else:
            return None
    return items or None


def _figure(alt: str, src: str, refs, prefix: str, cites: Cites | None) -> str:
    if src.startswith("figures/"):
        src = f"{prefix}assets/{src}"
    cap = f'<figcaption>{inline(alt, refs, cites)}</figcaption>' if alt.strip() else ""
    return (
        f'<figure class="fig"><img src="{esc(src)}" '
        f'alt="{esc(" ".join(plain(alt).split()))}" loading="lazy" />'
        f"{cap}</figure>"
    )


def _table(rows: list[str], caption: str, refs, cites: Cites | None) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[1:]
    thead = "".join(f"<th>{inline(c, refs, cites)}</th>" for c in head)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{inline(c, refs, cites)}</td>" for c in row) + "</tr>"
        for row in body
    )
    cap = f"<figcaption>{inline(caption, refs, cites)}</figcaption>" if caption else ""
    return (
        f'<figure class="tbl">{cap}<div class="table-wrap"><table>'
        f"<thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div></figure>"
    )


def render_blocks(
    lines: list[str], refs: dict[str, tuple[str, str]], prefix: str, cites: Cites | None
) -> str:
    groups: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.strip():
            cur.append(ln.rstrip())
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    out: list[str] = []
    pending_cap = ""

    def flush_cap() -> None:
        nonlocal pending_cap
        if pending_cap:
            out.append(f"<p>{inline(pending_cap, refs, cites)}</p>")
            pending_cap = ""

    for g in groups:
        m = FIGURE.match(g[0]) if len(g) == 1 else None
        if m:
            flush_cap()
            out.append(_figure(m.group(1), m.group(2), refs, prefix, cites))
            continue

        cm = TABLE_CAP.match(g[0])
        # a caption on its own line attaches to the table in the next block
        if cm and len(g) == 1:
            flush_cap()
            pending_cap = cm.group(1)
            continue

        caption, rows = pending_cap, g
        pending_cap = ""
        if cm and len(g) > 1:
            caption, rows = cm.group(1), g[1:]
        if len(rows) >= 2 and all(TABLE_ROW.match(r) for r in rows):
            data = [r for r in rows if not TABLE_SEP.match(r)]
            if data:
                out.append(_table(data, caption, refs, cites))
                continue
        if caption:  # turned out not to be a table; keep the text
            out.append(f"<p>{inline(caption, refs, cites)}</p>")

        bullets = _list_items(g)
        if bullets:
            items = "".join(f"<li>{inline(b, refs, cites)}</li>" for b in bullets)
            out.append(f"<ul>{items}</ul>")
            continue

        if all(x.lstrip().startswith(">") for x in g):
            text = " ".join(x.strip().lstrip(">").strip() for x in g)
            out.append(f'<blockquote class="callout">{inline(text, refs, cites)}</blockquote>')
            continue

        out.append(f"<p>{inline(' '.join(x.strip() for x in g), refs, cites)}</p>")
    flush_cap()
    return "".join(out)


# --------------------------------------------------------------------------- page shell

THEME_BOOT = (
    "<script>(function(){try{var t=localStorage.getItem('theme');"
    "if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>"
)


def topbar(modules: list[Module], prefix: str, active: str | None) -> str:
    tabs = []
    for mod in modules:
        cls = "tab tab--active" if mod.slug == active else "tab"
        tabs.append(
            f'<a class="{cls}" href="{prefix}{mod.slug}/index.html">'
            f'{esc(mod.en)}</a>'
        )
    return f"""<header class="topbar">
  <button class="menu-btn" id="menu-btn" aria-label="Toggle section list" aria-expanded="false">
    <span></span><span></span><span></span>
  </button>
  <a class="brand" href="{prefix}index.html">{SITE_TITLE}</a>
  <nav class="tabs">{"".join(tabs)}</nav>
  <div class="search">
    <input id="search-input" type="search" placeholder="Search    /" autocomplete="off"
           aria-label="Search the outline" />
    <div class="search-results" id="search-results" hidden></div>
  </div>
  <button class="theme-btn" id="theme-btn" aria-label="Toggle dark mode">
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>
  </button>
</header>"""


def sidebar(mod: Module, current: str | None) -> str:
    items = [
        f'<li><a class="sb-link{" sb-link--active" if current is None else ""}"'
        f' href="index.html">Overview</a></li>'
    ]
    for sec in mod.sections:
        cls = "sb-link sb-link--active" if sec.num == current else "sb-link"
        items.append(
            f'<li><a class="{cls}" href="{sec.page}">'
            f'<span class="sb-n">{sec.num}</span>'
            f'<span class="sb-t">{esc(plain(sec.raw))}</span></a></li>'
        )
    return f"""<aside class="sidebar" id="sidebar">
  <div class="sb-head">{esc(mod.en)}</div>
  <ul class="sb-list">{"".join(items)}</ul>
</aside>"""


def page(
    *,
    modules: list[Module],
    depth: int,
    title: str,
    layout: str,
    main: str,
    active: str | None = None,
    left: str = "",
    right: str = "",
) -> str:
    prefix = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>{title}</title>
{THEME_BOOT}
<link rel="stylesheet" href="{prefix}assets/site.css" />
<script>window.SITE_ROOT={json.dumps(prefix)};</script>
</head>
<body>
{topbar(modules, prefix, active)}
<div class="layout layout--{layout}">
{left}
<main class="content">
{main}
</main>
{right}
</div>
<div class="scrim" id="scrim" hidden></div>
<script src="{prefix}assets/site.js" defer></script>
</body>
</html>
"""


# --------------------------------------------------------------------------- pages


def render_home(front: Front, modules: list[Module], refs: dict[str, tuple[str, str]]) -> str:
    by_slug = {m.slug: m for m in modules}
    cards = []
    for cell, question, scope in front.table:
        m = URLLINK.match(cell.strip())
        if not m:
            continue
        href = m.group(2).lstrip("#")
        mod = by_slug.get(href)
        if mod is None:
            continue
        sub = f'<div class="card-sub">{esc(mod.subtitle)}</div>' if mod.subtitle else ""
        cards.append(
            f'<a class="card" href="{mod.slug}/index.html">'
            f'<div class="card-h">{esc(mod.en)}</div>'
            f'{sub}'
            f'<div class="card-q">{inline(question, refs)}</div>'
            f'<div class="card-s">{inline(scope, refs)}</div>'
            f'<div class="card-n">{len(mod.sections)} sections · '
            f'{sum(len(s.subs) for s in mod.sections)} topics</div></a>'
        )

    intro = "".join(f"<p>{inline(p, refs)}</p>" for p in front.intro)

    main = f"""<h1 class="home-h1">{inline(front.title, refs)}</h1>
<div class="prose">{intro}</div>
<div class="cards">{"".join(cards)}</div>"""
    return page(
        modules=modules,
        depth=0,
        title=esc(plain(front.title)),
        layout="home",
        main=main,
    )


def render_module(mod: Module, modules: list[Module], refs: dict[str, tuple[str, str]]) -> str:
    note = f'<blockquote class="note">{inline(mod.note, refs)}</blockquote>' if mod.note else ""
    sub_html = f'<span class="h1-sub">{esc(mod.subtitle)}</span>' if mod.subtitle else ""
    rows = "".join(
        f'<a class="row" href="{sec.page}">'
        f'<span class="row-n">{sec.num}</span>'
        f'<span class="row-t">{inline(sec.raw, refs)}</span>'
        f'<span class="row-c">{len(sec.subs)}</span></a>'
        for sec in mod.sections
    )
    main = f"""<nav class="crumbs"><span>{esc(mod.en)}</span></nav>
<h1>{esc(mod.en)}{sub_html}</h1>
{note}
<div class="rows">{rows}</div>"""
    return page(
        modules=modules,
        depth=1,
        title=f"{esc(mod.en)} · {SITE_TITLE}",
        layout="module",
        main=main,
        active=mod.slug,
        left=sidebar(mod, None),
    )


def render_section(
    mod: Module,
    sec: Section,
    modules: list[Module],
    refs: dict[str, tuple[str, str]],
    prev: tuple[str, str, str] | None,
    nxt: tuple[str, str, str] | None,
) -> str:
    prefix = "../"  # section pages always live one level down
    cites = Cites()

    sources = (
        f'<div class="sources"><span class="sources-k">Sources</span>'
        f'{inline(sec.sources, refs).replace("Sources: ", "", 1)}</div>'
        if sec.sources
        else ""
    )

    lead = render_blocks(sec.body, refs, prefix, cites)
    lead_html = f'<div class="sec-body">{lead}</div>' if lead else ""

    parts = []
    for sub in sec.subs:
        parts.append(
            f'<h2 class="sub" id="{anchor(sub.num)}">'
            f'<span class="sub-n">{sub.num}</span>'
            f'<span class="sub-t">{inline(sub.raw, refs)}</span>'
            f'<a class="hash" href="#{anchor(sub.num)}" aria-label="Link to {esc(plain(sub.raw))}">#</a>'
            f"</h2>"
        )
        blocks = render_blocks(sub.body, refs, prefix, cites)
        if blocks:
            parts.append(f'<div class="sub-body">{blocks}</div>')
    subs = "".join(parts)

    def link(side: str, item: tuple[str, str, str] | None) -> str:
        if item is None:
            return '<span class="pn-empty"></span>'
        href, num, label = item
        arrow = "←" if side == "prev" else "→"
        return (
            f'<a class="pn pn--{side}" href="{href}">'
            f'<span class="pn-k">{arrow} {"Previous" if side == "prev" else "Next"}</span>'
            f'<span class="pn-t">{num} {esc(label)}</span></a>'
        )

    main = "\n".join(
        x
        for x in (
            f'<nav class="crumbs"><a href="index.html">{esc(mod.en)}</a>'
            f'<span class="sep">/</span><span>§{sec.num}</span></nav>',
            f"<h1>{inline(sec.raw, refs)}</h1>",
            sources,
            lead_html,
            f'<div class="subs">{subs}</div>',
            cites.render(),
            f'<nav class="prevnext">{link("prev", prev)}{link("next", nxt)}</nav>',
        )
        if x
    )

    toc = "".join(
        f'<li><a href="#{anchor(sub.num)}" data-toc="{anchor(sub.num)}">'
        f'<span class="toc-n">{sub.num}</span>{esc(plain(sub.raw))}</a></li>'
        for sub in sec.subs
    )
    right = f"""<aside class="toc" id="toc">
  <div class="toc-head">On this page</div>
  <ul class="toc-list">{toc}</ul>
</aside>"""

    return page(
        modules=modules,
        depth=1,
        title=f"{sec.num} {esc(plain(sec.raw))} · {esc(mod.en)}",
        layout="section",
        main=main,
        active=mod.slug,
        left=sidebar(mod, sec.num),
        right=right,
    )


# --------------------------------------------------------------------------- build


def build() -> int:
    md = README.read_text(encoding="utf-8")
    front, modules, refs = parse(md)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    for name in ("site.css", "site.js"):
        shutil.copyfile(ASSET_SRC / name, OUT / "assets" / name)

    figures = 0
    fig_src = ASSET_SRC / "figures"
    if fig_src.is_dir():
        shutil.copytree(fig_src, OUT / "assets" / "figures")
        figures = sum(1 for p in (OUT / "assets" / "figures").iterdir() if p.is_file())

    flat: list[tuple[Module, Section]] = [(m, s) for m in modules for s in m.sections]

    pages = 0
    (OUT / "index.html").write_text(render_home(front, modules, refs), encoding="utf-8")
    pages += 1

    for mod in modules:
        (OUT / mod.slug).mkdir(parents=True, exist_ok=True)
        (OUT / mod.slug / "index.html").write_text(
            render_module(mod, modules, refs), encoding="utf-8"
        )
        pages += 1

    for i, (mod, sec) in enumerate(flat):
        def ref_to(j: int) -> tuple[str, str, str] | None:
            if not 0 <= j < len(flat):
                return None
            m2, s2 = flat[j]
            href = s2.page if m2 is mod else f"../{m2.slug}/{s2.page}"
            return href, s2.num, plain(s2.raw)

        (OUT / mod.slug / sec.page).write_text(
            render_section(mod, sec, modules, refs, ref_to(i - 1), ref_to(i + 1)),
            encoding="utf-8",
        )
        pages += 1

    index = []
    for mod in modules:
        index.append({"n": "", "t": plain(mod.en), "u": f"{mod.slug}/index.html",
                      "m": mod.en, "k": "module"})
        for sec in mod.sections:
            index.append({"n": sec.num, "t": plain(sec.raw), "u": f"{mod.slug}/{sec.page}",
                          "m": mod.en, "k": "section"})
            for sub in sec.subs:
                index.append({"n": sub.num, "t": plain(sub.raw),
                              "u": f"{mod.slug}/{sec.page}#{anchor(sub.num)}",
                              "m": mod.en, "k": "sub"})
    (OUT / "assets" / "search.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    n_sec = len(flat)
    n_sub = sum(len(s.subs) for _, s in flat)
    n_body = sum(1 for _, s in flat if s.body and any(x.strip() for x in s.body)) + sum(
        1 for _, s in flat for b in s.subs if b.body and any(x.strip() for x in b.body)
    )
    print(f"modules:     {len(modules)}")
    print(f"sections:    {n_sec}")
    print(f"subsections: {n_sub}")
    print(f"with body:   {n_body}")
    print(f"figures:     {figures}")
    print(f"bib entries: {len(BIB)}")
    print(f"pages:       {pages}")
    print(f"search:      {len(index)} entries")
    print(f"output:      {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(build())
    except BuildError as exc:
        print(f"build_site: {exc}", file=sys.stderr)
        sys.exit(1)
