#!/usr/bin/env python3
"""Lint README.md, CONTRIBUTING.md and docs/ against the docs conventions.

The conventions are the "Writing docs" section of CONTRIBUTING.md. This script uses
only the standard library and is read-only: it never writes a file.

Usage::

    python scripts/check_docs.py [PATH ...] [--links-only] [--pending FILE]
                                 [--self-test] [--no-warnings]

PATH is a Markdown file or a directory (searched for ``*.md``; ``.venv``, ``target``,
dot-directories and CLAUDE.md are skipped). The default set is README.md,
CONTRIBUTING.md and docs/**/*.md.

Output is one ``path:line: [rule] message`` line per problem, then a per-rule summary.
The exit status is 1 when there is any error (2 for a bad argument). Warnings never fail
the run. A file outside the repository is linted as a snippet: every rule applies except
the one-H1 rule and the theory-page rules (those apply to docs/theory/*.md only).

``--links-only``
    Run only the link, anchor and image checks.
``--pending FILE``
    Allow forward links to content that does not exist yet. FILE holds one
    repository-relative ``path`` or ``path#anchor`` per line (``*`` globs are allowed in
    the anchor; ``#`` at the start of a line begins a comment). A listed path allows
    every link into that file.
``--self-test``
    Lint the built-in fixtures: the good ones must be clean and each bad one must
    trigger its rule. Exits non-zero if any fixture misbehaves.
``--no-warnings``
    Print errors only.

To skip one line, append ``<!-- docs-lint: ignore -->`` to the END of that line (never
at the start of a line, never on an alert marker line) and say why in the PR.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import os
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------
# Rules: id -> (severity, scope, summary). Scope "links" rules also run under --links-only.
# ---------------------------------------------------------------------------------------

ERROR, WARNING = "error", "warning"
LINKS, STYLE = "links", "style"

RULES: dict[str, tuple[str, str, str]] = {
    # Links, anchors and images
    "link-missing-file": (ERROR, LINKS, "relative link to a file that does not exist"),
    "link-missing-anchor": (ERROR, LINKS, "#anchor that is neither a heading slug nor an id"),
    "link-forbidden": (ERROR, LINKS, "link to CLAUDE.md or up.sh, which the mirror strips"),
    "anchor-duplicate": (ERROR, LINKS, "explicit id defined twice in one file"),
    "anchor-collision": (ERROR, LINKS, "explicit id equal to an automatic heading slug"),
    "anchor-id-format": (ERROR, LINKS, "explicit id that is not lowercase ASCII [a-z0-9-]"),
    "anchor-in-details": (ERROR, LINKS, "linked-to heading or id inside <details>"),
    "anchor-pinned": (ERROR, LINKS, "pinned anchor (inbound links depend on it) is missing"),
    "image-missing": (ERROR, LINKS, "embedded image or inventory row names a missing file"),
    "image-no-scene": (ERROR, LINKS, "docs/images PNG without a capture scene"),
    "image-no-inventory": (ERROR, LINKS, "docs/images PNG without a Figure gallery inventory row"),
    "png-size": (WARNING, LINKS, "PNG over 800 KB"),
    # Alerts
    "alert-placement": (ERROR, STYLE, "alert marker not alone, not top level, or mid-quote"),
    "alert-type": (ERROR, STYLE, "unknown or lowercase alert type"),
    "alert-status": (ERROR, STYLE, "status alert without its **Status: <Label>.** line"),
    "alert-label": (ERROR, STYLE, "IMPORTANT or TIP without a bold, non-status label"),
    "alert-count": (ERROR, STYLE, "more than 1 IMPORTANT, 1 TIP or 3 alerts on a page"),
    "alert-consecutive": (ERROR, STYLE, "two alerts in a row"),
    "alert-length": (WARNING, STYLE, "status alert text over 60 words"),
    # Math
    "math-display": (ERROR, STYLE, "$$ display math; use a ```math fence"),
    "math-bare-dollar": (ERROR, STYLE, "bare $ math; use $`...`$ or <span>$</span>"),
    "math-delimiter": (ERROR, STYLE, "\\(...\\) math, which GitHub does not render"),
    "math-angle": (ERROR, STYLE, "< or > inside math; write \\lt or \\gt"),
    # Mermaid
    "mermaid-type": (ERROR, STYLE, "diagram type other than flowchart or pie"),
    "mermaid-init": (ERROR, STYLE, "%%{init}%% directive"),
    "mermaid-style": (ERROR, STYLE, "style line; use the shared classDefs"),
    "mermaid-classdef": (ERROR, STYLE, "classDef: fill without color, spacing, or not canonical"),
    "mermaid-count": (ERROR, STYLE, "too many diagrams on the page"),
    "mermaid-filename": (ERROR, STYLE, "file name in a diagram label"),
    "mermaid-nodes": (WARNING, STYLE, "diagram over 12 nodes"),
    "mermaid-unquoted": (WARNING, STYLE, "unquoted node label"),
    # Headings and theory pages
    "heading-h1": (ERROR, STYLE, "page without exactly one H1"),
    "heading-skip": (ERROR, STYLE, "skipped heading level"),
    "heading-content": (ERROR, STYLE, "&, code, math, arrow, tag or status tag in a heading"),
    "theory-h2": (ERROR, STYLE, "theory H2s not an ordered subset of the template"),
    "theory-status": (ERROR, STYLE, "theory page without exactly one status alert before H2"),
    "theory-pager": (ERROR, STYLE, "theory page without a one-line pager after ---"),
    # Structure traps
    "structure-table-glued": (ERROR, STYLE, "non-table line glued under a table row"),
    "structure-list-glued": (ERROR, STYLE, "unindented line glued under a list item"),
    "structure-setext": (ERROR, STYLE, "--- or === directly under a text line"),
    "structure-table-cells": (ERROR, STYLE, "table row cell count differs from the header"),
    "structure-table-pipe": (ERROR, STYLE, "raw | inside code or math in a table row"),
    "structure-card": (ERROR, STYLE, "blank line inside an HTML figure card"),
    "structure-details": (ERROR, STYLE, "no blank line after </summary>"),
    "anchor-standalone": (ERROR, STYLE, "standalone <a id> line not above a bold entry"),
    # Figures and HTML
    "figure-markdown-image": (ERROR, STYLE, "Markdown ![]() image; use an HTML figure card"),
    "figure-img": (ERROR, STYLE, "<img> without alt (<= 125 chars, no raw |) or a size"),
    "figure-thumb-link": (ERROR, STYLE, "table thumbnail <img> not wrapped in <a href>"),
    "figure-pixel-width": (ERROR, STYLE, "pixel width in a figure card"),
    "html-style-attr": (ERROR, STYLE, "style= attribute (GitHub strips it)"),
    "html-banned-tag": (ERROR, STYLE, "<figure>, <figcaption> or <small> (GitHub strips them)"),
    "html-markdown": (ERROR, STYLE, "Markdown inside an HTML block, where it is not rendered"),
    "link-line-anchor": (ERROR, STYLE, "link to a source line number"),
    "figure-caption-length": (WARNING, STYLE, "caption over 35 words (20 in 3-up cells)"),
    # Typography
    "typo-dash": (ERROR, STYLE, "ASCII -- in prose"),
    "typo-angstrom": (ERROR, STYLE, "angstrom or a bare A unit"),
    "typo-deg": (ERROR, STYLE, "deg or degrees after a number"),
    "typo-plusminus": (ERROR, STYLE, "+/-"),
    "typo-tilde": (ERROR, STYLE, "single ~ (approximately)"),
    "typo-spelling": (ERROR, STYLE, "Gruneisen, Mossbauer or British spelling"),
    "typo-eponym": (ERROR, STYLE, "hyphenated multi-person name"),
    "typo-formula": (ERROR, STYLE, "ASCII chemical formula or isotope in prose"),
    "typo-stale": (ERROR, STYLE, "stale phrase"),
    # Paragraphs, tables and the ignore marker
    "para-length": (WARNING, STYLE, "paragraph or list item over 100 words"),
    "table-columns": (WARNING, STYLE, "table over 5 columns"),
    "ignore-marker": (ERROR, STYLE, "misplaced <!-- docs-lint: ignore --> marker"),
}

# ---------------------------------------------------------------------------------------
# Configuration. Every per-file exemption, limit and phrase list lives here.
# ---------------------------------------------------------------------------------------

IGNORE_MARKER = "<!-- docs-lint: ignore -->"
DEFAULT_PATHS = ("README.md", "CONTRIBUTING.md", "docs")
SKIP_DIRS = frozenset({".venv", "target"})  # every dot-directory is skipped too
GALLERY = "docs/images/README.md"
IMAGES_DIR = "docs/images"
SCENE_SOURCES = (  # where a docs/images PNG gets its capture scene: literal file names
    ("scripts/capture_screenshots.py", r'"([\w.-]+\.png)"'),
    ("crates/moire-desktop/src/bin/capture.rs", r'filename:\s*"([\w.-]+\.png)"'),
)
FORBIDDEN_LINKS = frozenset({"CLAUDE.md", "up.sh"})

LIMITS = {
    "paragraph_words": 100,
    "alert_words": 60,
    "caption_words": 35,
    "caption_words_narrow": 20,  # cells of 3-up rows (the maximum when cells carry captions)
    "alt_chars": 125,
    "table_columns": 5,
    "mermaid_nodes": 12,
    "png_kb": 800,
}


@dataclass(frozen=True)
class FileRules:
    mermaid_max: int = 1
    alert_total: int | None = 3
    off: frozenset[str] = frozenset()


# Per-file exemptions, keyed by repository-relative path. Keep them all in this table.
FILE_RULES: dict[str, FileRules] = {
    # The front door carries no diagrams.
    "README.md": FileRules(mermaid_max=0),
    # Contributor diagrams name files by design. The style guide quotes banned forms only
    # inside code spans and fences, which every rule skips anyway.
    "CONTRIBUTING.md": FileRules(mermaid_max=3, off=frozenset({"mermaid-filename"})),
    # One status alert per web-page section; the At a glance table and the feature matrix
    # are wider than 5 columns by design.
    "docs/user-guide.md": FileRules(alert_total=None, off=frozenset({"table-columns"})),
    # The 6-column Inventory and the 7-column materials table are wide by design.
    "docs/images/README.md": FileRules(off=frozenset({"table-columns"})),
    "docs/materials.md": FileRules(off=frozenset({"table-columns"})),
}

# Anchors that inbound links (other pages, the code, outside bookmarks) depend on.
REF_IDS = (
    "abrikosov-1957", "agterberg-2020", "aishwarya-2023", "aoki-2020", "bardeen-1957",
    "beenakker-2013", "bessas-2012", "bistritzer-macdonald-2011", "blakslee-1970",
    "blonder-1982", "cao-2018", "cao-2018b", "castro-neto-2009", "ding-2025", "fradkin-2015",
    "fu-kane-2008", "gu-2023", "guinea-2010", "hasan-2010", "hasan-kane-2010", "huang-2019",
    "huder-2018", "kengle-2024", "khalaf-2019", "khasanov-2010", "khasanov-2010b", "kong-2025",
    "koshino-2018", "levy-2010", "lindsay-2008", "lindsay-2013", "lindsay-broido-2008",
    "liu-2009", "meyer-2007", "nachtigal-2022", "park-2021", "qi-2008", "sau-2010",
    "shirage-2009", "song-2019", "tarnopolsky-2019", "vozmediano-2010", "wang-2019",
    "wang-2026", "weng-2021", "wu-2024", "yan-2026", "zhu-2026",
)
PINNED_ANCHORS: dict[str, tuple[str, ...]] = {
    "CONTRIBUTING.md": ("keeping-python--rust-physics-in-sync", "screenshot-capture"),
    "docs/theory/gap-modulation.md": ("cpdm-versus-pdw",),
    "docs/theory/isotope-effects.md": ("highly-speculative-synthetic-and-exotic-isotopes",),
    "docs/references.md": tuple(f"ref-{r}" for r in REF_IDS),
}

ALERT_TYPES = ("NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION")
STATUS_LABELS = {
    "NOTE": ("Validated",),
    "WARNING": ("Mixed", "Speculative"),
    "CAUTION": ("Highly speculative",),
}
THEORY_ORDER = (  # reading order: core pipeline, graphene track, speculative extensions
    "moire-patterns",
    "gap-modulation",
    "fourier-analysis",
    "graphene-stacks",
    "bm-model",
    "curvature",
    "magnetic",
    "topological",
    "isotope-effects",
    "spin-fluctuation-pairing",
    "pair-density-waves",
)
THEORY_H2 = (
    "Overview",
    "Model",
    "Speculative components",
    "Assumptions and validity",
    "Implementation",
    "References",
)
MERMAID_TYPES = ("flowchart", "pie")
CANONICAL_CLASSDEFS = {
    "spec": "fill:#fff8c5,stroke:#9a6700,color:#1f2328",
    "hispec": "fill:#ffebe9,stroke:#cf222e,color:#1f2328",
    "mixed": "stroke:#bf8700,stroke-width:2px,stroke-dasharray:5 3",
    "zone": "fill:none,stroke:#8c959f,stroke-dasharray:4 3",
}

# Stale phrases: (phrase, where). "prose" skips code, math and URLs; "raw" also searches
# fences, code spans and URLs. Matching is case-sensitive; add each spelling to catch.
STALE_PHRASES: tuple[tuple[str, str], ...] = (
    ("Theory overview", "prose"),
    ("Theory Documentation", "prose"),
    ("Documentation index", "prose"),
    ("Historical context", "prose"),
    ("Python only", "prose"),
    ("the README's", "prose"),
    ("The README's", "prose"),
    ("README-level", "prose"),
    ("README-originated", "prose"),
    ("Post-cutoff", "prose"),
    ("iron-pnictide consensus", "prose"),
    ("iron-chalcogenide consensus", "prose"),
    ("(spec.)", "prose"),
    ("Status: mixed", "prose"),
    ("55.7 Å", "prose"),  # the strained-Sb₂Te₃ period is 55.9 Å
    ("4.38 Å", "prose"),  # Bi₂Te₃ is a = 4.386 Å
    ("github.com/user/", "raw"),  # the old placeholder clone URL
    (" of 10</sub>", "raw"),  # 11 theory pages since the SF module
    (" of 10`", "raw"),  # the same breadcrumb quoted in a code span
    ("Ten physics deep dives", "prose"),
    ("ten deep dives", "prose"),
    ("39 figures", "prose"),  # 44 since the SF scenes
)
EPONYMS = (  # multi-person names take an en dash: Fu–Kane
    "Bistritzer-MacDonald",
    "Fu-Kane",
    "Debye-Waller",
    "Ginzburg-Landau",
    "Bardeen-Cooper",
    "Cooper-Schrieffer",
    "Fulde-Ferrell",
    "Larkin-Ovchinnikov",
    "Blonder-Tinkham",
    "Tinkham-Klapwijk",
    "Clogston-Chandrasekhar",
    "Kosterlitz-Thouless",
    "Bogoliubov-de",
    "Jones-Zener",
    "Shockley-Chambers",
)
ELEMENTS = frozenset(
    (
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    )
)

# ---------------------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------------------

QUOTE_RE = re.compile(r"^ {0,3}> ?")
FENCE_RE = re.compile(r"^( *)(`{3,}|~{3,})(.*)$")
LIST_ITEM_RE = re.compile(r"^( *)([-*+]|\d{1,9}[.)])(?:( +)(.*))?$")
THEMATIC_RE = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
SETEXT_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
DELIM_RE = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
HTML_BLOCK6_RE = re.compile(
    r"^ {0,3}</?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col"
    r"|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form"
    r"|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem"
    r"|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td"
    r"|tfoot|th|thead|title|tr|track|ul)(?:[ \t>]|/>|$)",
    re.IGNORECASE,
)
HTML_BLOCK7_RE = re.compile(
    r"^ {0,3}(?:<(?!script|style|pre)[A-Za-z][A-Za-z0-9-]*"
    r"(?:\s+[A-Za-z_:][\w.:-]*(?:\s*=\s*(?:[^\s\"'=<>`]+|'[^']*'|\"[^\"]*\"))?)*\s*/?>"
    r"|</[A-Za-z][A-Za-z0-9-]*\s*>)[ \t]*$",
)
TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9-]*)\b([^<>]*)>")
SPLIT_IMG_RE = re.compile(r"<img\b[^<>]*$", re.I)  # an <img> still open at the end of a line
ATTR_RE = re.compile(r"([A-Za-z_:][\w:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+))")
ATTR_VALUE_RE = re.compile(r"\b([A-Za-z_:][\w:.-]*)\s*=\s*(?:\"[^\"]*\"|'[^']*')")
MD_LINK_RE = re.compile(
    r"(!?)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(\s*(<[^>\n]*>|[^\s()<>]*(?:\([^\s()]*\)[^\s()<>]*)*)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
REF_DEF_RE = re.compile(r"^ {0,3}\[([^\]]+)\]:\s*(<[^>]*>|\S+)")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+")
AUTOLINK_RE = re.compile(r"<(?:https?|mailto|ftp):[^>\s]*>")
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
IGNORE_RE = re.compile(r"<!--\s*docs-lint:\s*ignore\s*-->")
IGNORE_END_RE = re.compile(r"<!--\s*docs-lint:\s*ignore\s*-->$")
ALERT_RE = re.compile(r"^(\s*(?:(?:[-*+]|\d{1,9}[.)])\s+)*)((?:>[ \t]*)+)\[!([A-Za-z]+)\](.*)$")
STATUS_RE = re.compile(r"^\*\*Status: ([^*]+?)\.\*\*")
ID_ATTR_RE = re.compile(r"<a\b[^>]*?\b(?:id|name)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.I)
ANCHOR_ONLY_RE = re.compile(
    r"^\s*(?:<a\s+(?:id|name)\s*=\s*(?:\"[^\"]*\"|'[^']*')\s*>\s*</a>\s*)+$", re.I
)
EMPHASIS_US_RE = re.compile(r"(?<![\w\\])(_{1,2})(?=\S)(.+?)(?<=\S)\1(?!\w)")
EMPHASIS_STAR_RE = re.compile(r"(?<!\\)(\*{1,3})(?=[^\s*])(.+?)(?<=[^\s\\*])\1")
ARROW_RE = re.compile(r"[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f\u2b00-\u2b2f]|->|<-|=>")
FILE_NAME_RE = re.compile(
    r"\b[\w.-]+\.(?:py|rs|md|json|toml|png|svg|wgsl|js|ts|css|html|txt|ya?ml|sh|lock|csv)\b"
)
FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?[0-9]*){2,}\b")
ISOTOPE_RE = re.compile(r"\b([0-9]{2,3})([A-Z][a-z]?)\b")
WORD_RE = re.compile(r"[^\W_](?:[\w'’.-]*[^\W_])?")

TYPO_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("typo-dash", re.compile(r"--"), "ASCII '--': write – for ranges or — for asides"),
    (
        "typo-angstrom",
        re.compile(r"\b[Aa]ngstroms?\b|\b\d+(?:\.\d+)?[ \t]?A\b(?![-'’])"),
        "write Å, not 'angstrom' or a bare A after a number",
    ),
    ("typo-deg", re.compile(r"\bdeg\b|\d[ \t-]*degrees?\b"), "write °, not 'deg'"),
    ("typo-plusminus", re.compile(r"\+/-"), "write ±, not '+/-'"),
    (
        "typo-tilde",
        re.compile(r"(?<!~)~(?!~)"),
        "a single ~ can strike text through on GitHub: write ≈, or ∼ for an order of magnitude",
    ),
    (
        "typo-spelling",
        re.compile(r"\bGrue?neisen|\bMoe?ssbauer|\bcentre[sd]?\b|\bneighbour|\bcolour", re.I),
        "write Grüneisen, Mössbauer and US spelling (center, neighbor, color)",
    ),
    (
        "typo-eponym",
        re.compile("|".join(r"\b" + re.escape(e) + r"\b" for e in EPONYMS)),
        "multi-person names take an en dash: Bistritzer–MacDonald, Fu–Kane, Debye–Waller",
    ),
)

# ---------------------------------------------------------------------------------------
# File access: the real repository, or the in-memory tree the self-test builds
# ---------------------------------------------------------------------------------------


class Fs:
    """Read-only file access below ``root``; ``files`` makes it an in-memory tree."""

    def __init__(self, root: str, files: dict[str, str | bytes | int] | None = None) -> None:
        self.root = os.path.normpath(os.path.abspath(root))
        self.mem: dict[str, str | bytes | int] | None = None
        if files is not None:
            self.mem = {self.abs(k): v for k, v in files.items()}

    def abs(self, path: str) -> str:
        return os.path.normpath(os.path.join(self.root, path))

    def rel(self, path: str) -> str | None:
        """Repository-relative POSIX path, or None outside the repository."""
        try:
            r = os.path.relpath(path, self.root)
        except ValueError:
            return None
        if r == os.pardir or r.startswith(os.pardir + os.sep) or os.path.isabs(r):
            return None
        return r.replace(os.sep, "/")

    def is_file(self, path: str) -> bool:
        return os.path.isfile(path) if self.mem is None else path in self.mem

    def is_dir(self, path: str) -> bool:
        if self.mem is None:
            return os.path.isdir(path)
        prefix = path.rstrip(os.sep) + os.sep
        return any(k.startswith(prefix) for k in self.mem)

    def exists(self, path: str) -> bool:
        return self.is_file(path) or self.is_dir(path)

    def read_text(self, path: str) -> str:
        if self.mem is None:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        value = self.mem[path]
        return value if isinstance(value, str) else ""

    def size(self, path: str) -> int:
        if self.mem is None:
            return os.path.getsize(path)
        value = self.mem[path]
        if isinstance(value, int):
            return value
        return len(value.encode("utf-8") if isinstance(value, str) else value)

    def files_in(self, path: str) -> list[str]:
        """Files directly inside directory ``path``."""
        if self.mem is None:
            if not os.path.isdir(path):
                return []
            names = (os.path.join(path, n) for n in os.listdir(path))
            return sorted(p for p in names if os.path.isfile(p))
        return sorted(k for k in self.mem if os.path.dirname(k) == path)

    def markdown_under(self, path: str) -> list[str]:
        """``*.md`` below ``path``, skipping .venv, target, dot-directories and CLAUDE.md."""
        found: list[str] = []
        if self.mem is not None:
            prefix = path.rstrip(os.sep) + os.sep
            for k in self.mem:
                if k.startswith(prefix) and k.endswith(".md"):
                    parts = os.path.relpath(k, path).split(os.sep)
                    hidden = any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1])
                    if not hidden and parts[-1] not in FORBIDDEN_LINKS:
                        found.append(k)
            return sorted(found)
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            )
            found += [
                os.path.join(dirpath, f)
                for f in sorted(filenames)
                if f.endswith(".md") and f not in FORBIDDEN_LINKS
            ]
        return found


# ---------------------------------------------------------------------------------------
# Inline scanning: code spans, $`math`$ spans and HTML comments
# ---------------------------------------------------------------------------------------

Seg = tuple[str, str, str]  # (kind, raw, inner); kind is text, code, math or comment


def _closing_backticks(s: str, start: int, width: int) -> int:
    k, n = start, len(s)
    while True:
        k = s.find("`", k)
        if k < 0:
            return -1
        end = k
        while end < n and s[end] == "`":
            end += 1
        if end - k == width:
            return k
        k = end


def scan_inline(s: str, in_comment: bool = False) -> tuple[list[Seg], bool]:
    """Split one line into segments; the flag says whether an HTML comment stays open."""
    segs: list[Seg] = []
    i, n = 0, len(s)
    if in_comment:
        end = s.find("-->")
        if end < 0:
            return [("comment", s, "")], True
        segs.append(("comment", s[: end + 3], ""))
        i = end + 3
    start = i
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "`":
            j = i
            while j < n and s[j] == "`":
                j += 1
            width = j - i
            close = _closing_backticks(s, j, width)
            if close < 0:
                i = j
                continue
            after = close + width
            math = i - 1 >= start and s[i - 1] == "$" and after < n and s[after] == "$"
            cut = i - 1 if math else i
            if cut > start:
                segs.append(("text", s[start:cut], s[start:cut]))
            if math:
                segs.append(("math", s[i - 1 : after + 1], s[j:close]))
                i = after + 1
            else:
                segs.append(("code", s[i:after], s[j:close]))
                i = after
            start = i
            continue
        if c == "<" and s.startswith("<!--", i):
            if i > start:
                segs.append(("text", s[start:i], s[start:i]))
            end = s.find("-->", i + 4)
            if end < 0:
                segs.append(("comment", s[i:], ""))
                return segs, True
            segs.append(("comment", s[i : end + 3], ""))
            i = end + 3
            start = i
            continue
        i += 1
    if n > start:
        segs.append(("text", s[start:], s[start:]))
    return segs, False


def visible_text(segs: list[Seg]) -> str:
    """The line without HTML comments (code spans keep their backticks)."""
    return "".join(raw if kind != "comment" else " " for kind, raw, _ in segs)


def code_free(segs: list[Seg]) -> str:
    """The line with code spans, math spans and comments blanked out."""
    return "".join(raw if kind == "text" else " " for kind, raw, _ in segs)


def prose_text(bare: str) -> str:
    """Code-free text without link destinations, URLs or HTML attribute values."""
    text = MD_LINK_RE.sub(lambda m: m.group(1) + "[" + m.group(2) + "]", bare)
    text = REF_DEF_RE.sub(" ", text)
    text = AUTOLINK_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = ATTR_VALUE_RE.sub(r'\1=""', text)
    return text.replace("<span>$</span>", " ")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def strip_tags(text: str) -> str:
    """Text without tags: <br> and block tags separate words; inline tags (T<sub>c</sub>) join."""
    text = re.sub(r"<(?:br|/?(?:p|div|td|th|tr|table|li|ul|ol|details|summary))\b[^>]*>", " ",
                  text, flags=re.I)
    return html.unescape(TAG_RE.sub("", text))


# ---------------------------------------------------------------------------------------
# GitHub heading slugs
# ---------------------------------------------------------------------------------------


def rendered_text(markdown: str) -> str:
    """Approximate the text GitHub renders for inline Markdown (used for heading slugs)."""
    parts: list[str] = []
    for kind, raw, inner in scan_inline(markdown)[0]:
        if kind == "comment":
            continue
        if kind in ("code", "math"):
            if inner.startswith(" ") and inner.endswith(" ") and inner.strip():
                inner = inner[1:-1]
            parts.append(inner)
            continue
        s = re.sub(r"!\[(?:[^\[\]]|\[[^\[\]]*\])*\]\([^)]*\)", "", raw)
        s = re.sub(r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\[[^\]]*\]", r"\1", s)
        s = re.sub(r"<((?:https?|mailto):[^>\s]*)>", r"\1", s)
        s = TAG_RE.sub("", s)
        s = EMPHASIS_STAR_RE.sub(r"\2", s)
        s = EMPHASIS_US_RE.sub(r"\2", s)
        s = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", s)
        parts.append(html.unescape(s))
    return "".join(parts).strip()


def slugify(text: str) -> str:
    """GitHub's slug rule: lowercase; keep L*, M*, Nd, Pc, '-' and ' '; spaces to '-'."""
    kept = []
    for ch in text.lower():
        if ch in " -":
            kept.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] in "LM" or cat in ("Nd", "Pc"):
            kept.append(ch)
    return "".join(kept).replace(" ", "-")


class Slugger:
    """github-slugger: repeats get -1, -2, … and never reuse an earlier slug."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def slug(self, text: str) -> str:
        base = slugify(text)
        result = base
        while result in self.seen:
            self.seen[base] += 1
            result = f"{base}-{self.seen[base]}"
        self.seen[result] = 0
        return result


def github_slugs(headings: Iterable[str]) -> list[str]:
    """Slugs for a page's headings, in order (inline Markdown allowed)."""
    slugger = Slugger()
    return [slugger.slug(rendered_text(h)) for h in headings]


# ---------------------------------------------------------------------------------------
# Line model
# ---------------------------------------------------------------------------------------


@dataclass
class Line:
    n: int
    raw: str
    kind: str = "text"  # blank | text | html | fence | code | comment
    quote: int = 0  # blockquote depth
    content: str = ""  # raw line without blockquote markers
    segs: list[Seg] = field(default_factory=list)
    text: str = ""  # content without HTML comments
    bare: str = ""  # text without code spans, math spans and comments
    fence: str = ""  # info word of the fence this line belongs to
    role: str = ""  # fence lines: open | body | close
    details: bool = False  # inside <details>…</details>
    ignored: bool = False  # carries an honored ignore marker
    marker: str = ""  # why an ignore marker on this line is not honored
    item: bool = False  # opens a list item
    listed: bool = False  # list item or its continuation
    indent: int = 0
    table: str = ""  # header | delim | row | badheader
    level: int = 0  # ATX heading level

    @property
    def blank(self) -> bool:
        return not self.content.strip()


def _strip_quotes(raw: str, limit: int = 99) -> tuple[str, int]:
    depth = 0
    while depth < limit:
        m = QUOTE_RE.match(raw)
        if not m:
            break
        raw = raw[m.end() :]
        depth += 1
    return raw, depth


def split_cells(row: str) -> list[str]:
    """GFM cells: split on every unescaped |, even inside code spans."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return re.split(r"(?<!\\)\|", s)


def _ignore_marker(ln: Line, alert_line: bool) -> None:
    """Honor a trailing ignore marker; record why any other placement is refused."""
    markers = [raw for kind, raw, _ in ln.segs if kind == "comment" and IGNORE_RE.fullmatch(raw)]
    if not markers:
        return
    stripped = ln.raw.rstrip()
    last = IGNORE_END_RE.search(stripped)
    before = stripped[: last.start()] if last else ""
    if len(markers) > 1 or last is None:
        ln.marker = "put the ignore marker at the END of the line"
    elif not before.lstrip("> \t").strip():
        ln.marker = "the ignore marker cannot start a line (it opens an HTML block there)"
    elif alert_line:
        ln.marker = "the ignore marker on an alert marker line turns the alert into a quote"
    else:
        ln.ignored = True


def parse_lines(source: str) -> list[Line]:
    out: list[Line] = []
    fence: tuple[str, int, int, str, int] | None = None  # char, width, depth, info, indent
    in_comment = in_html = False
    list_cols: list[int] = []
    details = 0
    prev: Line | None = None
    for n, raw in enumerate(source.splitlines(), 1):
        ln = Line(n, raw)
        ln.content, ln.quote = _strip_quotes(raw)
        ln.details = details > 0
        if fence is not None:
            fch, fwidth, fdepth, finfo, findent = fence
            body, got = _strip_quotes(raw, fdepth)
            if got == fdepth:
                ln.kind, ln.fence, ln.role = "fence", finfo, "body"
                closing = rf"^ {{0,{findent}}}{re.escape(fch)}{{{fwidth},}}[ \t]*$"
                if re.match(closing, body):
                    ln.role, fence = "close", None
                if IGNORE_RE.search(raw) and finfo in ("math", "mermaid"):
                    ln.marker = f"the ignore marker breaks a ```{finfo} fence"
                out.append(ln)
                prev = ln
                continue
            fence = None  # the enclosing blockquote ended, and the fence with it
        if in_comment:
            ln.segs, in_comment = scan_inline(ln.content, in_comment=True)
            if in_comment or not visible_text(ln.segs).strip():
                ln.kind = "comment"
                out.append(ln)
                continue
        elif not raw.strip():
            ln.kind = "blank"
            in_html = False
            out.append(ln)
            prev = ln
            continue
        else:
            lead = len(ln.content) - len(ln.content.lstrip(" "))
            allow = list_cols[-1] + 3 if list_cols and lead >= list_cols[-1] else 3
            fm = FENCE_RE.match(ln.content)
            backtick_info = fm is not None and fm.group(2)[0] == "`" and "`" in fm.group(3)
            if fm and not in_html and len(fm.group(1)) <= allow and not backtick_info:
                info = fm.group(3).strip().split()
                word = info[0].strip("{}.").lower() if info else ""
                fence = (fm.group(2)[0], len(fm.group(2)), ln.quote, word, allow)
                ln.kind, ln.fence, ln.role, ln.indent = "fence", word, "open", lead
                ln.listed = bool(list_cols)
                if prev is not None and prev.kind == "blank":
                    while list_cols and lead < list_cols[-1]:
                        list_cols.pop()
                out.append(ln)
                prev = ln
                continue
            ln.segs, in_comment = scan_inline(ln.content)
        ln.text = visible_text(ln.segs)
        ln.bare = code_free(ln.segs)
        if not ln.text.strip():
            if any(kind == "comment" for kind, _, _ in ln.segs):
                ln.kind = "comment"  # a comment-only line renders nothing
                _ignore_marker(ln, alert_line=False)
            else:
                prev = ln  # an empty blockquote line ('>')
            out.append(ln)
            continue
        lead = len(ln.text) - len(ln.text.lstrip(" "))
        ln.indent = lead
        indented_code = (
            lead >= 4
            and not list_cols
            and not in_html
            and (prev is None or prev.kind in ("blank", "code"))
        )
        if indented_code:
            ln.kind = "code"  # an indented code block: never linted
            out.append(ln)
            prev = ln
            continue
        alert_line = bool(ALERT_RE.match(ln.raw))
        _ignore_marker(ln, alert_line)
        prev_para = (
            prev is not None
            and prev.kind == "text"
            and not prev.blank
            and not prev.level
            and not prev.table
        )
        if in_html:
            ln.kind = "html"
        elif HTML_BLOCK6_RE.match(ln.text) or (HTML_BLOCK7_RE.match(ln.text) and not prev_para):
            ln.kind = "html"
            in_html = True
        else:
            hm = HEADING_RE.match(ln.text)
            if hm:
                ln.level = len(hm.group(1))
        lm = LIST_ITEM_RE.match(ln.text)
        if ln.kind == "text" and lm and not THEMATIC_RE.match(ln.text) and not ln.level:
            width = len(lm.group(2))
            spaces = len(lm.group(3) or " ")
            if spaces > 4 or not (lm.group(4) or "").strip():
                spaces = 1
            while list_cols and list_cols[-1] > len(lm.group(1)):
                list_cols.pop()
            list_cols.append(len(lm.group(1)) + width + spaces)
            ln.item = ln.listed = True
        elif list_cols:
            if prev is not None and prev.kind == "blank":
                while list_cols and lead < list_cols[-1]:
                    list_cols.pop()
            ln.listed = bool(list_cols)
        opens = len(re.findall(r"<details\b", ln.bare, re.I))
        closes = len(re.findall(r"</details\s*>", ln.bare, re.I))
        ln.details = details > 0 or opens > 0
        details = max(0, details + opens - closes)
        out.append(ln)
        prev = ln
    _mark_tables(out)
    return out


def split_img_tags(lines: list[Line]) -> dict[int, list[tuple[int, str]]]:
    """<img> tags whose attributes continue on later lines, each joined into one tag.

    Maps a tag's first line number to (column, joined tag), so the checks that scan one
    line at a time see the whole tag and report it on its first line. A tag continues
    over text and html lines up to its closing >; a blank line or another tag ends it.
    """
    found: dict[int, list[tuple[int, str]]] = {}
    for i, ln in enumerate(lines):
        m = SPLIT_IMG_RE.search(ln.bare) if ln.kind in ("text", "html") else None
        if not m:
            continue
        parts = [m.group(0).strip()]
        for nxt in lines[i + 1 :]:
            if nxt.kind not in ("text", "html") or nxt.blank:
                break
            end = nxt.bare.find(">")
            if "<" in (nxt.bare if end < 0 else nxt.bare[:end]):
                break  # another tag opens before this one closes
            if end >= 0:
                parts.append(nxt.bare[: end + 1].strip())
                found.setdefault(ln.n, []).append((m.start(), " ".join(parts)))
                break
            parts.append(nxt.bare.strip())
    return found


def _block_start(ln: Line) -> bool:
    """A line that starts a new block (ends a table body)."""
    return bool(ln.level or ln.item or THEMATIC_RE.match(ln.text) or ln.kind != "text")


def _mark_tables(lines: list[Line]) -> None:
    i = 0
    while i < len(lines) - 1:
        ln, nxt = lines[i], lines[i + 1]
        if (
            ln.kind == "text"
            and not ln.table
            and not ln.level
            and not ln.blank
            and "|" in ln.text
            and nxt.kind == "text"
            and nxt.quote == ln.quote
            and "|" in nxt.text
            and DELIM_RE.match(nxt.text)
        ):
            if len(split_cells(ln.text)) != len(split_cells(nxt.text)):
                ln.table = "badheader"
                i += 2
                continue
            ln.table, nxt.table = "header", "delim"
            j = i + 2
            while j < len(lines):
                row = lines[j]
                if row.kind != "text" or row.blank or row.quote != ln.quote:
                    break
                if _block_start(row) and not row.content.lstrip().startswith("|"):
                    break
                row.table = "row"
                j += 1
            i = j
            continue
        i += 1


# ---------------------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------------------


@dataclass
class Heading:
    line: int
    level: int
    markdown: str
    text: str
    slug: str
    has_id: bool
    details: bool


@dataclass
class Anchor:
    name: str
    line: int
    details: bool
    explicit: bool


@dataclass
class Link:
    line: int
    target: str
    image: bool


class Doc:
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.lines = parse_lines(source)
        self.headings: list[Heading] = []
        self.anchors: list[Anchor] = []
        self.links: list[Link] = []
        self.split_imgs = split_img_tags(self.lines)
        slugger = Slugger()
        for ln in self.lines:
            if ln.kind not in ("text", "html"):
                continue
            if ln.level:
                hm = HEADING_RE.match(ln.text)
                markdown = (hm.group(2) or "") if hm else ""
                text = rendered_text(markdown)
                slug = slugger.slug(text)
                has_id = bool(ID_ATTR_RE.search(markdown))
                self.headings.append(
                    Heading(ln.n, ln.level, markdown, text, slug, has_id, ln.details)
                )
                self.anchors.append(Anchor(slug, ln.n, ln.details, False))
            for m in ID_ATTR_RE.finditer(ln.bare):
                name = m.group(1) if m.group(1) is not None else m.group(2)
                self.anchors.append(Anchor(name, ln.n, ln.details, True))
            for m in MD_LINK_RE.finditer(ln.bare):
                self.links.append(Link(ln.n, m.group(3), m.group(1) == "!"))
            rd = REF_DEF_RE.match(ln.bare)
            if rd:
                self.links.append(Link(ln.n, rd.group(2), False))
            for tag in TAG_RE.finditer(ln.bare):
                if tag.group(1):
                    continue
                name = tag.group(2).lower()
                for attr in ATTR_RE.finditer(tag.group(3)):
                    key = attr.group(1).lower()
                    if key in ("href", "src"):
                        value = next(v for v in attr.groups()[1:] if v is not None)
                        self.links.append(Link(ln.n, value, name == "img" and key == "src"))
            for _col, tag in self.split_imgs.get(ln.n, ()):
                for attr in ATTR_RE.finditer(tag[4:]):
                    if attr.group(1).lower() == "src":
                        value = next(v for v in attr.groups()[1:] if v is not None)
                        self.links.append(Link(ln.n, value, True))
        self.names = {a.name for a in self.anchors}

    def line(self, n: int) -> Line:
        return self.lines[n - 1]


# ---------------------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    path: str
    line: int
    rule: str
    message: str

    @property
    def severity(self) -> str:
        return RULES[self.rule][0]

    def format(self) -> str:
        prefix = "warning: " if self.severity == WARNING else ""
        return f"{self.path}:{self.line}: [{self.rule}] {prefix}{self.message}"


class Checker:
    def __init__(
        self,
        fs: Fs,
        *,
        links_only: bool = False,
        warnings: bool = True,
        pending: Iterable[str] = (),
    ) -> None:
        self.fs = fs
        self.links_only = links_only
        self.warnings = warnings
        self.pending_files: set[str] = set()
        self.pending_anchors: list[tuple[str, str]] = []
        for entry in pending:
            entry = entry.strip()
            if not entry or entry.startswith("#"):
                continue
            path, _, anchor = entry.partition("#")
            path = path.strip()
            if path.startswith("./"):
                path = path[2:]
            if anchor:
                self.pending_anchors.append((path, anchor.strip().lower()))
            else:
                self.pending_files.add(path)
        self.pending_hits = 0
        self.docs: dict[str, Doc | None] = {}
        self.problems: list[Problem] = []
        self._scenes: set[str] | None = None
        self._big_reported: set[str] = set()

    # -- helpers -----------------------------------------------------------------------

    def doc(self, path: str) -> Doc | None:
        if path not in self.docs:
            try:
                self.docs[path] = Doc(path, self.fs.read_text(path))
            except (OSError, UnicodeDecodeError, KeyError):
                self.docs[path] = None
        return self.docs[path]

    def display(self, path: str) -> str:
        rel = self.fs.rel(path)
        if rel is not None:
            return rel
        try:
            here = os.path.relpath(path)
        except ValueError:
            return path
        return path if here.startswith(os.pardir) else here

    def scenes(self) -> set[str]:
        if self._scenes is None:
            found: set[str] = set()
            for rel, pattern in SCENE_SOURCES:
                path = self.fs.abs(rel)
                if self.fs.is_file(path):
                    found |= set(re.findall(pattern, self.fs.read_text(path)))
            self._scenes = found
        return self._scenes

    def default_files(self) -> list[str]:
        files: list[str] = []
        for rel in DEFAULT_PATHS:
            path = self.fs.abs(rel)
            if self.fs.is_dir(path):
                files += self.fs.markdown_under(path)
            elif self.fs.is_file(path):
                files.append(path)
        return files

    def expand(self, paths: Iterable[str]) -> list[str]:
        files: list[str] = []
        for p in paths:
            path = os.path.normpath(os.path.abspath(p)) if self.fs.mem is None else self.fs.abs(p)
            if self.fs.is_dir(path):
                files += self.fs.markdown_under(path)
            elif self.fs.is_file(path):
                files.append(path)
            else:
                raise FileNotFoundError(p)
        return list(dict.fromkeys(files))

    def pending_allows(self, dest: str, anchor: str | None) -> bool:
        rel = self.fs.rel(dest)
        if rel is None:
            return False
        if rel in self.pending_files:
            return True
        if anchor is None:
            return False
        return any(p == rel and fnmatch.fnmatchcase(anchor, a) for p, a in self.pending_anchors)

    def resolve(self, doc: Doc, target: str) -> tuple[str, str | None, str] | None:
        """(destination path, anchor or None, cleaned target) for a relative link."""
        t = target.strip()
        if t.startswith("<") and t.endswith(">"):
            t = t[1:-1].strip()
        if not t or t.startswith("//") or SCHEME_RE.match(t):
            return None
        path_part, hash_sign, frag = t.partition("#")
        path_part = unquote(path_part.split("?", 1)[0])
        if not path_part:
            dest = doc.path
        elif path_part.startswith("/"):
            dest = self.fs.abs(path_part.lstrip("/"))
        else:
            dest = os.path.normpath(os.path.join(os.path.dirname(doc.path), path_part))
        anchor = unquote(frag).lower() if hash_sign and frag else None
        return dest, anchor, t

    # -- driver ------------------------------------------------------------------------

    def run(self, paths: Iterable[str] | None = None) -> list[Problem]:
        files = self.expand(paths) if paths else self.default_files()
        inbound: set[tuple[str, str]] = set()
        for path in dict.fromkeys(files + self.default_files()):
            doc = self.doc(path)
            if doc is None:
                continue
            for link in doc.links:
                resolved = self.resolve(doc, link.target)
                if resolved and resolved[1]:
                    inbound.add((resolved[0], resolved[1]))
        for path in files:
            doc = self.doc(path)
            if doc is None:
                self.add(path, 1, "link-missing-file", "cannot read this file as UTF-8 text")
                continue
            self.check_doc(doc, inbound)
        if self.fs.abs(GALLERY) in files:
            self.check_image_folder()
        rank = {self.display(path): i for i, path in enumerate(files)}
        return sorted(
            dict.fromkeys(self.problems),
            key=lambda p: (rank.get(p.path, len(rank)), p.path, p.line, p.rule),
        )

    def add(self, path: str, line: int, rule: str, message: str, doc: Doc | None = None) -> None:
        severity, scope, _ = RULES[rule]
        if self.links_only and scope != LINKS:
            return
        if severity == WARNING and not self.warnings:
            return
        rel = self.fs.rel(path)
        if rel is not None and rule in FILE_RULES.get(rel, FileRules()).off:
            return
        if doc is not None and 1 <= line <= len(doc.lines) and doc.line(line).ignored:
            return
        self.problems.append(Problem(self.display(path), line, rule, message))

    # -- per document ------------------------------------------------------------------

    def check_doc(self, doc: Doc, inbound: set[tuple[str, str]]) -> None:
        rel = self.fs.rel(doc.path)
        rules = FILE_RULES.get(rel or "", FileRules())

        def emit(line: int, rule: str, message: str) -> None:
            self.add(doc.path, line, rule, message, doc)

        self.check_links(doc, emit)
        self.check_anchors(doc, emit, inbound, rel)
        if self.links_only:
            return
        theory = (
            rel is not None
            and rel.startswith("docs/theory/")
            and rel.count("/") == 2
            and rel != "docs/theory/README.md"
        )
        self.check_lines(doc, emit)
        self.check_headings(doc, emit, theory)
        self.check_alerts(doc, emit, rules, theory)
        self.check_tables(doc, emit)
        self.check_lists(doc, emit)
        self.check_setext(doc, emit)
        self.check_cards(doc, emit)
        self.check_fences(doc, emit, rules)
        self.check_paragraphs(doc, emit)
        if theory:
            self.check_pager(doc, emit)

    def check_links(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        for link in doc.links:
            target = link.target.strip().strip("<>")
            base = os.path.basename(unquote(target.split("#", 1)[0].split("?", 1)[0]).rstrip("/"))
            if base in FORBIDDEN_LINKS:
                emit(link.line, "link-forbidden", f"never link {base}: the public mirror strips it")
                continue
            resolved = self.resolve(doc, link.target)
            if resolved is None:
                continue
            dest, anchor, clean = resolved
            if anchor and re.fullmatch(r"l\d+(?:-l\d+)?", anchor) and not dest.endswith(".md"):
                emit(link.line, "link-line-anchor", f"no line anchors in source links: {clean}")
            if not self.fs.exists(dest):
                if self.pending_allows(dest, None):
                    self.pending_hits += 1
                    continue
                rule = "image-missing" if link.image else "link-missing-file"
                emit(link.line, rule, f"missing file: {clean}")
                continue
            if link.image and dest.lower().endswith(".png"):
                self.check_png(dest, doc, link.line, emit)
            if anchor is None or not dest.endswith(".md") or not self.fs.is_file(dest):
                continue
            target_doc = self.doc(dest)
            if target_doc is None or anchor in target_doc.names:
                continue
            if self.pending_allows(dest, anchor):
                self.pending_hits += 1
                continue
            emit(link.line, "link-missing-anchor", f"missing anchor: {clean}")

    def check_png(
        self, dest: str, doc: Doc, line: int, emit: Callable[[int, str, str], None]
    ) -> None:
        rel = self.fs.rel(dest)
        name = os.path.basename(dest)
        if rel == f"{IMAGES_DIR}/{name}" and name not in self.scenes():
            emit(line, "image-no-scene", f"{name} has no capture scene ({_scene_hint()})")
        self.check_png_size(dest)

    def check_png_size(self, dest: str) -> None:
        if dest in self._big_reported or not self.fs.is_file(dest):
            return
        self._big_reported.add(dest)
        kb = self.fs.size(dest) / 1024
        if kb > LIMITS["png_kb"]:
            self.add(dest, 1, "png-size", f"{kb:.0f} KB (soft cap {LIMITS['png_kb']} KB)")

    def check_anchors(
        self,
        doc: Doc,
        emit: Callable[[int, str, str], None],
        inbound: set[tuple[str, str]],
        rel: str | None,
    ) -> None:
        slugs = {a.name for a in doc.anchors if not a.explicit}
        seen: dict[str, int] = {}
        for a in doc.anchors:
            if (doc.path, a.name.lower()) in inbound and a.details:
                emit(a.line, "anchor-in-details", f"#{a.name} is linked to but sits in <details>")
            if not a.explicit:
                continue
            if not re.fullmatch(r"[a-z0-9-]+", a.name):
                emit(a.line, "anchor-id-format", f"id '{a.name}' must be lowercase ASCII [a-z0-9-]")
            if a.name in seen:
                emit(a.line, "anchor-duplicate",
                     f"id '{a.name}' is already defined on line {seen[a.name]}")
            else:
                seen[a.name] = a.line
            if a.name in slugs:
                emit(
                    a.line,
                    "anchor-collision",
                    f"id '{a.name}' equals a heading slug; add an id only when the slug differs",
                )
        for name in PINNED_ANCHORS.get(rel or "", ()):
            if name not in doc.names:
                emit(1, "anchor-pinned", f"pinned anchor #{name} is missing (inbound links use it)")
        if self.links_only:
            return
        for i, ln in enumerate(doc.lines):
            if ln.kind == "text" and ANCHOR_ONLY_RE.match(ln.bare):
                nxt = doc.lines[i + 1] if i + 1 < len(doc.lines) else None
                if nxt is None or not nxt.content.lstrip().startswith("**"):
                    emit(
                        ln.n,
                        "anchor-standalone",
                        "a standalone <a id> line must sit directly above a bold reference or "
                        'glossary entry; elsewhere put the id inside the heading: '
                        '## <a id="old-slug"></a>Heading',
                    )

    def check_lines(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        in_card = 0
        for ln in doc.lines:
            if ln.marker:
                emit(ln.n, "ignore-marker", ln.marker)
            if ln.kind == "fence":
                for phrase, where in STALE_PHRASES:
                    if where == "raw" and phrase in ln.raw:
                        emit(ln.n, "typo-stale", f"stale phrase '{phrase}'")
                continue
            if ln.kind not in ("text", "html"):
                continue
            bare = ln.bare
            prose = prose_text(bare)
            # math
            if "$$" in prose:
                emit(ln.n, "math-display", "$$ display math: use a ```math fence")
            elif "$" in prose:
                emit(ln.n, "math-bare-dollar", "bare $: inline math is $`...`$; a literal $ is "
                     "<span>$</span>")
            if re.search(r"\\\(.*?\\\)", prose):
                emit(ln.n, "math-delimiter", "\\(...\\) does not render on GitHub: use $`...`$")
            for kind, _, inner in ln.segs:
                if kind == "math" and re.search(r"[<>]", inner):
                    emit(ln.n, "math-angle", "< or > inside math: write \\lt or \\gt")
            # HTML and figures
            if re.search(r"<[A-Za-z][^<>]*\sstyle\s*=", bare):
                emit(ln.n, "html-style-attr", "style= is stripped by GitHub")
            if re.search(r"<(?:figure|figcaption|small)\b", bare, re.I):
                emit(ln.n, "html-banned-tag", "GitHub strips <figure>, <figcaption> and <small>")
            if re.search(r"!\[", prose):
                emit(ln.n, "figure-markdown-image", "use an HTML figure card, not ![]()")
            if ln.kind == "html":
                found = re.search(r"\*\*[^*\s][^*]*\*\*|`[^`]+`|\]\([^)]*\)", ln.text)
                if found:
                    emit(
                        ln.n,
                        "html-markdown",
                        f"Markdown is not rendered inside an HTML block ('{found.group(0)[:30]}');"
                        " use <b>, <code> or <a href>",
                    )
            in_card += len(re.findall(r"<table\b", bare, re.I))
            split = doc.split_imgs.get(ln.n, [])  # tags continued on later lines, joined
            imgs = [(m.start(), m.group(0)) for m in re.finditer(r"<img\b[^>]*>", bare, re.I)]
            for pos, tag in imgs + split:
                self.check_img(ln, tag, pos, in_card > 0, emit)
            if in_card:
                tags = [m.group(0) for m in re.finditer(r"<(?:td|th|img)\b[^>]*>", bare, re.I)]
                for tag in tags + [tag for _pos, tag in split]:
                    width = re.search(r"\bwidth\s*=\s*[\"']?\s*([^\"'\s>]+)", tag, re.I)
                    if width and not width.group(1).endswith("%"):
                        emit(ln.n, "figure-pixel-width",
                             f"width=\"{width.group(1)}\" in a card: use a percentage")
            in_card = max(0, in_card - len(re.findall(r"</table\s*>", bare, re.I)))
            # typography
            delim = bool(DELIM_RE.match(ln.text) or THEMATIC_RE.match(ln.text)
                         or SETEXT_RE.match(ln.text))
            for rule, pattern, message in TYPO_RULES:
                if rule == "typo-dash" and delim:
                    continue
                m = pattern.search(prose)
                if m:
                    emit(ln.n, rule, f"{message}: {_snippet(prose, m.start(), m.end())}")
            formula = self.find_formula(prose)
            if formula:
                emit(
                    ln.n,
                    "typo-formula",
                    f"ASCII '{formula}': write Sb₂Te₃-style subscripts and ¹²⁵Te-style "
                    "superscripts (identifiers belong in code spans)",
                )
            for phrase, where in STALE_PHRASES:
                if phrase in (prose if where == "prose" else ln.text):
                    emit(ln.n, "typo-stale", f"stale phrase '{phrase}'")

    @staticmethod
    def find_formula(text: str) -> str | None:
        for m in FORMULA_RE.finditer(text):
            parts = re.findall(r"([A-Z][a-z]?)([0-9]*)", m.group(0))
            if (
                len(parts) >= 2
                and all(sym in ELEMENTS for sym, _ in parts)
                and any(num for _, num in parts)
            ):
                return m.group(0)
        for m in ISOTOPE_RE.finditer(text):
            if m.group(2) in ELEMENTS:
                return m.group(0)
        return None

    def check_img(
        self,
        ln: Line,
        tag: str,
        pos: int,
        in_card: bool,
        emit: Callable[[int, str, str], None],
    ) -> None:
        attrs = {
            a.group(1).lower(): next((v for v in a.groups()[1:] if v is not None), "")
            for a in ATTR_RE.finditer(tag[4:])
        }
        alt = attrs.get("alt")
        if not alt or not alt.strip():
            emit(ln.n, "figure-img", "<img> needs non-empty alt text")
        else:
            if "|" in alt:
                emit(ln.n, "figure-img", "raw | in alt text breaks tables: write &#124;")
            length = len(html.unescape(alt))
            if length > LIMITS["alt_chars"]:
                emit(ln.n, "figure-img", f"alt text is {length} characters (max 125)")
        if "width" not in attrs and "height" not in attrs:
            emit(ln.n, "figure-img", "<img> needs width= (cards) or height= (thumbnails)")
        if ln.table and "height" in attrs and not in_card:
            cell = ln.bare[: pos].rsplit("|", 1)[-1]
            if not re.search(r"<a\s[^>]*href\s*=[^>]*>\s*$", cell, re.I):
                emit(
                    ln.n,
                    "figure-thumb-link",
                    "wrap the thumbnail in <a href=\"page.md\">…</a>, or GitHub links the PNG",
                )

    def check_headings(
        self, doc: Doc, emit: Callable[[int, str, str], None], theory: bool
    ) -> None:
        h1 = [h.line for h in doc.headings if h.level == 1]
        h1 += [ln.n for ln in doc.lines if ln.kind in ("text", "html") and "<h1" in ln.bare.lower()]
        h1.sort()
        if len(h1) != 1 and self.fs.rel(doc.path) is not None:  # snippets outside the repo
            line = h1[1] if len(h1) > 1 else 1  # are not pages
            emit(line, "heading-h1", f"a page has exactly one H1 (found {len(h1)})")
        prev = 0
        for h in doc.headings:
            if prev and h.level > prev + 1:
                emit(h.line, "heading-skip", f"heading level jumps from H{prev} to H{h.level}")
            prev = h.level
            md = h.markdown
            if "SPECULATIVE" in md:
                emit(h.line, "heading-content", "no status tags in headings")
            if h.has_id:
                continue
            found = []
            if "&" in md:
                found.append("&")
            if "`" in md:
                found.append("code")
            if "$" in md:
                found.append("math")
            if ARROW_RE.search(md):
                found.append("an arrow")
            if TAG_RE.search(md):
                found.append("an HTML tag")
            if found:
                emit(
                    h.line,
                    "heading-content",
                    f"heading contains {', '.join(found)}: reword it, or keep the old anchor "
                    'with an explicit id: ## <a id="old-slug"></a>Heading',
                )
        if theory:
            h2 = [h for h in doc.headings if h.level == 2]
            position = -1
            for h in h2:
                name = h.text
                if name not in THEORY_H2:
                    emit(h.line, "theory-h2",
                         f"'{name}' is not a template H2 ({' · '.join(THEORY_H2)})")
                    continue
                index = THEORY_H2.index(name)
                if index <= position:
                    emit(h.line, "theory-h2", f"'{name}' is out of template order or repeated")
                position = max(position, index)

    def check_alerts(
        self,
        doc: Doc,
        emit: Callable[[int, str, str], None],
        rules: FileRules,
        theory: bool,
    ) -> None:
        lines = doc.lines
        counts: Counter[str] = Counter()
        total = 0
        first_h2 = next((h.line for h in doc.headings if h.level == 2), len(lines) + 1)
        status_before_h2 = 0
        alert_lines: set[int] = set()
        for i, ln in enumerate(lines):
            if ln.kind != "text":
                continue
            m = ALERT_RE.match(ln.raw)
            if not m or not ln.text.strip():
                continue
            kind = m.group(3)
            total += 1
            prev = lines[i - 1] if i else None
            if m.group(1) or m.group(2).count(">") != 1 or ln.details:
                emit(ln.n, "alert-placement",
                     "alerts render only at the top level: not in a list, quote or <details>")
            elif prev is not None and prev.quote > 0 and prev.kind == "text":
                emit(ln.n, "alert-placement",
                     "the [!TYPE] marker must open its own blockquote (leave a blank line)")
            if m.group(4).strip():
                emit(ln.n, "alert-placement", "the [!TYPE] marker must be alone on its line")
            if kind not in ALERT_TYPES:
                emit(ln.n, "alert-type", f"unknown alert [!{kind}]: use one of "
                     f"{', '.join(ALERT_TYPES)} (capitals)")
                continue
            counts[kind] += 1
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            following = nxt.content.lstrip() if nxt is not None and nxt.quote > 0 else ""
            j = i - 1
            while j >= 0 and lines[j].kind in ("blank", "comment"):
                j -= 1
            if j >= 0 and lines[j].n in alert_lines:
                emit(ln.n, "alert-consecutive", "two alerts in a row: put text between them")
            k = i + 1
            while k < len(lines) and lines[k].quote > 0 and lines[k].kind != "blank":
                k += 1
            alert_lines.update(lines[x].n for x in range(i, k))
            if kind in STATUS_LABELS:
                s = STATUS_RE.match(following)
                allowed = STATUS_LABELS[kind]
                if not s or s.group(1) not in allowed:
                    emit(ln.n, "alert-status",
                         f"[!{kind}] must be followed by '> **Status: "
                         f"{' | '.join(allowed)}.**'")
                if theory and ln.n < first_h2:
                    status_before_h2 += 1
                body = " ".join(lines[x].content for x in range(i + 1, k))
                body = re.sub(r"^\s*\*\*Status:[^*]*\*\*", "", body)
                words = count_words(strip_tags(code_free(scan_inline(body)[0])))
                if words > LIMITS["alert_words"]:
                    emit(ln.n, "alert-length",
                         f"status alert has {words} words after the label (max 60)")
            elif kind == "IMPORTANT":
                if not re.match(r"^\*\*(?!Status\b)[^*]+\*\*", following):
                    emit(ln.n, "alert-label",
                         "[!IMPORTANT] opens with a bold label that is not a status")
            elif kind == "TIP" and not re.match(r"^\*\*[^*]+\*\*", following):
                emit(ln.n, "alert-label", "[!TIP] opens with a bold label, e.g. **Try it:**")
            if kind in ("IMPORTANT", "TIP") and counts[kind] > 1:
                emit(ln.n, "alert-count", f"at most one [!{kind}] per page")
            if rules.alert_total is not None and total == rules.alert_total + 1:
                emit(ln.n, "alert-count", f"at most {rules.alert_total} alerts per page")
        if theory and status_before_h2 != 1:
            emit(1, "theory-status",
                 f"a theory page has exactly one status alert before its first H2 "
                 f"(found {status_before_h2})")

    def check_tables(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        lines = doc.lines
        header_cells = 0
        for i, ln in enumerate(lines):
            if ln.table == "badheader":
                emit(ln.n, "structure-table-cells",
                     "header and delimiter rows differ in cell count: GitHub shows plain text")
                continue
            if not ln.table:
                continue
            if ln.table == "header":
                header_cells = len(split_cells(ln.text))
                if header_cells > LIMITS["table_columns"]:
                    emit(ln.n, "table-columns",
                         f"table has {header_cells} columns (max {LIMITS['table_columns']})")
            starts_pipe = ln.content.lstrip().startswith("|")
            if ln.table == "row" and not starts_pipe:
                emit(ln.n, "structure-table-glued",
                     "this line joins the table above as a row: leave a blank line after tables")
            elif ln.table == "row" and len(split_cells(ln.text)) != header_cells:
                emit(ln.n, "structure-table-cells",
                     f"row has {len(split_cells(ln.text))} cells, header has {header_cells} "
                     "(a raw | splits cells, even in code: write &#124;)")
            for kind, _, inner in ln.segs:
                if kind in ("code", "math") and re.search(r"(?<!\\)\|", inner):
                    emit(ln.n, "structure-table-pipe",
                         "a | inside code or math still splits the cell: write &#124; or \\lvert")
                    break
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if (
                ln.table in ("header", "delim", "row")
                and nxt is not None
                and not nxt.table
                and nxt.kind in ("text", "html", "fence")
                and not nxt.blank
            ):
                emit(nxt.n, "structure-table-glued",
                     "leave a blank line after a table (a glued line joins it as a row)")

    def check_lists(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        lines = doc.lines
        for i in range(len(lines) - 1):
            ln, nxt = lines[i], lines[i + 1]
            # a list item line, or an indented continuation of one
            if ln.kind != "text" or ln.blank or not (ln.item or (ln.listed and ln.indent)):
                continue
            if nxt.kind in ("blank", "comment") or nxt.blank or nxt.item:
                continue
            if nxt.quote != ln.quote or nxt.indent > 0:
                continue
            emit(nxt.n, "structure-list-glued",
                 "line glued under a list item joins the item (or needs a blank line before it)")

    def check_setext(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        lines = doc.lines
        for i in range(1, len(lines)):
            ln, prev = lines[i], lines[i - 1]
            if ln.kind not in ("text", "html") or not SETEXT_RE.match(ln.text):
                continue
            if ln.table or prev.blank or prev.kind in ("blank", "comment", "fence"):
                continue
            if prev.kind == "html":
                emit(ln.n, "structure-setext",
                     f"'{ln.text.strip()}' under an HTML block is swallowed by it: leave a blank "
                     "line before it")
            elif not (prev.level or prev.listed or prev.table or THEMATIC_RE.match(prev.text)):
                emit(ln.n, "structure-setext",
                     f"'{ln.text.strip()}' under a text line turns it into a heading: leave a "
                     "blank line before it")

    def check_cards(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        lines = doc.lines
        depth = 0
        block: list[Line] = []
        for i, ln in enumerate(lines):
            if ln.kind in ("text", "html"):
                if "</summary>" in ln.bare.lower():
                    nxt = lines[i + 1] if i + 1 < len(lines) else None
                    if nxt is not None and nxt.kind != "blank":
                        emit(ln.n, "structure-details",
                             "leave a blank line after </summary> or the Markdown inside will "
                             "not render")
                opened = len(re.findall(r"<table\b", ln.bare, re.I))
                closed = len(re.findall(r"</table\s*>", ln.bare, re.I))
                if depth or opened:
                    block.append(ln)
                depth = max(0, depth + opened - closed)
                if not depth and block:
                    self.check_captions(block, emit)
                    block = []
            elif ln.kind == "blank" and depth:
                emit(ln.n, "structure-card", "no blank lines inside an HTML figure card")

    def check_captions(self, block: list[Line], emit: Callable[[int, str, str], None]) -> None:
        text = "\n".join(ln.bare for ln in block)
        starts = []
        offset = 0
        for ln in block:
            starts.append(offset)
            offset += len(ln.bare) + 1

        def line_at(pos: int) -> int:
            index = 0
            for k, s in enumerate(starts):
                if s <= pos:
                    index = k
            return block[index].n

        for tr in re.finditer(r"<tr\b.*?</tr\s*>", text, re.S | re.I):
            cells = list(re.finditer(r"<td\b[^>]*>(.*?)</td\s*>", tr.group(0), re.S | re.I))
            narrow = len(cells) >= 3
            limit = LIMITS["caption_words_narrow" if narrow else "caption_words"]
            for cell in cells:
                body = cell.group(1)
                m = re.search(r"<br\s*/?>\s*<sub>", body, re.I)
                if not m:
                    continue
                caption = _balanced_sub(body, m.end())
                caption = re.split(r"<br\s*/?>", caption, flags=re.I)[-1]
                caption = re.sub(r"^\s*<b>.*?</b>", "", caption, flags=re.S | re.I)
                caption = re.sub(r"<b>\((?:HIGHLY )?SPECULATIVE\)</b>", "", caption)
                words = count_words(strip_tags(caption))
                if words > limit:
                    pos = tr.start() + cell.start(1) + m.start()
                    emit(line_at(pos), "figure-caption-length",
                         f"caption has {words} words (max {limit}"
                         f"{' in 3-up cells' if narrow else ''})")

    def check_fences(
        self, doc: Doc, emit: Callable[[int, str, str], None], rules: FileRules
    ) -> None:
        lines = doc.lines
        diagrams = 0
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.kind != "fence" or ln.role != "open":
                i += 1
                continue
            j = i + 1
            while j < len(lines) and lines[j].kind == "fence" and lines[j].role == "body":
                j += 1
            body = lines[i + 1 : j]
            if ln.fence == "math":
                for b in body:
                    if re.search(r"[<>]", b.content):
                        emit(b.n, "math-angle", "< or > inside math: write \\lt or \\gt")
            elif ln.fence == "mermaid":
                diagrams += 1
                if diagrams == rules.mermaid_max + 1:
                    limit = rules.mermaid_max
                    emit(ln.n, "mermaid-count",
                         "no Mermaid diagrams on this page" if limit == 0
                         else f"at most {limit} Mermaid diagram{'s' if limit > 1 else ''} "
                         "per page")
                self.check_mermaid(body, emit)
            i = j + 1 if j < len(lines) and lines[j].role == "close" else j

    def check_mermaid(self, body: list[Line], emit: Callable[[int, str, str], None]) -> None:
        nodes: set[str] = set()
        subgraphs: set[str] = set()
        kind = ""
        for b in body:
            code = b.content.strip()
            if not code:
                continue
            if code.startswith("%%"):
                if code.startswith("%%{"):
                    emit(b.n, "mermaid-init", "no %%{init}%% directives (they pin one theme)")
                continue
            if not kind:
                kind = code.split()[0]
                if kind not in MERMAID_TYPES:
                    hint = " (write 'flowchart')" if kind == "graph" else ""
                    emit(b.n, "mermaid-type",
                         f"'{kind}' diagrams are not allowed: flowchart or pie only{hint}")
                continue
            word = code.split()[0]
            if word == "style":
                emit(b.n, "mermaid-style", "no style lines: apply a shared classDef with 'class'")
                continue
            if word == "classDef":
                self.check_classdef(b, emit)
                continue
            if word in ("class", "linkStyle", "click", "direction", "end", "title"):
                continue
            file_name = FILE_NAME_RE.search(code)
            if file_name:
                emit(b.n, "mermaid-filename",
                     f"file name in a label ('{file_name.group(0)}'): use words")
            if kind != "flowchart":
                continue
            if word == "subgraph":
                sm = re.match(r"subgraph\s+([A-Za-z_][\w-]*)", code)
                if sm:
                    subgraphs.add(sm.group(1))
                continue
            unquoted = re.sub(r'"[^"]*"', '""', code)
            unquoted = re.sub(r"\|[^|]*\|", " ", unquoted)
            if re.search(r"[A-Za-z_][\w-]*\s*(?:\[\[|\[\(|\(\(|\(\[|\{\{|\[/|\[\\|[\[({>])"
                         r"\s*(?![\"\[\](){}/\\>\s])", unquoted):
                emit(b.n, "mermaid-unquoted", 'quote node labels: A["Label"]')
            shapes = re.sub(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|>[^\]]*\]", " ", unquoted)
            shapes = re.sub(r":::\s*[\w-]+", " ", shapes)
            shapes = re.sub(r"(?:--|==|-\.)[^->=.]*?(?:-->|==>|\.->|---)", " --> ", shapes)
            for token in re.findall(r"(?<![\w-])([A-Za-z_][\w]*)(?![\w-])", shapes):
                if token not in ("end", "subgraph", "direction", "TB", "TD", "BT", "RL", "LR"):
                    nodes.add(token)
        count = len(nodes - subgraphs)
        if count > LIMITS["mermaid_nodes"] and body:
            emit(body[0].n - 1 if body[0].n > 1 else 1, "mermaid-nodes",
                 f"diagram has {count} nodes (max {LIMITS['mermaid_nodes']})")

    @staticmethod
    def check_classdef(b: Line, emit: Callable[[int, str, str], None]) -> None:
        m = re.match(r"^\s*classDef(\s+)([\w,-]+)(\s+)(\S.*?)\s*;?\s*$", b.content)
        if not m:
            emit(b.n, "mermaid-classdef", "malformed classDef")
            return
        if m.group(1) != " " or m.group(3) != " ":
            emit(b.n, "mermaid-classdef", "write classDef lines with single spaces")
        names, style = m.group(2).split(","), m.group(4)
        fill = re.search(r"(?:^|,)\s*fill\s*:\s*([^,]+)", style)
        no_fill = fill is None or fill.group(1).strip().lower() in ("none", "transparent")
        if not no_fill and not re.search(r"(?:^|,)\s*color\s*:", style):
            emit(b.n, "mermaid-classdef",
                 "a classDef that sets fill: must also set color: (dark mode)")
        for name in names:
            canon = CANONICAL_CLASSDEFS.get(name)
            if canon is not None and style.replace(" ", "") != canon.replace(" ", ""):
                emit(b.n, "mermaid-classdef", f"use the canonical line: classDef {name} {canon}")

    def check_paragraphs(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        chunk: list[Line] = []

        def flush() -> None:
            if chunk:
                text = " ".join(
                    "".join(raw if kind == "text" else " x " for kind, raw, _ in c.segs
                            if kind != "comment")
                    for c in chunk
                )
                words = count_words(strip_tags(prose_text(text)))
                if words > LIMITS["paragraph_words"]:
                    emit(chunk[0].n, "para-length",
                         f"{'list item' if chunk[0].item else 'paragraph'} has {words} words "
                         f"(max {LIMITS['paragraph_words']})")
                chunk.clear()

        for ln in doc.lines:
            paragraph = (
                ln.kind == "text"
                and not ln.blank
                and not ln.level
                and not ln.table
                and not ALERT_RE.match(ln.raw)
                and not SETEXT_RE.match(ln.text)
            )
            if not paragraph:
                flush()
                continue
            if ln.item or (chunk and chunk[-1].quote != ln.quote):
                flush()
            chunk.append(ln)
        flush()

    def check_pager(self, doc: Doc, emit: Callable[[int, str, str], None]) -> None:
        lines = doc.lines
        rules = [
            i
            for i, ln in enumerate(lines)
            if ln.kind == "text"
            and THEMATIC_RE.match(ln.text)
            and (i == 0 or lines[i - 1].kind == "blank")
        ]
        if not rules:
            emit(len(lines) or 1, "theory-pager",
                 "end the page with ---, then the one-line pager "
                 "[← Prev](…) · [Theory index](README.md) · [Next →](…)")
            return
        after = lines[rules[-1] + 1 :]
        pagers = [ln for ln in after if re.search(r"\]\(README\.md(?:#[^)]*)?\)", ln.bare)]
        if not pagers:
            emit(lines[rules[-1]].n, "theory-pager",
                 "after the last ---, add the one-line pager linking [Theory index](README.md)")
            return
        if all(ln.table for ln in pagers):
            emit(pagers[0].n, "theory-pager", "the pager is one line, not a table")
            return
        page = os.path.basename(doc.path)[:-3]
        if page not in THEORY_ORDER:
            return
        pager = next(ln for ln in pagers if not ln.table)
        k = THEORY_ORDER.index(page)
        neighbors = THEORY_ORDER[max(0, k - 1) : k] + THEORY_ORDER[k + 1 : k + 2]
        missing = [
            n
            for n in neighbors
            if not re.search(rf"\]\({re.escape(n)}\.md(?:#[^)]*)?\)", pager.bare)
        ]
        if missing:
            emit(pager.n, "theory-pager",
                 "the pager links the previous and next pages in reading order; missing "
                 + ", ".join(f"{n}.md" for n in missing))

    # -- repository-level image checks -------------------------------------------------

    def check_image_folder(self) -> None:
        folder = self.fs.abs(IMAGES_DIR)
        pngs = [p for p in self.fs.files_in(folder) if p.lower().endswith(".png")]
        scenes = self.scenes()
        for png in pngs:
            name = os.path.basename(png)
            if name not in scenes:
                self.add(png, 1, "image-no-scene", f"{name} has no capture scene ({_scene_hint()})")
            self.check_png_size(png)
        gallery = self.doc(self.fs.abs(GALLERY))
        if gallery is None:
            return
        section = None
        for h in gallery.headings:
            if section is None and h.level == 2 and "inventory" in h.text.lower():
                section = (h.line, len(gallery.lines) + 1)
            elif section is not None and h.level <= 2 and h.line > section[0]:
                section = (section[0], h.line)
                break
        if section is None:
            return
        listed: dict[str, int] = {}
        for ln in gallery.lines[section[0] : section[1] - 1]:
            if ln.table not in ("row",):
                continue
            first = split_cells(ln.text)[0]
            for name in re.findall(r"([\w.-]+\.png)", first):
                listed.setdefault(name, ln.n)
        names = {os.path.basename(p) for p in pngs}
        for png in pngs:
            name = os.path.basename(png)
            if name not in listed:
                self.add(self.fs.abs(GALLERY), section[0], "image-no-inventory",
                         f"{name} has no row in the Inventory table")
        for name, line in listed.items():
            if name not in names:
                self.add(self.fs.abs(GALLERY), line, "image-missing",
                         f"inventory row for {name}, which is not in {IMAGES_DIR}/", gallery)


def _snippet(text: str, start: int, end: int, pad: int = 18) -> str:
    """The match with a little context, for locating it on a long line."""
    a, b = max(0, start - pad), min(len(text), end + pad)
    inner = " ".join(text[a:b].split())
    return f'"{"…" if a else ""}{inner}{"…" if b < len(text) else ""}"'


def _scene_hint() -> str:
    return 'a literal "….png" in capture_screenshots.py or filename: "….png" in capture.rs'


def _balanced_sub(text: str, start: int) -> str:
    """Content of the <sub> opened just before ``start``, honoring nested <sub>."""
    depth, i = 1, start
    for m in re.finditer(r"<(/?)sub\b[^>]*>", text[start:], re.I):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return text[start : start + m.start()]
        i = start + m.end()
    return text[start:] if i >= start else ""


# ---------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------


def report(problems: list[Problem], checker: Checker, out: Callable[[str], None]) -> int:
    for p in problems:
        out(p.format())
    errors = [p for p in problems if p.severity == ERROR]
    warnings = [p for p in problems if p.severity == WARNING]
    if problems:
        out("")
        out("Summary by rule:")
        for rule, count in sorted(Counter(p.rule for p in problems).items()):
            out(f"  {rule:<24} {count:>5}  {RULES[rule][0]}  ({RULES[rule][2]})")
    if checker.pending_hits:
        out(f"{checker.pending_hits} link(s) allowed by --pending (forward links).")
    files = len({p.path for p in problems})
    out(f"{len(errors)} error(s), {len(warnings)} warning(s) in {files} file(s).")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint README.md, CONTRIBUTING.md and docs/ against the docs conventions.",
    )
    parser.add_argument("paths", nargs="*", metavar="PATH", help="files or directories")
    parser.add_argument("--links-only", action="store_true",
                        help="run only the link, anchor and image checks")
    parser.add_argument("--pending", metavar="FILE",
                        help="allow forward links listed in FILE (path or path#anchor lines)")
    parser.add_argument("--self-test", action="store_true",
                        help="lint the built-in fixtures and exit")
    parser.add_argument("--no-warnings", action="store_true", help="print errors only")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    pending: list[str] = []
    if args.pending:
        try:
            with open(args.pending, encoding="utf-8") as fh:
                pending = fh.read().splitlines()
        except OSError as exc:
            print(f"check_docs.py: cannot read --pending file: {exc}", file=sys.stderr)
            return 2
    checker = Checker(
        Fs(ROOT), links_only=args.links_only, warnings=not args.no_warnings, pending=pending
    )
    try:
        problems = checker.run(args.paths or None)
    except FileNotFoundError as exc:
        print(f"check_docs.py: no such file or directory: {exc}", file=sys.stderr)
        return 2
    return report(problems, checker, print)


# ---------------------------------------------------------------------------------------
# Self-test fixtures
# ---------------------------------------------------------------------------------------

SELFTEST_PNGS = (
    "viewer-2d.png", "viewer-gap-2d.png", "fourier.png", "sweep-twist.png",
    "graphene-pattern.png", "graphene-bands.png", "graphene-dos.png", "magnetic-vortex-2d.png",
    "magnetic-currents-3d.png", "phase-diagram.png", "pdw-tsweep.png", "pdw-fft.png",
)
SELFTEST_RUST_PNGS = ("rust-desktop-2d.png", "rust-desktop-3d.png", "rust-density-3d.png")


def _selftest_base() -> dict[str, str | bytes | int]:
    scenes = ", ".join(f'Scene("{p}")' for p in SELFTEST_PNGS)
    rust = ", ".join(f'Scene {{ filename: "{p}" }}' for p in SELFTEST_RUST_PNGS)
    rows = "\n".join(
        f"| <code>{p}</code> | {p[:-4]} · Python | app view | 1440 × 1440 | Caption. | "
        "[Moire patterns](../theory/moire-patterns.md) |"
        for p in SELFTEST_PNGS + SELFTEST_RUST_PNGS
    )
    files: dict[str, str | bytes | int] = {
        "scripts/capture_screenshots.py": f"SCENES = [{scenes}]\n",
        "crates/moire-desktop/src/bin/capture.rs": f"const SCENES: &[Scene] = &[{rust}];\n",
        "src/waytogocoop/computation/superconducting.py": "",
        "crates/moire-core/src/density.rs": "",
        "docs/theory/README.md": "# Theory Index\n\n## Core pipeline\n\n## Graphene track\n",
        "docs/theory/moire-patterns.md": "# Moire Patterns\n",
        "docs/theory/fourier-analysis.md": "# Fourier Analysis\n",
        "docs/theory/magnetic.md": "# Magnetic Field Effects\n\n## Speculative components\n",
        "docs/theory/pair-density-waves.md": "# Pair Density Waves\n",
        "docs/theory/isotope-effects.md": "# Isotope Effects\n",
        "docs/glossary.md": (
            "# Glossary\n\n## Superconductivity\n\n<a id=\"term-cpdm\"></a>\n"
            "**Cooper-pair density modulation (CPDM)** — an imprinted modulation.\n\n"
            "## Modulated pairing\n\nText.\n"
        ),
        "docs/references.md": (
            "# References\n\n<a id=\"ref-wang-2026\"></a>\n"
            "**Z. Wang et al.**, Nature 652, 335 (2026).\n"
        ),
        "docs/images/README.md": (
            "# Figure Gallery\n\n<sub>[Documentation](../README.md)</sub>\n\n"
            "## Inventory\n\n<details>\n<summary><b>Inventory:</b> all figures</summary>\n\n"
            "| Image | Scene · generator | Kind | Frame (px) | Canonical caption | Used in |\n"
            "|---|---|---|---|---|---|\n" + rows + "\n\n</details>\n"
        ),
        "docs/README.md": "# Good Job Coop Documentation\n",
        "README.md": "# Good Job Coop!\n",
    }
    for p in SELFTEST_PNGS + SELFTEST_RUST_PNGS:
        files[f"docs/images/{p}"] = 2048
    return files


GOOD_THEORY_PAGE = """\
# Gap Modulation and the CPDM State

<sub>[Good Job Coop](../../README.md) › [Documentation](../README.md) › [Theory index](README.md) \
· Core pipeline · 2 of 11</sub>

*How the moire pattern becomes a spatially modulated superconducting gap Δ(r), and why the \
modulation fades when Cooper pairs are larger than the moire.*

> [!NOTE]
> **Status: Validated.** BCS-style gap modulation with moire amplitude scaling, cross-checked by \
mirrored Python and Rust tests. The field-tunable CPDM built on it is speculative ([Magnetic \
field](magnetic.md#speculative-components)).

**Builds on** [Moire patterns](moire-patterns.md) · **In the apps** web \
[`/viewer`](../user-guide.md#viewer), [`/comparison`](../user-guide.md#comparison), \
[`/sweep`](../user-guide.md#sweep) · desktop **Density Modulation** tab

## Overview

The gap follows the moire: $`\\Delta(\\mathbf{r}) = \\Delta_{\\text{avg}} + \
\\delta\\Delta\\,m(\\mathbf{r})`$, where a [CPDM](../glossary.md#term-cpdm) is imprinted rather \
than spontaneous. Run `capture_screenshots.py --only viewer-gap-2d` to refresh the figure.

<table>
<tr>
<td width="100%" align="center" valign="top"><img src="../images/viewer-gap-2d.png" width="60%" \
alt="Square heatmap of the gap Δ(r) over the moire, larger Δ in red"><br><sub><b>Gap map.</b> \
Δ(r) for Sb₂Te₃ on FeTe; larger Δ in red.</sub></td>
</tr>
</table>

### CPDM versus PDW

| State | Origin | Onset |
|:--|:--|--:|
| CPDM | imprinted by the moire | $`T_c`$ |
| PDW | spontaneous | $`T_{\\text{PDW}} \\gt T_c`$ |

## Model

### Amplitude

```math
\\delta\\Delta \\propto e^{-\\xi / L_m}, \\qquad \\xi \\lt L_m
```

## Speculative components

> [!WARNING]
> **Status: Speculative.** The field-tunable CPDM borrows its field scale from the magnetic \
module. Treat outputs as exploratory illustrations, not quantitative predictions.

The field-tunable CPDM **(SPECULATIVE)** follows the upper critical field.

## Assumptions and validity

- A single harmonic of the moire sets the modulation.
- The coherence length ξ stays fixed at 20 Å.

## Implementation

Python paths are relative to `src/waytogocoop/`; Rust paths to `crates/moire-core/src/`.

| Concept | Python | Rust |
|---|---|---|
| Gap field Δ(r) | \
[`computation/superconducting.py`](../../src/waytogocoop/computation/superconducting.py) · \
`gap_modulation` | [`density.rs`](../../crates/moire-core/src/density.rs) · \
`compute_density_modulation` |

<details>
<summary><b>Cross-language notes:</b> parity anchors and edge cases</summary>

Both languages agree to 1 × 10⁻¹² meV on the default grid.

</details>

## References

- [Wang et al., *Nature* 652, 335 (2026)](../references.md#ref-wang-2026) — the heterostructure \
and its two modulated gaps.

---

**Related:** [Pair density waves](pair-density-waves.md) — the spontaneous cousin of a CPDM · \
[Magnetic field](magnetic.md) — vortex cores punch holes in Δ(r) · [Isotope \
effects](isotope-effects.md) — isotope mass rescales Δ

**Terms:** [Superconductivity](../glossary.md#superconductivity) · [Modulated \
pairing](../glossary.md#modulated-pairing)

[← Moire patterns](moire-patterns.md) · [Theory index](README.md) · [Fourier analysis \
→](fourier-analysis.md)
"""

GOOD_CARDS_PAGE = """\
# Card Layouts

<sub>[Good Job Coop](../README.md) › [Documentation](README.md)</sub>

Hero:

<table>
<tr>
<td width="100%" valign="top"><img src="images/phase-diagram.png" width="100%" alt="Map over \
field B and gap Δ with a curved boundary between trivial and topological \
regions"><br><sub><b>Fu–Kane phase map.</b> The boundary crosses Δ<sub>avg</sub> near 1.8 T. \
<b>(SPECULATIVE)</b></sub></td>
</tr>
</table>

Pair:

<table>
<tr>
<td width="50%" valign="top"><img src="images/graphene-bands.png" width="100%" alt="Moire bands \
with two nearly flat bands at zero energy"><br><sub><b>Flat bands.</b> Bands of twisted bilayer \
graphene at θ = 1.08°.</sub></td>
<td width="50%" valign="top"><img src="images/graphene-dos.png" width="100%" alt="Density of \
states with a sharp peak at zero energy"><br><sub><b>Density of states.</b> The flat bands pile \
states into one peak.</sub></td>
</tr>
</table>

Parity pair:

<table>
<tr>
<td width="61%" valign="top">
<img src="images/pdw-tsweep.png" width="100%" alt="Peak intensities against temperature with the \
window between the two onsets shaded">
<br><sub><b>Cheshire-cat window.</b> Between T<sub>c</sub> and T<sub>PDW</sub> only q and h \
survive. <b>(SPECULATIVE)</b></sub>
</td>
<td width="39%" valign="top">
<img src="images/pdw-fft.png" width="100%" alt="Square Fourier map with bright q and h peaks and \
empty rings where the p peaks were">
<br><sub><b>Spectrum at 3 K.</b> Top four decades shown. <b>(SPECULATIVE)</b></sub>
</td>
</tr>
</table>

Triptych:

<table>
<tr>
<td width="33%" valign="top"><a href="theory/moire-patterns.md"><img src="images/viewer-2d.png" \
width="100%" alt="Moire interference pattern of an Sb₂Te₃ overlayer on FeTe"></a><br><sub><b>Two \
lattices interfere.</b> Sb₂Te₃ on FeTe</sub></td>
<td width="33%" valign="top"><a href="theory/fourier-analysis.md"><img src="images/fourier.png" \
width="100%" alt="Fourier power spectrum with a few sharp peaks"></a><br><sub><b>Its Fourier \
fingerprint.</b> A handful of sharp peaks</sub></td>
<td width="33%" valign="top"><a href="theory/README.md"><img src="images/graphene-pattern.png" \
width="100%" alt="Twisted bilayer graphene with a triangular lattice of bright AA \
regions"></a><br><sub><b>A second platform.</b> Twisted bilayer graphene, 1.08°</sub></td>
</tr>
</table>

Square solo:

<table>
<tr>
<td width="100%" align="center" valign="top"><img src="images/fourier.png" width="60%" \
alt="Square log-power Fourier map with bright peaks at the moire wavevectors"><br><sub><b>Power \
spectrum.</b> log₁₀ &#124;F&#124;² of the paper-default pattern.</sub></td>
</tr>
</table>

Square solo again, with the tag's attributes on three lines:

<table>
<tr>
<td width="100%" align="center" valign="top">
<img src="images/fourier.png"
  width="60%"
  alt="Square log-power Fourier map with bright peaks at the moire wavevectors">
<br><sub><b>Power spectrum.</b> The same card, one attribute per line.</sub>
</td>
</tr>
</table>

Gallery rows, the second inside details:

<table>
<tr>
<td width="25%" valign="top"><a href="images/viewer-2d.png"><img src="images/viewer-2d.png" \
width="100%" alt="Moire intensity heatmap"></a><br><sub><b>viewer-2d</b> · web · app \
view<br>Moire pattern of Sb₂Te₃ on FeTe.</sub></td>
<td width="25%" valign="top"><a href="images/fourier.png"><img src="images/fourier.png" \
width="100%" alt="Fourier power map"></a><br><sub><b>fourier</b> · web · app view<br>Its power \
spectrum.</sub></td>
<td width="25%" valign="top"><a href="images/rust-desktop-3d.png"><img \
src="images/rust-desktop-3d.png" width="100%" alt="Desktop 3D moire \
surface"></a><br><sub><b>rust-desktop-3d</b> · desktop · app view<br>Shaded 3D surface.</sub></td>
<td width="25%" valign="top"><a href="images/rust-density-3d.png"><img \
src="images/rust-density-3d.png" width="100%" alt="Desktop pair-density \
surface"></a><br><sub><b>rust-density-3d</b> · desktop · app view<br>Gap surface, larger Δ in \
red.</sub></td>
</tr>
</table>

<details>
<summary><b>More screenshots:</b> three Wide figures</summary>

<table>
<tr>
<td width="33%" valign="top"><img src="images/sweep-twist.png" width="100%" alt="Moire period \
against twist angle"><br><sub><b>Twist sweep.</b> Period falls with twist.</sub></td>
<td width="33%" valign="top"><img src="images/magnetic-currents-3d.png" width="100%" \
alt="Screening-current cones around vortex cores"><br><sub><b>Screening view.</b> At \
B<sub>z</sub> = 4 T. <b>(SPECULATIVE)</b></sub></td>
<td width="33%" valign="top"><img src="images/magnetic-vortex-2d.png" width="100%" alt="Vortex \
cores marked over the gap map"><br><sub><b>Vortex view.</b> Seven cores. \
<b>(SPECULATIVE)</b></sub></td>
</tr>
</table>

</details>

Index thumbnails:

| Page | Status | What it covers |
|---|---|---|
| <a href="theory/moire-patterns.md"><img src="images/viewer-2d.png" height="72" alt="Moire \
intensity heatmap"></a><br>[Moire patterns](theory/moire-patterns.md) | Validated | Plane-wave \
superposition; twist and mismatch periods |
| <a href="theory/pair-density-waves.md"><img src="images/pdw-tsweep.png" height="72" alt="PDW \
peaks against temperature"></a><br>[Pair density waves](theory/pair-density-waves.md) | \
**Speculative** | Landau model of the UTe₂ PDW |
"""

GOOD_ALERTS_PAGE = """\
# User Guide

<sub>[Good Job Coop](../README.md) › [Documentation](README.md) · [At a glance](#at-a-glance)</sub>

## At a glance

> [!NOTE]
> **Status: Validated.** Plane-wave superposition and both period formulas, locked by unit tests.

The viewer shows the pattern.

> [!WARNING]
> **Status: Mixed.** Stacking registries are established; the flat-band dome is speculative.

The graphene page adds stacks.

> [!WARNING]
> **Status: Speculative.** Simplified models not validated for these heterostructures.

The magnetic page adds vortices.

> [!CAUTION]
> **Status: Highly speculative.** Radioactive and hypothetical isotope masses are what-ifs.

The exotic mode is a toggle.

> [!IMPORTANT]
> **Scale estimate only.** The 1D formula sets the scale, not the exact superlattice.

Read the period with care.

> [!TIP]
> **Try it:** open the **Moire Viewer** with the *Sb₂Te₃ / FeTe* preset.

### <a id="viewer"></a>Moire Viewer

The default view.

### <a id="sweep"></a>Parameter Sweep

The sweep view.

### <a id="comparison"></a>Substrate Comparison

Press <kbd>Ctrl</kbd>+<kbd>S</kbd> to save; a literal dollar sign is <span>$</span>5.

<!-- An HTML comment is not rendered -- and is not linted. -->
"""

GOOD_MATH_PAGE = """\
# Math Forms

<sub>[Good Job Coop](../README.md) › [Documentation](README.md)</sub>

Inline math such as $`\\theta \\lt 1.2^\\circ`$ renders, and so does a fence:

```math
\\begin{aligned}
L &= \\frac{a_1 a_2}{\\lvert a_1 - a_2 \\rvert} \\\\
  &\\approx 36.7\\ \\text{Å}
\\end{aligned}
```

| Quantity | Formula |
|---|---|
| Period | $`\\lvert G \\rvert^{-1}`$ |

Single symbols may stay Unicode: θ = 1.08°, Δ₁ = 2.58 meV, ±0.15, 5 × 10⁻⁶, 50–200 Å.
"""

GOOD_MERMAID_PAGE = """\
# Module Map

<sub>[Good Job Coop](../README.md) › [Documentation](README.md)</sub>

```mermaid
flowchart TD
    subgraph CORE["Core pipeline"]
        direction LR
        MOIRE["Moire patterns"] --> GAP["Gap modulation"] --> FFT["Fourier analysis"]
    end
    subgraph SPEC["Speculative extensions"]
        direction LR
        MAG["Magnetic field"] --> TOP["Topological proximity"]
        ISO["Isotope effects"] ~~~ PDW["Pair density waves"]
    end
    subgraph GRA["Graphene track"]
        direction LR
        STACK["Graphene stacks"] --> BM["BM model"] --> CURV["Curvature"]
    end
    CORE --> GRA
    CORE -.-> SPEC
    class CORE,GRA,SPEC zone
    class STACK,CURV mixed
    class MAG,TOP,PDW spec
    class ISO hispec
    classDef spec fill:#fff8c5,stroke:#9a6700,color:#1f2328
    classDef hispec fill:#ffebe9,stroke:#cf222e,color:#1f2328
    classDef mixed stroke:#bf8700,stroke-width:2px,stroke-dasharray:5 3
    classDef zone fill:none,stroke:#8c959f,stroke-dasharray:4 3
```
"""

GOOD_CONTRIBUTING = """\
# Contributing to Good Job Coop

## <a id="keeping-python--rust-physics-in-sync"></a>Keeping Python and Rust physics in sync

Run `python scripts/capture_screenshots.py --only viewer-2d,fourier` or `cargo run --release -- \
--only rust-desktop-2d`.

## Screenshot capture

The theory-page skeleton, quoted verbatim:

~~~markdown
# Gap Modulation and the CPDM State

> [!NOTE]
> **Status: Validated.** text

## Overview
~~~

Never write `$$`, a bare `$x$`, `angstrom`, `Gruneisen`, `+/-` or `(spec.)`; write `` $`…`$ `` \
instead.

The old heading Theory overview is gone. <!-- docs-lint: ignore -->

| Write | Not |
|:--|--:|
| 36.7 Å | `36.7 A` |

```mermaid
flowchart LR
    A["scripts/capture_screenshots.py"] --> B["docs/images/*.png"]
```

```mermaid
flowchart LR
    C["colormaps_data.json"] --> D["colormap_lut()"]
```

```mermaid
flowchart LR
    E["app.py"] --> F["computation/*.py"]
```
"""

GOOD_FIXTURES: dict[str, str] = {
    "docs/theory/gap-modulation.md": GOOD_THEORY_PAGE,
    "docs/cards.md": GOOD_CARDS_PAGE,
    "docs/user-guide.md": GOOD_ALERTS_PAGE,
    "docs/math.md": GOOD_MATH_PAGE,
    "docs/mermaid.md": GOOD_MERMAID_PAGE,
    "CONTRIBUTING.md": GOOD_CONTRIBUTING,
}

H = "# Page\n\n"  # every bad fixture starts with a valid H1
CARD = '<table>\n<tr>\n<td width="50%" valign="top">{}</td>\n</tr>\n</table>\n'
IMG = '<img src="images/fourier.png" width="100%" alt="Fourier map">'
# A card whose <img> attributes sit on separate lines, as CONTRIBUTING's examples write it.
SPLIT_CARD = (
    '<table>\n<tr>\n<td width="50%" valign="top">\n{}\n<br><sub><b>Map.</b> A caption.</sub>\n'
    "</td>\n</tr>\n</table>\n"
)

# (rule, path, text, extra files). Each text must trigger its rule.
BAD_FIXTURES: tuple[tuple[str, str, str, dict[str, str | bytes | int]], ...] = (
    ("link-missing-file", "docs/x.md", H + "[Gone](nope.md)\n", {}),
    ("link-missing-anchor", "docs/x.md", H + "[Gap](theory/README.md#nope)\n", {}),
    ("link-forbidden", "docs/x.md", H + "[Notes](../CLAUDE.md) and [mirror](../up.sh)\n", {}),
    ("anchor-duplicate", "docs/x.md",
     H + '<a id="ref-a"></a>\n**A.**\n\n<a id="ref-a"></a>\n**B.**\n', {}),
    ("anchor-collision", "docs/x.md", H + '## <a id="overview"></a>Overview\n', {}),
    ("anchor-id-format", "docs/x.md", H + '<a id="Ref_A"></a>\n**A.**\n', {}),
    ("anchor-in-details", "docs/x.md",
     H + "[Hidden](#hidden)\n\n<details>\n<summary><b>More:</b> x</summary>\n\n"
     "### Hidden\n\nText.\n\n</details>\n", {}),
    ("anchor-pinned", "docs/references.md", "# References\n\nNo anchors left.\n", {}),
    ("image-missing", "docs/x.md", H + CARD.format(IMG.replace("fourier", "gone")), {}),
    ("image-missing", "docs/x.md",
     H + SPLIT_CARD.format('<img src="images/gone.png"\n  width="100%"\n  alt="Fourier map">'), {}),
    ("image-no-scene", "docs/images/README.md", "# Figure Gallery\n",
     {"docs/images/orphan.png": 10}),
    ("image-no-inventory", "docs/images/README.md",
     "# Figure Gallery\n\n## Inventory\n\n| Image | Caption |\n|---|---|\n"
     "| `fourier.png` | Spectrum. |\n", {}),
    ("png-size", "docs/x.md", H + CARD.format(IMG), {"docs/images/fourier.png": 900 * 1024}),
    ("alert-placement", "docs/x.md",
     H + "- item\n\n  > [!NOTE]\n  > **Status: Validated.** Text.\n", {}),
    ("alert-type", "docs/x.md", H + "> [!DANGER]\n> **Careful.** Text.\n", {}),
    ("alert-status", "docs/x.md", H + "> [!NOTE]\n> This is fine.\n", {}),
    ("alert-label", "docs/x.md", H + "> [!IMPORTANT]\n> **Status: Validated.** Text.\n", {}),
    ("alert-count", "docs/x.md",
     H + "> [!TIP]\n> **Try it:** a.\n\nText.\n\n> [!TIP]\n> **Try it:** b.\n", {}),
    ("alert-consecutive", "docs/x.md",
     H + "> [!NOTE]\n> **Status: Validated.** A.\n\n> [!TIP]\n> **Try it:** b.\n", {}),
    ("alert-length", "docs/x.md",
     H + "> [!NOTE]\n> **Status: Validated.** " + "word " * 61 + "\n", {}),
    ("math-display", "docs/x.md", H + "$$\n\\Delta = a\n+ b\n$$\n", {}),
    ("math-bare-dollar", "docs/x.md", H + "The value $x$ here.\n", {}),
    ("math-delimiter", "docs/x.md", H + "The value \\(x^2\\) here.\n", {}),
    ("math-angle", "docs/x.md", H + "Here $`a < b`$ holds.\n", {}),
    ("mermaid-type", "docs/x.md", H + "```mermaid\ngraph LR\n  A[\"a\"] --> B[\"b\"]\n```\n", {}),
    ("mermaid-init", "docs/x.md",
     H + "```mermaid\n%%{init: {'theme': 'dark'}}%%\nflowchart LR\n  A[\"a\"]\n```\n", {}),
    ("mermaid-style", "docs/x.md",
     H + "```mermaid\nflowchart LR\n  A[\"a\"]\n  style A fill:#ffd\n```\n", {}),
    ("mermaid-classdef", "docs/x.md",
     H + "```mermaid\nflowchart LR\n  A[\"a\"]\n  classDef warm fill:#fff8c5,stroke:#9a6700\n"
     "```\n", {}),
    ("mermaid-count", "docs/x.md",
     H + ("```mermaid\nflowchart LR\n  A[\"a\"]\n```\n\n" * 2), {}),
    ("mermaid-filename", "docs/x.md",
     H + "```mermaid\nflowchart LR\n  A[\"moire.py\"] --> B[\"out\"]\n```\n", {}),
    ("mermaid-nodes", "docs/x.md",
     H + "```mermaid\nflowchart LR\n"
     + "".join(f'  N{k}["n{k}"] --> N{k + 1}["n{k + 1}"]\n' for k in range(13)) + "```\n", {}),
    ("mermaid-unquoted", "docs/x.md", H + "```mermaid\nflowchart LR\n  A[Label] --> B\n```\n", {}),
    ("heading-h1", "docs/x.md", H + "# Second title\n", {}),
    ("heading-skip", "docs/x.md", H + "## Two\n\n#### Four\n", {}),
    ("heading-content", "docs/x.md", H + "## Foo & bar\n", {}),
    ("theory-h2", "docs/theory/x.md",
     "# X\n\n> [!NOTE]\n> **Status: Validated.** A.\n\n## Background\n\n---\n\n"
     "[← A](a.md) · [Theory index](README.md) · [B →](b.md)\n", {}),
    ("theory-status", "docs/theory/x.md",
     "# X\n\n## Overview\n\n---\n\n[← A](a.md) · [Theory index](README.md) · [B →](b.md)\n", {}),
    ("theory-pager", "docs/theory/x.md",
     "# X\n\n> [!NOTE]\n> **Status: Validated.** A.\n\n## Overview\n\nText.\n", {}),
    ("structure-table-glued", "docs/x.md",
     H + "| A | B |\n|---|---|\n| 1 | 2 |\n<sub>Legend: one, two</sub>\n", {}),
    ("structure-list-glued", "docs/x.md",
     H + "- one\n- two\n**Theory:** [Moire patterns](theory/moire-patterns.md)\n", {}),
    ("structure-setext", "docs/x.md", H + "Some text\n---\n", {}),
    ("structure-table-cells", "docs/x.md",
     H + "| A | B |\n|---|---|\n| plain |F| here | 2 |\n", {}),
    ("structure-table-pipe", "docs/x.md", H + "| A | B |\n|---|---|\n| `a|b` | 2 |\n", {}),
    ("structure-card", "docs/x.md",
     H + "<table>\n<tr>\n\n<td width=\"50%\">" + IMG + "</td>\n</tr>\n</table>\n", {}),
    ("structure-details", "docs/x.md",
     H + "<details>\n<summary><b>More:</b> x</summary>\nText.\n\n</details>\n", {}),
    ("anchor-standalone", "docs/x.md", H + '<a id="old-slug"></a>\n## Heading\n', {}),
    ("figure-markdown-image", "docs/x.md", H + "![Fourier map](images/fourier.png)\n", {}),
    ("figure-img", "docs/x.md",
     H + CARD.format(IMG.replace("Fourier map", "Fourier | map")), {}),
    ("figure-img", "docs/x.md",
     H + SPLIT_CARD.format('<img src="images/fourier.png"\n  width="100%">'), {}),
    ("figure-img", "docs/x.md",
     H + SPLIT_CARD.format(
         '<img src="images/fourier.png"\n  width="100%"\n  alt="' + "x" * 200 + '">'
     ), {}),
    ("figure-thumb-link", "docs/x.md",
     H + "| Page | Status |\n|---|---|\n| <img src=\"images/fourier.png\" height=\"72\" "
     "alt=\"Map\"><br>[Fourier](theory/fourier-analysis.md) | Validated |\n", {}),
    ("figure-pixel-width", "docs/x.md", H + CARD.format(IMG).replace('"50%"', '"400"'), {}),
    ("html-style-attr", "docs/x.md",
     H + CARD.format(IMG).replace("valign", 'style="color:red" valign'), {}),
    ("html-banned-tag", "docs/x.md", H + "Text <small>aside</small>.\n", {}),
    ("html-markdown", "docs/x.md", H + CARD.format(IMG + "<br>**Bold** caption"), {}),
    ("link-line-anchor", "docs/x.md",
     H + "[`density.rs`](../crates/moire-core/src/density.rs#L10)\n", {}),
    ("figure-caption-length", "docs/x.md",
     H + CARD.format(IMG + "<br><sub><b>Lead.</b> " + "word " * 36 + "</sub>"), {}),
    ("typo-dash", "docs/x.md", H + "A range of 50--200 nm.\n", {}),
    ("typo-angstrom", "docs/x.md", H + "The period is 36.7 A.\n", {}),
    ("typo-deg", "docs/x.md", H + "Twist by 1.08 deg.\n", {}),
    ("typo-plusminus", "docs/x.md", H + "Within +/- 0.15.\n", {}),
    ("typo-tilde", "docs/x.md", H + "A period of ~36.7 Å.\n", {}),
    ("typo-spelling", "docs/x.md", H + "The Gruneisen parameter near the zone centre.\n", {}),
    ("typo-eponym", "docs/x.md", H + "The Fu-Kane criterion.\n", {}),
    ("typo-formula", "docs/x.md", H + "Sb2Te3 on FeTe, and 125Te.\n", {}),
    ("typo-stale", "docs/x.md", H + "See the Theory overview.\n", {}),
    ("typo-stale", "docs/x.md",
     H + "- **Theory pages:** `[Good Job Coop](../README.md) › [Theory index](theory/README.md) · "
     "Core pipeline · 2 of 10`\n", {}),
    ("typo-stale", "docs/x.md",
     H + "<sub>[Good Job Coop](../README.md) › [Theory index](theory/README.md) · Core pipeline · "
     "2 of 10</sub>\n", {}),
    ("para-length", "docs/x.md", H + "word " * 101 + "\n", {}),
    ("table-columns", "docs/x.md", H + "| a | b | c | d | e | f |\n" + "|---" * 6 + "|\n", {}),
    ("ignore-marker", "docs/x.md", H + "<!-- docs-lint: ignore --> text at the start\n", {}),
)

SLUG_CASES = (  # (heading, GitHub slug), checked against GitHub's renderer
    ("Sb₂Te₃ on FeTe", "sbte-on-fete"),
    ("STM/STS gap mapping", "stmsts-gap-mapping"),
    ("Fu–Kane criterion and phase diagram", "fukane-criterion-and-phase-diagram"),
    ("Keeping Python ↔ Rust physics in sync", "keeping-python--rust-physics-in-sync"),
    ("Assumptions & validity", "assumptions--validity"),
    ("The K → Γ → M → K′ path", "the-k--γ--m--k-path"),
    ("Sb₂Te₃ and ¹²⁵Te", "sbte-and-te"),
    ('<a id="x"></a>Keeping `snake_case` and *emphasis*', "keeping-snake_case-and-emphasis"),
)


def self_test() -> int:
    failures = 0

    def fail(message: str) -> None:
        nonlocal failures
        failures += 1
        print(f"FAIL {message}")

    base = _selftest_base()
    for heading, expected in SLUG_CASES:
        got = github_slugs([heading])[0]
        if got != expected:
            fail(f"slug {heading!r}: got {got!r}, expected {expected!r}")
    dupes = github_slugs(["Overview", "Overview", "Overview 1"])
    if dupes != ["overview", "overview-1", "overview-1-1"]:
        fail(f"duplicate slugs: got {dupes}")

    files = dict(base)
    files.update(GOOD_FIXTURES)
    checker = Checker(Fs("/selftest", files))
    problems = checker.run(list(GOOD_FIXTURES) + [GALLERY])
    for p in problems:
        fail(f"good fixture flagged: {p.format()}")
    if not problems:
        print(f"ok   {len(GOOD_FIXTURES) + 1} good fixtures lint clean")

    covered: set[str] = set()
    for rule, path, text, extra in BAD_FIXTURES:
        files = dict(base)
        files.update(extra)
        files[path] = text
        found = {p.rule for p in Checker(Fs("/selftest", files)).run([path])}
        covered.add(rule)
        if rule not in found:
            fail(f"bad fixture for [{rule}] not flagged (got {sorted(found) or 'nothing'})")
    missing = set(RULES) - covered
    for rule in sorted(missing):
        fail(f"no bad fixture for [{rule}]")
    if failures:
        print(f"self-test: {failures} failure(s)")
        return 1
    print(f"ok   {len(BAD_FIXTURES)} bad fixtures each trigger their rule")
    print(f"ok   {len(SLUG_CASES) + 1} slug cases match GitHub")
    print("self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
