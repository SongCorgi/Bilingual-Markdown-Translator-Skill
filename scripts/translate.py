#!/usr/bin/env python3
"""
Markdown Bilingual Translator — preserves formulas, LaTeX commands, code blocks,
table/chart structure, link references, and other formatting through a
protect-translate-restore pipeline.  Chunks long documents at section boundaries
and uses the DeepSeek API with a low temperature for consistent terminology.

Output formats:
  --bilingual (default)  English original as body text with Chinese translation
                          in <small> directly below each sentence; pairs are
                          separated by blank lines for clear visual grouping.
  --no-bilingual         emit only the translated document.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# ── constants ────────────────────────────────────────────────────────────────

API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_CHARS = 8000  # per chunk; safe with max_tokens=16384 (2× margin)
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ── protection ───────────────────────────────────────────────────────────────
#
# The order matters: longer / outer patterns must be matched before shorter /
# inner ones so that, e.g., a fenced code block containing backticks is
# protected as one unit.

# Each entry: (category_name, regex_pattern, flags)
PROTECTION_RULES: List[Tuple[str, str, int]] = [
    # YAML / TOML frontmatter
    ("FRONTMATTER", r"^---[\s\S]*?^---\s*", re.MULTILINE),
    # Fenced code blocks (``` … ``` or ~~~ … ~~~)
    ("CODEBLOCK", r"```[\s\S]*?```|~~~[\s\S]*?~~~", 0),
    # Display math ($$ … $$) — must precede inline $
    ("MATHBLOCK", r"\$\$[\s\S]*?\$\$", 0),
    # Pandoc-style display math \[ … \]
    ("MATHBLOCK_PANDOC", r"\\\[[\s\S]*?\\\]", 0),
    # Pandoc-style inline math \( … \)
    ("INLINEMATH_PANDOC", r"\\\([\s\S]*?\\\)", 0),
    # Inline math $…$ — must precede LATEXENV and LaTeX cmd scanner so
    # LaTeX commands inside math are NOT individually extracted
    ("INLINEMATH", r"\$[^$\n]+\$(?!\d)", 0),
    # LaTeX environments \begin{env}…\end{env}
    ("LATEXENV", r"\\begin\{([^}]*)\}[\s\S]*?\\end\{\1\}", 0),
    # HTML comment blocks
    ("HTMLCOMMENT", r"<!--[\s\S]*?-->", 0),
    # Whole images ![alt](url "title")
    ("IMAGE", r"!\[[^\]]*\]\([^)]*\)", 0),
    # Reference-style images ![alt][ref]
    ("IMAGEREF", r"!\[[^\]]*\]\[[^\]]*\]", 0),
    # Link targets: ](url) and ][ref] — protect URLs/refs, leave [text for translation
    ("LINKTARGET", r"\]\([^)]*\)|\]\[[^\]]*\]", 0),
    # Inline code
    ("INLINECODE", r"`[^`\n]+`", 0),
    # HTML inline/block tags
    ("HTMLTAG", r"<[^>\n]+>", 0),
    # Footnote references [^1]
    ("FOOTNOTE", r"\[\^[^\]]+\]", 0),
]


def _scan_latex_cmds(text: str, counter: int, placeholders: Dict[str, str]) -> Tuple[str, int]:
    """Protect LaTeX commands with correct nested-brace handling.

    Covers four categories of backslash-prefixed constructs:

    1. **Alpha commands** — ``\\alpha``, ``\\textbf{\\emph{nested}}``,
       ``\\frac{a}{b}``, ``\\sqrt[3]{x}``.  Uses a depth counter for
       ``{…}`` and ``[…]`` so nested braces are handled correctly.
    2. **``\\verb``** — ``\\verb|code|``, ``\\verb*+code+``.  Consumes
       everything up to the matching delimiter.
    3. **Non-letter escapes** — ``\\\\`` (line break), ``\\,`` (thin space),
       ``\\#``, ``\\{``, ``\\%``, ``\\_``, ``\\&`` (escaped special chars),
       ``\\␣`` (forced space).
    4. **Trailing backslash** — a lone ``\\`` at end-of-input is kept as-is
       (no following character to consume).

    Returns ``(sanitised_text, updated_counter)``.
    """
    result: List[str] = []
    i = 0
    n = len(text)

    # ── helpers (inline for readability) ──────────────────────────────────
    def _consume_balanced(bracket_open: str, bracket_close: str) -> None:
        """Consume *text[i:]* until the matching close-bracket is found,
        respecting arbitrary nesting."""
        nonlocal i
        depth = 1
        i += 1  # skip the opening bracket
        while i < n and depth > 0:
            ch = text[i]
            if ch == bracket_open:
                depth += 1
            elif ch == bracket_close:
                depth -= 1
            i += 1
        # i points one past the closing bracket

    def _emit(start: int) -> None:
        nonlocal counter
        key = f"⟨LATEXCMD:{counter}⟩"
        counter += 1
        placeholders[key] = text[start:i]
        result.append(key)

    # ── main loop ─────────────────────────────────────────────────────────
    while i < n:
        if text[i] != "\\" or i + 1 >= n:
            result.append(text[i])
            i += 1
            continue

        # Backslash at the very last character — keep as-is.
        if i + 1 >= n:
            result.append(text[i])
            i += 1
            continue

        # ── branch 1: alpha commands ──────────────────────────────────
        if text[i + 1].isalpha():
            start = i
            i += 1  # skip backslash
            while i < n and text[i].isalpha():
                i += 1

            # --- \verb / \verb* (delimiter-based, no braces) ----------
            cmd_bytes = text[start:i].encode()  # safe: ASCII letters + backslash
            if cmd_bytes == b"\\verb" or cmd_bytes == b"\\verb*":
                if i < n and not text[i].isspace() and not text[i].isalpha():
                    delim = text[i]
                    i += 1  # skip opening delimiter
                    while i < n and text[i] != delim:
                        i += 1
                    if i < n:
                        i += 1  # skip closing delimiter
                _emit(start)
                continue

            # --- normal LaTeX command: star, optional [...], mandatory {...}
            if i < n and text[i] == "*":
                i += 1
            while i < n and text[i] == "[":
                _consume_balanced("[", "]")
            while i < n and text[i] == "{":
                _consume_balanced("{", "}")

            _emit(start)
            continue

        # ── branch 2: non-letter escapes (\\, \,, \#, \{, \% etc.) ──
        # Skip if the *next* rule (MATHBLOCK_PANDOC / INLINEMATH_PANDOC)
        # already consumed this — but those run earlier, so backslash +
        # bracket here is genuinely an escaped literal like "\{hello\}".
        start = i
        i += 2  # consume backslash + one following character
        _emit(start)

    return "".join(result), counter


def protect(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace protected spans with opaque placeholders.

    Returns (sanitised_text, placeholder_map).
    """
    placeholders: Dict[str, str] = {}
    counter = 0

    for cat, pattern, flags in PROTECTION_RULES:
        def _repl(m: re.Match, _cat=cat) -> str:
            nonlocal counter
            key = f"⟨{_cat}:{counter}⟩"
            counter += 1
            placeholders[key] = m.group(0)
            return key

        text = re.sub(pattern, _repl, text, flags=flags)

    # After all regex rules have run (including INLINEMATH), scan for
    # LaTeX commands that appear in prose — not inside math blocks.
    text, counter = _scan_latex_cmds(text, counter, placeholders)

    return text, placeholders


def restore(text: str, placeholders: Dict[str, str]) -> str:
    """Inverse of :func:`protect`."""
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text



# ── chunking ─────────────────────────────────────────────────────────────────


def split_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Split *text* at ``##`` / ``###`` boundaries, respecting *max_chars*.

    Never breaks inside a paragraph.  When a section is too large and must be
    split across multiple chunks (safety net), the section heading is prepended
    to each continuation chunk so the model always has structural context.
    """
    if len(text) <= max_chars:
        return [text.strip()]

    # Split on atx headings level 2-3
    sections = re.split(r"(?=^#{2,3}\s)", text, flags=re.MULTILINE)

    chunks: List[str] = []
    buf = ""

    for sec in sections:
        if len(buf) + len(sec) > max_chars and buf:
            chunks.append(buf.strip())
            buf = sec
        else:
            buf += sec

    if buf.strip():
        chunks.append(buf.strip())

    # Safety net: if any single chunk still exceeds the limit, force-split
    # on paragraph boundaries (double-newline).  Continuation chunks get the
    # section heading prepended so the model never loses context.
    heading_re = re.compile(r"^#{2,}\s")
    final: List[str] = []

    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue

        paras = re.split(r"\n\n+", c)

        # Extract the section heading if the chunk starts with one
        heading = ""
        body_start = 0
        if paras and heading_re.match(paras[0]):
            heading = paras[0]
            body_start = 1

        sub = heading  # first sub-chunk always carries the heading
        for p in paras[body_start:]:
            candidate = sub + ("\n\n" + p if sub else p)
            if len(candidate) > max_chars and sub:
                final.append(sub.strip())
                # Continuation chunk: prepend heading for context
                sub = (heading + "\n\n" + p) if heading else p
            else:
                sub = candidate

        if sub.strip():
            final.append(sub.strip())

    return final


def _dedupe_headings(text: str) -> str:
    """Remove duplicate headings that appear due to safety-net chunk splits.

    Safety-net continuation chunks have the section heading prepended for
    context.  After translation and bilingual formatting, the same heading
    can appear twice (once at the start of the section and again at each
    continuation chunk boundary) — sometimes in different languages (the
    first chunk's heading was translated, the continuation's was not).

    We match duplicates by section number (``## 3. …``) when present,
    falling back to exact string matching for unnumbered headings
    (e.g. ``## References``).
    """
    lines = text.split("\n")
    result: List[str] = []
    seen: set[str] = set()
    # Section-number pattern:  "## 3."  or  "## 5.1."
    _SEC_NUM = re.compile(r"^(#{2,}\s+)(\d+(?:\.\d+)*)[.\s]")

    def _key(line: str) -> str:
        """Return a dedup key: section number if present, else the full text."""
        m = _SEC_NUM.match(line)
        if m:
            return m.group(1) + m.group(2)
        return line

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(#{2,}\s+.+)$", stripped)
        if m:
            k = _key(m.group(1))
            if k not in seen:
                result.append(line)
                seen.add(k)
        else:
            result.append(line)

    return "\n".join(result)


# ── terminology extraction (English source only) ───────────────────────────


# ── word lists for terminology extraction ─────────────────────────────────
#
# Three tiers with different roles:
#
# _FUNC_WORDS  — pure grammatical glue.  Never start or end a candidate
#                n-gram, and a single-word term must NOT be in this list.
# _NOISE_WORDS — bibliographic / publisher / venue boilerplate.  A candidate
#                that is wholly composed of these is rejected.
# _COMMON_EN   — full combined set used for the edge-word filter and the
#                single-word proper-noun requirement.
#
# Academic content words like "function", "process", "space", "matrix" are
# deliberately kept OUT of these lists — they are often the head noun of a
# technical term and must not be filtered away.

_FUNC_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "as",
    "and", "or", "but", "nor", "not", "no", "so", "if", "then", "than",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "us", "he", "she", "his", "her", "you", "your", "i", "my",
    "has", "have", "had", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "do", "does", "did", "also", "only", "just",
    "very", "much", "such", "each", "all", "some", "any", "both", "few",
    "more", "most", "other", "one", "two", "three", "first", "second",
    "new", "well", "now", "yet", "still", "even", "too", "thus", "hence",
    "since", "while", "where", "when", "which", "who", "whom", "whose",
    "what", "how", "why", "about", "into", "onto", "upon", "within",
    "without", "through", "during", "between", "among", "using", "used",
    "based", "given", "shown", "found", "seen", "known", "called", "made",
    "et", "al", "cf", "eg", "ie", "etc", "vs", "via",
    "paper", "section", "figure", "table", "lemma", "theorem", "proof",
    "proposition", "corollary", "remark", "definition", "example", "case",
    "note", "result", "approach",
    "introduction", "conclusion", "abstract", "reference", "appendix",
    "related", "previous", "present", "propose", "show", "consider",
    "follow", "obtain", "give", "take", "let", "see", "use", "assume",
    "denote", "define", "satisfy", "imply", "yield", "provide", "apply",
    "derive", "compute", "estimate", "prove", "discuss", "describe",
    "particular", "general", "following", "respectively", "however",
    "therefore", "moreover", "furthermore", "nevertheless", "indeed",
    "rather", "instead", "otherwise", "namely", "specifically",
    "sufficiently", "arbitrary", "appropriate", "corresponding",
    "necessary", "sufficient", "important", "significant",
})

_NOISE_WORDS: frozenset[str] = frozenset({
    "university", "press", "cambridge", "international", "conference",
    "berlin", "usa", "ny", "london", "springer", "verlag",
    "eds", "vol", "pp", "doi", "published", "proceedings", "symposium",
    "workshop", "journal", "transaction", "review", "letters",
    "annals", "bulletin", "communications", "preprint", "arxiv",
    "copyright", "rights", "reserved", "permission", "license",
    "department", "college", "institute", "laboratory", "school",
    "centre", "center", "faculty", "academy", "society",
})

_COMMON_EN: frozenset[str] = _FUNC_WORDS | _NOISE_WORDS


def extract_terms_en(
    text: str, top_n: int = 50, max_words: int = 200_000
) -> List[str]:
    """Extract domain-specific English terms from the full source document.

    Uses three signals that English provides for free:
    1. **Capitalisation** — proper nouns have title-case forms (*Malliavin*).
    2. **Acronyms** — ALLCAPS tokens like *SDE*, *ODE*.
    3. **Co-occurrence frequency** — multi-word terms that appear together
       repeatedly (*score function*, *Brownian motion*).

    Runs entirely locally (zero API cost).  For very large documents the
    scan is capped at *max_words* words (≈ 1 MB); the introduction and
    early sections contain the vast majority of key terms anyway.

    Returns a list of term strings, most-frequent first.
    """
    # Remove protected-placeholder-like spans so they don't pollute n-grams
    clean = text
    for pattern in [
        r"\$\$[\s\S]*?\$\$",          # display math
        r"\$[^$\n]+\$",               # inline math
        r"```[\s\S]*?```",            # fenced code
        r"⟨[^⟩]+⟩",                   # protection placeholders
        r"\[[0-9,\s-]+\]",            # citation brackets [1,2,3]
    ]:
        clean = re.sub(pattern, " ", clean)

    words = re.findall(r"[A-Za-z][A-Za-z-]*[A-Za-z]|[A-Za-z]", clean)

    # Safety cap for huge documents
    if len(words) > max_words:
        words = words[:max_words]

    # ── identify proper-noun candidates ─────────────────────────────────
    form_map: Dict[str, set[str]] = {}  # lower → {surface forms}
    for w in words:
        form_map.setdefault(w.lower(), set()).add(w)

    proper: set[str] = set()
    for lower, forms in form_map.items():
        if len(lower) <= 1 or lower in _COMMON_EN:
            continue
        # Title-case form (e.g. "Malliavin")
        has_title = any(
            f[0].isupper() and len(f) > 1 and not f.isupper() for f in forms
        )
        # Acronym form (e.g. "SDE")
        is_acronym = any(f.isupper() and len(f) >= 2 for f in forms)
        if has_title or is_acronym:
            proper.add(lower)

    # ── extract n-gram candidates ───────────────────────────────────────
    candidates: Counter[str] = Counter()
    for n in range(1, 6):
        for i in range(len(words) - n + 1):
            ngram_words = words[i : i + n]
            lower_words = [w.lower() for w in ngram_words]

            # Edge-word filter: first / last word must not be a pure
            # function word (grammatical glue like "the", "of", "using").
            if lower_words[0] in _FUNC_WORDS or lower_words[-1] in _FUNC_WORDS:
                continue
            # Skip if every word is noise (publisher boilerplate)
            if all(w in _NOISE_WORDS for w in lower_words):
                continue

            # ── admission gate (two pathways) ──────────────────────────
            has_proper = any(w in proper for w in lower_words)
            all_content = all(w not in _COMMON_EN for w in lower_words)

            if n == 1:
                # Single-word: MUST have a proper-noun signal
                if not has_proper:
                    continue
            else:
                # Multi-word: proper-noun signal OR every word is a
                # content word (not in the common list).  This second
                # pathway catches all-lowercase technical terms like
                # "score function", "diffusion process", etc.
                if not has_proper and not all_content:
                    continue

            candidates[" ".join(lower_words)] += 1

    # ── deduplicate: remove longer n-grams that are just contextual
    #    variants of a much more frequent shorter term.
    def _subsumed(term: str, count: int, all_terms: list) -> bool:
        twords = term.split()
        if len(twords) <= 2:
            return False
        for shorter, scount in all_terms:
            sw = shorter.split()
            if len(sw) >= len(twords):
                continue
            if not all(w in twords for w in sw):
                continue
            # Near-identical frequency → keep the LONGER term (more
            # specific).  Only subsume when the shorter is *substantially*
            # more frequent (1.5×), meaning the longer is merely a
            # contextual phrase built around the shorter.
            if abs(scount - count) <= 2:
                return False
            if scount >= count * 1.5:
                return True
        return False

    raw = [(t, c) for t, c in candidates.most_common(top_n * 2) if c > 1]
    filtered = [(t, c) for t, c in raw if not _subsumed(t, c, raw)]

    return [t for t, _ in filtered[:top_n]]


# ── API helpers ──────────────────────────────────────────────────────────────


def _build_system_prompt(
    src: str,
    tgt: str,
    glossary: Dict[str, str] | None = None,
) -> str:
    prompt = (
        f"You are an expert translator specialised in academic and technical documents.\n"
        f"Translate the following Markdown content from {src} to {tgt}.\n\n"
        f"ABSOLUTE RULES — follow them exactly:\n"
        f"1. NEVER modify, translate, or remove placeholders like ⟨XXX:N⟩.\n"
        f"   They are opaque tokens — copy them verbatim.\n"
        f"2. Preserve ALL Markdown syntax: headings, list markers, table pipes,\n"
        f"   bold/italic markers, link syntax [text](url), etc.\n"
        f"3. Keep the EXACT same paragraph AND sentence structure.\n"
        f"   Every blank line in the input MUST appear in the output.\n"
        f"   Never merge paragraphs that are separated by blank lines.\n"
        f"   Never add blank lines that do not exist in the input.\n"
        f"4. Translate only natural-language prose.  Each input sentence MUST\n"
        f"   map to exactly one sentence — never merge, split, or reorder.\n"
        f"5. Output the translation ONLY — no preamble, no postscript, no notes.\n"
        f"6. Use natural, idiomatic {tgt}. Match the register of the original\n"
        f"   (formal ↔ formal, colloquial ↔ colloquial)."
    )
    if glossary:
        lines = "\n".join(
            f"     {src_term} → {tgt_term}"
            for src_term, tgt_term in sorted(glossary.items())
        )
        prompt += (
            f"\n7. TERMINOLOGY GLOSSARY — use these exact translations for\n"
            f"   domain-specific terms.  When a term below appears in the\n"
            f"   source text you MUST use the given {tgt} equivalent:\n\n"
            f"{lines}"
        )
    return prompt


def translate_glossary(
    terms: List[str],
    src_lang: str,
    tgt_lang: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> Dict[str, str]:
    """Translate a list of domain-specific terms in a single lightweight
    API call.  Returns ``{source_term: target_term}``."""
    if not terms:
        return {}

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    term_list = "\n".join(f"- {t}" for t in terms)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a terminology translator.  Translate each English "
                    f"technical term below into idiomatic {tgt_lang}.  Output ONLY "
                    f"one mapping per line in the format:\n"
                    f"  English term → {tgt_lang} translation\n"
                    f"Keep the translations concise and consistent.  For acronyms "
                    f"(SDE, ODE, …) keep the Latin letters and add a Chinese explanation."
                ),
            },
            {"role": "user", "content": term_list},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    raw = body["choices"][0]["message"]["content"]

    # Parse "term → translation" lines
    glossary: Dict[str, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if "→" in line:
            src, tgt = line.split("→", 1)
            glossary[src.strip()] = tgt.strip()
        elif ": " in line and not line.startswith("-"):
            src, tgt = line.split(": ", 1)
            glossary[src.strip().lstrip("- ")] = tgt.strip()
    return glossary


def translate_chunk(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    glossary: Dict[str, str] | None = None,
) -> str:
    """Call the DeepSeek chat-completion API for one chunk.  Retries on
    transient errors.

    Blank-line paragraph separators (``\\n\\n``) are replaced with an
    opaque placeholder (``⟨SEP⟩``) before the API call so the model
    cannot merge adjacent paragraphs.  The placeholder is restored to
    blank lines in the translation output.
    """
    SEP = "⟨SEP⟩"

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    # Protect blank-line separators so the API preserves paragraph structure
    protected = re.sub(r"\n\n+", f" {SEP} ", text)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_system_prompt(src_lang, tgt_lang, glossary)},
            {"role": "user", "content": protected},
        ],
        "temperature": 0.2,
        "max_tokens": 16384,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    token_limit_bumped = False

    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"Unexpected API response: {json.dumps(body)[:300]}")

            choice = body["choices"][0]
            result = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "")

            # Detect truncation — check finish_reason and unclosed placeholders
            truncated = finish_reason == "length"
            if not truncated:
                n_open = result.count("⟨")
                n_close = result.count("⟩")
                truncated = n_open != n_close

            if truncated and not token_limit_bumped and payload["max_tokens"] < 32768:
                payload["max_tokens"] *= 2
                data = json.dumps(payload).encode("utf-8")
                token_limit_bumped = True
                # Don't count truncation against MAX_RETRIES — retry immediately
                continue

            # Restore blank-line separators and collapse extra blank lines
            # the API may add its own \n\n, which would inflate the
            # paragraph count and break bilingual formatting downstream
            result = re.sub(rf"\s*{re.escape(SEP)}\s*", "\n\n", result)
            result = re.sub(r"\n{3,}", "\n\n", result)
            return result
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse API response: {e}")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            last_err = RuntimeError(f"HTTP {e.code}: {err_body}")
            if e.code == 429:  # rate-limit — wait and retry
                time.sleep(RETRY_DELAY * attempt)
            elif e.code >= 500:
                time.sleep(RETRY_DELAY)
            else:
                raise last_err
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            time.sleep(RETRY_DELAY)
        except Exception as e:
            # Catch-all for unexpected HTTP errors (IncompleteRead, etc.)
            last_err = e
            time.sleep(RETRY_DELAY)

    raise last_err  # type: ignore[misc]


_BATCH_MAX = 25  # sentences per sub-batch for JSON array translation


def _translate_one_batch(
    sentences: List[str],
    api_key: str,
    model: str,
    src: str,
    tgt: str,
    glossary: Dict[str, str] | None = None,
) -> List[str]:
    """Translate a single batch of sentences (≤ _BATCH_MAX) via JSON array."""
    n = len(sentences)
    json_input = json.dumps(sentences, ensure_ascii=False)

    system = (
        f"You are an expert translator specialised in academic and technical documents.\n"
        f"Translate each English sentence below to {tgt}.\n\n"
        f"RULES:\n"
        f"1. You will receive a JSON array of {n} English sentences.\n"
        f"2. Translate EACH sentence individually and return a JSON array of exactly {n} translations.\n"
        f"3. Keep the SAME order — output[i] = translation of input[i].\n"
        f"4. NEVER merge, split, or reorder sentences.\n"
        f"5. Preserve ALL ⟨XXX:N⟩ placeholders verbatim — copy them exactly.\n"
        f"6. Preserve Markdown syntax (headings, links, bold/italic, etc.).\n"
        f"7. Use natural, idiomatic {tgt} matching the register of the original.\n"
        f"8. Output ONLY the JSON array — no preamble, no markdown fences."
    )
    if glossary:
        lines = "\n".join(
            f"  {s} → {t}" for s, t in sorted(glossary.items())
        )
        system += f"\n\nTERMINOLOGY GLOSSARY:\n{lines}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json_input},
        ],
        "temperature": 0.2,
        "max_tokens": 16384,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if the model wrapped the JSON
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            translations = json.loads(raw)

            if not isinstance(translations, list):
                raise ValueError(
                    f"Expected JSON array, got {type(translations).__name__}"
                )
            if len(translations) != n:
                raise ValueError(
                    f"Length mismatch: expected {n}, got {len(translations)}"
                )

            return translations
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            last_err = e
            time.sleep(RETRY_DELAY * attempt)
        except urllib.error.HTTPError as e:
            last_err = RuntimeError(f"HTTP {e.code}")
            if e.code == 429:
                time.sleep(RETRY_DELAY * attempt)
            elif e.code >= 500:
                time.sleep(RETRY_DELAY)
            else:
                raise last_err
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            time.sleep(RETRY_DELAY)

    raise last_err  # type: ignore[misc]


def _llm_batch_translate_sentences(
    sentences: List[str],
    api_key: str,
    model: str,
    src: str,
    tgt: str,
    glossary: Dict[str, str] | None = None,
) -> List[str]:
    """Translate a list of English sentences to Chinese in one or more
    API calls.  Sub-batches into groups of _BATCH_MAX sentences so the
    JSON arrays stay manageable."""
    all_translations: List[str] = []
    for start in range(0, len(sentences), _BATCH_MAX):
        batch = sentences[start : start + _BATCH_MAX]
        all_translations.extend(
            _translate_one_batch(batch, api_key, model, src, tgt, glossary)
        )
    return all_translations


# ── bilingual formatting (per chunk) ────────────────────────────────────────

_ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "e.g.", "i.e.",
    "etc.", "vs.", "Fig.", "Eq.", "Eqs.", "Figs.", "cf.",
    "al.", "et al.", "i.i.d.", "w.r.t.", "resp.", "approx.",
    "Inc.", "Ltd.", "vol.", "no.", "p.", "pp.", "Ch.", "Sec.",
]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences.  Protects abbreviations from false splits."""
    if not text or not text.strip():
        return []

    abbr_map: Dict[str, str] = {}
    for i, abbr in enumerate(_ABBREVIATIONS):
        if abbr in text:
            ph = f"⟨SABBR:{i}⟩"
            abbr_map[ph] = abbr
            text = text.replace(abbr, ph)

    # Protect heading-number periods (## 1., ### 2.1., etc.) so the
    # regex does not split inside "## 1. Introduction"
    hdot_map: Dict[str, str] = {}

    def _protect_hdot(m: re.Match) -> str:
        ph = f"⟨HDOT:{len(hdot_map)}⟩"
        hdot_map[ph] = m.group(0)
        return ph

    text = re.sub(
        r"(^#{1,6}\s+\d+(?:\.\d+)*)\.",
        _protect_hdot,
        text,
        flags=re.MULTILINE,
    )

    parts = re.split(
        # Standard sentence boundaries: .!? + space + Capital/heading
        r"(?<=[.!?])\s+(?=[A-Z#])"
        # Chinese/Japanese sentence boundaries
        r"|(?<=[。！？])"
        # Placeholder at end of a sentence on its own line — the next
        # line starts with a capital letter, signalling a new sentence
        # even though the placeholder absorbed the original period.
        # \n+ (not \s+) so inline placeholders like ⟨HDOT:N⟩ are
        # never split from their following word.
        r"|(?<=⟩)\n+(?=[A-Z#])",
        text.strip(),
    )

    result: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for ph, abbr in abbr_map.items():
            part = part.replace(ph, abbr)
        # Restore heading-number dots
        for ph, orig in hdot_map.items():
            part = part.replace(ph, orig)
        result.append(part)

    return result


def _classify_sentence(s: str) -> str:
    """Classify a sentence as 'structural' or 'prose'.

    'structural' — standalone math/code blocks, LaTeX environments,
                   placeholders that should NOT be translated.
    'prose'      — everything else including headings (heading formatting
                   is handled by _format_bilingual_chunk based on the
                   leading ``#`` prefix, not by this classification).
    """
    s_stripped = s.strip()
    if not s_stripped:
        return "structural"
    if re.match(
        r"^⟨(?:MATHBLOCK|CODEBLOCK|LATEXENV|FRONTMATTER|HTMLCOMMENT):\d+⟩$",
        s_stripped,
    ):
        return "structural"
    if re.match(r"^(\$\$|```|~~~)", s_stripped):
        return "structural"
    if re.match(r"^\\(?:begin|end)\{", s_stripped):
        return "structural"
    return "prose"


_MATH_CMD = re.compile(r"^\\(?:begin|end|displaystyle|\[|\])\b")


def _separate_trailing_structural(sentences: List[str]) -> List[str]:
    """Split sentences that have a structural placeholder appended at the
    end (e.g. ``text\n⟨MATHBLOCK:N⟩``).  Without this, the placeholder
    becomes part of the prose sentence and the Chinese translation appears
    after the math block instead of right after the English text."""
    result: List[str] = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
        if re.match(
            r"^⟨(?:MATHBLOCK|CODEBLOCK|LATEXENV|FRONTMATTER|HTMLCOMMENT):\d+⟩$",
            s_stripped,
        ):
            result.append(s)
            continue
        # Trailing structural placeholder at end of text
        m = re.match(
            r"^([\s\S]*?)\n+(⟨(?:MATHBLOCK|CODEBLOCK|LATEXENV|FRONTMATTER|HTMLCOMMENT):\d+⟩)$",
            s_stripped,
        )
        if m:
            before = m.group(1).strip()
            ph = m.group(2)
            if before:
                result.append(before)
            result.append(ph)
            continue
        result.append(s)
    return result


def _format_bilingual_chunk(
    orig_sentences: List[str],
    trans_sentences: List[str],
    start_num: int = 1,
) -> Tuple[str, int]:
    """Create bilingual output with English/Chinese sentence pairs.

    Each prose sentence pair gets a tightly-coupled group::

        English text
        中文翻译

    Consecutive groups are separated by ``***``.  Structural blocks
    (math, code) and headings are emitted standalone, also separated
    by ``***`` from surrounding content.

    Returns ``(formatted_text, next_available_number)`` so numbering
    can continue across chunks.
    """
    if len(orig_sentences) != len(trans_sentences):
        raise ValueError(
            f"Sentence count mismatch: "
            f"{len(orig_sentences)} orig vs {len(trans_sentences)} trans"
        )

    out: List[str] = []
    n = start_num
    first = True

    def _separator() -> None:
        if not first:
            out.append("***")
            out.append("")

    for os_, ts in zip(orig_sentences, trans_sentences):
        os_stripped = os_.strip()
        if not os_stripped:
            continue

        # Structural: math/code blocks, LaTeX environments — original only
        if re.match(
            r"^(⟨(?:MATHBLOCK|CODEBLOCK|LATEXENV|FRONTMATTER|HTMLCOMMENT):\d+⟩"
            r"|\$\$|```|~~~|\\begin|\\end)",
            os_stripped,
        ):
            _separator()
            out.append(os_)
            out.append("")
            first = False

        # Heading — emit the translation (heading markdown preserved)
        elif re.match(r"^#{1,6}\s", os_stripped):
            _separator()
            out.append(ts)
            out.append("")
            first = False

        # Prose — tightly-coupled English / Chinese pair
        else:
            _separator()
            out.append(os_)
            out.append(ts)
            out.append("")
            n += 1
            first = False

    return "\n".join(out).strip(), n


def _fix_adjacent_inline_math(text: str) -> str:
    """Separate adjacent inline-math blocks whose ``$$...$$`` concatenation
    creates an accidental display-math opener.

    When the LLM drops the space between two ``⟨INLINEMATH:N⟩`` placeholders,
    the restored text becomes ``$a$$b$`` — the middle ``$$`` looks like
    display-math to every downstream regex.  Real display-math ``$$`` is
    *always* at the start of a line (it originates from a ``⟨MATHBLOCK:N⟩``
    placeholder on its own line), so we only touch inline occurrences.
    """
    # Pattern: $content$ immediately followed by $content$ on the SAME line.
    # [^$\n] ensures we stay within a single line and don't cross $ boundaries.
    # Loop because a single replacement can leave a second adjacent pair
    # (e.g. "$a$$b$$c$" → "$a$ $b$$c$" needs a second pass for "$b$$c$").
    pat = re.compile(r"\$([^$\n]+)\$\$([^$\n]+)\$")
    for _ in range(5):  # at most 5 adjacent blocks in practice
        new_text = pat.sub(r"$\1$ $\2$", text)
        if new_text == text:
            break
        text = new_text
    return text


def _validate_and_fix_inline_math(
    text: str,
    api_key: str,
    model: str,
    src: str,
    tgt: str,
    glossary: Dict[str, str] | None = None,
) -> str:
    """Check each ``[EN-N]`` / ``[ZH-N]`` pair for inline-math count
    mismatches and re-translate the offending sentences individually.

    When the LLM processes large JSON batches (25 sentences), it
    occasionally shuffles ``⟨INLINEMATH:N⟩`` placeholders across
    sentence boundaries.  Single-sentence translation is immune to
    this because there is only one sentence in the batch.
    """
    # Split the bilingual output into blocks: each block is
    #   [EN-N] line
    #   [ZH-N] line
    #   (blank line)
    #   *** separator
    # Structural blocks and headings also use *** separators but
    # lack the [EN-N]/[ZH-N] anchors, so they are easy to skip.
    lines = text.split("\n")
    out_lines: List[str] = []
    i = 0
    fixed_count = 0

    while i < len(lines):
        line = lines[i]
        en_m = re.match(r"^\[EN-(\d+)\]\s+(.+)", line)
        if not en_m:
            out_lines.append(line)
            i += 1
            continue

        en_num = en_m.group(1)
        en_body = en_m.group(2)
        en_dollar = en_body.count("$")

        # Next line should be the matching [ZH-N]
        zh_line = ""
        zh_body = ""
        zh_dollar = -1
        if i + 1 < len(lines):
            zh_m = re.match(r"^\[ZH-" + en_num + r"\]\s+(.+)", lines[i + 1])
            if zh_m:
                zh_body = zh_m.group(1)
                zh_line = lines[i + 1]
                zh_dollar = zh_body.count("$")

        if zh_dollar >= 0 and en_dollar == zh_dollar:
            # Counts match — keep as-is
            out_lines.append(line)
            out_lines.append(zh_line)
            i += 2
            continue

        # Mismatch or missing ZH — re-translate this single sentence
        print(
            f"  ⚠  [#{en_num}] inline-math mismatch "
            f"(EN:{en_dollar} ZH:{zh_dollar}) — re-translating",
            file=sys.stderr,
        )

        try:
            new_translations = _translate_one_batch(
                [en_body], api_key, model, src, tgt, glossary
            )
            new_zh = new_translations[0]
            new_zh_dollar = new_zh.count("$")
            if new_zh_dollar == en_dollar:
                out_lines.append(line)
                out_lines.append(f"[ZH-{en_num}] {new_zh}")
                fixed_count += 1
                i += 2
                continue
            else:
                print(
                    f"  ⚠  [#{en_num}] re-translate also mismatched "
                    f"(EN:{en_dollar} ZH:{new_zh_dollar}) — keeping original",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"  ⚠  [#{en_num}] re-translate failed: {exc}",
                file=sys.stderr,
            )

        # Keep original (either re-translate failed or still mismatched)
        out_lines.append(line)
        if zh_line:
            out_lines.append(zh_line)
            i += 2
        else:
            i += 1

    if fixed_count:
        print(
            f"  ✓  fixed {fixed_count} inline-math mismatches",
            file=sys.stderr,
        )

    return "\n".join(out_lines)


def _sanitize_math_delimiters(text: str) -> str:
    """Ensure ``$$`` display-math fences are bare on their own line.
    The LLM punctuation fix may append a period after a ⟨MATHBLOCK:N⟩
    placeholder, which after restore becomes ``$$.`` — breaking every
    subsequent display-math block in renderers that expect exactly ``$$``."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("$$") and len(s) > 2:
            # Keep only the $$, drop trailing characters
            trailing = s[2:]
            # If trailing is pure punctuation/whitespace, strip it
            if trailing.strip() and all(c in ".!?,;: " for c in trailing):
                lines[i] = line.replace(s, "$$", 1)
    return "\n".join(lines)


def _clean_latex(text: str) -> str:
    """Post-process LaTeX formulas for Obsidian compatibility.

    * Compresses redundant token-spaces inside ``$...$`` and ``$$...$$``
      (e.g. ``$u _ { k }$`` → ``$u_{k}$``).
    * Externalises trailing punctuation from inline math
      (e.g. ``$x^2.$`` → ``$x^2$ .``).
    * Converts bare ``<`` / ``>`` inside math to ``\\lt`` / ``\\gt``
      to prevent HTML-tag parser deadlocks.
    * Ensures ``$$`` fences are isolated on their own lines with
      blank-line separation from surrounding text.
    """

    # ── step 1: inline math $...$ (single $, never $$) ────────────────
    def _clean_inline(m: re.Match) -> str:
        body = m.group(1)
        body = re.sub(r"\s+([_{}^])", r"\1", body)
        body = re.sub(r"([_{}^])\s+", r"\1", body)
        body = re.sub(r"(?<!\\)<(?![a-zA-Z])", r"\\lt ", body)
        body = re.sub(r"(?<!\\)>(?![a-zA-Z])", r"\\gt ", body)
        body = body.rstrip()
        m2 = re.match(r"^([\s\S]*?)([.,;:!?。，；：！？]+)$", body)
        if m2:
            return f"${m2.group(1).rstrip()}${m2.group(2)}"
        return f"${body}$"

    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)\$(?!\$)", _clean_inline, text)

    # ── step 2: display math $$...$$ ──────────────────────────────────
    # Space compression is safe to apply globally: _, {, }, ^ are LaTeX
    # token delimiters that never appear adjacent to spaces in prose.
    # Doing this globally avoids $$-pairing issues when the bilingual
    # output has stray $$ markers around prose text.
    text = re.sub(r"\s+([_{}^])", r"\1", text)
    text = re.sub(r"([_{}^])\s+", r"\1", text)

    # < > → \lt \gt only inside display-math blocks identified via
    # line-by-line state machine (which pairs $$ on their own line).
    # This avoids corrupting HTML tags or Markdown outside math.
    lines = text.split("\n")
    in_dm = False
    dm_start = -1

    for i, line in enumerate(lines):
        is_dd = line.strip() == "$$"
        if is_dd and not in_dm:
            in_dm = True
            dm_start = i
        elif is_dd and in_dm:
            in_dm = False
            content = "\n".join(lines[dm_start + 1 : i])
            # Only convert <> in blocks that look like real LaTeX math
            if "\\" in content:
                content = re.sub(
                    r"(?<!\\)<(?![a-zA-Z])", r"\\lt ", content
                )
                content = re.sub(
                    r"(?<!\\)>(?![a-zA-Z])", r"\\gt ", content
                )
                # Replace content lines
                lines[dm_start + 1 : i] = [content]
        # else: in_dm and not $$ — accumulate (done implicitly in lines)

    text = "\n".join(lines)

    # ── step 3: $$ fence isolation (blank-line padding) ──────────────
    text = re.sub(r"(\S)\s*\$\$", r"\1\n\n$$", text)
    text = re.sub(r"\$\$\s*(\S)", r"$$\n\n\1", text)
    text = re.sub(r"([^\n])\n\$\$", r"\1\n\n$$", text)
    text = re.sub(r"\$\$\n([^\n])", r"$$\n\n\1", text)

    # ── step 4: collapse excessive blank runs ────────────────────────
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bilingual Markdown translator (DeepSeek API)"
    )
    parser.add_argument("input", help="Path to the source Markdown file")
    parser.add_argument(
        "-o", "--output", help="Output path (default: <input>_bilingual.md)"
    )
    parser.add_argument(
        "--from", dest="src", default="en", help="Source language code (default: en)"
    )
    parser.add_argument(
        "--to", dest="tgt", default="zh", help="Target language code (default: zh)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--api-key", help="DeepSeek API key (or set DEEPSEEK_API_KEY)")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Max chars per API chunk (default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--no-bilingual",
        action="store_true",
        help="Emit translation only (no original paragraphs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show chunk plan and protected blocks without calling the API",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Parallel translation workers (default: 64)",
    )

    args = parser.parse_args()

    # --- resolve input -------------------------------------------------------
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"Error: file not found — {args.input}", file=sys.stderr)
        sys.exit(1)

    original = in_path.read_text(encoding="utf-8")
    if not args.quiet:
        print(f"📄  {in_path}  ({len(original):,} chars)")

    # --- protect -------------------------------------------------------------
    protected, placeholders = protect(original)
    n_protected = len(placeholders)
    if not args.quiet:
        print(f"🔒  {n_protected} protected blocks")

    if args.dry_run:
        _print_dry_run(protected, placeholders, args)
        return

    # --- chunk ---------------------------------------------------------------
    chunks = split_chunks(protected, args.max_chars)
    if not args.quiet:
        print(f"✂   {len(chunks)} chunk(s)  (max {args.max_chars:,} chars each)")

    # --- glossary (English source only) ------------------------------------
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    glossary: Dict[str, str] = {}
    if args.src == "en":
        if not args.quiet:
            print("📖  extracting terminology …", end=" ", flush=True)
        terms = extract_terms_en(original)
        if not args.quiet:
            print(f"{len(terms)} terms found")
        if terms and api_key:
            if not args.quiet:
                print(f"📝  translating {len(terms)} terms …", end=" ", flush=True)
            try:
                glossary = translate_glossary(terms, args.src, args.tgt, args.model, api_key)
                if not args.quiet:
                    print(f"{len(glossary)} mapped")
            except Exception as exc:
                if not args.quiet:
                    print(f"(skipped: {exc})")

    # --- translate -----------------------------------------------------------
    if not api_key:
        print(
            "Error: DEEPSEEK_API_KEY not set.  Use --api-key or export the env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- translate (parallel per chunk) ------------------------------------
    if not args.quiet:
        print(f"🔧  translating {len(chunks)} chunk(s) "
              f"with {args.workers} workers …")

    results: Dict[int, Tuple[List[str], List[str]]] = {}

    def _process_chunk(
        idx: int, txt: str
    ) -> Tuple[int, List[str], List[str]]:
        """Split protected chunk into sentences, batch-translate prose
        sentences, return pre-aligned (orig_sentences, trans_sentences)
        — both arrays have the same length.  Still protected with
        placeholders at this point."""
        # Split on blank lines first so headings stay separate from
        # body text; then sentence-split each paragraph independently.
        paragraphs = re.split(r"\n\s*\n", txt.strip())
        sentences: List[str] = []
        for para in paragraphs:
            para_sents = _split_sentences(para.strip())
            if para_sents:
                sentences.extend(para_sents)
        sentences = _separate_trailing_structural(sentences)

        # Identify which sentences need translation
        translatable_indices: List[int] = []
        translatable_sentences: List[str] = []
        for i, s in enumerate(sentences):
            if _classify_sentence(s) == "prose":
                translatable_indices.append(i)
                translatable_sentences.append(s)

        # Batch translate translatable sentences
        if translatable_sentences:
            translations = _llm_batch_translate_sentences(
                translatable_sentences, api_key, args.model,
                args.src, args.tgt,
                glossary=glossary if glossary else None,
            )

        # Reassemble: build translated array same length as original
        trans_sentences = list(sentences)  # start with originals
        for j, i in enumerate(translatable_indices):
            trans_sentences[i] = translations[j]

        return idx, sentences, trans_sentences

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_chunk, i, c): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            idx, orig_sents, trans_sents = future.result()
            results[idx] = (orig_sents, trans_sents)
            if not args.quiet:
                print(f"  ✓  [{idx + 1}/{len(chunks)}]")

    # --- reassemble ----------------------------------------------------------
    if args.no_bilingual:
        # Join all translated sentences into one text, then restore
        all_trans: List[str] = []
        for i in range(len(chunks)):
            _, trans_sents = results[i]
            all_trans.extend(trans_sents)
        full_translation = "\n\n".join(all_trans)
        full_translation = _dedupe_headings(full_translation)
        out_text = restore(full_translation, placeholders)
        out_text = _fix_adjacent_inline_math(out_text)
        out_text = _sanitize_math_delimiters(out_text)
    else:
        # Per-chunk bilingual: sentence arrays are already 1:1 aligned.
        # Restore placeholders in each sentence, then format with
        # global numbering across all chunks.
        bilingual_parts: List[str] = []
        num = 1
        for i in range(len(chunks)):
            orig_sents, trans_sents = results[i]
            orig_restored = [restore(s, placeholders) for s in orig_sents]
            trans_restored = [restore(s, placeholders) for s in trans_sents]
            text, num = _format_bilingual_chunk(
                orig_restored, trans_restored, start_num=num
            )
            bilingual_parts.append(text)
        out_text = "\n\n".join(bilingual_parts)
        out_text = _dedupe_headings(out_text)
        out_text = _fix_adjacent_inline_math(out_text)
        out_text = _sanitize_math_delimiters(out_text)
        out_text = _clean_latex(out_text)

    # --- output --------------------------------------------------------------
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = in_path.parent / f"{in_path.stem}_bilingual{in_path.suffix}"

    out_path.write_text(out_text, encoding="utf-8")
    if not args.quiet:
        print(f"✅  {out_path}  ({len(out_text):,} chars)")


def _print_dry_run(text: str, placeholders: Dict[str, str], args) -> None:
    """Show chunk plan and placeholder catalogue."""
    chunks = split_chunks(text, args.max_chars)
    print(f"\n{'─' * 60}")
    print(f"  {len(chunks)} chunk(s) preview")
    print(f"{'─' * 60}")
    for i, c in enumerate(chunks):
        preview = c[:200].replace("\n", "\\n")
        print(f"\n  [{i + 1}] {len(c):,} chars")
        print(f"  {preview}{' …' if len(c) > 200 else ''}")
        # Count placeholders in this chunk
        ph_counts = {}
        for key in placeholders:
            if key in c:
                cat = key.split(":")[0].lstrip("⟨")
                ph_counts[cat] = ph_counts.get(cat, 0) + 1
        if ph_counts:
            print(f"  Placeholders: {dict(ph_counts)}")

    print(f"\n{'─' * 60}")
    print(f"  Catalogue ({len(placeholders)} total)")
    print(f"{'─' * 60}")
    for key in sorted(placeholders.keys(), key=lambda k: (k.split(":")[0], int(k.split(":")[1].rstrip("⟩")))):
        val = placeholders[key]
        preview = val[:80].replace("\n", "\\n")
        print(f"  {key}  →  {preview}{' …' if len(val) > 80 else ''}")


if __name__ == "__main__":
    main()
