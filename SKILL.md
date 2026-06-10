---
name: markdown-translator
description: Translate Markdown documents into bilingual (source↔target) format while preserving formulas, code blocks, tables, charts, links, and all other formatting. Handles long documents via section-aware chunking. Uses the DeepSeek API (deepseek-chat / deepseek-v4-flash) as the translation engine. Invoke when the user asks to translate a .md file, create a bilingual version of a document, or convert technical/academic markdown between languages.
---

# Markdown Bilingual Translator

Preserves **formulas (LaTeX)**, **LaTeX commands** (`\textbf`, `\frac`, `\alpha`, `\begin`/`\end` environments, etc.), **code blocks**, **tables**, **images**, **links**, **HTML**, and **frontmatter** during translation so the output stays structurally identical to the input.  Long documents are split at heading boundaries (`##` / `###`) and translated chunk-by-chunk via the DeepSeek API.

Output is **sentence-by-sentence bilingual**: each original sentence (blockquoted) is immediately followed by its translation, with blank lines between pairs for comfortable reading in Typora and similar renderers.

## When to Use

- User asks to translate a `.md` file (English ↔ Chinese, or any language pair)
- User wants a "bilingual" / "对照" version of a document
- User needs to translate academic papers, technical docs, or README files
- User mentions preserving formulas / code / tables during translation

## Prerequisites (run once)

```bash
# Ensure Python 3.9+ is available
python3 --version

# The script uses only stdlib — no pip installs needed.

# Set your DeepSeek API key (get one at https://platform.deepseek.com/api_keys)
export DEEPSEEK_API_KEY="sk-…"
```

## Usage

```bash
# Basic: translate en → zh, bilingual output
python3 ~/.claude/skills/markdown-translator/scripts/translate.py doc.md

# Specify languages and output path
python3 ~/.claude/skills/markdown-translator/scripts/translate.py \
    doc.md --from en --to zh -o doc_cn.md

# Translation only (no bilingual interleaving)
python3 ~/.claude/skills/markdown-translator/scripts/translate.py \
    doc.md --no-bilingual

# Preview chunk plan without calling the API
python3 ~/.claude/skills/markdown-translator/scripts/translate.py \
    doc.md --dry-run

# Customise chunk size (default 5000 chars)
python3 ~/.claude/skills/markdown-translator/scripts/translate.py \
    doc.md --max-chars 3000
```

### Model selection

The default model is `deepseek-chat`.  To use a specific model:

```bash
python3 … --model deepseek-chat       # latest (v4)
python3 … --model deepseek-reasoner   # reasoning model
```

Set the `DEEPSEEK_API_KEY` environment variable or pass `--api-key` explicitly.

## Translation Strategy

The script uses a **protect → chunk → translate → restore** pipeline:

1. **Protect** — code blocks, inline code, LaTeX math (`$` / `$$`, `\(` / `\)`, `\[` / `\]`), LaTeX commands (`\textbf{…}`, `\frac{…}{…}`, `\begin{…}…\end{…}`, etc.), images, HTML tags, and frontmatter are replaced with opaque placeholders (`⟨CODEBLOCK:0⟩`, `⟨MATHBLOCK:1⟩`, `⟨LATEXCMD:2⟩`, etc.)
2. **Chunk** — the sanitised text is split at `##` / `###` boundaries, keeping each chunk under the `--max-chars` limit.  If a single section still exceeds the limit it is further split on paragraph breaks.
3. **Translate** — each chunk is sent to the DeepSeek API with a system prompt that enforces placeholder preservation, paragraph-structure fidelity, and register matching
4. **Restore** — placeholders in the translation output are swapped back to their original content
5. **Format** — in bilingual mode, sentences are interleaved (original blockquoted, translation plain) with blank lines between pairs.  Falls back gracefully to paragraph-level or section-level alignment when sentence/paragraph counts drift.

## Bilingual Output Format

```
## Section Heading

> Original English sentence that introduces the topic.

介绍该主题的中文翻译句子。

> Another English sentence with **bold** and *italic* text.

另一个带有**粗体**和*斜体*的中文翻译句子。

> A third sentence containing inline math $x^2$ and a LaTeX command.

包含行内公式 $x^2$ 和 LaTeX 命令的第三个句子。
```

- Original text is blockquoted (`> ` prefix)
- Translation follows immediately as plain text
- A blank line separates each sentence pair for Typora readability
- Protected elements (code, math, LaTeX commands, images) appear identically in both versions
- When sentence counts don't align (rare), the section falls back to paragraph-level alignment

## Error Handling

| Symptom | Fix |
|---------|-----|
| `DEEPSEEK_API_KEY is not set` | `export DEEPSEEK_API_KEY="sk-…"` or pass `--api-key` |
| `HTTP 429` | Script auto-retries with exponential backoff; wait and re-run |
| `HTTP 401` | API key is invalid or expired — check at platform.deepseek.com |
| Translation quality is poor on a specific chunk | Re-run with `--max-chars 2000` for smaller, higher-fidelity chunks |
| Paragraph alignment is off | Re-run with `--no-bilingual` for a clean single-language output |

## Tips for Best Results

- **Chunk size**: 3000–5000 chars per chunk gives the best balance of context vs. quality.  Smaller chunks = better consistency but more API calls.
- **Language codes**: Use ISO 639-1 codes (`en`, `zh`, `ja`, `ko`, `fr`, `de`, …).  For Chinese specifically, the model understands `zh` as simplified Chinese.
- **Technical content**: The default temperature of 0.2 keeps terminology stable across chunks.
- **Review**: Always skim the output — the interleaved format makes it easy to spot-check translations against the original.

## Script Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--from` / `--to` | `en` / `zh` | Source and target language codes |
| `--model` | `deepseek-chat` | DeepSeek model name |
| `--max-chars` | `5000` | Characters per API chunk |
| `--no-bilingual` | off | Emit translation only (no original) |
| `--dry-run` | off | Show chunk plan without API calls |
| `--api-key` | `$DEEPSEEK_API_KEY` | Override API key |
| `--quiet` / `-q` | off | Suppress progress output |
