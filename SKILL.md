---
name: markdown-translator
description: Translate Markdown documents into bilingual (source↔target) format while preserving formulas, code blocks, tables, charts, links, and all other formatting. Handles long documents via section-aware chunking. Uses the DeepSeek API (deepseek-chat / deepseek-v4-flash) as the translation engine. Invoke when the user asks to translate a .md file, create a bilingual version of a document, or convert technical/academic markdown between languages.
---

# Markdown Bilingual Translator

Preserves **formulas (LaTeX)**, **LaTeX commands** (`\textbf`, `\frac`, `\alpha`, `\begin`/`\end` environments, etc.), **code blocks**, **tables**, **images**, **links**, **HTML**, and **frontmatter** during translation so the output stays structurally identical to the input.  Long documents are split at heading boundaries (`##` / `###`) and translated chunk-by-chunk via the DeepSeek API.

Output is **sentence-by-sentence bilingual**.  Each English sentence and its Chinese translation form a tight visual group separated by ``***`` dividers.  LaTeX formulas are cleaned for Obsidian compatibility.

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

The default model is `deepseek-v4-flash`.  To use a specific model:

```bash
python3 … --model deepseek-v4-flash   # fast, cost-effective (default)
python3 … --model deepseek-chat       # latest general-purpose
python3 … --model deepseek-reasoner   # reasoning model
```

Set the `DEEPSEEK_API_KEY` environment variable or pass `--api-key` explicitly.

## Translation Strategy

The script uses a **protect → chunk → split → translate → restore → clean** pipeline:

1. **Protect** — code blocks, inline code, LaTeX math (`$` / `$$`, `\(` / `\)`, `\[` / `\]`), LaTeX commands (`\textbf{…}`, `\frac{…}{…}`, `\begin{…}…\end{…}`, etc.), images, HTML tags, and frontmatter are replaced with opaque placeholders (`⟨CODEBLOCK:0⟩`, `⟨MATHBLOCK:1⟩`, `⟨LATEXCMD:2⟩`, etc.)
2. **Chunk** — the sanitised text is split at `##` / `###` boundaries, keeping each chunk under the `--max-chars` limit
3. **Split** — each chunk is sentence-split with regex (abbreviation-aware, heading-number protected), structural blocks separated from prose, and translatable sentences identified
4. **Translate** — prose sentences are batch-translated via JSON array API call, guaranteeing output length = input length for strict 1:1 alignment.  Structural blocks (math, code) and headings are preserved as-is
5. **Restore** — placeholders in the aligned sentence pairs are swapped back to their original content
6. **Clean** — LaTeX formulas are post-processed for Obsidian: spaces compressed, punctuation externalised, `<`/`>` converted to `\lt`/`\gt`, `$$` fences isolated with blank-line padding

## Bilingual Output Format

```
## Section Heading (translated)

Score-based diffusion generative models have recently emerged as a standard tool.
基于得分的扩散生成模型最近已成为标准工具。
***
These models aim at learning the score function via SDEs.
这些模型旨在通过 SDE 学习得分函数。
***
$$
u_{k}(t) = \sum_{j=1}^{m} (\gamma^{-1})_{jk} D_{t} X_{T}^{j}.
$$
***
Our approach combines Malliavin derivatives with a novel Bismut-type formula.
我们的方法将马利亚万导数与新的 Bismut 型公式相结合。
```

- Each English↔Chinese pair is a tight visual group (no blank line between them)
- ``***`` horizontal rule separates consecutive sentence groups
- Structural blocks (display math, code, LaTeX environments) appear standalone between dividers
- Headings are translated and emitted once
- LaTeX formulas are cleaned: no token-spaces, punctuation externalised, ``<``/``>`` escaped

## Error Handling

| Symptom | Fix |
|---------|-----|
| `DEEPSEEK_API_KEY is not set` | `export DEEPSEEK_API_KEY="sk-…"` or pass `--api-key` |
| `HTTP 429` | Script auto-retries with exponential backoff; wait and re-run |
| `HTTP 401` | API key is invalid or expired — check at platform.deepseek.com |
| Translation quality is poor on a specific chunk | Re-run with `--max-chars 2000` for smaller, higher-fidelity chunks |
| Sentence alignment is off (rare) | Re-run with `--no-bilingual` for a clean single-language output |

## Tips for Best Results

- **Chunk size**: 5000–8000 chars per chunk gives the best balance of context vs. quality.  Smaller chunks = better consistency but more API calls.
- **Language codes**: Use ISO 639-1 codes (`en`, `zh`, `ja`, `ko`, `fr`, `de`, …).  For Chinese specifically, the model understands `zh` as simplified Chinese.
- **Technical content**: The default temperature of 0.2 keeps terminology stable across chunks.
- **Review**: Always skim the output — the interleaved format makes it easy to spot-check translations against the original.

## Script Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--from` / `--to` | `en` / `zh` | Source and target language codes |
| `--model` | `deepseek-chat` | DeepSeek model name |
| `--max-chars` | `8000` | Characters per API chunk |
| `--no-bilingual` | off | Emit translation only (no original) |
| `--dry-run` | off | Show chunk plan without API calls |
| `--api-key` | `$DEEPSEEK_API_KEY` | Override API key |
| `--quiet` / `-q` | off | Suppress progress output |
