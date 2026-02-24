"""
Response-cleaning utilities for LLM simulation outputs.

Can be used as a module:
    from response_cleaning import clean_entry, clean_response

Or run as a script to clean *_random_response.json files before LLM_judge.py:
    python response_cleaning.py results_joined/ --output-dir results_cleaned/
    python response_cleaning.py results_joined/ --output-dir results_cleaned/ --clean-level 1
    python response_cleaning.py results_joined/ --output-dir results_cleaned/ --dry-run

If --output-dir is omitted the files are modified in-place.

Cleaning levels
---------------
  0  No-op — text returned unchanged.
  1  Discard-only — responses consisting entirely of bracket/bold tags are
     replaced with '' (e.g. "[Conversation]", "**Please respond to the user.**").
  2  Full cleaning (default) — level-1 discard + prefix stripping:
       A/B  @User_* / @anon_user_* prefix (incl. **bold** wrapper)
       C    User_XXXX: prefix
       D    [Output] prefix
       E    any other [tag] or [tag]: prefix when real content follows
       G    u/User_* / u/anon_user_* prefix (Reddit style, incl. **bold** wrapper)

Only our own-user handle formats (User_XXXX, anon_user_XXXX) are stripped.
References to other users (e.g. @ElizabethScudder:, u/ConservativeVoice:) are kept.

See response_cleaning.md for detailed documentation, examples, and statistics.
"""

import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Rule A+B: @User_* / @anon_user_* prefix  (e.g. "@User_2C21F7: text")
# group(1) captures the full handle including @, used for author comparison and dedup
_RE_AT_HANDLE = re.compile(r'^(@(?:User_|anon_user_)\w+)\s*:?\s*', re.IGNORECASE)

# Rule A+B (bold variant): **@User_XXXX**: or **@User_XXXX:** — unwrapped before Rule A/B
_RE_BOLD_AT_UNWRAP = re.compile(r'^\*\*(@(?:User_|anon_user_)\w+):?\*\*\s*:?\s*', re.IGNORECASE)

# Rule C: bare User_XXXX: prefix  (e.g. "User_06D3E3: text")
# group(1) captures the handle for author comparison
_RE_BARE_USER = re.compile(r'^(User_[A-Za-z0-9]+)\s*:\s*')

# Rule D: [Output] prefix
_RE_OUTPUT_TAG = re.compile(r'^\[Output\]\s*', re.IGNORECASE)

# Rule E: any remaining [tag] or [tag]: prefix when real content follows
_RE_BRACKET_PREFIX = re.compile(r'^\[[^\]]+\]\s*:?\s*(?=\S)')

# Rule G: u/User_* or u/anon_user_* prefix (Reddit style, e.g. "u/User_05B2: text")
# group(1) captures the full handle including u/, used for author comparison and dedup
_RE_U_HANDLE = re.compile(r'^([Uu]/(?:User_|anon_user_)\w+)\s*:?\s*')

# Rule G (bold variant): **u/User_XXXX**: — unwrapped before Rule G
_RE_BOLD_U_UNWRAP = re.compile(r'^\*\*([Uu]/(?:User_|anon_user_)\w+):?\*\*\s*:?\s*')

# Used by Rule F to test whether anything real remains after removing all [...] and **...**
_RE_BRACKET_STRIP = re.compile(r'\[[^\]]*\]')
_RE_BOLD_STRIP    = re.compile(r'\*\*[^*\n]+\*\*')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_bracket_only(text: str) -> bool:
    """Return True if text contains nothing beyond bracket/bold tags and punctuation.

    Covers both [tag] patterns and **bold label** patterns (e.g. leaked instructions).
    """
    stripped = _RE_BRACKET_STRIP.sub('', text)
    stripped = _RE_BOLD_STRIP.sub('', stripped)
    stripped = stripped.strip(' \t\n\r.,;:!?*#-/')
    return len(stripped) == 0


def clean_response(text: str, level: int = 2, author: str = '') -> str:
    """Clean LLM generation artifacts from a response string.

    Parameters
    ----------
    text   : the raw response string
    level  : cleaning level (0 = none, 1 = discard-only, 2 = full)
    author : the user ID of the response author (e.g. 'User_2C21F7').
             When provided, handle-prefix rules (A/B, C, G) only strip if
             the handle matches the author.  References to other users are
             left intact.  Pass '' to disable the check (strip any handle).

    Returns the cleaned string, or '' if the response is discarded.
    """
    if level == 0:
        return text if isinstance(text, str) else ''

    if not text or not isinstance(text, str):
        return text or ''
    text = text.strip()
    if not text:
        return ''

    # Rule F (first pass): whole response is bracket/bold-tag junk → discard
    if is_bracket_only(text):
        return ''

    if level == 1:
        return text

    # level 2: strip prefix artifacts
    author_lower = author.lower() if author else ''

    def _handle_matches(handle_id: str) -> bool:
        """True if handle_id matches the author, or if no author is set."""
        return (not author_lower) or (handle_id.lower() == author_lower)

    # Pre-process: unwrap **bold** from our-user handles so Rules A/B and G apply normally
    m = _RE_BOLD_AT_UNWRAP.match(text)
    if m and _handle_matches(m.group(1).lstrip('@').lower()):
        text = (m.group(1) + ' ' + text[m.end():]).strip()
    m = _RE_BOLD_U_UNWRAP.match(text)
    if m and _handle_matches(re.sub(r'^[Uu]/', '', m.group(1)).lower()):
        text = (m.group(1) + ' ' + text[m.end():]).strip()

    # Rule A/B: @User_* / @anon_user_* prefix — only if handle matches author.
    # Strip once; if the exact same handle repeats immediately, strip a second time.
    m = _RE_AT_HANDLE.match(text)
    if m and _handle_matches(m.group(1).lstrip('@').lower()):
        first_handle = m.group(1).lower()
        text = text[m.end():].strip()
        m2 = _RE_AT_HANDLE.match(text)
        if m2 and m2.group(1).lower() == first_handle:
            text = text[m2.end():].strip()

    # Rule C: bare User_XXXX: prefix — only if handle matches author.
    m = _RE_BARE_USER.match(text)
    if m and _handle_matches(m.group(1).lower()):
        text = text[m.end():].strip()

    # Rule D: [Output] prefix
    text = _RE_OUTPUT_TAG.sub('', text).strip()

    # Rule E: any remaining [tag][:] prefix when followed by real content
    text = _RE_BRACKET_PREFIX.sub('', text).strip()

    # Rule G: u/User_* / u/anon_user_* prefix — only if handle matches author.
    m = _RE_U_HANDLE.match(text)
    if m and _handle_matches(re.sub(r'^[Uu]/', '', m.group(1)).lower()):
        first_handle = m.group(1).lower()
        text = text[m.end():].strip()
        m2 = _RE_U_HANDLE.match(text)
        if m2 and m2.group(1).lower() == first_handle:
            text = text[m2.end():].strip()

    # Rule F (second pass): discard if stripping left nothing real
    if is_bracket_only(text):
        return ''

    return text


def clean_entry(entry: dict, level: int = 2) -> dict:
    """Apply clean_response to all response fields of a simulation entry.

    Modifies and returns the entry dict in-place. Fields cleaned:
      - 'response'            : single randomly-selected response
      - 'all_valid_responses' : list of all candidate responses

    The entry's 'user' field is used as the author for handle-prefix rules:
    only handles matching the author are stripped; references to other users
    are left intact.
    """
    if level == 0:
        return entry

    author = entry.get('user', '')

    if 'response' in entry:
        entry['response'] = clean_response(entry['response'], level, author=author)

    if 'all_valid_responses' in entry:
        entry['all_valid_responses'] = [
            clean_response(r, level, author=author) for r in entry['all_valid_responses']
        ]

    return entry


def classify_response(text: str) -> str:
    """Classify a response by the first cleaning rule that would fire.

    Returns one of: 'at_user', 'u_handle', 'bare_user', 'brk_out', 'brk_text',
                    'brk_only', 'clean'
    Classification is based on pattern type only (no author check).
    """
    if not text or not isinstance(text, str):
        return 'clean'
    text = text.strip()
    if not text:
        return 'clean'
    if is_bracket_only(text):
        return 'brk_only'
    if _RE_BOLD_AT_UNWRAP.match(text) or _RE_AT_HANDLE.match(text):
        return 'at_user'
    if _RE_BOLD_U_UNWRAP.match(text) or _RE_U_HANDLE.match(text):
        return 'u_handle'
    if _RE_BARE_USER.match(text):
        return 'bare_user'
    if _RE_OUTPUT_TAG.match(text):
        return 'brk_out'
    if _RE_BRACKET_PREFIX.match(text):
        return 'brk_text'
    return 'clean'


# ---------------------------------------------------------------------------
# Script entry point — clean *_random_response.json files in-place
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from collections import defaultdict
    from pathlib import Path

    CATEGORIES = ['at_user', 'u_handle', 'bare_user', 'brk_out', 'brk_text', 'brk_only', 'clean']

    parser = argparse.ArgumentParser(
        description="Clean response artifacts in *_random_response.json files (in-place)."
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Root directory to search for *_random_response.json files (e.g. results_joined/)"
    )
    parser.add_argument(
        "--clean-level", type=int, choices=[0, 1, 2], default=2,
        metavar="{0,1,2}",
        help="Cleaning level (0=none, 1=discard-only, 2=full; default: 2)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Write cleaned files here (mirroring input structure). "
             "If omitted, files are modified in-place."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without modifying files"
    )
    parser.add_argument(
        "--stats-file", type=str, default="cleaning_stats.txt",
        help="Path to write per-model/platform statistics (default: cleaning_stats.txt)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    files = sorted(input_dir.rglob("*_random_response.json"))

    if not files:
        print(f"No *_random_response.json files found in {input_dir}")
        raise SystemExit(1)

    # stats[model][platform][category] = count
    stats: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    n_files = 0
    n_entries = 0

    for fpath in files:
        # Extract platform and model from path:
        # .../results_joined/{platform}/{vendor}/{ModelName}__{flags}__random_response.json
        platform = fpath.parts[-3]
        vendor   = fpath.parts[-2]
        model_name = fpath.stem.split('__')[0]
        model = f"{vendor}/{model_name}"

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        # Classify before cleaning, then clean
        cleaned = []
        for entry in data:
            cat = classify_response(entry.get('response', ''))
            stats[model][platform][cat] += 1
            cleaned.append(clean_entry(entry, args.clean_level))

        if output_dir is not None:
            out_path = output_dir / fpath.relative_to(input_dir)
        else:
            out_path = fpath

        if args.dry_run:
            print(f"[DRY RUN] Would clean {len(cleaned)} entries -> {out_path}")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)
            print(f"Cleaned {len(cleaned)} entries -> {out_path}")

        n_files += 1
        n_entries += len(cleaned)

    print(f"\nDone: {n_files} files, {n_entries} entries (clean_level={args.clean_level})")

    # -----------------------------------------------------------------------
    # Build aggregated totals
    # -----------------------------------------------------------------------
    all_models    = sorted(stats.keys())
    all_platforms = sorted({p for m in stats for p in stats[m]})

    # totals_by_model[model][cat], totals_by_platform[platform][cat]
    totals_by_model:    dict = defaultdict(lambda: defaultdict(int))
    totals_by_platform: dict = defaultdict(lambda: defaultdict(int))
    grand_total:        dict = defaultdict(int)

    for model in all_models:
        for platform in stats[model]:
            for cat in CATEGORIES:
                n = stats[model][platform][cat]
                totals_by_model[model][cat]       += n
                totals_by_platform[platform][cat] += n
                grand_total[cat]                  += n

    total_responses = sum(grand_total.values())

    # -----------------------------------------------------------------------
    # Formatting helpers
    # -----------------------------------------------------------------------
    def pct(n, total):
        return f"{n:>6,} ({100*n/total if total else 0:>5.1f}%)"

    col_w   = 22   # category column width
    name_w  = max(len(m) for m in all_models) + 2

    def header_line():
        cols = "".join(f"{'':>{col_w}}" for _ in CATEGORIES[:-1]) + f"{'':>{col_w}}"
        return "-" * (name_w + col_w * len(CATEGORIES))

    def col_header():
        return (f"{'':>{name_w}}" +
                "".join(f"{c:>{col_w}}" for c in CATEGORIES))

    def row(name, counts, total):
        return (f"{name:>{name_w}}" +
                "".join(pct(counts[c], total) for c in CATEGORIES) +
                f"   {total:,}")

    # -----------------------------------------------------------------------
    # Write stats file
    # -----------------------------------------------------------------------
    lines = []
    lines.append("PREFIX-PATTERN STATISTICS (response field only)")
    lines.append(f"Generated from: {input_dir.resolve()}")
    lines.append(f"Total responses classified: {total_responses:,}")
    lines.append(f"clean_level applied: {args.clean_level}")
    lines.append("")

    sep = "-" * (name_w + col_w * len(CATEGORIES) + 6)

    # Table 1: by model
    lines.append("=" * 10 + " TABLE 1: BY MODEL " + "=" * 10)
    lines.append("")
    lines.append(col_header())
    lines.append(sep)
    for model in all_models:
        total = sum(totals_by_model[model].values())
        lines.append(row(model, totals_by_model[model], total))
    lines.append(sep)
    lines.append(row("GRAND TOTAL", grand_total, total_responses))
    lines.append(sep)
    lines.append("")

    # Table 2: by platform
    lines.append("=" * 10 + " TABLE 2: BY PLATFORM " + "=" * 10)
    lines.append("")
    lines.append(col_header())
    lines.append(sep)
    for platform in all_platforms:
        total = sum(totals_by_platform[platform].values())
        lines.append(row(platform, totals_by_platform[platform], total))
    lines.append(sep)
    lines.append(row("GRAND TOTAL", grand_total, total_responses))
    lines.append(sep)
    lines.append("")

    # Table 3: by model x platform
    lines.append("=" * 10 + " TABLE 3: BY MODEL x PLATFORM " + "=" * 10)
    for platform in all_platforms:
        lines.append("")
        lines.append(f"  --- Platform: {platform.upper()} ---")
        lines.append(col_header())
        lines.append(sep)
        for model in all_models:
            if platform not in stats[model]:
                continue
            total = sum(stats[model][platform].values())
            lines.append(row(model, stats[model][platform], total))
        plat_total = sum(totals_by_platform[platform].values())
        lines.append(sep)
        lines.append(row("GRAND TOTAL", totals_by_platform[platform], plat_total))
        lines.append(sep)

    stats_path = Path(args.stats_file)
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nStatistics written to {stats_path}")
