"""German synonym expansion + semantic scoring for keyword matching."""

import math
import os
import re
from pathlib import Path

# External resources are optional and are never downloaded automatically.
SYNONYMS_FILE = Path(os.environ["DE_BENCH_SYNONYMS"]).expanduser() if os.environ.get("DE_BENCH_SYNONYMS") else None
DECOMPOUND_FILE = Path(os.environ["DE_BENCH_DECOMPOUND"]).expanduser() if os.environ.get("DE_BENCH_DECOMPOUND") else None


def resource_hashes() -> dict:
    import hashlib

    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None
        for name, path in (("synonyms", SYNONYMS_FILE), ("decompound", DECOMPOUND_FILE))
    }


# --- Synonym & Decompound Loading ---

_synonym_index: dict[str, set[str]] | None = None
_decompound_words: set[str] | None = None


def get_synonym_index() -> dict[str, set[str]]:
    global _synonym_index
    if _synonym_index is None:
        if SYNONYMS_FILE is None:
            return {}
        groups = []
        with open(SYNONYMS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Language:") or line.startswith("Charset:") or line.startswith("Thesaurus:"):
                    continue
                words = [w.strip().lower() for w in line.split(",") if w.strip()]
                if len(words) >= 2:
                    groups.append(set(words))
        index: dict[str, set[str]] = {}
        for group in groups:
            for word in group:
                if word not in index:
                    index[word] = set()
                index[word].update(group)
        _synonym_index = index
    return _synonym_index


def get_decompound_words() -> set[str]:
    global _decompound_words
    if _decompound_words is None:
        if DECOMPOUND_FILE is None:
            return set()
        words = set()
        with open(DECOMPOUND_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    words.add(line)
        _decompound_words = words
    return _decompound_words


# --- Keyword Classification ---

def is_exact_keyword(keyword: str) -> bool:
    """Determine if a keyword needs exact matching (§§, article numbers, abbreviations)."""
    # Paragraphs, articles, norms
    if re.match(r'^[§#]', keyword):
        return True
    if re.match(r'^art\.\s*\d', keyword, re.IGNORECASE):
        return True
    # ISO/DIN numbers
    if re.match(r'^(iso|din|asil|euro)\s', keyword, re.IGNORECASE):
        return True
    # Short abbreviations (48v, kwh, scr, etc.)
    if len(keyword) <= 4 and keyword.isalpha():
        return True
    return False


# --- Matching Strategies ---

def keyword_matches_exact(keyword: str, text: str) -> bool:
    """Exact substring match (case-insensitive)."""
    return keyword.lower() in text.lower()


def keyword_matches_stem(keyword: str, text: str) -> bool:
    """Match via German morphological stems + decompounding."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # Direct
    if keyword_lower in text_lower:
        return True

    # Synonym expansion
    index = get_synonym_index()
    if keyword_lower in index:
        for syn in index[keyword_lower]:
            if syn in text_lower:
                return True

    # Stem extraction from keyword, check in text
    stems = _extract_stems(keyword_lower)
    for stem in stems:
        if len(stem) >= 4 and stem in text_lower:
            return True

    # Decompound keyword, check parts in text
    parts = _decompound_word(keyword_lower)
    for part in parts:
        if len(part) >= 5 and part in text_lower:
            return True

    # Multi-word keywords: check each word separately
    if " " in keyword_lower:
        words = keyword_lower.split()
        for word in words:
            if len(word) >= 5 and word in text_lower:
                return True
            for stem in _extract_stems(word):
                if len(stem) >= 4 and stem in text_lower:
                    return True

    # Reverse: extract stems from text words and check if keyword stem appears
    # This catches: keyword "auswuchten" text "ausgewuchtet"
    # keyword stems: ["wuchten", "auswucht"] → check text
    # text word "ausgewuchtet" → stems: ["wucht", "ausgewuchtet"]
    keyword_stems = set(stems)
    keyword_stems.add(keyword_lower)
    # Find words in text that share a root with keyword
    text_words = re.findall(r'[a-zäöüß]+', text_lower)
    for tw in text_words:
        if len(tw) < 4:
            continue
        tw_stems = _extract_stems(tw)
        tw_stems.append(tw)
        # Check overlap between keyword stems and text word stems
        for ks in keyword_stems:
            for ts in tw_stems:
                # Shared root of 5+ characters
                if len(ks) >= 4 and len(ts) >= 4:
                    if ks in ts or ts in ks:
                        return True

    return False


def keyword_matches_semantic(keyword: str, text: str, threshold: float = 0.55) -> bool:
    """Match via embedding cosine similarity. Compares keyword against text chunks."""
    from openai import OpenAI
    client = OpenAI()

    # Embed keyword and relevant text chunks (sentences containing related words)
    # For efficiency, only embed the keyword + full response (not per-sentence)
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=[keyword, text[:8000]]  # limit text to avoid token overflow
    )
    embs = [d.embedding for d in r.data]
    sim = _cosine(embs[0], embs[1])
    return sim >= threshold


# --- Hybrid Matching (the main entry point) ---

def keyword_matches(keyword: str, text: str) -> bool:
    """Hybrid matching: exact for §§/codes, stem+synonym for concepts."""
    if is_exact_keyword(keyword):
        return keyword_matches_exact(keyword, text)
    else:
        return keyword_matches_stem(keyword, text)


def keyword_matches_hybrid(keyword: str, text: str, use_semantic: bool = False) -> tuple[bool, str]:
    """Full hybrid matching with method reporting.
    
    Returns: (matched: bool, method: str)
    Method is one of: 'exact', 'synonym', 'stem', 'semantic', 'none'
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # 1. Exact match (always tried first)
    if keyword_lower in text_lower:
        return True, "exact"

    # 2. For §§ and codes, only exact works
    if is_exact_keyword(keyword):
        return False, "none"

    # 3. Synonym lookup
    index = get_synonym_index()
    if keyword_lower in index:
        for syn in index[keyword_lower]:
            if syn in text_lower:
                return True, "synonym"

    # 4. Stem matching
    stems = _extract_stems(keyword_lower)
    for stem in stems:
        if len(stem) >= 5 and stem in text_lower:
            return True, "stem"

    # 5. Decompound matching
    parts = _decompound_word(keyword_lower)
    for part in parts:
        if len(part) >= 5 and part in text_lower:
            return True, "decompound"

    # 6. Semantic (optional, costs API call)
    if use_semantic:
        if keyword_matches_semantic(keyword, text, threshold=0.60):
            return True, "semantic"

    return False, "none"


# --- Helper functions ---

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_stems(word: str) -> list[str]:
    """Extract possible stems by removing common German suffixes/prefixes."""
    stems = []
    # Suffix removal (nominalizations, adjectives, verbs, etc.)
    suffixes = [
        "ierung", "isation", "igkeit", "schaft", "ation",
        "keit", "heit", "ung", "lich", "isch", "nis", "tum", "bar", "sam",
        "iert", "ieren", "tion", "ieren", "ung", "en", "er", "te", "et",
    ]
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[:-len(suffix)]
            if len(stem) >= 3:
                stems.append(stem)
    # Prefix removal
    prefixes = ["un", "ver", "be", "er", "zer", "ent", "ge", "miss", "aus", "ab", "an", "auf", "ein", "vor", "zu"]
    for prefix in sorted(prefixes, key=len, reverse=True):
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            stems.append(word[len(prefix):])
    # Combined: prefix + suffix removal
    for prefix in prefixes:
        if word.startswith(prefix):
            inner = word[len(prefix):]
            for suffix in suffixes:
                if inner.endswith(suffix) and len(inner) - len(suffix) >= 3:
                    stem = inner[:-len(suffix)]
                    if len(stem) >= 3:
                        stems.append(stem)
    return list(set(stems))


def _decompound_word(word: str) -> list[str]:
    """Split German compound using the decompound dictionary."""
    components = get_decompound_words()
    parts = []
    fugen = ['', 's', 'n', 'en', 'er', 'es']

    for i in range(3, len(word) - 2):
        left = word[:i]
        remainder = word[i:]

        left_valid = left in components

        if not left_valid:
            continue

        for fuge in fugen:
            if remainder.startswith(fuge):
                right = remainder[len(fuge):]
                if len(right) >= 3 and right in components:
                    parts.append(left)
                    parts.append(right)

    return list(set(parts))
