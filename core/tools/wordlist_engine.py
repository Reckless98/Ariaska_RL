#!/usr/bin/env python3
"""
core/tools/wordlist_engine.py — Phase 10.1C: Ultra-Intelligent Wordlist Mutation Engine

Produces high-quality, target-aware password mutations on demand. Uses
heuristic ranking, deduplication, and context-aware generation instead
of massive stored lists.

Architecture:
    WordlistMutationEngine receives context (discovered usernames, hostnames,
    service banners, organization strings, paths) and generates scored
    mutations via composable ops:
      - case variants, leetspeak, separators, digit patterns
      - keyboard patterns, locale patterns, service-aware defaults
      - auth-target-aware ranking (ssh vs web form vs smb)

Usage:
    from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
    engine = WordlistMutationEngine(seed=42)
    ctx = MutationContext(
        base_words=["admin", "tomcat"],
        target_service="ssh",
        discovered_users=["msfadmin"],
        hostname="metasploitable",
    )
    for word, score in engine.generate(ctx, top_k=100):
        print(f"{word} (score={score:.2f})")
"""

import hashlib
import itertools
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.wordlist_engine")


@dataclass
class MutationContext:
    """Context for target-aware wordlist generation."""
    base_words: List[str] = field(default_factory=list)
    discovered_users: List[str] = field(default_factory=list)
    hostname: str = ""
    domain: str = ""
    org_name: str = ""
    service_banners: List[str] = field(default_factory=list)
    web_paths: List[str] = field(default_factory=list)
    target_service: str = ""  # ssh, http, ftp, smb, mysql, etc.
    min_length: int = 1
    max_length: int = 32
    complexity_required: bool = False  # Uppercase + digit + special
    known_passwords: List[str] = field(default_factory=list)  # For pattern expansion
    locale: str = "en"


@dataclass
class MutationTelemetry:
    """Telemetry for wordlist generation."""
    wordlist_generated_count: int = 0
    mutation_ops_used: List[str] = field(default_factory=list)
    estimated_candidates: int = 0
    actual_candidates: int = 0
    cache_hit: bool = False
    generation_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_count": self.wordlist_generated_count,
            "ops_used": self.mutation_ops_used,
            "estimated": self.estimated_candidates,
            "actual": self.actual_candidates,
            "cache_hit": self.cache_hit,
            "gen_time_ms": self.generation_time_ms,
        }


# ============================================================================
# MUTATION OPERATIONS
# ============================================================================

# Common digit patterns for suffixes/prefixes
DIGIT_SUFFIXES = [
    "", "1", "2", "3", "12", "123", "1234", "!",
    "01", "007", "69", "77", "99", "00",
    "2024", "2025", "2026", "23", "24", "25", "26",
]

SEASONS = ["spring", "summer", "fall", "winter", "autumn"]
SEASON_YEARS = [f"{s}{y}" for s in SEASONS for y in ["24", "25", "2024", "2025"]]

SEPARATORS = ["", "_", "-", ".", "@"]

# Leetspeak mappings (light → only common substitutions)
LEET_LIGHT = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
LEET_HEAVY = {
    "a": "@", "b": "8", "c": "(", "e": "3", "g": "9",
    "i": "!", "l": "1", "o": "0", "s": "$", "t": "+", "z": "2",
}

# Keyboard adjacency walk starts (qwerty)
KEYBOARD_PATTERNS = [
    "qwerty", "qwer", "asdf", "zxcv", "1234", "12345",
    "qwertyuiop", "asdfghjkl", "1qaz", "2wsx", "3edc",
    "qazwsx", "!@#$", "password", "p@ssw0rd", "passw0rd",
]

# Service-specific defaults
SERVICE_DEFAULTS: Dict[str, List[str]] = {
    "ssh": ["root", "admin", "user", "ubuntu", "test", "guest"],
    "ftp": ["anonymous", "ftp", "admin", "user"],
    "http": ["admin", "administrator", "root", "user", "test", "guest"],
    "tomcat": ["tomcat", "admin", "manager", "role1", "both"],
    "jenkins": ["admin", "jenkins", "user"],
    "mysql": ["root", "admin", "mysql", "dbadmin", "dba"],
    "postgres": ["postgres", "admin", "pgsql"],
    "smb": ["administrator", "admin", "guest", "user"],
    "vnc": ["password", "vnc", "admin", "letmein"],
    "telnet": ["root", "admin", "user"],
    "redis": ["", "redis", "admin", "password"],  # Redis often has no auth or simple
    "mongodb": ["admin", "root", "mongo"],
    "mssql": ["sa", "admin", "mssql"],
    "oracle": ["sys", "system", "scott", "admin"],
    "ofbiz": ["ofbiz", "admin", "ofbizadmin"],
}

# Common password bases (not full passwords — mutation seeds)
COMMON_BASES = [
    "password", "admin", "root", "letmein", "welcome",
    "changeme", "secret", "master", "dragon", "monkey",
    "shadow", "sunshine", "trustno1", "iloveyou", "batman",
    "access", "hello", "charlie", "donald", "login",
    "princess", "qwerty", "abc123", "football", "baseball",
    "starwars", "passw0rd", "pass", "toor", "test",
]


def _case_variants(word: str) -> Generator[str, None, None]:
    """Generate case variants: lower, upper, title, swapcase."""
    yield word.lower()
    yield word.upper()
    yield word.capitalize()
    if len(word) > 1:
        yield word[0].lower() + word[1:].upper()


def _leetspeak(word: str, heavy: bool = False) -> Generator[str, None, None]:
    """Apply leetspeak substitutions."""
    mapping = LEET_HEAVY if heavy else LEET_LIGHT
    result = word.lower()
    for char, replacement in mapping.items():
        result = result.replace(char, replacement)
    if result != word.lower():
        yield result


def _digit_suffixes(word: str) -> Generator[str, None, None]:
    """Append common digit patterns."""
    for suffix in DIGIT_SUFFIXES:
        yield f"{word}{suffix}"


def _digit_prefixes(word: str) -> Generator[str, None, None]:
    """Prepend common digit patterns."""
    for prefix in ["1", "12", "123", "!"]:
        yield f"{prefix}{word}"


def _separator_combos(words: List[str]) -> Generator[str, None, None]:
    """Join word pairs with separators."""
    if len(words) < 2:
        return
    for w1, w2 in itertools.combinations(words[:6], 2):
        for sep in SEPARATORS:
            yield f"{w1}{sep}{w2}"
            yield f"{w2}{sep}{w1}"


def _season_combos(word: str) -> Generator[str, None, None]:
    """Combine word with season+year patterns."""
    for sy in SEASON_YEARS[:8]:  # Top 8 most likely
        yield f"{word}{sy}"
        yield f"{sy}{word}"


def _locale_patterns(word: str, locale: str) -> Generator[str, None, None]:
    """Generate locale-specific patterns."""
    if locale in ("sr", "rs", "serbian"):
        # Serbian common substitutions: c→č, s→š, z→ž, etc.
        subs = {"c": "č", "s": "š", "z": "ž", "dj": "đ"}
        result = word
        for k, v in subs.items():
            result = result.replace(k, v)
        if result != word:
            yield result
    # Always include the original
    yield word


def _service_aware_defaults(service: str) -> Generator[str, None, None]:
    """Yield service-specific default credentials."""
    defaults = SERVICE_DEFAULTS.get(service.lower(), [])
    for d in defaults:
        yield d
        # Common password patterns with service name
        yield f"{d}:{d}"  # user:pass same
        for suffix in ["", "1", "123", "!", "@123"]:
            yield f"{d}{suffix}"


# ============================================================================
# SCORING HEURISTICS
# ============================================================================

def _score_mutation(
    word: str,
    ctx: MutationContext,
    op_name: str,
) -> float:
    """Score a mutation candidate by likelihood (0.0 - 1.0).

    Higher score = more likely to be correct password.
    """
    score = 0.3  # Base score

    word_lower = word.lower()

    # Boost if contains username or hostname fragments
    for user in ctx.discovered_users:
        if user.lower() in word_lower:
            score += 0.25
            break

    if ctx.hostname and ctx.hostname.lower() in word_lower:
        score += 0.15

    if ctx.org_name and ctx.org_name.lower() in word_lower:
        score += 0.2

    if ctx.domain and ctx.domain.lower() in word_lower:
        score += 0.15

    # Length heuristic (most passwords 6-12 chars)
    if 6 <= len(word) <= 12:
        score += 0.1
    elif len(word) < 4:
        score -= 0.1

    # Complexity bonus
    has_upper = any(c.isupper() for c in word)
    has_digit = any(c.isdigit() for c in word)
    has_special = any(not c.isalnum() for c in word)
    if has_upper and has_digit:
        score += 0.05
    if has_special:
        score += 0.05

    # Service-specific bumps
    if ctx.target_service:
        svc_defaults = SERVICE_DEFAULTS.get(ctx.target_service.lower(), [])
        if word_lower in [d.lower() for d in svc_defaults]:
            score += 0.3

    # Known password pattern expansion
    for kp in ctx.known_passwords:
        if kp.lower() in word_lower or word_lower in kp.lower():
            score += 0.35
            break

    # Op-specific adjustments
    if op_name == "service_default":
        score += 0.15
    elif op_name == "keyboard_pattern":
        score -= 0.05
    elif op_name == "heavy_leet":
        score -= 0.05

    return min(1.0, max(0.0, score))


# ============================================================================
# WORDLIST MUTATION ENGINE
# ============================================================================

class WordlistMutationEngine:
    """Target-aware wordlist mutation engine.

    Generates scored, deduplicated password candidates by composing
    mutation operations on base words with contextual scoring.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._cache: Dict[str, str] = {}  # context_hash -> file_path
        self._telemetry = MutationTelemetry()

    @property
    def telemetry(self) -> MutationTelemetry:
        return self._telemetry

    def generate(
        self,
        ctx: MutationContext,
        top_k: int = 500,
        max_total: int = 10000,
    ) -> List[Tuple[str, float]]:
        """Generate scored mutation candidates.

        Args:
            ctx: Mutation context with base words, target info, etc.
            top_k: Return only top-K highest-scored candidates
            max_total: Maximum raw candidates before dedup + scoring

        Returns:
            List of (word, score) tuples, sorted by score descending.
        """
        start = time.time()
        ops_used: List[str] = []
        seen: Set[str] = set()
        scored: List[Tuple[str, float]] = []

        def _add(word: str, op: str) -> None:
            if len(scored) >= max_total:
                return
            if not word or len(word) < ctx.min_length or len(word) > ctx.max_length:
                return
            if ctx.complexity_required:
                has_upper = any(c.isupper() for c in word)
                has_digit = any(c.isdigit() for c in word)
                if not (has_upper and has_digit):
                    return
            key = word.lower()  # Dedup key
            if key in seen:
                return
            seen.add(key)
            score = _score_mutation(word, ctx, op)
            scored.append((word, score))
            if op not in ops_used:
                ops_used.append(op)

        # Collect all base words
        all_bases: List[str] = list(ctx.base_words)
        all_bases.extend(ctx.discovered_users)
        all_bases.extend(ctx.known_passwords)
        if ctx.hostname:
            all_bases.append(ctx.hostname)
        if ctx.org_name:
            all_bases.append(ctx.org_name)
        if ctx.domain:
            all_bases.append(ctx.domain)
        # Add common bases
        all_bases.extend(COMMON_BASES)

        # Deduplicate base list
        unique_bases = list(dict.fromkeys(all_bases))

        # 1. Service defaults (highest priority)
        if ctx.target_service:
            for word in _service_aware_defaults(ctx.target_service):
                _add(word, "service_default")

        # 2. Raw bases
        for word in unique_bases:
            _add(word, "raw")

        # 3. Case variants
        for word in unique_bases:
            for variant in _case_variants(word):
                _add(variant, "case")

        # 4. Digit suffixes/prefixes
        for word in unique_bases[:20]:  # Limit to top bases
            for variant in _digit_suffixes(word):
                _add(variant, "digit_suffix")
            for variant in _digit_prefixes(word):
                _add(variant, "digit_prefix")

        # 5. Leetspeak (light)
        for word in unique_bases[:15]:
            for variant in _leetspeak(word, heavy=False):
                _add(variant, "leet_light")

        # 6. Leetspeak (heavy)
        for word in unique_bases[:10]:
            for variant in _leetspeak(word, heavy=True):
                _add(variant, "leet_heavy")

        # 7. Separator combos
        combo_words = (ctx.discovered_users + [ctx.hostname] + ctx.base_words)[:8]
        combo_words = [w for w in combo_words if w]
        for variant in _separator_combos(combo_words):
            _add(variant, "separator")

        # 8. Season combos
        for word in unique_bases[:8]:
            for variant in _season_combos(word):
                _add(variant, "season")

        # 9. Keyboard patterns
        for kp in KEYBOARD_PATTERNS:
            _add(kp, "keyboard_pattern")

        # 10. Locale patterns
        for word in unique_bases[:10]:
            for variant in _locale_patterns(word, ctx.locale):
                _add(variant, "locale")

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[:top_k]

        # Update telemetry
        elapsed_ms = int((time.time() - start) * 1000)
        self._telemetry.wordlist_generated_count += 1
        self._telemetry.mutation_ops_used = ops_used
        self._telemetry.estimated_candidates = len(seen)
        self._telemetry.actual_candidates = len(result)
        self._telemetry.generation_time_ms = elapsed_ms

        logger.info(
            "Wordlist generated: %d candidates (from %d raw) in %dms, ops=%s",
            len(result), len(seen), elapsed_ms, ops_used,
        )

        return result

    def generate_to_file(
        self,
        ctx: MutationContext,
        top_k: int = 500,
        output_path: Optional[str] = None,
    ) -> Tuple[str, MutationTelemetry]:
        """Generate wordlist and write to a temp file.

        Args:
            ctx: Mutation context
            top_k: Number of top candidates
            output_path: Optional explicit path; uses temp file if None

        Returns:
            (file_path, telemetry)
        """
        # Check cache
        cache_key = self._context_hash(ctx)
        if cache_key in self._cache:
            cached_path = self._cache[cache_key]
            if os.path.exists(cached_path):
                self._telemetry.cache_hit = True
                logger.debug("Wordlist cache hit: %s", cached_path)
                return cached_path, self._telemetry

        candidates = self.generate(ctx, top_k=top_k)

        if output_path is None:
            fd, output_path = tempfile.mkstemp(
                prefix="ariaska_wl_", suffix=".txt",
            )
            os.close(fd)

        with open(output_path, "w") as f:
            for word, _score in candidates:
                f.write(f"{word}\n")

        self._cache[cache_key] = output_path
        logger.info("Wordlist written to %s (%d entries)", output_path, len(candidates))

        return output_path, self._telemetry

    def _context_hash(self, ctx: MutationContext) -> str:
        """Create a deterministic hash for caching."""
        key_parts = [
            ",".join(sorted(ctx.base_words)),
            ",".join(sorted(ctx.discovered_users)),
            ctx.hostname, ctx.domain, ctx.org_name,
            ctx.target_service, str(ctx.min_length),
            str(ctx.max_length), str(ctx.complexity_required),
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def cleanup_temp_files(self) -> int:
        """Remove cached temp wordlist files. Returns count removed."""
        removed = 0
        for key, path in list(self._cache.items()):
            try:
                if os.path.exists(path) and path.startswith(tempfile.gettempdir()):
                    os.unlink(path)
                    removed += 1
            except OSError:
                pass
            del self._cache[key]
        return removed
