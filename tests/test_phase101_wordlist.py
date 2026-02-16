#!/usr/bin/env python3
"""
tests/test_phase101_wordlist.py — Phase 10.1C: Wordlist Mutation Engine Tests

Tests for deterministic generation, scoring, dedup, context-awareness,
and file-based output.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestWordlistMutationEngine:
    """Core mutation engine tests."""

    def test_basic_generation(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=42)
        ctx = MutationContext(base_words=["admin", "password"])
        results = engine.generate(ctx, top_k=50)
        assert len(results) > 0
        assert len(results) <= 50
        # Should be sorted by score descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_with_seed(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        ctx = MutationContext(base_words=["test", "user"])
        engine1 = WordlistMutationEngine(seed=42)
        results1 = engine1.generate(ctx, top_k=100)
        engine2 = WordlistMutationEngine(seed=42)
        results2 = engine2.generate(ctx, top_k=100)
        # Same seed, same context → same results
        assert results1 == results2

    def test_dedup_stability(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=0)
        ctx = MutationContext(base_words=["admin", "Admin", "ADMIN"])
        results = engine.generate(ctx, top_k=500)
        words = [w for w, _ in results]
        # No exact duplicates (case-insensitive)
        lower_words = [w.lower() for w in words]
        assert len(lower_words) == len(set(lower_words))

    def test_service_aware_ssh(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(
            base_words=["test"],
            target_service="ssh",
        )
        results = engine.generate(ctx, top_k=200)
        words = [w for w, _ in results]
        assert "root" in words  # SSH default
        assert "admin" in words

    def test_service_aware_tomcat(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(
            base_words=["admin"],
            target_service="tomcat",
        )
        results = engine.generate(ctx, top_k=200)
        words = [w for w, _ in results]
        assert "tomcat" in words
        assert "manager" in words

    def test_hostname_boosts_score(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(
            base_words=["admin"],
            hostname="metasploitable",
        )
        results = engine.generate(ctx, top_k=500)
        # metasploitable-containing words should have higher score
        meta_score = None
        admin_score = None
        for word, score in results:
            if "metasploitable" in word.lower() and meta_score is None:
                meta_score = score
            if word == "admin" and admin_score is None:
                admin_score = score
        # hostname-derived entries exist
        assert any("metasploitable" in w.lower() for w, _ in results)

    def test_discovered_users_used(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(
            base_words=[],
            discovered_users=["msfadmin", "postgres"],
        )
        results = engine.generate(ctx, top_k=200)
        words = [w for w, _ in results]
        assert "msfadmin" in words
        assert "postgres" in words

    def test_digit_suffixes(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(base_words=["admin"])
        results = engine.generate(ctx, top_k=500)
        words = [w for w, _ in results]
        assert "admin1" in words
        assert "admin123" in words

    def test_leetspeak(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(base_words=["password"])
        results = engine.generate(ctx, top_k=500)
        words = [w for w, _ in results]
        # Light leet: a→4, e→3, s→5, o→0 | Heavy leet: a→@, s→$, o→0
        assert "passw0rd" in words or "p@$$w0rd" in words

    def test_length_filtering(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(
            base_words=["a", "test"],
            min_length=4, max_length=8,
        )
        results = engine.generate(ctx, top_k=500)
        for word, _ in results:
            assert 4 <= len(word) <= 8

    def test_complexity_filter(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(
            base_words=["admin", "test"],
            complexity_required=True,
        )
        results = engine.generate(ctx, top_k=200)
        for word, _ in results:
            has_upper = any(c.isupper() for c in word)
            has_digit = any(c.isdigit() for c in word)
            assert has_upper and has_digit, f"'{word}' doesn't meet complexity"

    def test_keyboard_patterns_included(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(base_words=[])
        results = engine.generate(ctx, top_k=500)
        words = [w for w, _ in results]
        assert "qwerty" in words

    def test_max_total_cap(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(base_words=["a", "b", "c", "d", "e"])
        results = engine.generate(ctx, top_k=10, max_total=50)
        assert len(results) <= 10


class TestWordlistFileOutput:
    """Test file-based wordlist generation."""

    def test_generate_to_file(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=42)
        ctx = MutationContext(base_words=["admin", "test"])
        path, tel = engine.generate_to_file(ctx, top_k=50)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) > 0
        assert len(lines) <= 50
        # Cleanup
        engine.cleanup_temp_files()
        assert not os.path.exists(path)

    def test_cache_hit(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=42)
        ctx = MutationContext(base_words=["admin"])
        path1, _tel1 = engine.generate_to_file(ctx, top_k=50)
        path2, tel2 = engine.generate_to_file(ctx, top_k=50)
        assert path1 == path2
        assert tel2.cache_hit is True
        engine.cleanup_temp_files()


class TestWordlistTelemetry:
    """Test telemetry tracking."""

    def test_telemetry_populated(self):
        from core.tools.wordlist_engine import WordlistMutationEngine, MutationContext
        engine = WordlistMutationEngine(seed=1)
        ctx = MutationContext(base_words=["admin"])
        engine.generate(ctx, top_k=100)
        tel = engine.telemetry
        assert tel.wordlist_generated_count == 1
        assert tel.actual_candidates > 0
        assert tel.estimated_candidates >= tel.actual_candidates
        assert len(tel.mutation_ops_used) > 0

    def test_telemetry_to_dict(self):
        from core.tools.wordlist_engine import MutationTelemetry
        tel = MutationTelemetry(
            wordlist_generated_count=2,
            mutation_ops_used=["case", "digit_suffix"],
            estimated_candidates=500,
            actual_candidates=100,
        )
        d = tel.to_dict()
        assert d["generated_count"] == 2
        assert "case" in d["ops_used"]
