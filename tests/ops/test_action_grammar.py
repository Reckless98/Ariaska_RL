"""Phase 42 Stage 3: ActionGrammar unit tests."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestActionGrammar:
    """Tests for the ActionGrammar command sequencing module."""

    def _make_grammar(self, load_defaults: bool = True):
        from core.ops.action_grammar import ActionGrammar
        return ActionGrammar(load_defaults=load_defaults)

    def test_init_with_defaults(self):
        """ActionGrammar loads default rules on init."""
        g = self._make_grammar()
        stats = g.get_stats()
        assert stats["default_rules"] >= 5

    def test_init_without_defaults(self):
        """ActionGrammar can start empty."""
        g = self._make_grammar(load_defaults=False)
        stats = g.get_stats()
        assert stats["default_rules"] == 0

    def test_add_rule(self):
        """add_rule registers a rule."""
        from core.ops.action_grammar import GrammarRule
        g = self._make_grammar(load_defaults=False)
        g.add_rule(GrammarRule(
            rule_id="test_rule",
            precursor_patterns=["nmap"],
            preferred_templates=["nikto_scan"],
            priority=1.0,
        ))
        stats = g.get_stats()
        assert stats["default_rules"] == 1

    def test_suggest_matching(self):
        """suggest returns templates when history matches."""
        g = self._make_grammar()
        suggestions = g.suggest(
            history=["nmap -sV -p- 10.0.0.1"],
            phase="RECON",
        )
        assert len(suggestions) > 0
        template_names = [s[0] for s in suggestions]
        assert any("searchsploit" in t or "nikto" in t for t in template_names)

    def test_suggest_no_match(self):
        """suggest returns empty when no history matches."""
        g = self._make_grammar()
        suggestions = g.suggest(
            history=["unknown_command_xyz"],
            phase="RECON",
        )
        assert len(suggestions) == 0

    def test_suggest_empty_history(self):
        """suggest handles empty history."""
        g = self._make_grammar()
        suggestions = g.suggest(history=[], phase="RECON")
        assert suggestions == []

    def test_credential_rule(self):
        """Credential discovery triggers login suggestions."""
        g = self._make_grammar()
        suggestions = g.suggest(
            history=["Found credential: admin:password123"],
        )
        assert len(suggestions) > 0
        names = [s[0] for s in suggestions]
        assert any("login" in n for n in names)

    def test_learned_rules(self):
        """add_learned_rule contributes to suggestions."""
        from core.ops.action_grammar import GrammarRule
        g = self._make_grammar(load_defaults=False)
        g.add_learned_rule(GrammarRule(
            rule_id="learned_1",
            precursor_patterns=["specialized_scan"],
            preferred_templates=["specialized_exploit"],
            priority=3.0,
        ))
        suggestions = g.suggest(history=["specialized_scan -a"])
        assert len(suggestions) == 1
        assert suggestions[0][0] == "specialized_exploit"

    def test_reset_clears_learned(self):
        """reset clears learned rules but keeps defaults."""
        from core.ops.action_grammar import GrammarRule
        g = self._make_grammar()
        default_count = g.get_stats()["default_rules"]
        g.add_learned_rule(GrammarRule(
            rule_id="temp",
            precursor_patterns=["x"],
            preferred_templates=["y"],
        ))
        g.reset()
        stats = g.get_stats()
        assert stats["default_rules"] == default_count
        assert stats["learned_rules"] == 0

    def test_top_k_limit(self):
        """suggest respects top_k limit."""
        g = self._make_grammar()
        suggestions = g.suggest(
            history=["nmap -sV target"],
            phase="RECON",
            top_k=2,
        )
        assert len(suggestions) <= 2
