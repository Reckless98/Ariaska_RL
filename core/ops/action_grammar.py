"""core/ops/action_grammar.py — Phase 42: Action Grammar sequencing.

Encodes valid action sequences (phase transitions and command ordering)
as a lightweight grammar. Used by SmartCoach to filter/prioritize
candidate commands based on what should logically follow the current
action history.

Example grammar rules:
- After "nmap -sV" (service scan), prefer "searchsploit" or "nikto"
- After "exploit" success, escalate to "privesc" commands
- After "credential discovered", try "ssh/ftp login"

Author: Phase 42 Contract
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.ops.action_grammar")


@dataclass
class GrammarRule:
    """Single grammar rule: after precursor patterns, prefer these actions."""
    rule_id: str
    precursor_patterns: List[str]  # Command prefixes that trigger this rule
    preferred_templates: List[str]  # Template names to prioritize
    phase: str = ""  # Optional phase constraint
    priority: float = 1.0  # Higher = stronger preference
    description: str = ""


class ActionGrammar:
    """Phase 42: Command sequencing grammar for SmartCoach.

    Maintains a set of grammar rules that map action history patterns
    to preferred next actions. Used to bias command selection toward
    logically coherent sequences.

    Methods:
        add_rule(): Register a grammar rule
        suggest(): Given history, return prioritized template names
        load_defaults(): Load built-in penetration testing grammar
        reset(): Clear learned rules (keeps defaults)
    """

    def __init__(self, load_defaults: bool = True) -> None:
        self._rules: Dict[str, GrammarRule] = {}
        self._learned_rules: Dict[str, GrammarRule] = {}
        self._match_count: int = 0
        self._total_queries: int = 0
        if load_defaults:
            self._load_defaults()
        logger.info(
            "ActionGrammar initialized with %d rules", len(self._rules)
        )

    def add_rule(self, rule: GrammarRule) -> None:
        """Register a grammar rule.

        Args:
            rule: The GrammarRule to add.
        """
        self._rules[rule.rule_id] = rule

    def add_learned_rule(self, rule: GrammarRule) -> None:
        """Register a dynamically learned grammar rule.

        Args:
            rule: The learned GrammarRule.
        """
        self._learned_rules[rule.rule_id] = rule

    def suggest(
        self,
        history: List[str],
        phase: str = "",
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Suggest preferred templates based on action history.

        Args:
            history: Recent command strings (most recent last).
            phase: Current attack phase for phase-specific rules.
            top_k: Maximum number of suggestions.

        Returns:
            List of (template_name, priority) tuples, sorted by priority desc.
        """
        self._total_queries += 1
        if not history:
            return []

        suggestions: Dict[str, float] = {}
        all_rules = {**self._rules, **self._learned_rules}

        for rule in all_rules.values():
            # Phase filter
            if rule.phase and phase and rule.phase != phase:
                continue

            # Check if any precursor pattern matches recent history
            matched = False
            for pattern in rule.precursor_patterns:
                for cmd in history[-5:]:  # Check last 5 commands
                    if pattern.lower() in cmd.lower():
                        matched = True
                        break
                if matched:
                    break

            if matched:
                self._match_count += 1
                for template in rule.preferred_templates:
                    existing = suggestions.get(template, 0.0)
                    suggestions[template] = max(existing, rule.priority)

        # Sort by priority descending
        sorted_suggs = sorted(
            suggestions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_suggs[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return grammar statistics.

        Returns:
            Dict with rule counts and match statistics.
        """
        return {
            "default_rules": len(self._rules),
            "learned_rules": len(self._learned_rules),
            "total_queries": self._total_queries,
            "total_matches": self._match_count,
            "match_rate": (
                self._match_count / max(self._total_queries, 1)
            ),
        }

    def reset(self) -> None:
        """Clear learned rules, keep defaults."""
        self._learned_rules.clear()
        self._match_count = 0
        self._total_queries = 0
        logger.debug("ActionGrammar reset (kept %d default rules)", len(self._rules))

    def _load_defaults(self) -> None:
        """Load built-in penetration testing grammar rules."""
        defaults = [
            GrammarRule(
                rule_id="recon_to_enum",
                precursor_patterns=["nmap -sV", "nmap -sC", "nmap -A"],
                preferred_templates=[
                    "searchsploit_service", "nikto_scan", "gobuster_dir",
                    "enum4linux", "smbclient_list",
                ],
                phase="RECON",
                priority=1.5,
                description="After service scan, enumerate findings",
            ),
            GrammarRule(
                rule_id="web_discovery_to_exploit",
                precursor_patterns=["gobuster", "dirb", "ffuf", "nikto"],
                preferred_templates=[
                    "curl_path", "sqlmap_url", "wpscan",
                    "hydra_http", "burp_scan",
                ],
                phase="ENUMERATION",
                priority=1.3,
                description="After web discovery, probe found paths",
            ),
            GrammarRule(
                rule_id="cred_to_login",
                precursor_patterns=["credential", "password", "hash"],
                preferred_templates=[
                    "ssh_login", "ftp_login", "smb_login",
                    "mysql_login", "psexec",
                ],
                priority=2.0,
                description="After credential discovery, try login",
            ),
            GrammarRule(
                rule_id="shell_to_privesc",
                precursor_patterns=["shell", "reverse_shell", "meterpreter"],
                preferred_templates=[
                    "linpeas", "linux_exploit_suggester", "sudo_l",
                    "find_suid", "kernel_exploit",
                ],
                priority=2.0,
                description="After shell, escalate privileges",
            ),
            GrammarRule(
                rule_id="smb_enum",
                precursor_patterns=["smbclient", "enum4linux", "smbmap"],
                preferred_templates=[
                    "smb_download", "crackmapexec", "impacket_secretsdump",
                ],
                phase="ENUMERATION",
                priority=1.2,
                description="After SMB enum, download/crack",
            ),
            GrammarRule(
                rule_id="exploit_to_stabilize",
                precursor_patterns=["exploit", "msf_exploit"],
                preferred_templates=[
                    "shell_upgrade", "pty_spawn", "stable_shell",
                ],
                phase="EXPLOITATION",
                priority=1.8,
                description="After exploit, stabilize shell",
            ),
            GrammarRule(
                rule_id="privesc_to_loot",
                precursor_patterns=["root", "admin", "SYSTEM"],
                preferred_templates=[
                    "cat_shadow", "hashdump", "flag_search",
                    "data_exfil", "collect_artifacts",
                ],
                priority=1.5,
                description="After privesc, collect loot",
            ),
        ]
        for rule in defaults:
            self._rules[rule.rule_id] = rule
