"""
Phase 9.3 Component Tests — TacticalCortex, ExecutiveCortex, CommandEnrichment,
Unknown-Target Generalization, Service Archetypes.

Tests all new components introduced in Phase 9.3:
  - ServiceArchetype + classify_service
  - UnknownTargetStrategy chain-of-thought reasoning
  - GeneralizationPlaybook generation
  - TacticalCortex per-step assessment
  - ExecutiveCortex episode-level planning
  - CommandTemplate enrichment (assigned_agents, not_when, etc.)
"""

import os
import pytest
from typing import Dict, Any, List

# Ensure dry-run mode for all tests
os.environ["ARIASKA_DRY_RUN"] = "1"


# ═══════════════════════════════════════════════════════════════════════════
# ServiceArchetype + classify_service
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceArchetypes:
    """Tests for ServiceArchetype enum and classify_service function."""

    def test_archetype_enum_has_expected_members(self):
        from core.knowledge.target_profiler import ServiceArchetype
        expected = {
            "AUTH", "WEB", "DATABASE", "FILE_SHARING", "REMOTE_EXEC",
            "MAIL", "DNS", "DIRECTORY", "MONITORING", "MESSAGING",
            "VNC_RDP", "CONTAINER", "CI_CD", "CUSTOM_APP",
        }
        actual = {m.name for m in ServiceArchetype}
        assert expected == actual

    def test_classify_ssh(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=22, service="ssh", product="OpenSSH")
        archetypes = classify_service(svc)
        assert ServiceArchetype.AUTH in archetypes

    def test_classify_web(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=80, service="http", product="Apache")
        archetypes = classify_service(svc)
        assert ServiceArchetype.WEB in archetypes

    def test_classify_database(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=3306, service="mysql", product="MySQL")
        archetypes = classify_service(svc)
        assert ServiceArchetype.DATABASE in archetypes

    def test_classify_smb_is_file_sharing(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=445, service="microsoft-ds")
        archetypes = classify_service(svc)
        assert ServiceArchetype.FILE_SHARING in archetypes

    def test_classify_unknown_port_is_custom_app(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=31337, service="")
        archetypes = classify_service(svc)
        assert ServiceArchetype.CUSTOM_APP in archetypes

    def test_classify_ftp_dual_archetype(self):
        """FTP should be both AUTH and FILE_SHARING."""
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=21, service="ftp", product="vsftpd")
        archetypes = classify_service(svc)
        assert ServiceArchetype.AUTH in archetypes
        assert ServiceArchetype.FILE_SHARING in archetypes

    def test_classify_rdp(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=3389, service="ms-wbt-server")
        archetypes = classify_service(svc)
        assert ServiceArchetype.VNC_RDP in archetypes

    def test_classify_ldap_is_directory(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=389, service="ldap")
        archetypes = classify_service(svc)
        assert ServiceArchetype.DIRECTORY in archetypes

    def test_classify_docker_api(self):
        from core.knowledge.target_profiler import classify_service, ServiceFingerprint, ServiceArchetype
        svc = ServiceFingerprint(port=2375, service="docker")
        archetypes = classify_service(svc)
        assert ServiceArchetype.CONTAINER in archetypes


# ═══════════════════════════════════════════════════════════════════════════
# TargetProfile archetype properties
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetProfileArchetypes:
    """Tests for the new archetype-related properties on TargetProfile."""

    @pytest.fixture
    def profile_with_archetypes(self):
        from core.knowledge.target_profiler import TargetProfile
        profile = TargetProfile(target_ip="10.0.0.1")
        profile.service_archetypes = {
            "web": [80, 8080],
            "auth": [22],
            "database": [3306, 5432, 27017],
        }
        return profile

    def test_archetype_summary_counts(self, profile_with_archetypes):
        summary = profile_with_archetypes.archetype_summary
        assert summary["web"] == 2
        assert summary["auth"] == 1
        assert summary["database"] == 3

    def test_dominant_archetypes_order(self, profile_with_archetypes):
        dominant = profile_with_archetypes.dominant_archetypes
        assert dominant[0] == "database"  # 3 ports
        assert dominant[1] == "web"  # 2 ports
        assert dominant[2] == "auth"  # 1 port

    def test_empty_archetypes(self):
        from core.knowledge.target_profiler import TargetProfile
        profile = TargetProfile()
        assert profile.archetype_summary == {}
        assert profile.dominant_archetypes == []

    def test_generalization_assessment_default_none(self):
        from core.knowledge.target_profiler import TargetProfile
        profile = TargetProfile()
        assert profile.generalization_assessment is None


# ═══════════════════════════════════════════════════════════════════════════
# UnknownTargetStrategy — Chain-of-thought
# ═══════════════════════════════════════════════════════════════════════════

class TestUnknownTargetStrategy:
    """Tests for chain-of-thought reasoning on unknown targets."""

    @pytest.fixture
    def linux_profile(self):
        from core.knowledge.target_profiler import (
            TargetProfile, ServiceFingerprint, TargetType, TargetProfiler,
        )
        profile = TargetProfile(
            target_ip="10.10.10.50",
            os_family="unknown",
            target_type=TargetType.UNKNOWN,
            confidence=0.1,
            services=[
                ServiceFingerprint(port=22, service="ssh", product="OpenSSH"),
                ServiceFingerprint(port=80, service="http", product="Apache"),
                ServiceFingerprint(port=3306, service="mysql", product="MySQL"),
            ],
            open_ports={22, 80, 3306},
        )
        profiler = TargetProfiler()
        profiler._classify_archetypes(profile)
        return profile

    @pytest.fixture
    def minimal_profile(self):
        from core.knowledge.target_profiler import (
            TargetProfile, ServiceFingerprint, TargetType, TargetProfiler,
        )
        profile = TargetProfile(
            target_ip="10.10.10.99",
            os_family="unknown",
            target_type=TargetType.UNKNOWN,
            services=[ServiceFingerprint(port=80, service="http")],
            open_ports={80},
        )
        profiler = TargetProfiler()
        profiler._classify_archetypes(profile)
        return profile

    @pytest.fixture
    def empty_profile(self):
        from core.knowledge.target_profiler import TargetProfile, TargetType
        return TargetProfile(target_type=TargetType.UNKNOWN)

    def test_assess_returns_all_keys(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        required_keys = {
            "reasoning_trace", "attack_priorities", "recommended_tools",
            "estimated_difficulty", "os_inference", "credential_opportunities",
            "has_privesc_path", "summary",
        }
        assert required_keys <= set(result.keys())

    def test_reasoning_trace_has_steps(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        trace = result["reasoning_trace"]
        assert len(trace) >= 5  # At least 5 steps
        assert any("STEP 1" in s for s in trace)
        assert any("STEP 3" in s for s in trace)

    def test_os_inference_detects_linux(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        assert result["os_inference"] == "linux"

    def test_priorities_sorted_descending(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        priorities = [p["priority"] for p in result["attack_priorities"]]
        assert priorities == sorted(priorities, reverse=True)

    def test_has_privesc_path_true_for_mixed(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        assert result["has_privesc_path"] is True

    def test_tools_not_empty(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        assert len(result["recommended_tools"]) > 0

    def test_tools_capped_at_20(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        assert len(result["recommended_tools"]) <= 20

    def test_empty_profile_returns_scan_recommendation(self, empty_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(empty_profile)
        assert result["estimated_difficulty"] == "unknown"
        assert "nmap" in result["recommended_tools"][0]

    def test_web_only_target_is_hard(self, minimal_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(minimal_profile)
        # Web-only with 1 port → hard
        assert result["estimated_difficulty"] == "hard"

    def test_difficulty_easy_for_backdoor_port(self):
        from core.knowledge.target_profiler import (
            TargetProfile, ServiceFingerprint, TargetType,
            TargetProfiler, UnknownTargetStrategy,
        )
        profile = TargetProfile(
            target_ip="10.10.10.1",
            os_family="linux",
            target_type=TargetType.UNKNOWN,
            services=[
                ServiceFingerprint(port=22, service="ssh"),
                ServiceFingerprint(port=80, service="http"),
                ServiceFingerprint(port=1524, service="ingreslock"),
                ServiceFingerprint(port=3306, service="mysql"),
                ServiceFingerprint(port=5432, service="postgresql"),
                ServiceFingerprint(port=512, service="exec"),
                ServiceFingerprint(port=513, service="login"),
                ServiceFingerprint(port=514, service="shell"),
                ServiceFingerprint(port=6667, service="irc"),
                ServiceFingerprint(port=5900, service="vnc"),
                ServiceFingerprint(port=2049, service="nfs"),
            ],
            open_ports={22, 80, 1524, 3306, 5432, 512, 513, 514, 6667, 5900, 2049},
        )
        profiler = TargetProfiler()
        profiler._classify_archetypes(profile)
        strategy = UnknownTargetStrategy()
        result = strategy.assess(profile)
        assert result["estimated_difficulty"] == "easy"

    def test_summary_is_string(self, linux_profile):
        from core.knowledge.target_profiler import UnknownTargetStrategy
        strategy = UnknownTargetStrategy()
        result = strategy.assess(linux_profile)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 20


# ═══════════════════════════════════════════════════════════════════════════
# GeneralizationPlaybook
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneralizationPlaybook:
    """Tests for the fallback playbook generator."""

    @pytest.fixture
    def linux_profile(self):
        from core.knowledge.target_profiler import (
            TargetProfile, ServiceFingerprint, TargetType, TargetProfiler,
        )
        profile = TargetProfile(
            target_ip="10.10.10.50",
            os_family="linux",
            target_type=TargetType.UNKNOWN,
            services=[
                ServiceFingerprint(port=22, service="ssh", product="OpenSSH"),
                ServiceFingerprint(port=80, service="http", product="Apache"),
                ServiceFingerprint(port=3306, service="mysql", product="MySQL"),
            ],
            open_ports={22, 80, 3306},
        )
        profiler = TargetProfiler()
        profiler._classify_archetypes(profile)
        return profile

    def test_from_profile_creates_chains(self, linux_profile):
        from core.knowledge.target_profiler import GeneralizationPlaybook
        playbook = GeneralizationPlaybook.from_profile(linux_profile)
        assert len(playbook.chains) >= 3  # At least one per archetype

    def test_chains_have_required_fields(self, linux_profile):
        from core.knowledge.target_profiler import GeneralizationPlaybook
        playbook = GeneralizationPlaybook.from_profile(linux_profile)
        for chain in playbook.chains:
            assert "name" in chain
            assert "steps" in chain
            assert "source" in chain
            assert chain["source"] == "generalization"

    def test_chains_have_steps_with_commands(self, linux_profile):
        from core.knowledge.target_profiler import GeneralizationPlaybook
        playbook = GeneralizationPlaybook.from_profile(linux_profile)
        for chain in playbook.chains:
            for step in chain["steps"]:
                assert "step" in step
                assert "command" in step

    def test_empty_profile_returns_nmap_chain(self):
        from core.knowledge.target_profiler import TargetProfile, GeneralizationPlaybook
        profile = TargetProfile(target_ip="10.0.0.1")
        playbook = GeneralizationPlaybook.from_profile(profile)
        assert len(playbook.chains) == 1
        assert "nmap" in playbook.chains[0]["steps"][0]["command"]

    def test_to_prompt_fragment_returns_string(self, linux_profile):
        from core.knowledge.target_profiler import GeneralizationPlaybook
        playbook = GeneralizationPlaybook.from_profile(linux_profile)
        fragment = playbook.to_prompt_fragment(max_chars=500)
        assert isinstance(fragment, str)
        assert len(fragment) <= 500

    def test_reasoning_populated(self, linux_profile):
        from core.knowledge.target_profiler import GeneralizationPlaybook
        playbook = GeneralizationPlaybook.from_profile(linux_profile)
        assert len(playbook.reasoning) > 10

    def test_credential_chain_added_when_multiple_auth(self):
        from core.knowledge.target_profiler import (
            TargetProfile, ServiceFingerprint, TargetType,
            TargetProfiler, GeneralizationPlaybook,
        )
        profile = TargetProfile(
            target_ip="10.10.10.1",
            os_family="linux",
            target_type=TargetType.UNKNOWN,
            services=[
                ServiceFingerprint(port=22, service="ssh"),
                ServiceFingerprint(port=80, service="http"),
                ServiceFingerprint(port=3306, service="mysql"),
                ServiceFingerprint(port=5432, service="postgresql"),
            ],
            open_ports={22, 80, 3306, 5432},
        )
        profiler = TargetProfiler()
        profiler._classify_archetypes(profile)
        playbook = GeneralizationPlaybook.from_profile(profile)
        chain_names = [c["name"] for c in playbook.chains]
        assert any("Credential" in n for n in chain_names)


# ═══════════════════════════════════════════════════════════════════════════
# TacticalCortex
# ═══════════════════════════════════════════════════════════════════════════

class TestTacticalCortex:
    """Tests for per-step tactical assessment."""

    @pytest.fixture
    def cortex(self):
        from core.cortex.tactical_cortex import TacticalCortex
        return TacticalCortex()

    @pytest.fixture
    def basic_state(self):
        return {
            "ports_discovered": True,
            "services_discovered": True,
            "shell_obtained": False,
        }

    @pytest.fixture
    def discovery_board(self):
        return {
            "ports": {22, 80, 3306},
            "services": {"ssh", "http", "mysql"},
            "credentials": set(),
            "shells": set(),
        }

    def test_cortex_init(self, cortex):
        from core.cortex.tactical_cortex import TacticalCortex
        assert isinstance(cortex, TacticalCortex)

    def test_assess_returns_assessment(self, cortex, basic_state, discovery_board):
        from core.cortex.tactical_cortex import TacticalAssessment
        from core.commands.command_registry import COMMAND_REGISTRY
        # Get any valid template
        templates = list(COMMAND_REGISTRY.values())
        template = templates[0]
        result = cortex.assess(
            command=template.template,
            template=template,
            state=basic_state,
            agent_role="offensive",
            discovery_board=discovery_board,
            current_phase="EXPLOITATION",
            detection_risk=0.2,
            step=5,
        )
        assert isinstance(result, TacticalAssessment)
        assert hasattr(result, "verdict")
        assert hasattr(result, "confidence")
        assert hasattr(result, "rules_evaluated")

    def test_assess_has_valid_verdict(self, cortex, basic_state, discovery_board):
        from core.cortex.tactical_cortex import TacticalVerdict
        from core.commands.command_registry import COMMAND_REGISTRY
        templates = list(COMMAND_REGISTRY.values())
        template = templates[0]
        result = cortex.assess(
            command=template.template,
            template=template,
            state=basic_state,
            agent_role="offensive",
            discovery_board=discovery_board,
            current_phase="EXPLOITATION",
            detection_risk=0.2,
            step=5,
        )
        assert result.verdict in (
            TacticalVerdict.APPROVE, TacticalVerdict.REDIRECT,
            TacticalVerdict.BLOCK, TacticalVerdict.ESCALATE,
        )

    def test_record_step_and_stats(self, cortex):
        cortex.record_step("nmap -sV target", "RECON", "recon")
        cortex.record_step("nmap -sV target", "RECON", "recon")
        stats = cortex.get_stats()
        assert stats["steps_assessed"] == 2

    def test_reset_episode(self, cortex):
        cortex.record_step("nmap -sV target", "RECON", "recon")
        cortex.reset_episode()
        stats = cortex.get_stats()
        assert stats["steps_assessed"] == 0

    def test_llm_budget_respected(self, cortex):
        from core.cortex.tactical_cortex import TacticalCortex
        # Default max is 5 LLM calls per episode
        assert cortex._llm_calls_this_episode == 0
        assert cortex._max_llm_calls == 5


# ═══════════════════════════════════════════════════════════════════════════
# ExecutiveCortex
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutiveCortex:
    """Tests for episode-level strategic planning."""

    @pytest.fixture
    def cortex(self):
        from core.cortex.executive_cortex import ExecutiveCortex
        return ExecutiveCortex()

    @pytest.fixture
    def discovery_board(self):
        return {
            "ports": {22, 80},
            "services": {"ssh", "http"},
            "credentials": set(),
            "shells": set(),
            "phase": "RECON",
        }

    def test_cortex_init(self, cortex):
        from core.cortex.executive_cortex import ExecutiveCortex
        assert isinstance(cortex, ExecutiveCortex)

    def test_create_plan_returns_attack_plan(self, cortex, discovery_board):
        from core.cortex.executive_cortex import AttackPlan
        plan = cortex.create_plan(
            initial_state=discovery_board,
            target_type="unknown",
            max_steps=120,
        )
        assert isinstance(plan, AttackPlan)
        assert len(plan.objectives) > 0

    def test_plan_has_phases(self, cortex, discovery_board):
        plan = cortex.create_plan(
            initial_state=discovery_board,
            target_type="unknown",
            max_steps=120,
        )
        phases = [obj.phase for obj in plan.objectives]
        assert "RECON" in phases

    def test_get_phase_guidance(self, cortex, discovery_board):
        cortex.create_plan(
            initial_state=discovery_board,
            target_type="unknown",
            max_steps=120,
        )
        guidance = cortex.get_phase_guidance("RECON")
        assert isinstance(guidance, dict)

    def test_record_step(self, cortex, discovery_board):
        cortex.create_plan(
            initial_state=discovery_board,
            target_type="unknown",
            max_steps=120,
        )
        cortex.record_step("nmap -sV target", "RECON", discovery_board)

    def test_end_episode_returns_summary(self, cortex, discovery_board):
        cortex.create_plan(
            initial_state=discovery_board,
            target_type="unknown",
            max_steps=120,
        )
        summary = cortex.end_episode()
        assert isinstance(summary, dict)

    def test_phase_templates_exist(self):
        from core.cortex.executive_cortex import ExecutiveCortex
        cortex = ExecutiveCortex()
        # Should have templates for standard phases
        assert hasattr(cortex, "_PHASE_TEMPLATES") or len(cortex._phase_templates) > 0


# ═══════════════════════════════════════════════════════════════════════════
# CommandTemplate Enrichment
# ═══════════════════════════════════════════════════════════════════════════

class TestCommandEnrichment:
    """Tests for Phase 9.3 command enrichment fields."""

    def test_all_commands_have_assigned_agents(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        missing = []
        for name, tmpl in COMMAND_REGISTRY.items():
            if not tmpl.assigned_agents:
                missing.append(name)
        assert len(missing) == 0, f"Commands missing assigned_agents: {missing[:10]}"

    def test_all_commands_have_not_when(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        missing = []
        for name, tmpl in COMMAND_REGISTRY.items():
            if not tmpl.not_when:
                missing.append(name)
        assert len(missing) == 0, f"Commands missing not_when: {missing[:10]}"

    def test_most_commands_have_follows_after(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        with_follows = sum(1 for t in COMMAND_REGISTRY.values() if t.follows_after)
        total = len(COMMAND_REGISTRY)
        ratio = with_follows / total
        assert ratio >= 0.9, f"Only {with_follows}/{total} have follows_after ({ratio:.1%})"

    def test_some_commands_have_enables(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        with_enables = sum(1 for t in COMMAND_REGISTRY.values() if t.enables)
        assert with_enables >= 100, f"Only {with_enables} commands have enables"

    def test_assigned_agents_are_valid_roles(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        valid_roles = {"scout", "red", "shadow", "blue", "orion"}
        for name, tmpl in COMMAND_REGISTRY.items():
            for agent in tmpl.assigned_agents:
                assert agent in valid_roles, f"{name} has invalid agent '{agent}'"

    def test_get_usage_context_includes_agents(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        for name, tmpl in list(COMMAND_REGISTRY.items())[:5]:
            ctx = tmpl.get_usage_context()
            assert "AGENTS:" in ctx, f"{name} missing AGENTS in usage context"

    def test_command_count_unchanged(self):
        """Enrichment should not add/remove commands."""
        from core.commands.command_registry import COMMAND_REGISTRY
        assert len(COMMAND_REGISTRY) == 255, f"Expected 255 commands, got {len(COMMAND_REGISTRY)}"


# ═══════════════════════════════════════════════════════════════════════════
# TargetProfiler integration with archetypes
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetProfilerIntegration:
    """Tests for TargetProfiler using archetypes and generalization."""

    def test_profile_from_state_classifies_archetypes(self):
        from core.knowledge.target_profiler import TargetProfiler
        state = {
            "target_ip": "10.0.0.1",
            "port_22": True,
            "port_80": True,
            "port_3306": True,
            "platform": "linux",
        }
        profiler = TargetProfiler()
        profile = profiler.profile_from_state(state)
        # Should have classified archetypes
        assert len(profile.service_archetypes) > 0

    def test_unknown_target_gets_generalization(self):
        from core.knowledge.target_profiler import TargetProfiler, TargetType
        state = {
            "target_ip": "10.0.0.99",
            "port_9999": True,
            "platform": "unknown",
        }
        profiler = TargetProfiler()
        profile = profiler.profile_from_state(state)
        # Unknown target should get generalization assessment
        if profile.target_type == TargetType.UNKNOWN or profile.confidence < 0.3:
            assert profile.generalization_assessment is not None

    def test_ms2_target_no_generalization(self):
        from core.knowledge.target_profiler import TargetProfiler, TargetType
        # Simulate MS2 with many signature ports
        state = {
            "target_ip": "192.168.56.101",
            "platform": "linux",
        }
        for p in [21, 22, 23, 25, 80, 139, 445, 512, 513, 514, 1099, 1524, 2049, 3306, 5432, 5900, 6667, 8180]:
            state[f"port_{p}"] = True
        profiler = TargetProfiler()
        profile = profiler.profile_from_state(state)
        # MS2 should be classified with high confidence, no generalization needed
        if profile.target_type == TargetType.METASPLOITABLE2 and profile.confidence >= 0.3:
            assert profile.generalization_assessment is None

    def test_select_knowledge_includes_archetypes(self):
        from core.knowledge.target_profiler import TargetProfiler
        state = {
            "target_ip": "10.0.0.1",
            "port_22": True,
            "port_80": True,
            "platform": "linux",
        }
        profiler = TargetProfiler()
        profile = profiler.profile_from_state(state)
        knowledge = profiler.select_knowledge(profile)
        assert "service_archetypes" in knowledge or "dominant_archetypes" in knowledge
