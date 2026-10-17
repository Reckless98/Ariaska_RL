"""Regression coverage for knowledge-pack decision rules in real mentor prompts."""

import json

import pytest

from core.commands.command_registry import COMMAND_REGISTRY, render_command
from core.commands.learned_commands import LearnedCommandStore
from core.knowledge import knowledge_packs
from core.llm.gateway_chat_client import GatewayChatClient
from core.llm.smart_mentor import AttackContext, SmartMentor


@pytest.fixture
def mentor(tmp_path):
    return SmartMentor(
        llm_client=None,
        learned_store=LearnedCommandStore(str(tmp_path / "learned.json")),
    )


@pytest.mark.parametrize("profile", ["generic", "htb", "metasploitable2", "metasploitable3"])
def test_system_prompt_renders_actual_knowledge_pack_rules(mentor, profile):
    mentor.target_profile = profile
    prompt = mentor._build_system_prompt()
    rules_section = prompt.split("=== EXPERT DECISION RULES (IF → THEN) ===", 1)[1]
    rules_section = rules_section.split("=== CVE QUICK REFERENCE ===", 1)[0]
    assert knowledge_packs.HTB_DECISION_RULES
    for rule in knowledge_packs.HTB_DECISION_RULES:
        assert f"[P{rule['priority']}] IF: {rule['if']}\n" in rules_section
        assert f"     THEN: {rule['then']}\n" in rules_section
        assert f"     WHY: {rule['reasoning']}\n" in rules_section


@pytest.mark.parametrize("include_current_keys", [False, True])
def test_legacy_rule_keys_keep_existing_behavior(mentor, monkeypatch, include_current_keys):
    rule = {
        "if_condition": "legacy condition",
        "then_action": "legacy action",
        "priority": 7,
        "reasoning": "legacy reason",
    }
    if include_current_keys:
        rule.update({"if": "current condition", "then": "current action"})
    monkeypatch.setattr(knowledge_packs, "HTB_DECISION_RULES", [rule])
    # This formatter itself requires current keys; isolate the legacy builder contract.
    monkeypatch.setattr(knowledge_packs, "get_mentor_knowledge_text", lambda _: "")
    prompt = mentor._build_system_prompt()
    assert "[P7] IF: legacy condition\n     THEN: legacy action\n" in prompt
    assert "     WHY: legacy reason\n" in prompt


def test_real_prompts_reach_gateway_and_return_parsed_response(mentor):
    template = COMMAND_REGISTRY["nmap_top_ports"]
    params = {"target": "192.0.2.10", "num_ports": "100"}

    class Gateway:
        def __init__(self):
            self.calls = []

        def gpt_request(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return json.dumps({
                "selected_command": template.name,
                "parameters": params,
                "reasoning": "offline regression fixture",
                "confidence": 0.83,
            })

    gateway = Gateway()
    mentor.llm_client = GatewayChatClient(gateway, agent_id="ScoutAgent")
    response = mentor.get_command(
        AttackContext(target=params["target"]), filtered_commands=[template]
    )
    assert len(gateway.calls) == 1
    assert "=== EXPERT DECISION RULES (IF → THEN) ===" in gateway.calls[0][1]["system_prompt"]
    assert response.template_name == template.name
    assert response.command == render_command(template, params)
    assert response.reasoning == "offline regression fixture"
    assert response.confidence == pytest.approx(0.83)
