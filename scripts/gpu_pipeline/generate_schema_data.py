#!/usr/bin/env python3
"""
generate_schema_data.py — Ariaska Schema-Perfect Synthetic Data Generator

Uses Qwen3-32B teacher (via vLLM) to generate schema-perfect training examples
for all 7 Ariaska LLM prompt types.

Outputs: /workspace/data/ariaska_schema_sft.jsonl
"""

import json
import os
import re
import random
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

# ── JSON Schema Definitions for All 7 Ariaska Prompt Types ──────────────────

SCHEMAS = {
    "microchain_classify": {
        "description": "MicroChain Stage 1: Classify tactical situation in one word",
        "output_type": "single_word",
        "valid_outputs": [
            "recon_gap", "enum_needed", "exploit_ready",
            "privesc_needed", "post_exploit", "lateral_move", "stalled"
        ],
    },
    "microchain_generate": {
        "description": "MicroChain Stage 2: Generate candidate commands as JSON array",
        "output_type": "json_array",
        "required_fields": [
            "command", "template_name", "reasoning", "evidence_used",
            "hypothesis", "test", "expected_observable", "stop_condition", "confidence"
        ],
        "field_types": {
            "command": "string",
            "template_name": "string",
            "reasoning": "string",
            "evidence_used": "list_of_strings",
            "hypothesis": "string",
            "test": "string",
            "expected_observable": "string",
            "stop_condition": "string",
            "confidence": "float_0_1",
        },
    },
    "microchain_score": {
        "description": "MicroChain Stage 3: Score candidates for phase fit",
        "output_type": "json_array",
        "required_fields": ["idx", "phase_fit", "evidence_support", "novelty"],
        "field_types": {
            "idx": "integer",
            "phase_fit": "float_0_1",
            "evidence_support": "float_0_1",
            "novelty": "float_0_1",
        },
    },
    "microchain_fast_local": {
        "description": "MicroChain fast local single-call: pick best command",
        "output_type": "json_object",
        "required_fields": ["command", "template_name", "reasoning", "score"],
        "field_types": {
            "command": "string",
            "template_name": "string",
            "reasoning": "string",
            "score": "float_0_1",
        },
    },
    "phase_guided": {
        "description": "PhaseGuidedLLM P34: Full structured guidance",
        "output_type": "json_object",
        "required_fields": [
            "phase_decision", "anomalies", "candidates",
            "selection", "distillation_packet"
        ],
        "nested_schemas": {
            "phase_decision": {
                "required": ["chosen_phase", "phase_confidence", "phase_goal",
                           "stay_conditions", "move_on_conditions",
                           "contradictions", "phase_tag"],
                "constraints": {"phase_tag": "P34", "phase_confidence": "float_0_1"},
            },
            "candidates": {
                "item_required": ["template_name", "family", "why",
                                "expected_outcome", "stop_condition",
                                "confidence", "risk", "tags"],
                "constraints": {"confidence": "float_0_1", "risk": "low|medium|high"},
            },
            "selection": {
                "required": ["best_template_name", "runner_up_template_name",
                           "selection_reason", "should_escalate_to_codex",
                           "escalation_reason"],
            },
            "distillation_packet": {
                "required": ["observation", "reasoning", "action_target",
                           "expected_outcome", "phase_target",
                           "confidence_target", "gating_notes", "phase_tag"],
                "constraints": {"phase_tag": "P34", "confidence_target": "float_0_1"},
            },
        },
    },
    "phase_guided_fast_local": {
        "description": "PhaseGuidedLLM fast local: stay/advance + 3 candidates",
        "output_type": "json_object",
        "required_fields": ["stay_or_advance", "reason", "candidates", "confidence"],
        "field_types": {
            "stay_or_advance": "enum_stay_advance",
            "reason": "string",
            "candidates": "list_of_strings",
            "confidence": "float_0_1",
        },
    },
    "smart_mentor": {
        "description": "SmartMentor: Select best command with reasoning",
        "output_type": "json_object",
        "required_fields": [
            "intent", "selected_command", "parameters", "reasoning",
            "expected_observation", "risk", "confidence",
            "next_phase_hint", "candidate_actions"
        ],
        "field_types": {
            "intent": "string",
            "selected_command": "template_name",  # MUST be template name, not raw command
            "parameters": "dict",
            "reasoning": "string",
            "expected_observation": "string",
            "risk": "enum_risk",
            "confidence": "float_0_1",
            "next_phase_hint": "string",
            "candidate_actions": "list_of_dicts",
        },
    },
    "coherence_classify": {
        "description": "CoherenceChain Step A: Classify phase from evidence",
        "output_type": "json_object",
        "required_fields": [
            "phase_guess", "phase_confidence", "key_evidence",
            "missing_evidence", "next_best_families"
        ],
        "field_types": {
            "phase_guess": "enum_phase",
            "phase_confidence": "float_0_1",
            "key_evidence": "list_of_strings",
            "missing_evidence": "list_of_strings",
            "next_best_families": "list_of_strings",
        },
    },
    "coherence_summarize": {
        "description": "CoherenceChain Step C: Compact state postcard",
        "output_type": "json_object",
        "required_fields": ["postcard", "evidence_counts"],
        "field_types": {
            "postcard": "string",
            "evidence_counts": "dict_str_int",
        },
    },
    "coherence_score": {
        "description": "CoherenceChain Step D: Coherence quality metrics",
        "output_type": "json_object",
        "required_fields": [
            "coherence_score", "novelty_score", "repeat_risk",
            "confidence_calibration"
        ],
        "field_types": {
            "coherence_score": "float_0_1",
            "novelty_score": "float_0_1",
            "repeat_risk": "float_0_1",
            "confidence_calibration": "float_0_1",
        },
    },
}

# ── Ariaska Command Templates (from command_registry.py) ───────────────────

COMMAND_TEMPLATES = [
    "nmap_fast_scan", "nmap_version_detection", "nmap_vuln_scan",
    "nmap_os_detection", "nmap_all_ports", "nmap_udp_scan",
    "nmap_script_scan", "nmap_aggressive_scan",
    "gobuster_dir", "gobuster_vhost", "gobuster_dns",
    "nikto_scan", "dirb_scan", "wfuzz_fuzz",
    "ffuf_dir", "ffuf_vhost",
    "hydra_ssh", "hydra_ftp", "hydra_http_post",
    "medusa_ssh", "crackmapexec_smb",
    "ssh_login", "ssh_key_login", "ssh_command",
    "ftp_anonymous", "ftp_login", "ftp_download",
    "smbclient_list", "smbclient_connect", "enum4linux",
    "sqlmap_test", "sqlmap_dump",
    "searchsploit", "exploit_metasploit",
    "msfconsole_exploit", "msfvenom_payload",
    "wget_download", "curl_request", "curl_post",
    "python_http_server", "nc_reverse_shell", "nc_listen",
    "find_suid", "find_writable", "sudo_check",
    "linpeas", "linenum", "pspy",
    "getcap_check", "cat_file", "ls_dir",
    "id_check", "whoami_check", "uname_check",
    "netstat_check", "ifconfig_check", "ps_check",
    "crontab_check", "etc_passwd", "etc_shadow",
    "docker_escape", "lxd_escape",
    "tcpdump_capture", "tshark_read",
    "mysql_login", "psql_login",
    "redis_cli", "mongo_login",
    "wpscan", "joomscan",
    "snmp_walk", "ldapsearch",
    "rpcclient", "rpcinfo",
    "impacket_psexec", "impacket_smbexec",
    "impacket_wmiexec", "impacket_secretsdump",
    "bloodhound_collect", "kerbrute_userenum",
    "getuserspns", "getnpusers",
    "chisel_proxy", "ligolo_proxy",
    "socat_forward", "ssh_tunnel",
    "exfil_base64", "exfil_dns", "exfil_http",
]

ATTACK_PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION",
    "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
    "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"
]

SERVICES = [
    "ssh:22", "http:80", "https:443", "ftp:21", "smb:445",
    "mysql:3306", "postgresql:5432", "redis:6379", "mongodb:27017",
    "dns:53", "snmp:161", "ldap:389", "kerberos:88",
    "rdp:3389", "winrm:5985", "vnc:5900", "telnet:23",
    "smtp:25", "pop3:110", "imap:143",
    "nfs:2049", "rpc:111",
]

PORTS = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 161,
         389, 443, 445, 512, 513, 514, 1099, 1433, 1524, 2049,
         3306, 3389, 5432, 5900, 5985, 6379, 8080, 8443, 8888, 27017]

FAMILIES = ["nmap", "web", "ssh", "smb", "privesc", "file", "misc", "dns"]

# ── Tactical Scenarios (diverse states for generation) ──────────────────────

def _random_discovery_board() -> Dict[str, Any]:
    """Generate a random but realistic discovery board."""
    n_ports = random.randint(0, 12)
    ports = sorted(random.sample(PORTS, min(n_ports, len(PORTS))))
    n_services = min(random.randint(0, n_ports), 8)
    services = random.sample(SERVICES, min(n_services, len(SERVICES)))
    
    creds = []
    if random.random() > 0.6:
        users = random.sample(["admin", "root", "user", "nathan", "www-data", "john", "tom", "ftpuser"], random.randint(1, 3))
        for u in users:
            creds.append(f"{u}:{''.join(random.choices('abcdefghijklmnop0123456789', k=8))}")
    
    shells = []
    if random.random() > 0.7:
        shells.append({"type": random.choice(["user", "root"]), "method": random.choice(["ssh", "reverse_shell", "webshell"])})
    
    vulns = []
    if random.random() > 0.5:
        vulns = random.sample(["CVE-2021-41773", "CVE-2021-3156", "CVE-2019-15107", 
                               "CVE-2017-0144", "CVE-2014-6271", "CVE-2021-44228",
                               "MS17-010", "vsftpd_backdoor", "distccd_rce"], random.randint(1, 3))
    
    web_paths = []
    if random.random() > 0.5 and any("http" in s for s in services):
        web_paths = random.sample(["/admin", "/login", "/api", "/uploads", "/backup",
                                   "/robots.txt", "/.git", "/config", "/data/0"], random.randint(1, 4))
    
    return {
        "ports": ports,
        "services": services,
        "credentials": creds,
        "shells": shells,
        "vulns": vulns,
        "web_paths": web_paths,
        "users": [c.split(":")[0] for c in creds],
        "flags_set": [],
    }


def _random_recent_commands(n: int = 5) -> List[str]:
    """Generate random recent commands."""
    cmd_pool = [
        "nmap -sV -sC 10.10.10.5",
        "nmap -p- -T4 10.10.10.5",
        "gobuster dir -u http://10.10.10.5 -w /usr/share/wordlists/dirb/common.txt",
        "nikto -h http://10.10.10.5",
        "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.10.10.5",
        "ssh user@10.10.10.5",
        "find / -perm -4000 -type f 2>/dev/null",
        "sudo -l",
        "cat /etc/passwd",
        "linpeas.sh",
        "curl http://10.10.10.5/data/0",
        "tshark -r capture.pcap -Y 'ftp' -T fields -e ftp.request.arg",
        "searchsploit apache 2.4",
        "sqlmap -u http://10.10.10.5/page?id=1 --batch",
        "smbclient -L //10.10.10.5 -N",
        "enum4linux -a 10.10.10.5",
        "crackmapexec smb 10.10.10.5 -u admin -p password",
        "getcap -r / 2>/dev/null",
        "cat /home/user/user.txt",
    ]
    return random.sample(cmd_pool, min(n, len(cmd_pool)))


def _random_available_templates(phase: str, n: int = 15) -> List[str]:
    """Return templates appropriate for a given phase."""
    phase_templates = {
        "RECON": ["nmap_fast_scan", "nmap_all_ports", "nmap_udp_scan", "nmap_os_detection", 
                  "nmap_version_detection", "gobuster_dir", "nikto_scan"],
        "ENUMERATION": ["nmap_version_detection", "nmap_vuln_scan", "nmap_script_scan",
                       "gobuster_dir", "nikto_scan", "ffuf_dir", "enum4linux",
                       "smbclient_list", "snmp_walk", "ldapsearch", "wpscan"],
        "EXPLOITATION": ["hydra_ssh", "hydra_ftp", "hydra_http_post", "ssh_login",
                        "ftp_anonymous", "sqlmap_test", "searchsploit", "exploit_metasploit",
                        "msfconsole_exploit", "curl_post", "nc_reverse_shell"],
        "PRIVILEGE_ESCALATION": ["find_suid", "sudo_check", "linpeas", "linenum",
                                "getcap_check", "pspy", "crontab_check", "docker_escape",
                                "etc_passwd", "etc_shadow"],
        "LATERAL_MOVEMENT": ["ssh_login", "ssh_key_login", "chisel_proxy", "ligolo_proxy",
                            "impacket_psexec", "impacket_wmiexec", "socat_forward", "ssh_tunnel"],
        "POST_EXPLOITATION": ["cat_file", "ls_dir", "id_check", "whoami_check", 
                             "ps_check", "netstat_check", "impacket_secretsdump",
                             "bloodhound_collect"],
        "EXFILTRATION": ["exfil_base64", "exfil_http", "exfil_dns", "cat_file",
                        "wget_download", "python_http_server"],
    }
    pool = phase_templates.get(phase, COMMAND_TEMPLATES[:20])
    return random.sample(pool, min(n, len(pool)))


# ── Prompt Builders for Each Schema Type ────────────────────────────────────

def build_microchain_classify_prompt(board: Dict, recent: List[str], phase: str, role: str) -> str:
    ports = list(board.get("ports", []))[:10]
    services = list(board.get("services", []))[:10]
    return (
        f"Classify the tactical situation in one word.\n"
        f"Phase: {phase}\nRole: {role}\n"
        f"Ports: {ports}\nServices: {services}\n"
        f"Recent commands: {recent[-5:]}\n"
        f"Options: recon_gap, enum_needed, exploit_ready, privesc_needed, "
        f"post_exploit, lateral_move, stalled\n"
        f"Reply with ONLY the situation label."
    )


def build_microchain_generate_prompt(board: Dict, phase: str, situation: str, 
                                      role: str, templates: List[str]) -> str:
    ports = list(board.get("ports", []))[:10]
    services = list(board.get("services", []))[:10]
    creds = list(board.get("credentials", []))[:3]
    return (
        f"Generate 3 candidate commands for phase={phase}, situation={situation}, "
        f"role={role}.\n"
        f"Known ports: {ports}\nKnown services: {services}\n"
        f"Known credentials: {creds}\n"
        f"Available templates: {templates}\n\n"
        f"Reply with ONLY a JSON array. EVERY object MUST include ALL fields:\n"
        f'[{{"command":"...", "template_name":"...", "reasoning":"...", '
        f'"evidence_used":["port_22","service_ssh"], '
        f'"hypothesis":"SSH may allow password auth", '
        f'"test":"template_name", '
        f'"expected_observable":"SSH banner or auth prompt", '
        f'"stop_condition":"auth denied or timeout", '
        f'"confidence":0.7}}]\n'
        f"No markdown, no explanation outside JSON."
    )


def build_microchain_score_prompt(candidates: List[Dict], phase: str,
                                   board: Dict, recent: List[str]) -> str:
    cmd_list = [{"idx": i, "cmd": c.get("command", "")[:80]} for i, c in enumerate(candidates)]
    ports = list(board.get("ports", []))[:10]
    return (
        f"Score these commands for phase={phase}.\n"
        f"Commands: {json.dumps(cmd_list)}\n"
        f"Known ports: {ports}\nRecent: {recent[-5:]}\n\n"
        f"Reply ONLY with JSON array of objects:\n"
        f'[{{"idx":0,"phase_fit":0.8,"evidence_support":0.7,"novelty":0.6}}]\n'
        f"Scores 0.0-1.0. No markdown."
    )


def build_microchain_fast_local_prompt(board: Dict, phase: str, role: str,
                                        stagnation: int, recent: List[str],
                                        templates: List[str]) -> str:
    ports = sorted(str(p) for p in list(board.get("ports", []))[:8])
    services = sorted(str(s) for s in list(board.get("services", []))[:8])
    creds = list(board.get("credentials", []))[:3]
    return (
        "/no_think\n"
        f"phase={phase} role={role} stagnation={stagnation}\n"
        f"ports={ports} services={services} creds={creds}\n"
        f"recent={recent[-5:]}\n"
        f"templates={templates[:10]}\n"
        "Pick the BEST next command. Reply ONLY JSON:\n"
        '{"command":"full command string","template_name":"from templates list",'
        '"reasoning":"why this command","score":0.0-1.0}'
    )


def build_phase_guided_prompt(board: Dict, phase: str, role: str,
                               stagnation: int, templates: List[str]) -> str:
    """Full PhaseGuidedLLM prompt."""
    input_data = {
        "episode_id": f"ep_{random.randint(1,100)}",
        "step_id": random.randint(1, 500),
        "agent_role": role,
        "current_phase": phase,
        "inferred_phase": phase,
        "phase_state": {
            "stagnation_steps": stagnation,
            "recent_discovery_deltas": [random.randint(0, 3) for _ in range(5)],
        },
        "discovery_board": {
            "ports": sorted(board.get("ports", [])),
            "services": sorted(board.get("services", [])),
            "credentials": board.get("credentials", []),
            "vulns": board.get("vulns", []),
            "shells": board.get("shells", []),
            "web_paths": board.get("web_paths", []),
        },
        "available_templates": [{"name": t} for t in templates[:20]],
        "last_output_excerpt": "Sample output from previous command...",
    }
    sys_prompt = (
        "You are PHASE GUIDE for Ariaska_RL pentesting lab. "
        "Output ONLY a raw JSON object. No prose, no markdown fences. "
        'Include "phase_tag":"P34" in phase_decision and distillation_packet. '
        "Keys: phase_decision, anomalies, candidates, selection, distillation_packet. "
        "phase_decision needs: chosen_phase, phase_confidence (0-1), phase_goal, stay_conditions[], move_on_conditions[], contradictions[], phase_tag. "
        "candidates[]: template_name, family, why, expected_outcome, stop_condition, confidence, risk, tags[]. "
        "selection: best_template_name, runner_up_template_name, selection_reason, should_escalate_to_codex, escalation_reason. "
        "distillation_packet: observation, reasoning, action_target{template_name,why}, expected_outcome, phase_target, confidence_target, gating_notes{expected_gate_result,reasons[]}, phase_tag."
    )
    user_prompt = (
        "Analyze the following tactical state and return STRICT JSON guidance.\n\n"
        f"```json\n{json.dumps(input_data, indent=2, default=str)}\n```\n\n"
        "Return your response as a single JSON object with keys: "
        "phase_decision, anomalies, candidates, selection, distillation_packet. "
        "Include phase_tag:'P34' in both phase_decision and distillation_packet."
    )
    return sys_prompt + "\n\n" + user_prompt


def build_phase_guided_fast_local_prompt(board: Dict, phase: str, role: str,
                                          stagnation: int, templates: List[str]) -> str:
    ports = sorted(str(p) for p in list(board.get("ports", []))[:8])
    services = sorted(str(s) for s in list(board.get("services", []))[:8])
    creds = list(board.get("credentials", []))[:3]
    shells = list(board.get("shells", []))[:3]
    return (
        "/no_think\n"
        f"phase={phase} inferred={phase} stagnation={stagnation}\n"
        f"ports={ports} services={services} creds={creds} shells={shells}\n"
        f"templates={templates[:10]}\n"
        "Should we STAY in current phase or ADVANCE? Suggest 3 candidate templates.\n"
        "Reply ONLY JSON:\n"
        '{"stay_or_advance":"stay|advance","reason":"why",'
        '"candidates":["tmpl1","tmpl2","tmpl3"],"confidence":0.0-1.0}'
    )


def build_smart_mentor_prompt(board: Dict, phase: str, role: str,
                               target: str, templates: List[str],
                               recent: List[str], failed: List[str]) -> str:
    """Build SmartMentor prompt (simplified system + user)."""
    sys = (
        "You are an elite pentester AI MENTOR. "
        "Select the BEST next command from AVAILABLE COMMANDS list. "
        "Output ONLY valid JSON.\n"
        "JSON format:\n"
        '{"intent":"strategic goal","selected_command":"TEMPLATE_NAME_from_list",'
        '"parameters":{"target":"IP","port":"80"},'
        '"reasoning":"detailed WHY - teach the agent to think like you",'
        '"expected_observation":"what success looks like",'
        '"risk":"low|medium|high","confidence":0.8,'
        '"next_phase_hint":"if success: X; if fail: Y",'
        '"candidate_actions":[{"command":"alt_template","why":"reason"}]}\n\n'
        "CRITICAL: selected_command MUST be a template_name from the list, NOT a raw command string."
    )
    
    context_lines = [
        f"TARGET: {target}",
        f"PHASE: {phase}",
        f"ROLE: {role}",
    ]
    if board.get("services"):
        context_lines.append(f"SERVICES: {board['services'][:8]}")
    if board.get("ports"):
        context_lines.append(f"PORTS: {board['ports'][:10]}")
    if board.get("credentials"):
        context_lines.append(f"CREDENTIALS: {board['credentials'][:3]}")
    if board.get("vulns"):
        context_lines.append(f"VULNS: {board['vulns'][:3]}")
    if board.get("shells"):
        context_lines.append(f"SHELLS: {board['shells']}")
    if recent:
        context_lines.append(f"\nRECENT COMMANDS (DO NOT REPEAT):")
        for cmd in recent[-5:]:
            context_lines.append(f"  ❌ {cmd[:60]}")
    
    context_lines.append(f"\nAVAILABLE COMMANDS:")
    for t in templates[:20]:
        context_lines.append(f"  • {t}")
    
    context_lines.append(f"\nSelect the BEST command. JSON only.")
    
    return sys + "\n\n" + "\n".join(context_lines)


def build_coherence_classify_prompt(board: Dict, phase: str, stagnation: int) -> str:
    ports = list(board.get("ports", []))[:10]
    services = list(board.get("services", []))[:10]
    creds = list(board.get("credentials", []))[:3]
    shells = list(board.get("shells", []))[:3]
    return (
        f"Classify the current phase from evidence.\n"
        f"Declared phase: {phase}, Stagnation: {stagnation}\n"
        f"Ports: {ports}\nServices: {services}\nCreds: {creds}\nShells: {shells}\n\n"
        f"Reply ONLY JSON:\n"
        f'{{"phase_guess":"RECON|ENUMERATION|EXPLOITATION|PRIVILEGE_ESCALATION|POST_EXPLOITATION|EXFILTRATION",'
        f'"phase_confidence":0.0-1.0,'
        f'"key_evidence":["evidence1","evidence2"],'
        f'"missing_evidence":["missing1"],'
        f'"next_best_families":["nmap","web","ssh"]}}\n'
        f"No markdown."
    )


def build_coherence_summarize_prompt(board: Dict, phase: str, stagnation: int) -> str:
    ports = list(board.get("ports", []))[:10]
    services = list(board.get("services", []))[:10]
    return (
        f"Summarize the current state as a compact postcard.\n"
        f"Phase: {phase}, Stagnation: {stagnation}\n"
        f"Ports: {ports}\nServices: {services}\n"
        f"Creds: {len(board.get('credentials', []))}\n"
        f"Shells: {len(board.get('shells', []))}\n\n"
        f"Reply ONLY JSON:\n"
        f'{{"postcard":"Phase: X. N ports, M services, stagnation=K.",'
        f'"evidence_counts":{{"ports":5,"services":3,"creds":0,"shells":0,"vulns":0}}}}\n'
        f"No markdown."
    )


def build_coherence_score_prompt(board: Dict, phase: str, recent: List[str]) -> str:
    return (
        f"Score the coherence of the current tactical state.\n"
        f"Phase: {phase}\n"
        f"Ports: {len(board.get('ports', []))}, Services: {len(board.get('services', []))}\n"
        f"Recent commands: {recent[-3:]}\n\n"
        f"Reply ONLY JSON:\n"
        f'{{"coherence_score":0.0-1.0,"novelty_score":0.0-1.0,'
        f'"repeat_risk":0.0-1.0,"confidence_calibration":0.0-1.0}}\n'
        f"No markdown."
    )


# ── Schema Validation ──────────────────────────────────────────────────────

def validate_json_output(text: str, schema_type: str) -> bool:
    """Validate that the output matches the expected schema."""
    schema = SCHEMAS.get(schema_type)
    if not schema:
        return False
    
    output_type = schema.get("output_type")
    
    # Clean text
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    text = re.sub(r",\s*([}\]])", r"\1", text)  # fix trailing commas
    
    if output_type == "single_word":
        return text.strip().lower() in schema.get("valid_outputs", [])
    
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    
    if output_type == "json_array":
        if not isinstance(parsed, list) or len(parsed) == 0:
            return False
        for item in parsed:
            if not isinstance(item, dict):
                return False
            for field in schema.get("required_fields", []):
                if field not in item:
                    return False
        return True
    
    if output_type == "json_object":
        if not isinstance(parsed, dict):
            return False
        for field in schema.get("required_fields", []):
            if field not in parsed:
                return False
        # Check nested schemas
        if "nested_schemas" in schema:
            for key, nested in schema["nested_schemas"].items():
                if key not in parsed:
                    return False
                val = parsed[key]
                if isinstance(val, dict):
                    for req in nested.get("required", []):
                        if req not in val:
                            return False
                    constraints = nested.get("constraints", {})
                    if "phase_tag" in constraints and val.get("phase_tag") != constraints["phase_tag"]:
                        return False
                elif isinstance(val, list) and "item_required" in nested:
                    for item in val:
                        if isinstance(item, dict):
                            for req in nested["item_required"]:
                                if req not in item:
                                    return False
        return True
    
    return False


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from model output."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


# ── Main Generator ──────────────────────────────────────────────────────────

class AriaskaSchemaGenerator:
    """Generate schema-perfect training data using Qwen3-32B teacher."""
    
    def __init__(self, model_path: str = "/workspace/models/qwen3-32b-awq"):
        self.model_path = model_path
        self.llm = None
        self.sampling_params = None
        self.target_ips = ["10.10.10.5", "10.10.10.15", "10.10.10.40", 
                          "10.10.10.79", "10.10.10.98", "10.10.10.117",
                          "192.168.1.10", "10.129.5.115", "10.129.95.185"]
        self.roles = ["offensive", "recon", "defensive", "stealth", "strategic"]
        self.stats = {k: {"generated": 0, "valid": 0, "failed": 0} for k in SCHEMAS}
        
    def init_vllm(self):
        """Initialize vLLM with teacher model."""
        from vllm import LLM, SamplingParams
        
        print(f"Loading teacher model from {self.model_path}...")
        self.llm = LLM(
            model=self.model_path,
            quantization="awq",
            max_model_len=8192,
            gpu_memory_utilization=0.88,
            trust_remote_code=True,
            dtype="float16",
        )
        # Default sampling params - low temp for deterministic schema output
        self.sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=600,
            stop=["```\n", "\n\n\n"],
        )
        print("Teacher model loaded successfully!")
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 600) -> List[str]:
        """Generate a batch of outputs from the teacher."""
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["```\n", "\n\n\n"],
        )
        outputs = self.llm.generate(prompts, params)
        results = []
        for output in outputs:
            text = output.outputs[0].text
            text = strip_think_tags(text)
            results.append(text)
        return results
    
    def generate_with_retry(self, prompt: str, schema_type: str,
                            max_tokens: int = 600, max_retries: int = 2) -> Optional[str]:
        """Generate with validation and retry."""
        from vllm import SamplingParams
        
        for attempt in range(max_retries + 1):
            temp = 0.3 + (attempt * 0.15)  # increase temp on retry
            params = SamplingParams(
                temperature=min(temp, 0.8),
                top_p=0.9,
                max_tokens=max_tokens,
            )
            outputs = self.llm.generate([prompt], params)
            text = strip_think_tags(outputs[0].outputs[0].text)
            
            if validate_json_output(text, schema_type):
                self.stats[schema_type]["valid"] += 1
                return text
            
            if attempt < max_retries:
                # Add retry instruction
                prompt = prompt + "\n\nPREVIOUS OUTPUT WAS INVALID. Output ONLY valid JSON. No markdown."
        
        self.stats[schema_type]["failed"] += 1
        return None
    
    def generate_all(self, output_path: str = "/workspace/data/ariaska_schema_sft.jsonl",
                     examples_per_type: Optional[Dict[str, int]] = None):
        """Generate all training examples across all schema types."""
        
        if examples_per_type is None:
            examples_per_type = {
                "microchain_classify": 1000,
                "microchain_generate": 2500,
                "microchain_score": 1500,
                "microchain_fast_local": 1500,
                "phase_guided": 2000,
                "phase_guided_fast_local": 1500,
                "smart_mentor": 2000,
                "coherence_classify": 800,
                "coherence_summarize": 800,
                "coherence_score": 800,
            }
        
        total = sum(examples_per_type.values())
        print(f"\n=== GENERATING {total} SCHEMA-PERFECT EXAMPLES ===\n")
        
        all_examples = []
        
        for schema_type, count in examples_per_type.items():
            print(f"\n--- {schema_type}: {count} examples ---")
            examples = self._generate_for_type(schema_type, count)
            all_examples.extend(examples)
            
            valid = self.stats[schema_type]["valid"]
            failed = self.stats[schema_type]["failed"]
            rate = valid / (valid + failed) * 100 if (valid + failed) > 0 else 0
            print(f"    Valid: {valid}, Failed: {failed}, Rate: {rate:.1f}%")
        
        # Write to JSONL
        random.shuffle(all_examples)  # mix types for better training
        with open(output_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        print(f"\n=== DONE: {len(all_examples)} examples written to {output_path} ===")
        self._print_stats()
        return all_examples
    
    def _generate_for_type(self, schema_type: str, count: int) -> List[Dict]:
        """Generate examples for a specific schema type — fast batch strategy.
        
        Strategy: Generate 1.5x target in large batches, validate, do ONE
        batch retry of all failures, accept whatever passes. No 1-by-1 retries.
        """
        examples = []
        batch_size = 64  # Larger batches for vLLM throughput
        
        # Generate 1.5x to account for validation failures
        target_gen = int(count * 1.5)
        
        # Determine max tokens per type
        max_tokens_map = {
            "microchain_classify": 30,   # Slightly more room
            "microchain_generate": 600,
            "microchain_score": 200,
            "microchain_fast_local": 200,
            "phase_guided": 1000,
            "phase_guided_fast_local": 200,
            "smart_mentor": 600,
            "coherence_classify": 250,
            "coherence_summarize": 250,
            "coherence_score": 150,
        }
        max_tok = max_tokens_map.get(schema_type, 400)
        
        # Phase 1: Generate all in batches
        all_outputs = []  # (output_text, meta)
        for batch_start in range(0, target_gen, batch_size):
            batch_end = min(batch_start + batch_size, target_gen)
            batch_prompts = []
            batch_meta = []
            
            for i in range(batch_start, batch_end):
                board = _random_discovery_board()
                phase = random.choice(ATTACK_PHASES)
                role = random.choice(self.roles)
                target = random.choice(self.target_ips)
                stagnation = random.randint(0, 25)
                recent = _random_recent_commands(random.randint(2, 8))
                templates = _random_available_templates(phase)
                
                prompt = self._build_prompt(schema_type, board, phase, role,
                                          target, stagnation, recent, templates)
                batch_prompts.append(prompt)
                batch_meta.append({
                    "schema_type": schema_type,
                    "phase": phase,
                    "role": role,
                    "board": board,
                    "prompt": prompt,
                })
            
            try:
                outputs = self.generate_batch(batch_prompts, max_tokens=max_tok)
                all_outputs.extend(zip(outputs, batch_meta))
            except Exception as e:
                print(f"  Batch generation failed: {e}")
                continue
            
            print(f"  Generated: {len(all_outputs)}/{target_gen}", end="\r")
        
        # Phase 2: Validate all outputs
        valid_examples = []
        failed_meta = []
        
        for output, meta in all_outputs:
            self.stats[schema_type]["generated"] += 1
            
            if validate_json_output(output, schema_type):
                self.stats[schema_type]["valid"] += 1
                example = {
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt(schema_type)},
                        {"role": "user", "content": meta["prompt"]},
                        {"role": "assistant", "content": output.strip()},
                    ],
                    "schema_type": schema_type,
                }
                valid_examples.append(example)
            else:
                self.stats[schema_type]["failed"] += 1
                failed_meta.append(meta)
        
        print(f"  Pass 1: {len(valid_examples)} valid / {len(all_outputs)} generated ({len(valid_examples)*100//max(len(all_outputs),1)}%)")
        
        # Phase 3: Batch retry failures if we need more examples
        if len(valid_examples) < count and failed_meta:
            retry_needed = min(len(failed_meta), count - len(valid_examples) + 50)
            retry_prompts = []
            retry_metas = []
            
            for meta in failed_meta[:retry_needed]:
                retry_prompt = meta["prompt"] + "\n\nIMPORTANT: Output ONLY valid JSON. No markdown, no code blocks, no explanation. Just the raw JSON."
                retry_prompts.append(retry_prompt)
                retry_metas.append(meta)
            
            # Batch retry with higher temperature
            for batch_start in range(0, len(retry_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(retry_prompts))
                batch = retry_prompts[batch_start:batch_end]
                batch_m = retry_metas[batch_start:batch_end]
                
                try:
                    from vllm import SamplingParams
                    params = SamplingParams(
                        temperature=0.5,  # Higher temp for retries
                        top_p=0.9,
                        max_tokens=max_tok,
                    )
                    outputs = self.llm.generate(batch, params)
                    
                    for output, meta in zip(outputs, batch_m):
                        text = strip_think_tags(output.outputs[0].text)
                        if validate_json_output(text, schema_type):
                            self.stats[schema_type]["valid"] += 1
                            example = {
                                "messages": [
                                    {"role": "system", "content": self._get_system_prompt(schema_type)},
                                    {"role": "user", "content": meta["prompt"]},
                                    {"role": "assistant", "content": text.strip()},
                                ],
                                "schema_type": schema_type,
                            }
                            valid_examples.append(example)
                except Exception as e:
                    print(f"  Retry batch failed: {e}")
            
            print(f"  Pass 2 (retry): {len(valid_examples)} total valid")
        
        # Take up to count examples
        examples = valid_examples[:count]
        print(f"  Final: {len(examples)} examples (target: {count})")
        return examples
    
    def _build_prompt(self, schema_type: str, board: Dict, phase: str, role: str,
                      target: str, stagnation: int, recent: List[str],
                      templates: List[str]) -> str:
        """Build prompt for a specific schema type."""
        if schema_type == "microchain_classify":
            return build_microchain_classify_prompt(board, recent, phase, role)
        elif schema_type == "microchain_generate":
            situation = random.choice(["recon_gap", "enum_needed", "exploit_ready", 
                                      "privesc_needed", "post_exploit"])
            return build_microchain_generate_prompt(board, phase, situation, role, templates)
        elif schema_type == "microchain_score":
            candidates = [
                {"command": f"nmap -sV {target}", "template_name": templates[0] if templates else "nmap_version_detection"},
                {"command": f"gobuster dir -u http://{target} -w common.txt", "template_name": templates[1] if len(templates) > 1 else "gobuster_dir"},
                {"command": f"nikto -h http://{target}", "template_name": templates[2] if len(templates) > 2 else "nikto_scan"},
            ]
            return build_microchain_score_prompt(candidates, phase, board, recent)
        elif schema_type == "microchain_fast_local":
            return build_microchain_fast_local_prompt(board, phase, role, stagnation, recent, templates)
        elif schema_type == "phase_guided":
            return build_phase_guided_prompt(board, phase, role, stagnation, templates)
        elif schema_type == "phase_guided_fast_local":
            return build_phase_guided_fast_local_prompt(board, phase, role, stagnation, templates)
        elif schema_type == "smart_mentor":
            return build_smart_mentor_prompt(board, phase, role, target, templates, recent, [])
        elif schema_type == "coherence_classify":
            return build_coherence_classify_prompt(board, phase, stagnation)
        elif schema_type == "coherence_summarize":
            return build_coherence_summarize_prompt(board, phase, stagnation)
        elif schema_type == "coherence_score":
            return build_coherence_score_prompt(board, phase, recent)
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")
    
    def _get_system_prompt(self, schema_type: str) -> str:
        """Get the system prompt for a schema type."""
        prompts = {
            "microchain_classify": "You are a tactical situation classifier for an autonomous pentesting system. Reply with ONLY a single classification word.",
            "microchain_generate": "You are a command candidate generator for an autonomous pentesting system. Output ONLY a valid JSON array of command candidates with ALL required fields: command, template_name, reasoning, evidence_used, hypothesis, test, expected_observable, stop_condition, confidence.",
            "microchain_score": "You are a command scoring engine for an autonomous pentesting system. Output ONLY a valid JSON array of score objects with fields: idx, phase_fit, evidence_support, novelty. All scores 0.0-1.0.",
            "microchain_fast_local": "You are a fast command selector for an autonomous pentesting system. Output ONLY a valid JSON object with fields: command, template_name, reasoning, score (0.0-1.0).",
            "phase_guided": (
                "You are PHASE GUIDE for Ariaska_RL pentesting lab. "
                "Output ONLY a raw JSON object. No prose, no markdown fences. "
                'Include "phase_tag":"P34" in phase_decision and distillation_packet. '
                "Keys: phase_decision, anomalies, candidates, selection, distillation_packet."
            ),
            "phase_guided_fast_local": "You are a fast phase advisor for an autonomous pentesting system. Output ONLY a valid JSON object with fields: stay_or_advance (\"stay\" or \"advance\"), reason, candidates (list of template names), confidence (0.0-1.0).",
            "smart_mentor": (
                "You are an elite pentester AI MENTOR. Select the BEST next command from the AVAILABLE COMMANDS list. "
                "CRITICAL: selected_command MUST be a template_name from the list, NOT a raw command string. "
                "Output ONLY valid JSON with fields: intent, selected_command, parameters, reasoning, "
                "expected_observation, risk, confidence, next_phase_hint, candidate_actions."
            ),
            "coherence_classify": "You are a state coherence classifier. Output ONLY valid JSON with fields: phase_guess, phase_confidence, key_evidence, missing_evidence, next_best_families.",
            "coherence_summarize": "You are a state summarizer. Output ONLY valid JSON with fields: postcard (one-line state summary), evidence_counts (dict of category:count).",
            "coherence_score": "You are a coherence scorer. Output ONLY valid JSON with fields: coherence_score, novelty_score, repeat_risk, confidence_calibration. All 0.0-1.0.",
        }
        return prompts.get(schema_type, "Output ONLY valid JSON.")
    
    def _print_stats(self):
        """Print generation statistics."""
        print("\n=== GENERATION STATISTICS ===")
        total_valid = 0
        total_failed = 0
        for schema_type, stats in self.stats.items():
            valid = stats["valid"]
            failed = stats["failed"]
            total = valid + failed
            rate = valid / total * 100 if total > 0 else 0
            total_valid += valid
            total_failed += failed
            print(f"  {schema_type:30s}: {valid:5d} valid / {total:5d} total ({rate:.1f}%)")
        
        overall = total_valid + total_failed
        rate = total_valid / overall * 100 if overall > 0 else 0
        print(f"  {'TOTAL':30s}: {total_valid:5d} valid / {overall:5d} total ({rate:.1f}%)")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Ariaska schema-perfect training data")
    parser.add_argument("--output", default="/workspace/data/ariaska_schema_sft.jsonl")
    parser.add_argument("--model", default=None, help="Path to teacher model (legacy)")
    parser.add_argument("--teacher-model", default="/workspace/models/qwen3-32b-awq",
                       help="Path to 32B teacher model")
    parser.add_argument("--num-examples", type=int, default=12000,
                       help="Approximate total examples to generate")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for example counts")
    args = parser.parse_args()
    
    # --model is legacy alias for --teacher-model
    model_path = args.model or args.teacher_model
    
    gen = AriaskaSchemaGenerator(model_path=model_path)
    gen.init_vllm()
    
    # Scale counts so total ≈ num_examples
    base_counts = {
        "microchain_classify": 1000,
        "microchain_generate": 2500,
        "microchain_score": 1500,
        "microchain_fast_local": 1500,
        "phase_guided": 2000,
        "phase_guided_fast_local": 1500,
        "smart_mentor": 2000,
        "coherence_classify": 800,
        "coherence_summarize": 800,
        "coherence_score": 800,
    }
    base_total = sum(base_counts.values())
    auto_scale = args.num_examples / base_total if base_total > 0 else 1.0
    effective_scale = args.scale * auto_scale
    scaled = {k: max(10, int(v * effective_scale)) for k, v in base_counts.items()}
    
    print(f"Target: ~{args.num_examples} examples (scale={effective_scale:.2f})")
    gen.generate_all(output_path=args.output, examples_per_type=scaled)
