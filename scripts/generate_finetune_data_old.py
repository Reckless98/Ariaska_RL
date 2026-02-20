#!/usr/bin/env python3
"""
Generate fine-tuning training data for Ariaska's pentesting mentor LLM.

Uses OpenAI to generate high-quality pentesting tactical advisor conversations
that will be used to QLoRA fine-tune a local model (Qwen3-14B).

The fine-tuned model becomes Ariaska's permanent local mentor, replacing
expensive cloud LLM calls with a specialized pentesting advisor.

Budget: ~$5 OpenAI spend
Output: JSONL with ChatML conversations for SFT/QLoRA

Usage:
    cd /home/zer0/Projects/Ariaska_RL
    .venv/bin/python scripts/generate_finetune_data.py
"""

import json
import os
import sys
import time
import hashlib
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ariaska.finetune_datagen")

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "ariaska_mentor_train.jsonl"
STATS_FILE = OUTPUT_DIR / "generation_stats.json"

MODEL = "gpt-4o-mini"  # Cost-efficient: ~$0.15/1M input, $0.60/1M output
MAX_BUDGET_USD = 5.00
COST_PER_1K_INPUT = 0.000150
COST_PER_1K_OUTPUT = 0.000600

# ── System Prompt (what the fine-tuned model will become) ───────────────────
SYSTEM_PROMPT = """You are Ariaska's tactical pentesting advisor — an expert autonomous penetration testing AI assistant.

Your role: Given a penetration test scenario (target state, discoveries, current phase, history), recommend the BEST next action with precise reasoning.

PHASES (in order): RECON → ENUMERATION → EXPLOITATION → PRIVILEGE_ESCALATION → LATERAL_MOVEMENT → POST_EXPLOITATION → EXFILTRATION → CLOSEOUT

RESPONSE FORMAT (always JSON):
{
  "command": "exact command to execute",
  "template_name": "matching CommandTemplate name",
  "reasoning": "2-3 sentences: why this command, what we expect to find",
  "confidence": 0.0-1.0,
  "phase_appropriate": true/false,
  "evidence_basis": ["list of discoveries supporting this choice"],
  "alternatives": [{"command": "alt cmd", "reason": "why alternative"}],
  "risk_level": "low|medium|high",
  "expected_discoveries": ["what new info this might reveal"]
}

RULES:
- Always match command to current phase (don't exploit during RECON)
- Base decisions on EVIDENCE — discovered ports, services, versions
- Prefer targeted attacks over blind fuzzing
- Consider stealth vs speed tradeoff
- Recommend phase transitions when current phase objectives are met
- Include credential reuse when credentials are discovered
- Prioritize known CVEs for discovered service versions"""


# ── Scenario Templates ──────────────────────────────────────────────────────

PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"
]

# Real services and versions for realistic scenarios
SERVICES = {
    "ftp": ["vsftpd 2.3.4", "ProFTPD 1.3.5", "Pure-FTPd 1.0.49", "FileZilla 0.9.60"],
    "ssh": ["OpenSSH 7.2p2", "OpenSSH 8.2p1", "OpenSSH 9.0", "Dropbear 2020.81"],
    "http": ["Apache 2.4.49", "Apache 2.4.50", "nginx 1.18.0", "IIS 10.0", "lighttpd 1.4.55"],
    "https": ["Apache 2.4.49 mod_ssl", "nginx 1.18.0 OpenSSL", "IIS 10.0 TLS"],
    "smb": ["Samba 4.6.2", "Samba 3.0.20", "Windows SMB 3.1.1"],
    "mysql": ["MySQL 5.7.33", "MySQL 8.0.23", "MariaDB 10.5.9"],
    "postgres": ["PostgreSQL 9.6.22", "PostgreSQL 13.3", "PostgreSQL 14.0"],
    "rdp": ["Microsoft Terminal Services", "xrdp 0.9.12"],
    "smtp": ["Postfix 3.5.6", "Exim 4.94", "Sendmail 8.15.2"],
    "dns": ["BIND 9.16.1", "dnsmasq 2.80", "PowerDNS 4.4.0"],
    "snmp": ["SNMPv1", "SNMPv2c community=public", "SNMPv3"],
    "ldap": ["OpenLDAP 2.4.57", "Active Directory"],
    "redis": ["Redis 6.0.9", "Redis 5.0.7"],
    "mongodb": ["MongoDB 4.4.4", "MongoDB 5.0.3"],
    "tomcat": ["Apache Tomcat 9.0.31", "Tomcat 8.5.51", "Tomcat 7.0.79"],
    "jenkins": ["Jenkins 2.289", "Jenkins 2.346.3"],
    "docker": ["Docker API 1.41", "Docker 20.10.7"],
    "vnc": ["VNC 3.3", "TightVNC 1.3.10", "RealVNC 5.3.2"],
    "telnet": ["Linux telnetd", "BusyBox telnetd"],
    "nfs": ["NFS v3", "NFS v4"],
    "irc": ["UnrealIRCd 3.2.8.1", "InspIRCd 3.0"],
    "java_rmi": ["Java RMI 1.8", "Java RMI 11"],
    "webdav": ["Apache mod_dav", "IIS WebDAV"],
    "wordpress": ["WordPress 5.7.2", "WordPress 6.0.1"],
    "phpmyadmin": ["phpMyAdmin 4.8.1", "phpMyAdmin 5.1.0"],
}

COMMON_PORTS = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    80: "http", 110: "pop3", 111: "rpcbind", 135: "msrpc", 139: "netbios",
    143: "imap", 443: "https", 445: "smb", 993: "imaps", 995: "pop3s",
    1099: "java_rmi", 1433: "mssql", 1521: "oracle", 2049: "nfs",
    3306: "mysql", 3389: "rdp", 5432: "postgres", 5900: "vnc",
    5985: "winrm", 6379: "redis", 6667: "irc", 8080: "http",
    8443: "https", 8888: "http", 9200: "elasticsearch", 27017: "mongodb",
}

VULNS = [
    "CVE-2021-41773 (Apache path traversal)",
    "CVE-2021-44228 (Log4Shell)",
    "CVE-2017-0144 (EternalBlue)",
    "CVE-2014-6271 (Shellshock)",
    "CVE-2019-0708 (BlueKeep)",
    "CVE-2021-3156 (sudo Baron Samedit)",
    "CVE-2021-4034 (PwnKit polkit)",
    "CVE-2020-1472 (Zerologon)",
    "CVE-2017-5638 (Apache Struts RCE)",
    "CVE-2019-5736 (Docker runc escape)",
    "CVE-2018-7600 (Drupalgeddon2)",
    "CVE-2015-1635 (MS15-034 HTTP.sys)",
    "CVE-2012-1823 (PHP-CGI RCE)",
    "CVE-2016-3714 (ImageTragick)",
    "CVE-2021-22205 (GitLab RCE)",
    "CVE-2023-44487 (HTTP/2 Rapid Reset)",
    "CVE-2024-3094 (xz backdoor)",
    "MS17-010 (EternalBlue SMB)",
    "MS08-067 (NetAPI)",
    "CVE-2011-2523 (vsftpd 2.3.4 backdoor)",
]

PRIVESC_METHODS = [
    "SUID binary exploitation", "sudo misconfiguration", "kernel exploit",
    "cron job hijacking", "writable /etc/passwd", "capabilities abuse",
    "PATH hijacking", "wildcard injection", "NFS no_root_squash",
    "docker group membership", "lxd group membership", "pkexec CVE-2021-4034",
    "sudo CVE-2021-3156", "writable systemd service", "LD_PRELOAD injection",
]

CTF_TYPES = [
    "HackTheBox easy Linux", "HackTheBox medium Linux", "HackTheBox hard Linux",
    "HackTheBox easy Windows", "HackTheBox medium Windows",
    "TryHackMe beginner", "TryHackMe intermediate",
    "VulnHub boot2root", "OSCP-style lab",
    "Active Directory domain", "Docker escape",
    "Web application CTF", "Binary exploitation CTF",
    "Cloud misconfiguration", "Kubernetes escape",
]

OS_TYPES = ["Linux (Ubuntu 20.04)", "Linux (Debian 11)", "Linux (CentOS 7)",
            "Windows Server 2019", "Windows 10", "FreeBSD 13"]


def generate_scenario() -> Dict[str, Any]:
    """Generate a random realistic pentesting scenario."""
    os_type = random.choice(OS_TYPES)
    is_windows = "Windows" in os_type
    ctf_type = random.choice(CTF_TYPES)

    # Random port/service combo (3-8 open ports)
    num_ports = random.randint(3, 8)
    port_choices = list(COMMON_PORTS.items())
    open_ports = random.sample(port_choices, min(num_ports, len(port_choices)))

    services_discovered = {}
    for port, svc_type in open_ports:
        if svc_type in SERVICES:
            services_discovered[port] = {
                "service": svc_type,
                "version": random.choice(SERVICES[svc_type])
            }
        else:
            services_discovered[port] = {"service": svc_type, "version": "unknown"}

    # Phase-dependent state
    phase = random.choice(PHASES)
    discoveries = []
    has_creds = False
    has_shell = False
    has_privesc = False

    if phase in ("ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
                 "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
        discoveries.append(f"Open ports: {', '.join(str(p) for p, _ in open_ports)}")
        for port, info in services_discovered.items():
            discoveries.append(f"Port {port}: {info['service']} {info['version']}")

    if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                 "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
        # Add some vuln discoveries
        n_vulns = random.randint(0, 2)
        for v in random.sample(VULNS, min(n_vulns, len(VULNS))):
            discoveries.append(f"Potential vulnerability: {v}")
        # Maybe add web findings
        if any(s["service"] in ("http", "https", "tomcat", "wordpress") for s in services_discovered.values()):
            web_findings = random.sample([
                "Web directory found: /admin/ (403 Forbidden)",
                "Web directory found: /backup/ (200 OK, directory listing)",
                "robots.txt reveals: /secret/, /api/v1/",
                "WordPress detected with outdated plugins",
                "Login page at /wp-login.php",
                "phpinfo.php exposed — PHP 7.4.3",
                "File upload form at /upload.php",
                "SQL injection in parameter 'id' at /search?id=1",
                "XSS reflected in /search?q=<script>",
                "Git repository exposed at /.git/",
            ], random.randint(1, 3))
            discoveries.extend(web_findings)

    if phase in ("PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                 "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
        has_creds = True
        cred_user = random.choice(["admin", "www-data", "tomcat", "mysql", "ftp", "user", "developer"])
        discoveries.append(f"Credentials found: {cred_user}:Password123!")
        has_shell = True
        discoveries.append(f"Low-privilege shell as {cred_user} on target")

    if phase in ("LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"):
        has_privesc = True
        discoveries.append(f"Root shell obtained via {random.choice(PRIVESC_METHODS)}")

    # History of recent commands (2-5)
    history_pool = {
        "RECON": ["nmap -sV -sC -p- {target}", "nmap -sU --top-ports 50 {target}",
                   "ping -c 3 {target}", "whois {target}"],
        "ENUMERATION": ["gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt",
                        "nikto -h http://{target}", "enum4linux -a {target}",
                        "smbclient -L //{target} -N", "snmpwalk -v2c -c public {target}"],
        "EXPLOITATION": ["msfconsole -q -x 'use exploit/multi/handler; set LHOST tun0; run'",
                         "python3 exploit.py {target}", "sqlmap -u 'http://{target}/page?id=1' --dump",
                         "hydra -l admin -P /usr/share/wordlists/rockyou.txt {target} ssh"],
        "PRIVILEGE_ESCALATION": ["sudo -l", "find / -perm -4000 2>/dev/null",
                                  "linpeas.sh", "cat /etc/crontab", "getcap -r / 2>/dev/null"],
    }
    hist_phase = phase if phase in history_pool else "RECON"
    history = random.sample(history_pool.get(hist_phase, history_pool["RECON"]),
                           min(random.randint(2, 4), len(history_pool[hist_phase])))

    return {
        "target_ip": f"10.10.{random.randint(10, 250)}.{random.randint(1, 254)}",
        "os": os_type,
        "ctf_type": ctf_type,
        "phase": phase,
        "step": random.randint(1, 50),
        "total_steps_so_far": random.randint(5, 100),
        "discoveries": discoveries,
        "services": {str(k): v for k, v in services_discovered.items()},
        "has_credentials": has_creds,
        "has_shell": has_shell,
        "has_privesc": has_privesc,
        "recent_commands": history,
        "stagnation_steps": random.choices([0, 0, 0, 2, 4, 7, 12], weights=[40, 20, 10, 10, 10, 5, 5])[0],
        "detection_risk": round(random.uniform(0.0, 0.8), 2),
    }


def build_user_prompt(scenario: Dict[str, Any]) -> str:
    """Build the user prompt from a scenario."""
    lines = [
        f"TARGET: {scenario['target_ip']} ({scenario['os']})",
        f"SCENARIO TYPE: {scenario['ctf_type']}",
        f"CURRENT PHASE: {scenario['phase']}",
        f"STEP: {scenario['step']}/{scenario['total_steps_so_far']}",
        f"DETECTION RISK: {scenario['detection_risk']}",
        f"STAGNATION: {scenario['stagnation_steps']} steps without progress",
        "",
        "DISCOVERIES:",
    ]
    for d in scenario["discoveries"]:
        lines.append(f"  - {d}")

    if not scenario["discoveries"]:
        lines.append("  (none yet)")

    lines.append("")
    lines.append("RECENT COMMANDS:")
    for cmd in scenario["recent_commands"]:
        lines.append(f"  > {cmd}")

    lines.append("")
    lines.append("What is the best next action? Provide your tactical recommendation as JSON.")
    return "\n".join(lines)


# ── Additional prompt templates for diversity ────────────────────────────────

def build_phase_transition_prompt(scenario: Dict[str, Any]) -> str:
    """Ask about whether to transition phase."""
    return f"""TARGET: {scenario['target_ip']} ({scenario['os']})
CURRENT PHASE: {scenario['phase']}
STEP: {scenario['step']}/{scenario['total_steps_so_far']}
STAGNATION: {scenario['stagnation_steps']} steps without progress

DISCOVERIES:
{chr(10).join('  - ' + d for d in scenario['discoveries']) or '  (none)'}

Should we stay in {scenario['phase']} or transition to the next phase?
Evaluate the evidence and provide your recommendation as JSON with:
- "stay_or_advance": "stay" or "advance"
- "reasoning": why
- "confidence": 0.0-1.0
- "missing_evidence": what we'd need to be more confident
- "next_action": recommended command regardless of phase decision"""


def build_evidence_eval_prompt(scenario: Dict[str, Any]) -> str:
    """Ask to evaluate evidence for exploitation readiness."""
    return f"""TARGET: {scenario['target_ip']} ({scenario['os']})

EVIDENCE COLLECTED:
{chr(10).join('  - ' + d for d in scenario['discoveries']) or '  (none)'}

SERVICES:
{json.dumps(scenario['services'], indent=2)}

Evaluate whether we have sufficient evidence to attempt exploitation.
Score each piece of evidence and provide JSON:
- "exploitation_readiness": 0.0-1.0
- "strongest_vector": best attack path with reasoning
- "evidence_gaps": what's missing
- "recommended_actions": [list of 3 commands to fill gaps or exploit]
- "risk_assessment": "low"|"medium"|"high" """


def build_stagnation_recovery_prompt(scenario: Dict[str, Any]) -> str:
    """Ask for stagnation recovery strategy."""
    scenario["stagnation_steps"] = random.randint(5, 15)
    return f"""TARGET: {scenario['target_ip']} ({scenario['os']})
CURRENT PHASE: {scenario['phase']}
STAGNATION: {scenario['stagnation_steps']} steps without any new discoveries

DISCOVERIES SO FAR:
{chr(10).join('  - ' + d for d in scenario['discoveries']) or '  (none)'}

COMMANDS ALREADY TRIED:
{chr(10).join('  > ' + c for c in scenario['recent_commands'])}

We are stuck. Provide a recovery strategy as JSON:
- "diagnosis": what's likely going wrong
- "pivot_strategy": new approach
- "commands": [3-5 specific commands to try, different from what failed]
- "reasoning": why these commands will break the stagnation
- "fallback": what to do if these also fail"""


def build_multi_agent_prompt(scenario: Dict[str, Any]) -> str:
    """Ask for multi-agent coordination advice."""
    agents = ["ScoutAgent", "RedAgent", "BlueAgent", "ShadowAgent", "OrionAgent"]
    return f"""TARGET: {scenario['target_ip']} ({scenario['os']})
CURRENT PHASE: {scenario['phase']}
DETECTION RISK: {scenario['detection_risk']}

AVAILABLE AGENTS: {', '.join(agents)}
DISCOVERIES:
{chr(10).join('  - ' + d for d in scenario['discoveries']) or '  (none)'}

Recommend which agents should be active and what each should do.
Provide JSON:
- "primary_agent": which agent should lead
- "agent_tasks": {{"AgentName": "specific task description"}}
- "activation_order": ["Agent1", "Agent2", ...]
- "coordination_notes": any cross-agent dependencies
- "stealth_adjustments": changes if detection risk is high"""


# ── Data Generation ──────────────────────────────────────────────────────────

PROMPT_BUILDERS = [
    (build_user_prompt, 0.40),             # Standard tactical advice
    (build_phase_transition_prompt, 0.15), # Phase decisions
    (build_evidence_eval_prompt, 0.15),    # Evidence evaluation
    (build_stagnation_recovery_prompt, 0.15), # Stagnation recovery
    (build_multi_agent_prompt, 0.15),       # Multi-agent coordination
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def generate_conversation(client: openai.OpenAI, scenario: Dict[str, Any],
                          prompt_builder) -> Optional[Dict[str, Any]]:
    """Generate one training conversation."""
    user_prompt = prompt_builder(scenario)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        assistant_msg = response.choices[0].message.content
        usage = response.usage

        # Build ChatML training sample
        conversation = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_msg},
            ],
            "metadata": {
                "phase": scenario["phase"],
                "ctf_type": scenario["ctf_type"],
                "os": scenario["os"],
                "prompt_type": prompt_builder.__name__,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
        }
        return conversation, usage.prompt_tokens, usage.completion_tokens

    except Exception as e:
        log.error(f"API error: {e}")
        return None


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    # Stats tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    total_samples = 0
    phase_counts = {p: 0 for p in PHASES}
    prompt_type_counts = {}
    dedup_hashes = set()

    # Load existing if resuming
    existing_samples = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    existing_samples.append(sample)
                    # Build dedup hash
                    user_msg = sample["messages"][1]["content"]
                    h = hashlib.md5(user_msg.encode()).hexdigest()
                    dedup_hashes.add(h)
        total_samples = len(existing_samples)
        log.info(f"Resuming: {total_samples} existing samples loaded")

    # Target: generate until budget exhausted
    # gpt-4o-mini: ~$0.15/1M input + $0.60/1M output
    # Average conversation: ~800 input + ~500 output tokens
    # Cost per sample: ~$0.00042
    # $5 budget: ~11,900 samples — but let's be conservative
    TARGET_SAMPLES = 8000
    BATCH_SIZE = 20  # Concurrent-ish generation

    log.info(f"=== Ariaska Mentor Fine-Tune Data Generation ===")
    log.info(f"Model: {MODEL}")
    log.info(f"Budget: ${MAX_BUDGET_USD:.2f}")
    log.info(f"Target: {TARGET_SAMPLES} samples")
    log.info(f"Output: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "a") as out_f:
        while total_samples < TARGET_SAMPLES and total_cost < MAX_BUDGET_USD:
            # Pick prompt type by weight
            r = random.random()
            cumulative = 0.0
            prompt_builder = build_user_prompt
            for builder, weight in PROMPT_BUILDERS:
                cumulative += weight
                if r <= cumulative:
                    prompt_builder = builder
                    break

            scenario = generate_scenario()

            # Dedup check
            user_prompt = prompt_builder(scenario)
            h = hashlib.md5(user_prompt.encode()).hexdigest()
            if h in dedup_hashes:
                continue
            dedup_hashes.add(h)

            result = generate_conversation(client, scenario, prompt_builder)
            if result is None:
                time.sleep(2)
                continue

            conversation, in_tok, out_tok = result
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            cost = (in_tok / 1000) * COST_PER_1K_INPUT + (out_tok / 1000) * COST_PER_1K_OUTPUT
            total_cost += cost
            total_samples += 1

            phase = scenario["phase"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            ptype = prompt_builder.__name__
            prompt_type_counts[ptype] = prompt_type_counts.get(ptype, 0) + 1

            # Write immediately (crash-safe)
            out_f.write(json.dumps(conversation) + "\n")
            out_f.flush()

            if total_samples % 50 == 0:
                log.info(
                    f"[{total_samples}/{TARGET_SAMPLES}] "
                    f"cost=${total_cost:.3f}/${MAX_BUDGET_USD:.2f} "
                    f"tokens={total_input_tokens+total_output_tokens:,} "
                    f"phase={phase} type={ptype}"
                )
                # Save stats
                stats = {
                    "total_samples": total_samples,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_cost_usd": round(total_cost, 4),
                    "budget_remaining": round(MAX_BUDGET_USD - total_cost, 4),
                    "phase_distribution": phase_counts,
                    "prompt_type_distribution": prompt_type_counts,
                    "model": MODEL,
                }
                with open(STATS_FILE, "w") as sf:
                    json.dump(stats, sf, indent=2)

            # Rate limiting — gpt-4o-mini allows high throughput
            # but let's be a bit careful
            if total_samples % 100 == 0:
                time.sleep(1)

    # Final stats
    stats = {
        "total_samples": total_samples,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": round(total_cost, 4),
        "phase_distribution": phase_counts,
        "prompt_type_distribution": prompt_type_counts,
        "model": MODEL,
        "completed": True,
    }
    with open(STATS_FILE, "w") as sf:
        json.dump(stats, sf, indent=2)

    log.info(f"=== COMPLETE ===")
    log.info(f"Total samples: {total_samples}")
    log.info(f"Total cost: ${total_cost:.4f}")
    log.info(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
    log.info(f"Output: {OUTPUT_FILE}")
    log.info(f"Phase distribution: {json.dumps(phase_counts)}")


if __name__ == "__main__":
    main()
