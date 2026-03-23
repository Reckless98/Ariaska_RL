#!/usr/bin/env python3
"""Benchmark ariaska-cybersec2 vs ariaska-cybersec4 across all schema types."""

import json
import time
import requests
import sys
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS = ["ariaska-cybersec2", "ariaska-cybersec4"]

# Test prompts covering all major schema types the models were trained on
TESTS = [
    {
        "name": "microchain_fast_local — command selection (RECON)",
        "schema": "microchain_fast_local",
        "messages": [
            {"role": "system", "content": "You are a fast command selector for Ariaska pentesting system. Output ONLY a JSON object with: command, template_name, reasoning, score (0.0-1.0). No markdown."},
            {"role": "user", "content": """Pick the single best next command.

Target: 10.10.10.40
Phase: RECON
Role: offensive
Stagnation: 2 steps
Discovery board:
{
 "ports": ["80", "443"],
 "services": ["http"],
 "credentials": [],
 "vulns": [],
 "shells": [],
 "users": [],
 "web_paths": ["/"]
}
Available commands: nmap_fast_scan, nmap_service_scan, gobuster_common, nikto_scan, whatweb_scan, curl_headers
Last 3 actions: nmap_fast_scan (found 80,443), curl_headers (got Apache/2.4.18)"""}
        ],
        "expected_keys": ["command", "template_name", "reasoning", "score"],
    },
    {
        "name": "microchain_fast_local — command selection (PRIVESC)",
        "schema": "microchain_fast_local",
        "messages": [
            {"role": "system", "content": "You are a fast command selector for Ariaska pentesting system. Output ONLY a JSON object with: command, template_name, reasoning, score (0.0-1.0). No markdown."},
            {"role": "user", "content": """Pick the single best next command.

Target: 10.10.10.79
Phase: PRIVILEGE_ESCALATION
Role: offensive
Stagnation: 0 steps
Discovery board:
{
 "ports": ["22", "80", "3306"],
 "services": ["ssh", "http", "mysql"],
 "credentials": ["www-data:Summer2023!"],
 "vulns": ["CVE-2021-4034"],
 "shells": ["www-data@target"],
 "users": ["root", "www-data", "admin"],
 "web_paths": ["/", "/wp-admin", "/wp-content/uploads"]
}
Available commands: linpeas_run, sudo_l_check, suid_find, kernel_exploit_suggester, cron_check, writable_dirs
Last 3 actions: reverse_shell (got www-data), id (uid=33), find_suid (found /usr/bin/pkexec)"""}
        ],
        "expected_keys": ["command", "template_name", "reasoning", "score"],
    },
    {
        "name": "smart_mentor — strategic command (stalled ENUM)",
        "schema": "smart_mentor",
        "messages": [
            {"role": "system", "content": "You are an elite pentester AI MENTOR for Ariaska system. Select the BEST next action. Output ONLY valid JSON with: intent, selected_command (MUST be a template_name like nmap_fast_scan), parameters (dict), reasoning (string with deep analysis), confidence (0.0-1.0)."},
            {"role": "user", "content": """Select the best next command. selected_command MUST be a template_name from available.

Target: 10.10.10.3
Phase: ENUMERATION
Role: offensive
Stagnation: 8 steps
Discovery board:
{
 "ports": ["21", "22", "139", "445", "3632"],
 "services": ["ftp", "ssh", "smb", "distcc"],
 "credentials": [],
 "vulns": [],
 "shells": [],
 "users": ["nobody"],
 "web_paths": []
}
Available commands: smbclient_list, enum4linux_scan, ftp_anonymous, distcc_exploit, hydra_ssh, nmap_vuln_scan
Last 3 actions: nmap_service_scan (found distcc 3632), smbclient_list (access denied), enum4linux_scan (found nobody user)

History: 8 enumeration steps with no exploitation progress. FTP allows anonymous but empty. SMB locked. distcc v1 on port 3632 is unusual."""}
        ],
        "expected_keys": ["intent", "selected_command", "parameters", "reasoning", "confidence"],
    },
    {
        "name": "microchain_classify — situation classification",
        "schema": "microchain_classify",
        "messages": [
            {"role": "system", "content": "You are a tactical situation classifier for Ariaska pentesting system. Respond with ONLY one word from: recon_gap, enum_needed, exploit_ready, privesc_needed, post_exploit, lateral_move, stalled"},
            {"role": "user", "content": """Classify the current tactical situation.

Target: 10.10.10.5
Phase: ENUMERATION
Role: offensive
Stagnation: 3 steps
Discovery board:
{
 "ports": ["135", "139", "445"],
 "services": ["smb"],
 "credentials": [],
 "vulns": ["CVE-2017-0144"],
 "shells": [],
 "users": [],
 "web_paths": []
}
Last 3 actions: nmap_vuln_scan (found MS17-010/CVE-2017-0144), smbclient_list (access denied), enum4linux_scan (null session blocked)"""}
        ],
        "expected_keys": None,  # Single word response
    },
    {
        "name": "microchain_generate — multi-command generation",
        "schema": "microchain_generate",
        "messages": [
            {"role": "system", "content": "You are a command generator for Ariaska pentesting system. Output ONLY a JSON array of 1-3 objects, each with these exact keys: command, template_name, reasoning, evidence_used, hypothesis, test, expected_outcome. No markdown."},
            {"role": "user", "content": """Classification: exploit_ready
Generate 1-3 command candidates.

Target: 10.10.10.40
Phase: EXPLOITATION
Role: offensive
Stagnation: 0 steps
Discovery board:
{
 "ports": ["80", "443", "8080"],
 "services": ["http", "https", "http-proxy"],
 "credentials": [],
 "vulns": ["shellshock"],
 "shells": [],
 "users": ["www-data"],
 "web_paths": ["/cgi-bin/", "/cgi-bin/test.cgi"]
}
Available commands: shellshock_exploit, curl_shellshock, reverse_shell_bash, metasploit_shellshock"""}
        ],
        "expected_keys": ["command", "template_name", "reasoning"],
    },
    {
        "name": "coherence_score — scoring coherence",
        "schema": "coherence_score",
        "messages": [
            {"role": "system", "content": "You are a coherence scorer for Ariaska pentesting system. Output ONLY a JSON object with: coherence_score (0.0-1.0), novelty_score (0.0-1.0), repeat_risk (0.0-1.0), confidence_calibration (0.0-1.0). No markdown."},
            {"role": "user", "content": """Score the coherence, novelty, repeat risk, and confidence calibration.

Target: 10.129.5.115
Phase: ENUMERATION
Role: recon
Stagnation: 12 steps
Discovery board:
{
 "ports": ["22", "80"],
 "services": ["ssh", "http"],
 "credentials": [],
 "vulns": [],
 "shells": [],
 "users": [],
 "web_paths": ["/", "/login"]
}
Proposed command: nmap_fast_scan -p- 10.129.5.115
Last 5 actions: nmap_fast_scan, nmap_fast_scan, gobuster_common, nmap_fast_scan, curl_headers"""}
        ],
        "expected_keys": ["coherence_score", "novelty_score", "repeat_risk", "confidence_calibration"],
    },
    {
        "name": "phase_guided — phase transition reasoning",
        "schema": "phase_guided",
        "messages": [
            {"role": "system", "content": "You are a phase advisor for Ariaska pentesting system. Output ONLY valid JSON with: recommended_phase, reasoning (deep analysis of why transition is warranted), confidence (0.0-1.0), key_evidence (list of specific findings supporting the decision)."},
            {"role": "user", "content": """Should we transition phases?

Target: 10.10.10.27
Current phase: RECON
Role: offensive
Steps in phase: 15
Discovery board:
{
 "ports": ["22", "80", "111", "139", "443", "445", "2049", "6697", "8067"],
 "services": ["ssh", "http", "rpcbind", "smb", "https", "nfs", "irc"],
 "credentials": [],
 "vulns": [],
 "shells": [],
 "users": [],
 "web_paths": ["/", "/ircd", "/manual"]
}
We've done 15 recon steps and found 9 ports with diverse services. NFS and IRC are high-value targets."""}
        ],
        "expected_keys": ["recommended_phase", "reasoning", "confidence"],
    },
]


def query_model(model: str, messages: list, timeout: int = 60) -> dict:
    """Query Ollama and return response with timing."""
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512}
        }, timeout=timeout)
        elapsed = time.time() - start
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1)
        tok_s = eval_count / (eval_duration / 1e9) if eval_duration else 0
        return {
            "content": content,
            "tokens": eval_count,
            "tok_s": round(tok_s, 1),
            "elapsed": round(elapsed, 1),
        }
    except Exception as e:
        return {"content": f"ERROR: {e}", "tokens": 0, "tok_s": 0, "elapsed": time.time() - start}


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def score_response(content: str, expected_keys: list | None, schema: str) -> dict:
    """Score response quality."""
    clean = strip_think(content)
    scores = {"valid_json": False, "has_keys": False, "reasoning_depth": "NONE", "reasoning_len": 0}

    # Classification schema — just check single word
    if schema == "microchain_classify":
        valid_classes = {"recon_gap", "enum_needed", "exploit_ready", "privesc_needed", "post_exploit", "lateral_move", "stalled"}
        word = clean.strip().lower()
        scores["valid_json"] = True  # N/A
        scores["has_keys"] = word in valid_classes
        scores["classification"] = word
        return scores

    # Try to parse JSON
    try:
        # Handle arrays vs objects
        parsed = json.loads(clean)
        scores["valid_json"] = True

        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]  # Check first element for key validation

        if expected_keys and isinstance(parsed, dict):
            scores["has_keys"] = all(k in parsed for k in expected_keys)
            # Check reasoning depth
            reasoning = parsed.get("reasoning", "")
            if isinstance(reasoning, str):
                scores["reasoning_len"] = len(reasoning)
                if len(reasoning) > 150:
                    scores["reasoning_depth"] = "DEEP"
                elif len(reasoning) > 60:
                    scores["reasoning_depth"] = "MEDIUM"
                elif len(reasoning) > 0:
                    scores["reasoning_depth"] = "SHALLOW"
    except json.JSONDecodeError:
        # Try to extract JSON from text
        import re
        m = re.search(r'(\{.*\}|\[.*\])', clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                scores["valid_json"] = True
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = parsed[0]
                if expected_keys and isinstance(parsed, dict):
                    scores["has_keys"] = all(k in parsed for k in expected_keys)
                    reasoning = parsed.get("reasoning", "")
                    if isinstance(reasoning, str):
                        scores["reasoning_len"] = len(reasoning)
                        if len(reasoning) > 150:
                            scores["reasoning_depth"] = "DEEP"
                        elif len(reasoning) > 60:
                            scores["reasoning_depth"] = "MEDIUM"
                        elif len(reasoning) > 0:
                            scores["reasoning_depth"] = "SHALLOW"
            except:
                pass

    return scores


def main():
    print("=" * 80)
    print("  ARIASKA MODEL BENCHMARK: cybersec2 vs cybersec4")
    print("  7 prompts × 2 models = 14 inferences")
    print("=" * 80)

    results = {m: [] for m in MODELS}

    for i, test in enumerate(TESTS):
        print(f"\n{'─' * 80}")
        print(f"  Test {i+1}/{len(TESTS)}: {test['name']}")
        print(f"{'─' * 80}")

        for model in MODELS:
            print(f"  → {model}...", end=" ", flush=True)
            resp = query_model(model, test["messages"])
            scores = score_response(resp["content"], test["expected_keys"], test["schema"])

            result = {**resp, **scores, "test": test["name"], "schema": test["schema"]}
            results[model].append(result)

            clean = strip_think(resp["content"])
            status = "✓" if scores["valid_json"] and scores["has_keys"] else "✗"
            depth = scores.get("reasoning_depth", "N/A")
            rlen = scores.get("reasoning_len", 0)

            print(f"{status} {resp['tokens']}tok {resp['tok_s']}t/s {resp['elapsed']}s | json={scores['valid_json']} keys={scores['has_keys']} depth={depth}({rlen}c)")

            # Show truncated response
            preview = clean[:200].replace('\n', ' ')
            print(f"    └─ {preview}{'...' if len(clean)>200 else ''}")

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")

    for model in MODELS:
        r = results[model]
        valid = sum(1 for x in r if x["valid_json"])
        keys = sum(1 for x in r if x["has_keys"])
        deep = sum(1 for x in r if x.get("reasoning_depth") == "DEEP")
        medium = sum(1 for x in r if x.get("reasoning_depth") == "MEDIUM")
        avg_rlen = sum(x.get("reasoning_len", 0) for x in r) / max(1, len([x for x in r if x.get("reasoning_len", 0) > 0]))
        avg_tok = sum(x["tok_s"] for x in r) / len(r)
        avg_tokens = sum(x["tokens"] for x in r) / len(r)

        print(f"\n  {model}:")
        print(f"    Valid JSON:     {valid}/{len(r)}")
        print(f"    Correct keys:   {keys}/{len(r)}")
        print(f"    Deep reasoning: {deep}/{len(r)} (MEDIUM: {medium})")
        print(f"    Avg reasoning:  {avg_rlen:.0f} chars")
        print(f"    Avg speed:      {avg_tok:.1f} tok/s")
        print(f"    Avg tokens:     {avg_tokens:.0f}")

    # Head-to-head
    print(f"\n{'─' * 80}")
    print("  HEAD-TO-HEAD (per test)")
    print(f"{'─' * 80}")

    cs2_wins, cs4_wins, ties = 0, 0, 0
    for i, test in enumerate(TESTS):
        r2 = results[MODELS[0]][i]
        r4 = results[MODELS[1]][i]

        # Score: +2 valid json, +2 correct keys, +1 deep reasoning, +0.5 medium
        def calc_score(r):
            s = 0
            if r["valid_json"]: s += 2
            if r["has_keys"]: s += 2
            if r.get("reasoning_depth") == "DEEP": s += 1
            elif r.get("reasoning_depth") == "MEDIUM": s += 0.5
            return s

        s2, s4 = calc_score(r2), calc_score(r4)
        winner = "cs2" if s2 > s4 else ("cs4" if s4 > s2 else "TIE")
        if winner == "cs2": cs2_wins += 1
        elif winner == "cs4": cs4_wins += 1
        else: ties += 1

        print(f"  {test['name'][:50]:50s}  cs2={s2:.1f}  cs4={s4:.1f}  → {winner}")

    print(f"\n  FINAL: cybersec2={cs2_wins}  cybersec4={cs4_wins}  ties={ties}")

    # Save results
    with open("results/benchmark_cs2_vs_cs4.json", "w") as f:
        json.dump({"tests": TESTS, "results": results, "summary": {
            "cs2_wins": cs2_wins, "cs4_wins": cs4_wins, "ties": ties
        }}, f, indent=2, default=str)
    print(f"\n  Results saved to results/benchmark_cs2_vs_cs4.json")


if __name__ == "__main__":
    main()
