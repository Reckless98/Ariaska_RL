#!/usr/bin/env python3
"""
core/knowledge/htb_analyzer.py — ARIASKA HTB Walkthrough Analyzer v1.0

Uses GPT-5.2 (base) to read PDF and Markdown walkthroughs from
``data/htb_walkthroughs/``, extract structured offensive knowledge,
and feed it into skill cards, kill chains, ChromaDB embeddings,
and the pentesting playbook system.

Extracts:
  - Phase transitions with reasoning (WHY recon → exploit)
  - Command sequences with decision logic (WHY this tool, not that)
  - Failure paths and pivots (what didn't work, human recovery)
  - Tool selection logic and human biases
  - Credentials, CVEs, services discovered
  - Attack chain structure (ordered steps)

Run via CLI:
    python ariaska_cli.py ingest-htb [--pdf-dir ./data/htb_walkthroughs/]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

logger = logging.getLogger("ariaska.htb_analyzer")
console = Console()

# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class HTBStep:
    """A single step extracted from an HTB walkthrough."""

    step_number: int
    phase: str  # recon, enumeration, exploitation, privesc, post_exploitation
    command: str  # The actual command run
    reasoning: str  # WHY this command was chosen
    output_summary: str  # What the output revealed
    alternatives_rejected: List[str] = field(default_factory=list)
    failure_mode: str = ""  # What could go wrong / what DID go wrong
    human_bias: str = ""  # Human thinking pattern observed
    discoveries: List[str] = field(default_factory=list)  # What was found
    tools_used: List[str] = field(default_factory=list)
    credentials_found: List[str] = field(default_factory=list)
    cves_referenced: List[str] = field(default_factory=list)
    next_step_reasoning: str = ""  # WHY the human moved to the next step


@dataclass
class HTBExtraction:
    """Full structured extraction from one HTB walkthrough."""

    box_name: str
    source_file: str
    difficulty: str  # easy, medium, hard
    platform: str  # linux, windows
    extraction_timestamp: float = field(default_factory=time.time)
    # Core data
    steps: List[HTBStep] = field(default_factory=list)
    attack_chain: List[str] = field(default_factory=list)  # Ordered command sequence
    phase_transitions: List[Dict[str, str]] = field(default_factory=list)
    key_decisions: List[Dict[str, str]] = field(default_factory=list)
    failure_paths: List[Dict[str, str]] = field(default_factory=list)
    # Extracted artefacts
    services: List[Dict[str, str]] = field(default_factory=list)
    credentials: List[Dict[str, str]] = field(default_factory=list)
    cves: List[str] = field(default_factory=list)
    # Summary
    summary: str = ""
    key_insight: str = ""  # The single most important lesson
    tool_preference_bias: str = ""  # Human tool over/under-use pattern
    # Metadata
    total_steps: int = 0
    initial_foothold: str = ""
    privesc_method: str = ""
    root_method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HTBExtraction":
        """Deserialize from dict."""
        steps = [HTBStep(**s) for s in data.pop("steps", [])]
        return cls(steps=steps, **data)


# ═══════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION (PDF + MD)
# ═══════════════════════════════════════════════════════════════════════


def _extract_text_from_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.error(f"PDF extraction failed for {filepath}: {e}")
        return ""


def _extract_text_from_md(filepath: str) -> str:
    """Read raw text from a Markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        logger.error(f"MD read failed for {filepath}: {e}")
        return ""


def _guess_box_name(filepath: str) -> str:
    """Guess HTB box name from filename."""
    name = Path(filepath).stem
    # Remove common prefixes
    for prefix in ["HTB", "htb", "HackTheBox", "hackthebox"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Clean up
    name = name.strip("-_ ")
    return name or Path(filepath).stem


def _guess_difficulty(text: str) -> str:
    """Guess box difficulty from walkthrough text."""
    lower = text[:2000].lower()
    if "hard" in lower or "insane" in lower:
        return "hard"
    if "medium" in lower:
        return "medium"
    return "easy"


def _guess_platform(text: str) -> str:
    """Guess OS platform from walkthrough text."""
    lower = text[:3000].lower()
    win_score = sum(1 for w in ["windows", "powershell", "mimikatz", ".exe",
                                 "active directory", "kerberos", "bloodhound"]
                    if w in lower)
    linux_score = sum(1 for w in ["linux", "bash", "sudo", "suid", "/etc/passwd",
                                   "linpeas", "ssh", "cron"]
                      if w in lower)
    return "windows" if win_score > linux_score else "linux"


# ═══════════════════════════════════════════════════════════════════════
# GPT-5.2 ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

# The extraction prompt — asks GPT-5.2 to think like a pentesting instructor
HTB_ANALYSIS_PROMPT = """You are an elite offensive security instructor analyzing a Hack The Box walkthrough.
Your goal is to extract STRUCTURED KNOWLEDGE that can train an AI agent to think like a human pentester.

Extract the following from this walkthrough as a JSON object:

{{
  "box_name": "<box name>",
  "difficulty": "easy|medium|hard",
  "platform": "linux|windows",
  "summary": "<2-3 sentence summary of the full attack path>",
  "key_insight": "<single most important lesson from this box>",
  "initial_foothold": "<how initial access was gained>",
  "privesc_method": "<how privileges were escalated>",
  "root_method": "<how root/admin was achieved>",
  "tool_preference_bias": "<any tool overuse or underuse pattern>",
  "services": [
    {{"port": "<port>", "service": "<service>", "version": "<version if known>"}}
  ],
  "credentials": [
    {{"username": "<user>", "password": "<pass>", "context": "<where found>"}}
  ],
  "cves": ["CVE-XXXX-XXXX"],
  "steps": [
    {{
      "step_number": 1,
      "phase": "recon|enumeration|exploitation|privesc|post_exploitation",
      "command": "<exact command run>",
      "reasoning": "<WHY this command was chosen at this point>",
      "output_summary": "<what the output revealed>",
      "alternatives_rejected": ["<other commands that could have been tried>"],
      "failure_mode": "<what could go wrong or what DID fail>",
      "human_bias": "<human thinking pattern, e.g. 'always starts with nmap -sC -sV'>",
      "discoveries": ["<what was discovered>"],
      "tools_used": ["<tool name>"],
      "credentials_found": ["<any creds found>"],
      "cves_referenced": ["<CVE if applicable>"],
      "next_step_reasoning": "<WHY the human moved to the next step>"
    }}
  ],
  "phase_transitions": [
    {{"from": "recon", "to": "enumeration", "reason": "<why transition happened>"}}
  ],
  "key_decisions": [
    {{"decision": "<what was decided>", "reasoning": "<why>", "alternatives": "<what else could have been done>"}}
  ],
  "failure_paths": [
    {{"attempted": "<what was tried>", "result": "<what happened>", "lesson": "<what was learned>"}}
  ],
  "attack_chain": ["<ordered list of commands that form the successful attack path>"]
}}

RULES:
- Extract EVERY command mentioned, including failed attempts
- For each command, explain WHY a human chose it (reasoning)
- Note phase transitions and explain the decision logic
- Identify human biases (e.g., always running gobuster before nikto)
- Extract ALL credentials, CVEs, services found
- Be specific about failure paths — what didn't work and why
- The "reasoning" field is the MOST IMPORTANT — explain the human thought process
- If the walkthrough doesn't mention something explicitly, infer from context
- Commands should be exact (with flags and arguments)
- Keep output_summary brief but informative

WALKTHROUGH TEXT:
{walkthrough_text}"""

# Maximum characters to send per API call (avoid token overflow)
MAX_TEXT_CHARS = 25000


def _chunk_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> List[str]:
    """Split long text into chunks for multiple API calls."""
    if len(text) <= max_chars:
        return [text]

    # Split at paragraph boundaries
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current)

    return chunks


def analyze_walkthrough(
    text: str,
    box_name: str,
    source_file: str,
    gpt_manager: Any,
) -> Optional[HTBExtraction]:
    """Analyze a single walkthrough using GPT-5.2 base model.

    Args:
        text: Raw walkthrough text (from PDF or MD).
        box_name: Name of the HTB box.
        source_file: Path to the source file.
        gpt_manager: Shared GPTManager instance.

    Returns:
        HTBExtraction with structured knowledge, or None on failure.
    """
    if not text or len(text.strip()) < 100:
        logger.warning(f"Skipping {box_name}: text too short ({len(text)} chars)")
        return None

    # Chunk if needed — for long walkthroughs, send the full text but truncated
    # GPT-5.2 can handle large contexts, so we send as much as possible
    analysis_text = text[:MAX_TEXT_CHARS]
    if len(text) > MAX_TEXT_CHARS:
        # Include beginning and end (end often has privesc/root)
        mid = MAX_TEXT_CHARS // 2
        analysis_text = text[:mid] + "\n\n[... middle section omitted for brevity ...]\n\n" + text[-mid:]

    prompt = HTB_ANALYSIS_PROMPT.format(walkthrough_text=analysis_text)

    try:
        # Use GPT-5.2 BASE (gpt-5.2-2025-12-11) for deep walkthrough analysis
        response = gpt_manager.gpt_request(
            prompt=prompt,
            task_type="walkthrough_analysis",  # Routes to gpt-5.2-2025-12-11
            agent_id="htb_analyzer",
            max_tokens=4000,
            model="gpt-5.2-2025-12-11",  # Base GPT-5.2 dated release for best reasoning
            timeout=120,  # 120s timeout for deep walkthrough analysis
        )

        if not response or len(response) < 50:
            logger.error(f"Empty GPT response for {box_name}")
            return None

        # Parse JSON from response (might be wrapped in markdown code blocks)
        json_str = response.strip()
        if json_str.startswith("```"):
            # Remove markdown code fences
            lines = json_str.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.strip().startswith("```"):
                    json_lines.append(line)
            json_str = "\n".join(json_lines)

        # Try to find JSON object in response
        brace_start = json_str.find("{")
        brace_end = json_str.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start:brace_end]

        data = json.loads(json_str)

        # Build HTBExtraction from parsed data
        steps = []
        for s in data.get("steps", []):
            steps.append(HTBStep(
                step_number=s.get("step_number", 0),
                phase=s.get("phase", "recon"),
                command=s.get("command", ""),
                reasoning=s.get("reasoning", ""),
                output_summary=s.get("output_summary", ""),
                alternatives_rejected=s.get("alternatives_rejected", []),
                failure_mode=s.get("failure_mode", ""),
                human_bias=s.get("human_bias", ""),
                discoveries=s.get("discoveries", []),
                tools_used=s.get("tools_used", []),
                credentials_found=s.get("credentials_found", []),
                cves_referenced=s.get("cves_referenced", []),
                next_step_reasoning=s.get("next_step_reasoning", ""),
            ))

        extraction = HTBExtraction(
            box_name=data.get("box_name", box_name),
            source_file=source_file,
            difficulty=data.get("difficulty", _guess_difficulty(text)),
            platform=data.get("platform", _guess_platform(text)),
            extraction_timestamp=time.time(),
            steps=steps,
            attack_chain=data.get("attack_chain", []),
            phase_transitions=data.get("phase_transitions", []),
            key_decisions=data.get("key_decisions", []),
            failure_paths=data.get("failure_paths", []),
            services=data.get("services", []),
            credentials=data.get("credentials", []),
            cves=data.get("cves", []),
            summary=data.get("summary", ""),
            key_insight=data.get("key_insight", ""),
            tool_preference_bias=data.get("tool_preference_bias", ""),
            total_steps=len(steps),
            initial_foothold=data.get("initial_foothold", ""),
            privesc_method=data.get("privesc_method", ""),
            root_method=data.get("root_method", ""),
        )

        logger.info(
            f"✅ Analyzed {box_name}: {len(steps)} steps, "
            f"{len(extraction.credentials)} creds, {len(extraction.cves)} CVEs"
        )
        return extraction

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed for {box_name}: {e}")
        # Try to salvage partial data
        return _build_fallback_extraction(text, box_name, source_file)
    except Exception as e:
        logger.error(f"GPT analysis failed for {box_name}: {e}")
        return _build_fallback_extraction(text, box_name, source_file)


def _build_fallback_extraction(
    text: str, box_name: str, source_file: str
) -> HTBExtraction:
    """Build a basic extraction from text when GPT analysis fails.

    Uses regex patterns to extract commands, IPs, ports, and credentials
    from the raw walkthrough text. Less accurate but always produces output.
    """
    steps: List[HTBStep] = []
    step_num = 0

    # Regex for common command patterns
    cmd_patterns = [
        r"(?:^|\n)\s*(?:\$|#|└──|┌──)\s*(.+?)(?:\n|$)",
        r"```(?:bash|sh|shell|console)?\n(.*?)```",
        r"(?:^|\n)((?:nmap|gobuster|ffuf|hydra|sqlmap|curl|wget|nikto|"
        r"searchsploit|msfconsole|ssh|ftp|smbclient|enum4linux|"
        r"crackmapexec|evil-winrm|linpeas|winpeas|sudo|find|cat|"
        r"python|nc|netcat|burp|wpscan|dirsearch|feroxbuster|john|"
        r"hashcat)\s+.+?)(?:\n|$)",
    ]

    seen_commands = set()
    for pattern in cmd_patterns:
        for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
            cmd = match.group(1).strip()
            if len(cmd) > 200 or len(cmd) < 5:
                continue
            if cmd in seen_commands:
                continue
            seen_commands.add(cmd)
            step_num += 1

            # Guess phase from command
            phase = "recon"
            tool = cmd.split()[0].lower() if cmd.split() else ""
            if tool in ("nmap", "masscan", "ping"):
                phase = "recon"
            elif tool in ("gobuster", "ffuf", "dirb", "nikto", "wpscan",
                         "enum4linux", "smbclient", "showmount"):
                phase = "enumeration"
            elif tool in ("hydra", "sqlmap", "msfconsole", "searchsploit",
                         "curl", "wget", "nc", "netcat", "python"):
                phase = "exploitation"
            elif tool in ("sudo", "find", "linpeas", "winpeas"):
                phase = "privesc"

            steps.append(HTBStep(
                step_number=step_num,
                phase=phase,
                command=cmd,
                reasoning="Extracted from walkthrough text (fallback mode)",
                output_summary="",
                tools_used=[tool] if tool else [],
            ))

    # Extract credentials via regex
    cred_patterns = [
        r"(?:username|user|login)[\s:=]+['\"]?(\w+)['\"]?\s*(?:password|pass|pwd)[\s:=]+['\"]?(\S+)['\"]?",
        r"(\w+):(\S+)@",
    ]
    credentials = []
    for pattern in cred_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            credentials.append({
                "username": match.group(1),
                "password": match.group(2),
                "context": "regex extraction",
            })

    # Extract CVEs
    cves = list(set(re.findall(r"CVE-\d{4}-\d{4,7}", text)))

    return HTBExtraction(
        box_name=box_name,
        source_file=source_file,
        difficulty=_guess_difficulty(text),
        platform=_guess_platform(text),
        extraction_timestamp=time.time(),
        steps=steps,
        attack_chain=[s.command for s in steps],
        credentials=credentials,
        cves=cves,
        summary=f"Fallback extraction from {box_name}: {len(steps)} commands found",
        key_insight="Fallback extraction — manual review recommended",
        total_steps=len(steps),
    )


# ═══════════════════════════════════════════════════════════════════════
# KNOWLEDGE INJECTION
# ═══════════════════════════════════════════════════════════════════════


def extraction_to_skill_cards(extraction: HTBExtraction) -> List[Dict[str, Any]]:
    """Convert an HTBExtraction into skill cards for the SkillLibrary.

    Each step with meaningful reasoning becomes a skill card with
    if_condition (when to apply) and then_action (what to do).
    """
    cards: List[Dict[str, Any]] = []

    for step in extraction.steps:
        if not step.reasoning or not step.command:
            continue
        if step.reasoning.startswith("Extracted from walkthrough"):
            continue  # Skip fallback-mode steps

        # Build condition from phase + discoveries + context
        condition_parts = [f"phase={step.phase}"]
        for disc in step.discoveries:
            condition_parts.append(disc)
        if step.cves_referenced:
            condition_parts.extend(step.cves_referenced)

        card_id = f"htb_{extraction.box_name.lower()}_{step.step_number}"
        confidence = 0.80  # HTB walkthroughs are reliable

        # Boost confidence for steps with explicit reasoning
        if len(step.reasoning) > 50:
            confidence = 0.85
        if step.credentials_found:
            confidence = 0.90

        cards.append({
            "id": card_id,
            "if_condition": f"HTB-{extraction.box_name}: {'; '.join(condition_parts)}",
            "then_action": f"{step.command} — {step.reasoning[:100]}",
            "confidence": confidence,
            "source": f"htb_walkthrough:{extraction.source_file}",
            "box_name": extraction.box_name,
            "phase": step.phase,
        })

    # Add a chain-level skill card
    if extraction.initial_foothold:
        cards.append({
            "id": f"htb_{extraction.box_name.lower()}_chain",
            "if_condition": (
                f"HTB-{extraction.box_name} attack chain: "
                f"platform={extraction.platform}, "
                f"foothold={extraction.initial_foothold[:50]}"
            ),
            "then_action": (
                f"Full chain: {extraction.initial_foothold} → "
                f"{extraction.privesc_method} → {extraction.root_method}"
            ),
            "confidence": 0.85,
            "source": f"htb_walkthrough:{extraction.source_file}",
            "box_name": extraction.box_name,
            "phase": "exploitation",
        })

    # Add failure-path skill cards (learn from mistakes)
    for fp in extraction.failure_paths:
        cards.append({
            "id": f"htb_{extraction.box_name.lower()}_fail_{hashlib.md5(fp.get('attempted', '').encode()).hexdigest()[:6]}",
            "if_condition": f"Considering: {fp.get('attempted', 'unknown')}",
            "then_action": f"CAUTION: {fp.get('lesson', 'This approach failed')}",
            "confidence": 0.75,
            "source": f"htb_walkthrough:{extraction.source_file}",
            "box_name": extraction.box_name,
            "phase": "exploitation",
        })

    return cards


def extraction_to_chromadb_docs(extraction: HTBExtraction) -> List[Dict[str, Any]]:
    """Convert extraction into documents for ChromaDB embedding.

    Creates rich text documents from each step's reasoning, decisions,
    and failure paths for semantic retrieval by SmartMentor RAG.
    """
    docs: List[Dict[str, Any]] = []

    # Step-level documents
    for step in extraction.steps:
        if not step.reasoning:
            continue

        text = (
            f"[HTB {extraction.box_name}] Phase: {step.phase}. "
            f"Command: {step.command}. "
            f"Reasoning: {step.reasoning}. "
        )
        if step.output_summary:
            text += f"Output: {step.output_summary}. "
        if step.next_step_reasoning:
            text += f"Next step: {step.next_step_reasoning}. "
        if step.failure_mode:
            text += f"Failure mode: {step.failure_mode}. "

        docs.append({
            "text": text,
            "metadata": {
                "source": f"htb:{extraction.box_name}",
                "phase": step.phase,
                "box": extraction.box_name,
                "difficulty": extraction.difficulty,
                "platform": extraction.platform,
                "type": "step_reasoning",
            },
        })

    # Decision-level documents
    for dec in extraction.key_decisions:
        text = (
            f"[HTB {extraction.box_name}] Decision: {dec.get('decision', '')}. "
            f"Reasoning: {dec.get('reasoning', '')}. "
            f"Alternatives: {dec.get('alternatives', '')}."
        )
        docs.append({
            "text": text,
            "metadata": {
                "source": f"htb:{extraction.box_name}",
                "box": extraction.box_name,
                "type": "decision_logic",
            },
        })

    # Failure-path documents
    for fp in extraction.failure_paths:
        text = (
            f"[HTB {extraction.box_name}] Failed attempt: {fp.get('attempted', '')}. "
            f"Result: {fp.get('result', '')}. "
            f"Lesson: {fp.get('lesson', '')}."
        )
        docs.append({
            "text": text,
            "metadata": {
                "source": f"htb:{extraction.box_name}",
                "box": extraction.box_name,
                "type": "failure_path",
            },
        })

    # Summary document
    if extraction.summary:
        docs.append({
            "text": (
                f"[HTB {extraction.box_name}] Summary: {extraction.summary}. "
                f"Key insight: {extraction.key_insight}. "
                f"Foothold: {extraction.initial_foothold}. "
                f"Privesc: {extraction.privesc_method}. "
                f"Root: {extraction.root_method}."
            ),
            "metadata": {
                "source": f"htb:{extraction.box_name}",
                "box": extraction.box_name,
                "type": "box_summary",
                "difficulty": extraction.difficulty,
                "platform": extraction.platform,
            },
        })

    return docs


def _inject_into_chromadb(docs: List[Dict[str, Any]]) -> int:
    """Inject documents into ChromaDB for RAG retrieval.

    Returns number of documents successfully embedded.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path="./chroma_storage")
        collection = client.get_or_create_collection("ariaska_kb")

        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        ids = [
            hashlib.md5(t.encode()).hexdigest()
            for t in texts
        ]

        # Embed in batches of 32
        embedded = 0
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            vectors = embedder.encode(batch_texts).tolist()
            collection.upsert(
                documents=batch_texts,
                embeddings=vectors,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            embedded += len(batch_texts)

        logger.info(f"ChromaDB: embedded {embedded} HTB documents")
        return embedded

    except Exception as e:
        logger.warning(f"ChromaDB injection failed (non-fatal): {e}")
        return 0


def _inject_into_skill_library(
    cards: List[Dict[str, Any]],
) -> int:
    """Inject skill cards into the SkillLibrary.

    Returns number of skills successfully promoted.
    """
    try:
        from core.postmortem.orion_postmortem import SkillCard
        from core.postmortem.skill_library import SkillLibrary

        library = SkillLibrary()
        promoted = 0

        for card in cards:
            skill = SkillCard(
                id=card["id"],
                if_condition=card["if_condition"],
                then_action=card["then_action"],
                confidence=card["confidence"],
            )
            if library.promote(skill, reason=f"HTB walkthrough: {card.get('box_name', 'unknown')}"):
                promoted += 1

        logger.info(f"SkillLibrary: promoted {promoted} HTB skill cards")
        return promoted

    except Exception as e:
        logger.warning(f"Skill library injection failed (non-fatal): {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════
# MAIN INGESTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def _get_cache_path(source_file: str) -> str:
    """Get cache path for a processed walkthrough."""
    name = Path(source_file).stem
    return os.path.join("data", "htb_extractions", f"{name}.json")


def _is_cached(source_file: str) -> bool:
    """Check if a walkthrough has already been processed."""
    cache_path = _get_cache_path(source_file)
    if not os.path.exists(cache_path):
        return False
    # Check if source file is newer than cache
    try:
        source_mtime = os.path.getmtime(source_file)
        cache_mtime = os.path.getmtime(cache_path)
        return cache_mtime >= source_mtime
    except OSError:
        return False


def _load_cached(source_file: str) -> Optional[HTBExtraction]:
    """Load a cached extraction."""
    cache_path = _get_cache_path(source_file)
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        return HTBExtraction.from_dict(data)
    except Exception as e:
        logger.warning(f"Cache load failed for {source_file}: {e}")
        return None


def _save_cache(extraction: HTBExtraction):
    """Save extraction to cache."""
    cache_path = _get_cache_path(extraction.source_file)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(extraction.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def ingest_htb_walkthroughs(
    walkthrough_dir: str = "data/htb_walkthroughs",
    gpt_manager: Any = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Main ingestion pipeline — scans directory, analyzes, and injects.

    Args:
        walkthrough_dir: Path to directory containing PDF and MD files.
        gpt_manager: Shared GPTManager instance. Created if None.
        force: If True, re-analyze even if cached.

    Returns:
        Summary dict with counts and results.
    """
    walkthrough_path = Path(walkthrough_dir)
    if not walkthrough_path.exists():
        console.print(f"[red]❌ Directory not found: {walkthrough_dir}[/red]")
        return {"error": f"Directory not found: {walkthrough_dir}"}

    # Find all PDF and MD files
    files = []
    for ext in ("*.pdf", "*.md"):
        files.extend(walkthrough_path.glob(ext))

    # Filter out README.md
    files = [f for f in files if f.name.lower() != "readme.md"]

    if not files:
        console.print(f"[yellow]⚠ No PDF or MD files found in {walkthrough_dir}[/yellow]")
        return {"error": "No files found"}

    console.print(Panel(
        f"[bold cyan]HTB Walkthrough Ingestion Pipeline[/bold cyan]\n"
        f"Directory: {walkthrough_dir}\n"
        f"Files found: {len(files)} (PDF + MD)\n"
        f"Model: gpt-5.2-2025-12-11 (base) for analysis\n"
        f"Force re-analyze: {force}",
        title="🔬 HTB Analyzer",
    ))

    # Create GPTManager if not provided
    if gpt_manager is None:
        from core.gpt_manager import GPTManager
        gpt_manager = GPTManager()

    results = {
        "total_files": len(files),
        "analyzed": 0,
        "cached": 0,
        "failed": 0,
        "skills_promoted": 0,
        "chromadb_docs": 0,
        "extractions": [],
    }

    all_skill_cards: List[Dict[str, Any]] = []
    all_chromadb_docs: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing walkthroughs...", total=len(files))

        for filepath in sorted(files):
            fname = filepath.name
            box_name = _guess_box_name(str(filepath))

            # Check cache
            if not force and _is_cached(str(filepath)):
                progress.update(task, description=f"[dim]Cached: {fname}[/dim]")
                extraction = _load_cached(str(filepath))
                if extraction:
                    results["cached"] += 1
                    # Still generate skill cards and ChromaDB docs from cache
                    all_skill_cards.extend(extraction_to_skill_cards(extraction))
                    all_chromadb_docs.extend(extraction_to_chromadb_docs(extraction))
                    results["extractions"].append({
                        "box": box_name,
                        "file": fname,
                        "status": "cached",
                        "steps": extraction.total_steps,
                    })
                    progress.advance(task)
                    continue

            progress.update(task, description=f"[cyan]Analyzing: {fname}[/cyan]")

            # Extract text
            if filepath.suffix.lower() == ".pdf":
                text = _extract_text_from_pdf(str(filepath))
            else:
                text = _extract_text_from_md(str(filepath))

            if not text or len(text.strip()) < 100:
                logger.warning(f"Skipping {fname}: insufficient text")
                results["failed"] += 1
                progress.advance(task)
                continue

            # Analyze with GPT-5.2
            extraction = analyze_walkthrough(
                text=text,
                box_name=box_name,
                source_file=str(filepath),
                gpt_manager=gpt_manager,
            )

            if extraction:
                _save_cache(extraction)
                results["analyzed"] += 1

                # Generate skill cards and ChromaDB docs
                skill_cards = extraction_to_skill_cards(extraction)
                chromadb_docs = extraction_to_chromadb_docs(extraction)

                all_skill_cards.extend(skill_cards)
                all_chromadb_docs.extend(chromadb_docs)

                results["extractions"].append({
                    "box": box_name,
                    "file": fname,
                    "status": "analyzed",
                    "steps": extraction.total_steps,
                    "skills": len(skill_cards),
                    "creds": len(extraction.credentials),
                    "cves": len(extraction.cves),
                })
            else:
                results["failed"] += 1
                results["extractions"].append({
                    "box": box_name,
                    "file": fname,
                    "status": "failed",
                })

            progress.advance(task)

    # ── Inject into subsystems ──────────────────────────────────────────
    console.print("\n[bold]Injecting knowledge into subsystems...[/bold]")

    if all_skill_cards:
        results["skills_promoted"] = _inject_into_skill_library(all_skill_cards)
        console.print(f"  ✅ SkillLibrary: {results['skills_promoted']} cards promoted")

    if all_chromadb_docs:
        results["chromadb_docs"] = _inject_into_chromadb(all_chromadb_docs)
        console.print(f"  ✅ ChromaDB: {results['chromadb_docs']} documents embedded")

    # ── Summary table ───────────────────────────────────────────────────
    table = Table(title="HTB Ingestion Results", show_header=True)
    table.add_column("Box", style="cyan")
    table.add_column("File", style="dim")
    table.add_column("Status", style="bold")
    table.add_column("Steps", justify="right")
    table.add_column("Skills", justify="right")

    for ext in results["extractions"]:
        status_style = {
            "analyzed": "[green]✅ Analyzed[/green]",
            "cached": "[blue]📦 Cached[/blue]",
            "failed": "[red]❌ Failed[/red]",
        }.get(ext.get("status", ""), ext.get("status", ""))

        table.add_row(
            ext.get("box", "?"),
            ext.get("file", "?"),
            status_style,
            str(ext.get("steps", "-")),
            str(ext.get("skills", "-")),
        )

    console.print(table)

    console.print(Panel(
        f"[bold green]Ingestion Complete[/bold green]\n"
        f"Analyzed: {results['analyzed']} | Cached: {results['cached']} | Failed: {results['failed']}\n"
        f"Skill Cards: {results['skills_promoted']} | ChromaDB Docs: {results['chromadb_docs']}",
        title="📊 Summary",
    ))

    return results
