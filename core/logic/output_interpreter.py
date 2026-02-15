"""
Output Interpreter - Modular, extensible command output analysis for ARIASKA RL.
"""

import re
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────
# 📦 Structured Output: ParsedResult
# ─────────────────────────────────────────────
@dataclass
class ParsedResult:
    success: bool = False
    phase: str = "unknown"
    description: str = ""
    risk_score: float = 0.0
    stealth_score: float = 1.0
    artifacts: List[str] = None
    entities: Dict[str, List[str]] = None
    output_excerpt: str = ""
    summary: str = ""
    error: Optional[str] = None

    def as_dict(self):
        return asdict(self)

    def summary_text(self):
        if self.error:
            return f"[red]Parse error:[/red] {self.error}"
        base = f"{self.description or 'No description'}"
        if self.success:
            base += " [green](Success)[/green]"
        if self.artifacts:
            base += f" | Artifacts: {', '.join(self.artifacts)}"
        if self.entities:
            ents = ", ".join(f"{k}: {v}" for k, v in self.entities.items() if v)
            if ents:
                base += f" | Entities: {ents}"
        return base

# ─────────────────────────────────────────────
# 🧩 Modular Parsers (Each tool gets a class)
# ─────────────────────────────────────────────
class BaseParser:
    def parse(self, output: str) -> ParsedResult:
        return ParsedResult(
            description="No parser implemented.",
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
            artifacts=[],
            entities={},
        )

class NmapParser(BaseParser):
    port_re = re.compile(r'(\d+)/(tcp|udp)\s+open\s+(\w+)', re.IGNORECASE)
    version_re = re.compile(r'(service version|service info)', re.IGNORECASE)

    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            phase="recon",
            description="Port scan completed",
            risk_score=0.3,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        ports = self.port_re.findall(output)
        if ports:
            result.artifacts = [f"{p[0]}/{p[1]}:{p[2]}" for p in ports]
            result.success = True
            result.description = f"Discovered {len(ports)} open ports"
        if self.version_re.search(output):
            result.description += " with service version information"
            result.risk_score = 0.5
        return result

class DirScanParser(BaseParser):
    dir_re = re.compile(r'(/[\w\.\-/]+)\s+\(Status:\s+(\d+)\)', re.IGNORECASE)

    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            phase="enumeration",
            description="Directory scan completed",
            risk_score=0.4,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        dirs = self.dir_re.findall(output)
        if dirs:
            result.artifacts = [f"{d[0]} (HTTP {d[1]})" for d in dirs]
            result.success = True
            result.description = f"Discovered {len(dirs)} web paths"
        return result

class BruteForceParser(BaseParser):
    cred_re = re.compile(r'login:\s*(\w+).*?password:\s*(\S+)', re.IGNORECASE | re.DOTALL)

    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            phase="exploit",
            description="Authentication attempt",
            risk_score=0.7,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        if "successful" in output.lower() or "password found" in output.lower() or "success" in output:
            result.success = True
            result.description = "Successful authentication"
            result.risk_score = 0.8
            creds = self.cred_re.findall(output)
            if creds:
                result.artifacts = [f"Credentials: {u}:{p}" for u, p in creds]
        return result

class ExploitToolParser(BaseParser):
    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            phase="exploit",
            description="Exploitation attempt",
            risk_score=0.8,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        if any(s in output.lower() for s in ["shell", "session opened", "vulnerability confirmed", "injection point found"]):
            result.success = True
            result.description = "Successful exploitation"
            result.risk_score = 0.9
            result.artifacts.append("Gained access")
        return result

class PrivescParser(BaseParser):
    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            phase="privesc",
            description="Privilege escalation attempt",
            risk_score=0.8,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        if "uid=0" in output.lower() or "root" in output.lower() or "#" in output:
            result.success = True
            result.description = "Successful privilege escalation"
            result.risk_score = 0.95
            result.artifacts.append("Root access")
        return result

class ExfilParser(BaseParser):
    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            phase="exfiltrate",
            description="Data transfer attempt",
            risk_score=0.9,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        if any(s in output.lower() for s in ["transfer complete", "bytes sent", "connection successful"]) or output.strip() == "":
            result.success = True
            result.description = "Successful data exfiltration"
            result.risk_score = 0.95
            result.artifacts.append("Data transferred")
        return result

class GenericParser(BaseParser):
    def parse(self, output: str) -> ParsedResult:
        result = ParsedResult(
            description="Command executed",
            risk_score=0.4,
            artifacts=[],
            entities={},
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
        )
        if (
            len(output) > 10 and
            "error" not in output.lower() and
            "not found" not in output.lower()
        ):
            result.success = True
        return result

# ─────────────────────────────────────────────
# 🧠 LLM Fallback Parser (Optional)
# ─────────────────────────────────────────────
class LLMParser(BaseParser):
    def parse(self, output: str) -> ParsedResult:
        # Optionally use LLM for ambiguous output parsing
        try:
            from core.gpt_manager import GPTManager
            gpt = GPTManager.get_instance()
            prompt = f"Analyze this command output and summarize: {output[:500]}"
            summary = gpt.gpt_request(prompt, task_type="output_parse")
            return ParsedResult(
                description="LLM summary",
                summary=summary,
                output_excerpt=output[:100] + "..." if len(output) > 100 else output,
                success="success" in summary.lower(),
                artifacts=[],
                entities={},
            )
        except Exception as e:
            return ParsedResult(
                description="LLM parse failed",
                output_excerpt=output[:100] + "..." if len(output) > 100 else output,
                error=str(e),
            )

# ─────────────────────────────────────────────
# 🧠 Main Dispatcher: analyze_output
# ─────────────────────────────────────────────
PARSER_MAP = {
    "nmap": NmapParser(),
    "gobuster": DirScanParser(),
    "ffuf": DirScanParser(),
    "hydra": BruteForceParser(),
    "crackmapexec": BruteForceParser(),
    "sqlmap": ExploitToolParser(),
    "msfconsole": ExploitToolParser(),
    "sudo": PrivescParser(),
    "su": PrivescParser(),
    "chmod": PrivescParser(),
    "scp": ExfilParser(),
    "zip": ExfilParser(),
    "tar": ExfilParser(),
    "nc": ExfilParser(),
    "netcat": ExfilParser(),
}

def analyze_output(command: str, output: str) -> Dict[str, Any]:
    """
    Analyze command output and return structured ParsedResult as dict.
    """
    try:
        if not isinstance(command, str):
            command = str(command)
        if not isinstance(output, str):
            output = str(output)
        cmd_base = command.split()[0].lower() if command.split() else ""
        parser = PARSER_MAP.get(cmd_base, GenericParser())
        result = parser.parse(output)
        # Extract entities from output (IPs, ports, usernames, etc.)
        result.entities = extract_entities(output)
        # Calculate stealth score based on command type and success
        result.stealth_score = calculate_stealth_score(command, result.success)
        # Human-readable summary for CLI/logging
        result.summary = result.summary_text()
        return result.as_dict()
    except Exception as e:
        return ParsedResult(
            description="Parse error",
            output_excerpt=output[:100] + "..." if len(output) > 100 else output,
            error=str(e),
        ).as_dict()

# ─────────────────────────────────────────────
# 🔎 Entity Extraction & Utility Functions
# ─────────────────────────────────────────────
def extract_entities(output: str) -> Dict[str, List[str]]:
    entities = {
        "ips": [],
        "ports": [],
        "usernames": [],
        "services": [],
        "paths": [],
    }
    # Extract IPs
    ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    entities["ips"] = ip_pattern.findall(output)
    # Extract ports
    port_pattern = re.compile(r'(\d{1,5})/(tcp|udp)')
    port_matches = port_pattern.findall(output)
    entities["ports"] = [match[0] for match in port_matches]
    # Extract usernames
    if "login:" in output.lower() or "username:" in output.lower():
        username_pattern = re.compile(r'(login|username):\s*(\w+)', re.IGNORECASE)
        username_matches = username_pattern.findall(output)
        entities["usernames"] = [match[1] for match in username_matches]
    # Extract services
    service_pattern = re.compile(r'(http[s]?|ftp|ssh|telnet|smtp|pop3|imap|rdp|smb|mysql|postgresql)', re.IGNORECASE)
    entities["services"] = service_pattern.findall(output)
    # Extract paths
    path_pattern = re.compile(r'(/[\w/\.-]+)')
    entities["paths"] = path_pattern.findall(output)
    return entities

def calculate_stealth_score(command: str, success: bool) -> float:
    # Simple heuristic: more stealth for non-loud commands and success
    loud_tools = ["nmap", "hydra", "ffuf", "gobuster"]
    base = 1.0
    if any(tool in command for tool in loud_tools):
        base -= 0.3
    if not success:
        base -= 0.2
    return max(0.0, min(1.0, base))

def detect_phase(command: str) -> str:
    # Heuristic phase detection
    if not isinstance(command, str):
        return "unknown"
    cmd = command.lower()
    if any(x in cmd for x in ["nmap", "masscan", "ping"]):
        return "recon"
    if any(x in cmd for x in ["gobuster", "enum4linux", "ffuf"]):
        return "enumeration"
    if any(x in cmd for x in ["hydra", "crackmapexec", "sqlmap", "msfconsole"]):
        return "exploit"
    if any(x in cmd for x in ["sudo", "su", "chmod", "linpeas", "winpeas"]):
        return "privesc"
    if any(x in cmd for x in ["scp", "zip", "tar", "nc", "netcat"]):
        return "exfiltrate"
    return "unknown"

# ─────────────────────────────────────────────
# 🧪 CLI/Debug: Test the interpreter
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_commands = [
        ("nmap -sV 10.10.10.10", """Starting Nmap 7.91 ( https://nmap.org ) 
Nmap scan report for 10.10.10.10
PORT   STATE SERVICE VERSION
22/tcp open  ssh     OpenSSH 7.6p1
80/tcp open  http    Apache httpd 2.4.29"""),
        ("gobuster dir -u http://10.10.10.10 -w wordlist.txt", """/admin (Status: 200)
/login (Status: 200)
/images (Status: 301)"""),
        ("hydra -l admin -P passwords.txt 10.10.10.10 ssh", """[22][ssh] host: 10.10.10.10   login: admin   password: Password123!"""),
        ("sudo su", "root@target:~# id\nuid=0(root) gid=0(root) groups=0(root)"),
        ("scp data.txt user@remote:/tmp/", "Transfer complete. 1024 bytes sent."),
        ("unknowncmd", "Some output that doesn't match any known pattern."),
    ]
    for cmd, out in test_commands:
        result = analyze_output(cmd, out)
        print(f"\nCommand: {cmd}")
        print(f"Analysis: {result['summary']}")
