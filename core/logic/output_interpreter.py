"""
Output Interpreter - Analyzes command outputs for the multi-agent system.
"""

import re
import random
from rich.console import Console

console = Console()

def analyze_output(command, output):
    """
    Analyze command output to extract success status, entities, risk score, etc.
    
    Args:
        command (str): The executed command
        output (str): The command output text
        
    Returns:
        dict: Analysis results containing success, entities, risk_score, etc.
    """
    if not isinstance(command, str):
        command = str(command)
    if not isinstance(output, str):
        output = str(output)
    
    # Default values
    result = {
        "success": False,
        "entities": {},
        "risk_score": 0.0,
        "stealth_score": 1.0,
        "phase": detect_phase(command),
        "artifacts": [],
        "output_excerpt": output[:100] + "..." if len(output) > 100 else output,
    }
    
    # Command-specific analyzers
    cmd_base = command.split()[0].lower() if command.split() else ""
    
    if cmd_base == "nmap":
        result.update(analyze_nmap(output))
    elif cmd_base == "gobuster" or cmd_base == "ffuf":
        result.update(analyze_dir_scan(output))
    elif cmd_base == "hydra" or cmd_base == "crackmapexec":
        result.update(analyze_brute_force(output))
    elif cmd_base in ["sqlmap", "msfconsole"]:
        result.update(analyze_exploit_tool(output))
    elif cmd_base in ["sudo", "su", "chmod"]:
        result.update(analyze_privesc(output))
    elif cmd_base in ["scp", "zip", "tar", "nc", "netcat"]:
        result.update(analyze_exfil(output))
    else:
        # Generic analysis
        result.update(analyze_generic(command, output))
    
    # Extract entities from output (IPs, ports, usernames, etc.)
    result["entities"] = extract_entities(output)
    
    # Detect success/failure markers in output
    result["success"] = (
        "success" in output.lower() or 
        "completed" in output.lower() or
        "[+]" in output or 
        "open port" in output.lower()
    ) and not (
        "failed" in output.lower() or
        "error" in output.lower() or
        "no such file" in output.lower()
    )
    
    # Calculate stealth score based on command type and success
    result["stealth_score"] = calculate_stealth_score(command, result["success"])
    
    return result

def detect_phase(command):
    """Detect which phase a command belongs to"""
    cmd_lower = command.lower()
    
    if any(x in cmd_lower for x in ["nmap", "masscan", "ping", "whois", "dig"]):
        return "recon"
    elif any(x in cmd_lower for x in ["gobuster", "ffuf", "enum4linux", "smbclient", "showmount"]):
        return "enumeration"
    elif any(x in cmd_lower for x in ["exploit", "hydra", "sqlmap", "msfconsole", "ssh", "ftp"]):
        return "exploit"
    elif any(x in cmd_lower for x in ["sudo", "su", "chmod", "chown", "linpeas", "winpeas"]):
        return "privesc"
    elif any(x in cmd_lower for x in ["scp", "zip", "tar", "nc", "netcat"]):
        return "exfiltrate"
    else:
        return "unknown"

def extract_entities(output):
    """Extract entities like IPs, ports, usernames, etc. from output"""
    entities = {
        "ips": [],
        "ports": [],
        "usernames": [],
        "services": [],
        "paths": [],
    }
    
    # Extract IPs
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    entities["ips"] = re.findall(ip_pattern, output)
    
    # Extract ports
    port_pattern = r'(\d{1,5})/(tcp|udp)'
    port_matches = re.findall(port_pattern, output)
    entities["ports"] = [match[0] for match in port_matches]
    
    # Extract usernames
    if "login:" in output.lower() or "username:" in output.lower():
        username_pattern = r'(login|username):\s*(\w+)'
        username_matches = re.findall(username_pattern, output, re.IGNORECASE)
        entities["usernames"] = [match[1] for match in username_matches]
    
    # Extract services
    service_pattern = r'(http[s]?|ftp|ssh|telnet|smtp|pop3|imap|rdp|smb|mysql|postgresql)'
    entities["services"] = re.findall(service_pattern, output, re.IGNORECASE)
    
    # Extract paths
    path_pattern = r'(/[\w/\.-]+)'
    entities["paths"] = re.findall(path_pattern, output)
    
    return entities

def calculate_stealth_score(command, success):
    """Calculate a stealth score for the command (0.0-1.0)"""
    cmd_lower = command.lower()
    
    # High-noise commands reduce stealth
    if any(x in cmd_lower for x in ["nmap -T5", "-A", "masscan", "medusa", "--min-rate=1000"]):
        return 0.2 if success else 0.4
    
    # Medium-noise commands
    if any(x in cmd_lower for x in ["nmap", "gobuster", "hydra", "sqlmap", "crackmapexec"]):
        return 0.5 if success else 0.7
    
    # Low-noise commands
    if any(x in cmd_lower for x in ["ping", "smbclient", "showmount", "ssh", "ftp"]):
        return 0.8 if success else 0.9
    
    # Default
    return 0.6

def analyze_nmap(output):
    """Analyze nmap scan output"""
    result = {
        "description": "Port scan completed",
        "risk_score": 0.3,
        "artifacts": [],
    }
    
    # Extract open ports
    port_pattern = r'(\d+)/(tcp|udp)\s+open\s+(\w+)'
    port_matches = re.findall(port_pattern, output)
    
    if port_matches:
        result["artifacts"] = [f"{match[0]}/{match[1]}:{match[2]}" for match in port_matches]
        result["success"] = True
        result["description"] = f"Discovered {len(port_matches)} open ports"
    
    # Check for service version info
    if "service version" in output.lower() or "service info" in output.lower():
        result["description"] += " with service version information"
        result["risk_score"] = 0.5
    
    return result

def analyze_dir_scan(output):
    """Analyze directory scanner output (gobuster, ffuf)"""
    result = {
        "description": "Directory scan completed",
        "risk_score": 0.4,
        "artifacts": [],
    }
    
    # Extract discovered directories
    dir_pattern = r'(/[\w\.\-/]+)\s+\(Status:\s+(\d+)\)'
    dir_matches = re.findall(dir_pattern, output)
    
    if dir_matches:
        result["artifacts"] = [f"{match[0]} (HTTP {match[1]})" for match in dir_matches]
        result["success"] = True
        result["description"] = f"Discovered {len(dir_matches)} web paths"
    
    return result

def analyze_brute_force(output):
    """Analyze brute force tool output (hydra, crackmapexec)"""
    result = {
        "description": "Authentication attempt",
        "risk_score": 0.7,
        "artifacts": [],
    }
    
    # Check for successful login
    if "successful" in output.lower() or "password found" in output.lower() or "SUCCESS" in output:
        result["success"] = True
        result["description"] = "Successful authentication"
        result["risk_score"] = 0.8
        
        # Try to extract credentials
        cred_pattern = r'login:\s*(\w+).*?password:\s*(\S+)'
        cred_matches = re.findall(cred_pattern, output, re.IGNORECASE | re.DOTALL)
        
        if cred_matches:
            result["artifacts"] = [f"Credentials: {match[0]}:{match[1]}" for match in cred_matches]
    
    return result

def analyze_exploit_tool(output):
    """Analyze exploitation tool output"""
    result = {
        "description": "Exploitation attempt",
        "risk_score": 0.8,
        "artifacts": [],
    }
    
    # Check for successful exploitation
    if (
        "shell" in output.lower() or 
        "session opened" in output.lower() or
        "vulnerability confirmed" in output.lower() or
        "injection point found" in output.lower()
    ):
        result["success"] = True
        result["description"] = "Successful exploitation"
        result["risk_score"] = 0.9
        result["artifacts"].append("Gained access")
    
    return result

def analyze_privesc(output):
    """Analyze privilege escalation command output"""
    result = {
        "description": "Privilege escalation attempt",
        "risk_score": 0.8,
        "artifacts": [],
    }
    
    # Check for successful privesc
    if (
        "uid=0" in output.lower() or
        "root" in output.lower() or
        "#" in output
    ):
        result["success"] = True
        result["description"] = "Successful privilege escalation"
        result["risk_score"] = 0.95
        result["artifacts"].append("Root access")
    
    return result

def analyze_exfil(output):
    """Analyze data exfiltration command output"""
    result = {
        "description": "Data transfer attempt",
        "risk_score": 0.9,
        "artifacts": [],
    }
    
    # Check for successful exfil
    if (
        "transfer complete" in output.lower() or
        "bytes sent" in output.lower() or
        "connection successful" in output.lower() or
        output.strip() == ""  # Many exfil commands return empty output when successful
    ):
        result["success"] = True
        result["description"] = "Successful data exfiltration"
        result["risk_score"] = 0.95
        result["artifacts"].append("Data transferred")
    
    return result

def analyze_generic(command, output):
    """Generic analysis for unrecognized commands"""
    result = {
        "description": "Command executed",
        "risk_score": 0.4,
        "artifacts": [],
    }
    
    # Simple heuristic for success detection
    if (
        len(output) > 10 and
        "error" not in output.lower() and
        "not found" not in output.lower()
    ):
        result["success"] = True
    
    # Generate a phase-appropriate description
    phase = detect_phase(command)
    if phase != "unknown":
        result["description"] = f"{phase.capitalize()} command executed"
    
    return result

# Test the interpreter if run directly
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
        ("hydra -l admin -P passwords.txt 10.10.10.10 ssh", """[22][ssh] host: 10.10.10.10   login: admin   password: Password123!""")
    ]
    
    for cmd, out in test_commands:
        result = analyze_output(cmd, out)
        print(f"\nCommand: {cmd}")
        print(f"Analysis: {result}")
