import os
import json
import threading
import re
import hashlib
import time
from collections import deque, defaultdict, Counter
from datetime import datetime
from rich.console import Console
from difflib import SequenceMatcher

console = Console()

"""
RedAgentBrain — Episodic Memory | Command Deduplication | Adaptive Strategy Generation
------------------------------------------------------------
• Logs every RedAgent step with full context
• Prevents redundant command execution via similarity detection
• Provides action diversity metrics and strategy evolution
• Enables enhanced learning through episodic memory analysis
• Detects repetitive patterns and encourages exploration
"""

class RedAgentBrain:
    """
    Enhanced episodic memory and action optimization for RedAgent.
    
    Features:
        - Command similarity detection to prevent redundancy
        - Action diversity metrics for optimization
        - GPT feedback loop for continual improvement
        - Pattern recognition for strategic evolution
        
    Methods:
        log_step: Log a single agent step with extensive metadata
        check_redundancy: Detect if a command is redundant based on history
        get_action_diversity: Calculate diversity metrics of recent actions
        get_strategy_advice: Generate strategic advice based on history
        log_gpt_feedback: Log GPT meta-analysis/feedback
    """
    def __init__(self, log_dir="logs/redagent_evolution", max_steps=200, similarity_threshold=0.85):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.step_log_path = os.path.join(self.log_dir, "steps.jsonl")
        self.gpt_log_path = os.path.join(self.log_dir, "gpt_feedback.jsonl")
        self.command_hash_path = os.path.join(self.log_dir, "command_hashes.json")
        
        # In-memory storage
        self.steps = deque(maxlen=max_steps)
        self.gpt_feedback = deque(maxlen=20)
        self.command_history = deque(maxlen=100)  # Recent commands
        self.command_counts = Counter()           # Command frequency counter
        self.command_hashes = set()               # Exact command hashes
        self.command_results = {}                 # Command -> result mapping
        self.phase_history = defaultdict(list)    # Phase -> commands mapping
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.redundancy_count = 0
        self.last_reward = 0
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
        
        # Load existing data
        self._load_from_disk()

    def _normalize_command(self, command):
        """
        Normalize commands for better comparison (remove IPs, timestamps, etc.)
        Enhanced version that catches more patterns and variations.
        """
        if not command:
            return ""
            
        # Convert to string if not already
        command = str(command).strip()
        
        # Case normalization
        command = command.lower()
        
        # Replace IP addresses with placeholder
        command = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP', command)
        
        # Replace port ranges
        command = re.sub(r'\b\d+-\d+\b', 'RANGE', command)
        
        # Replace timestamps/dates with placeholder
        command = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', 'DATE', command)
        command = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', 'TIME', command)
        
        # Replace output filenames (common in recon)
        command = re.sub(r'(-o|-output) *[A-Za-z0-9._/-]+\b', ' -oFILE ', command)
        
        # Replace port numbers
        command = re.sub(r'port \d+', 'port NUM', command)
        command = re.sub(r':\d+\b', ':NUM', command)
        
        # Replace common NMAP parameters
        command = re.sub(r'-[pPsS][A-Z0-9,]+\b', '-pNUM', command)
        
        # Replace common file paths
        command = re.sub(r'(/[a-zA-Z0-9._/-]+)\b', 'PATH', command)
        
        # Replace script arguments
        command = re.sub(r'--[a-z]+=[\w.]+', '--arg=VALUE', command)
        
        # Clean up excessive whitespace
        command = re.sub(r'\s+', ' ', command).strip()
        
        return command

    def _command_similarity(self, cmd1, cmd2):
        """Calculate similarity between two commands using SequenceMatcher"""
        if not cmd1 or not cmd2:
            return 0.0
            
        # Normalize commands
        norm_cmd1 = self._normalize_command(cmd1)
        norm_cmd2 = self._normalize_command(cmd2)
        
        # Use SequenceMatcher for string similarity
        return SequenceMatcher(None, norm_cmd1, norm_cmd2).ratio()

    def _get_command_hash(self, command):
        """Get hash of command for exact duplicate detection"""
        normalized = self._normalize_command(command)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def check_redundancy(self, command, phase=None):
        """
        Enhanced redundancy detection with smarter semantic analysis.
        
        Args:
            command (str): Command to check
            phase (str, optional): Current phase (recon, exploit, etc.)
            
        Returns:
            tuple: (is_redundant, reason, similar_command)
        """
        with self.lock:
            if not command:
                return True, "Empty command", None
            
            try:
                # Skip some commands that are always unique
                if any(unique_cmd in command.lower() for unique_cmd in [
                    "help", "--help", "clear", "history", "exit"
                ]):
                    return False, None, None
                    
                # Check for exact duplicates first (fastest check)
                cmd_hash = self._get_command_hash(command)
                if cmd_hash in self.command_hashes:
                    self.redundancy_count += 1
                    return True, "Exact duplicate command", command
                    
                # Advanced: Extract command type for smarter comparison
                cmd_type = command.split()[0] if ' ' in command else command
                
                # Check similarity with recent commands
                normalized = self._normalize_command(command)
                
                # First check the most recent 5 commands (with higher threshold)
                for recent_cmd in list(self.command_history)[-5:]:
                    similarity = self._command_similarity(command, recent_cmd)
                    if similarity >= self.similarity_threshold:
                        self.redundancy_count += 1
                        return True, f"Very similar to recent command (similarity: {similarity:.2f})", recent_cmd
                
                # Then check if this command type was already heavily used
                cmd_type_count = sum(1 for c in self.command_history if c.startswith(cmd_type))
                if cmd_type_count >= 10:
                    # For repeated command types (like nmap), use more specific checks
                    for recent_cmd in self.command_history:
                        if recent_cmd.startswith(cmd_type):
                            similarity = self._command_similarity(command, recent_cmd)
                            if similarity >= self.similarity_threshold - 0.05:  # Slightly lower threshold
                                self.redundancy_count += 1
                                return True, f"Command type '{cmd_type}' redundant (similarity: {similarity:.2f})", recent_cmd
                
                # Phase-specific redundancy (e.g., too many recon commands)
                if phase and phase in self.phase_history:
                    phase_commands = self.phase_history[phase]
                    if len(phase_commands) >= 10:  # Too many commands in same phase
                        # Check phase similarity
                        for phase_cmd in phase_commands[-5:]:  # Check last 5 commands in phase
                            similarity = self._command_similarity(command, phase_cmd)
                            if similarity >= self.similarity_threshold - 0.1:  # Lower threshold for phase
                                self.redundancy_count += 1
                                return True, f"Phase redundancy in {phase}", phase_cmd
                
                # Special handling for commands with predictable results
                if 'ping' in command.lower() and any(p in normalized for p in ['ping ip', 'ping -c']):
                    # Check if we've already done ping commands
                    ping_count = sum(1 for c in self.command_history if 'ping' in c.lower())
                    if ping_count >= 3:
                        self.redundancy_count += 1
                        return True, "Excessive ping commands", command
                
                # Not redundant
                return False, None, None
                
            except Exception as e:
                # Safety: On error, don't block execution
                print(f"⚠ Error in redundancy check: {e}")
                return False, None, None
                
    def get_action_diversity(self):
        """
        Calculate diversity metrics of recent actions.
        
        Returns:
            dict: Diversity metrics
        """
        with self.lock:
            if not self.command_history:
                return {"diversity_score": 0, "unique_ratio": 0, "top_commands": []}
                
            # Count command frequencies
            counter = Counter(self._normalize_command(cmd) for cmd in self.command_history)
            
            # Calculate diversity metrics
            total = len(self.command_history)
            unique = len(counter)
            unique_ratio = unique / total if total > 0 else 0
            
            # Top repeated commands
            top_commands = counter.most_common(5)
            
            # Simpson's diversity index (higher means more diverse)
            n = sum(counter.values())
            diversity = 1 - sum((c/n)**2 for c in counter.values()) if n > 0 else 0
            
            return {
                "diversity_score": diversity,
                "unique_ratio": unique_ratio,
                "total_commands": total,
                "unique_commands": unique,
                "redundancy_count": self.redundancy_count,
                "top_commands": top_commands
            }
    
    def log_step(self, **kwargs):
        """
        Log a single step with extensive metadata and update command history.
        """
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "redundancy_checked": kwargs.get("redundancy_checked", False),
                **kwargs
            }
            
            # Track command in history
            command = kwargs.get("command")
            if command:
                # Add to history
                self.command_history.append(command)
                self.command_counts[self._normalize_command(command)] += 1
                self.command_hashes.add(self._get_command_hash(command))
                
                # Add to phase history
                phase = kwargs.get("phase", "unknown")
                self.phase_history[phase].append(command)
                
                # Track command result
                output = kwargs.get("output")
                reward = kwargs.get("reward", 0)
                if output:
                    self.command_results[command] = {
                        "output": output[:200] + "..." if len(output) > 200 else output,
                        "reward": reward,
                        "timestamp": time.time()
                    }
                    
                # Track reward
                self.last_reward = reward
            
            # Add to step history
            self.steps.append(entry)
            
            # Write to disk
            try:
                with open(self.step_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                
                # Periodically save command hashes
                if len(self.command_hashes) % 10 == 0:
                    self._save_command_hashes()
            except Exception as e:
                console.print(f"[yellow]⚠ RedAgentBrain: Failed to log step: {e}[/yellow]")

    def log_gpt_feedback(self, prompt, gpt_feedback, summary, episode):
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "episode": episode,
                "prompt": prompt,
                "gpt_feedback": gpt_feedback,
                "summary": summary,
                "diversity_metrics": self.get_action_diversity()
            }
            self.gpt_feedback.append(entry)
            try:
                with open(self.gpt_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                console.print(f"[yellow]⚠ RedAgentBrain: Failed to log GPT feedback: {e}[/yellow]")

    def get_strategy_advice(self, current_state):
        """
        Generate strategic advice based on history and current state.
        
        Args:
            current_state (dict): Current agent state
            
        Returns:
            dict: Strategy advice
        """
        # Calculate diversity metrics
        diversity = self.get_action_diversity()
        
        # Default advice
        advice = {
            "exploration_needed": False,
            "phase_change_needed": False,
            "phase_suggestions": [],
            "recommended_actions": [],
            "diversity_score": diversity["diversity_score"],
            "redundancy_detected": self.redundancy_count > 5
        }
        
        # Analyze current state to provide advice
        current_phase = current_state.get("phase", "unknown")
        
        # Check if we're stuck in one phase
        phase_counts = Counter(entry.get("phase") for entry in self.steps if "phase" in entry)
        if phase_counts and current_phase in phase_counts and phase_counts[current_phase] > 15:
            advice["phase_change_needed"] = True
            
            # Suggest next logical phase
            phase_progression = {
                "recon": "enumeration",
                "enumeration": "exploit",
                "exploit": "privilege_escalation",
                "privilege_escalation": "persistence",
                "persistence": "exfiltration",
                "exfiltration": "cleanup"
            }
            if current_phase in phase_progression:
                advice["phase_suggestions"].append(phase_progression[current_phase])
        
        # Check if more exploration is needed
        if diversity["unique_ratio"] < 0.3:  # Less than 30% unique commands
            advice["exploration_needed"] = True
        
        # Recommend actions based on rewards
        successful_commands = []
        for cmd, data in self.command_results.items():
            if data.get("reward", 0) > 5:  # Consider high-reward commands
                successful_commands.append((cmd, data.get("reward", 0)))
        
        # Sort by reward and take top 3
        successful_commands.sort(key=lambda x: x[1], reverse=True)
        advice["recommended_actions"] = successful_commands[:3]
        
        return advice

    def load_recent_steps(self, n=20):
        """Return last n steps from memory (for dashboard/learning)"""
        with self.lock:
            return list(self.steps)[-n:]

    def load_recent_gpt_feedback(self, n=5):
        """Return recent GPT feedback entries"""
        with self.lock:
            return list(self.gpt_feedback)[-n:]

    def get_redundancy_stats(self):
        """Get statistics about command redundancy"""
        return {
            "redundancy_count": self.redundancy_count,
            "diversity_metrics": self.get_action_diversity()
        }

    def _save_command_hashes(self):
        """Save command hashes to disk for persistence"""
        try:
            with open(self.command_hash_path, "w") as f:
                json.dump(list(self.command_hashes), f)
        except Exception as e:
            console.print(f"[yellow]⚠ RedAgentBrain: Failed to save command hashes: {e}[/yellow]")

    def _load_from_disk(self):
        """Load existing data from disk"""
        try:
            # Load step logs
            if os.path.exists(self.step_log_path):
                with open(self.step_log_path, "r") as f:
                    lines = f.readlines()[-self.steps.maxlen:]
                    for line in lines:
                        try:
                            entry = json.loads(line)
                            self.steps.append(entry)
                            
                            # Also update command history
                            if "command" in entry:
                                cmd = entry["command"]
                                self.command_history.append(cmd)
                                self.command_counts[self._normalize_command(cmd)] += 1
                                self.command_hashes.add(self._get_command_hash(cmd))
                                
                                # Update phase history
                                if "phase" in entry:
                                    phase = entry["phase"]
                                    self.phase_history[phase].append(cmd)
                        except:
                            pass  # Skip invalid lines
            
            # Load GPT feedback
            if os.path.exists(self.gpt_log_path):
                with open(self.gpt_log_path, "r") as f:
                    lines = f.readlines()[-self.gpt_feedback.maxlen:]
                    for line in lines:
                        try:
                            self.gpt_feedback.append(json.loads(line))
                        except:
                            pass  # Skip invalid lines
            
            # Load command hashes
            if os.path.exists(self.command_hash_path):
                with open(self.command_hash_path, "r") as f:
                    hashes = json.load(f)
                    self.command_hashes.update(hashes)
                    
            console.print(f"[blue]📂 RedAgentBrain: Loaded {len(self.steps)} steps, {len(self.command_history)} commands[/blue]")
        except Exception as e:
            console.print(f"[yellow]⚠ RedAgentBrain: Failed to load logs: {e}[/yellow]")

    def reset_for_episode(self):
        """Reset temporary metrics for a new episode while preserving historical data"""
        self.redundancy_count = 0
        self.last_reward = 0

    def get_recommendations(self, command, output, state=None):
        """
        Generate intelligent command recommendations based on:
        - Current command and its output
        - Command history and success patterns
        - Current environment state
        - Current phase of operation
        
        Args:
            command (str): The command that was just executed
            output (str): The output of the command execution
            state (dict, optional): Current environment state
            
        Returns:
            list: List of recommendation dicts with command, params, and why fields
        """
        try:
            recommendations = []
            current_phase = state.get("phase", "recon") if state else "recon"
            
            # Get the normalized command type
            cmd_type = command.split()[0] if ' ' in command else command
            
            # Analysis of output for keywords indicating success/information
            has_success_indicators = any(kw in output.lower() for kw in [
                "success", "found", "open", "available", "discovered", 
                "vulnerable", "exploitable", "completed"
            ])
            
            # Find commands from history with good rewards in this phase
            successful_cmds = []
            for step in self.steps:
                if (step.get("phase") == current_phase and 
                    step.get("reward", 0) > 5.0 and
                    step.get("command") != command):
                    successful_cmds.append({
                        "command": step.get("command"),
                        "reward": step.get("reward", 0)
                    })
            
            # Sort by reward, highest first
            successful_cmds.sort(key=lambda x: x.get("reward", 0), reverse=True)
            
            # Phase-specific recommendations
            if current_phase == "recon":
                # After port scanning, suggest more detailed scanning or enumeration
                if "nmap" in command.lower() or "masscan" in command.lower():
                    # Check if we found ports
                    ports_found = re.search(r'(\d+)/\w+\s+open', output)
                    if ports_found:
                        recommendations.append({
                            "command": f"nmap -sV -p {ports_found.group(1)} {state.get('target_ip', '10.10.10.10')}",
                            "params": "-sV -p",
                            "why": f"Enumerate service on port {ports_found.group(1)}"
                        })
                        
                        # If HTTP port found
                        if any(port in output for port in ["80", "443", "8080", "8443"]):
                            recommendations.append({
                                "command": f"gobuster dir -u http://{state.get('target_ip', '10.10.10.10')} -w /usr/share/wordlists/dirb/common.txt",
                                "params": "-u -w",
                                "why": "Enumerate web directories on discovered HTTP port"
                            })
                            
                    else:
                        # Suggest different scan types
                        recommendations.append({
                            "command": f"nmap -sS -p- {state.get('target_ip', '10.10.10.10')}",
                            "params": "-sS -p-",
                            "why": "Try a full SYN scan on all ports"
                        })
                        
            elif current_phase == "enumeration":
                # Based on output, suggest follow-up enumeration
                if "gobuster" in command.lower() or "dirb" in command.lower():
                    # If found interesting directories in web scan
                    interesting_dirs = re.findall(r'(/\w+)\s+\(Status: 200\)', output)
                    if interesting_dirs:
                        recommendations.append({
                            "command": f"nikto -host http://{state.get('target_ip', '10.10.10.10')}{interesting_dirs[0]}",
                            "params": "-host",
                            "why": f"Run vulnerability scan on discovered directory {interesting_dirs[0]}"
                        })
                
                # If SMB service was mentioned
                if "smb" in output.lower() or "445" in output:
                    recommendations.append({
                        "command": f"enum4linux -a {state.get('target_ip', '10.10.10.10')}",
                        "params": "-a",
                        "why": "Enumerate SMB service details"
                    })
                
            elif current_phase == "exploit":
                # After finding vulnerabilities, suggest exploitation
                if "vulnerability" in output.lower() or "cve" in output.lower():
                    # Extract CVE if present
                    cve_match = re.search(r'(CVE-\d{4}-\d{4,})', output, re.IGNORECASE)
                    if cve_match:
                        cve = cve_match.group(1)
                        recommendations.append({
                            "command": f"searchsploit {cve}",
                            "params": cve,
                            "why": f"Search for exploits for {cve}"
                        })
                        recommendations.append({
                            "command": f"msfconsole -q -x \"search {cve}; exit;\"",
                            "params": cve,
                            "why": f"Check Metasploit for {cve} exploits"
                        })
                
                # If we see unauthorized login message, suggest brute force
                if any(kw in output.lower() for kw in ["login", "password", "auth", "credentials"]):
                    for service in ["ssh", "ftp", "http"]:
                        if service in output.lower():
                            recommendations.append({
                                "command": f"hydra -L /usr/share/wordlists/metasploit/common_users.txt -P /usr/share/wordlists/metasploit/common_passwords.txt {state.get('target_ip', '10.10.10.10')} {service}",
                                "params": "-L -P",
                                "why": f"Attempt credential brute force on {service}"
                            })
                            break
            
            elif current_phase == "privesc":
                # Suggest privilege escalation techniques
                recommendations.append({
                    "command": "sudo -l",
                    "params": "-l",
                    "why": "Check sudo permissions for current user"
                })
                recommendations.append({
                    "command": "find / -perm -u=s -type f 2>/dev/null",
                    "params": "-perm -u=s",
                    "why": "Find SUID binaries for privilege escalation"
                })
                
                # If we find indicators of kernel exploits
                if "kernel" in output.lower() or "linux version" in output.lower():
                    recommendations.append({
                        "command": "uname -a",
                        "params": "-a",
                        "why": "Get kernel version details for potential exploits"
                    })
            
            elif current_phase == "exfiltrate":
                # Suggest data exfiltration techniques
                recommendations.append({
                    "command": "find / -name \"*.txt\" -o -name \"*.pdf\" -o -name \"*.doc\" 2>/dev/null | grep -v \"proc\"",
                    "params": "-name",
                    "why": "Find potentially valuable documents"
                })
                recommendations.append({
                    "command": f"tar czf /tmp/data.tar.gz /home/",
                    "params": "czf",
                    "why": "Archive user home directories"
                })
            
            # Include successful historical commands if relevant
            if successful_cmds:
                for i, cmd_data in enumerate(successful_cmds[:2]):  # Add up to 2 historical commands
                    cmd = cmd_data.get("command")
                    # Only add if not too similar to current command or existing recommendations
                    if not any(self._command_similarity(cmd, r.get("command", "")) > 0.7 for r in recommendations):
                        recommendations.append({
                            "command": cmd,
                            "params": "",
                            "why": f"Previously successful command in {current_phase} phase"
                        })
            
            # Ensure we have at least 3 recommendations
            if len(recommendations) < 3:
                # Add general recommendations based on phase
                phase_defaults = {
                    "recon": [
                        {"command": f"nmap -sV -sS {state.get('target_ip', '10.10.10.10')}", "params": "-sV -sS", "why": "Service version detection scan"},
                        {"command": f"nmap -sU --top-ports 20 {state.get('target_ip', '10.10.10.10')}", "params": "-sU", "why": "UDP port scan for top ports"}
                    ],
                    "enumeration": [
                        {"command": f"nikto -host http://{state.get('target_ip', '10.10.10.10')}", "params": "-host", "why": "Web server vulnerability scan"},
                        {"command": f"whatweb {state.get('target_ip', '10.10.10.10')}", "params": "", "why": "Identify web technologies in use"}
                    ],
                    "exploit": [
                        {"command": f"searchsploit apache 2.4", "params": "", "why": "Search for Apache exploits"},
                        {"command": f"msfconsole -q -x \"search type:exploit platform:linux\"", "params": "", "why": "Browse Linux exploits in Metasploit"}
                    ],
                    "privesc": [
                        {"command": "linpeas.sh", "params": "", "why": "Run LinPEAS privilege escalation scanner"},
                        {"command": "cat /etc/passwd", "params": "", "why": "Examine user accounts on system"}
                    ],
                    "exfiltrate": [
                        {"command": "python3 -m http.server 8000", "params": "", "why": "Start web server to exfiltrate data"},
                        {"command": "zip -r /tmp/evidence.zip /etc/", "params": "-r", "why": "Archive system configuration files"}
                    ]
                }
                
                if current_phase in phase_defaults:
                    for default_rec in phase_defaults[current_phase]:
                        if not any(self._command_similarity(default_rec["command"], r.get("command", "")) > 0.7 for r in recommendations):
                            recommendations.append(default_rec)
                            if len(recommendations) >= 3:
                                break
            
            # Limit to maximum of 5 recommendations
            return recommendations[:5]
            
        except Exception as e:
            console.print(f"[red]❌ Error generating recommendations: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return [
                {"command": "nmap -sS -sV 10.10.10.10", "params": "-sS -sV", "why": "Basic recon scan (fallback)"},
                {"command": "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/dirb/common.txt", "params": "", "why": "Web directory enumeration (fallback)"}
            ]

    def log_novel_command(self, command, source, reason=""):
        """
        Log a novel command that was suggested or discovered
        
        Args:
            command (str): The command
            source (str): Where the command came from (e.g., "OrionAgent", "user", etc.)
            reason (str): Why this command is being logged
        """
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "source": source,
                "reason": reason,
                "is_novel": True
            }
            
            # Add to command history for future recommendations
            if command:
                self.command_history.append(command)
                
            # Write to disk
            try:
                with open(self.step_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                console.print(f"[yellow]⚠ RedAgentBrain: Failed to log novel command: {e}[/yellow]")

    def get_recent_steps(self, n=10):
        """Return the n most recent steps"""
        with self.lock:
            return list(self.steps)[-n:]
