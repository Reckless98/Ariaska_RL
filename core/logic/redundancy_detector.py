# core/logic/redundancy_detector.py — ARIASKA Redundancy Detection v11.0
# ♻️ Smart Command History Analysis | 🧠 Pattern Recognition | 🚫 Anti-Loop Protection

import re
import difflib
import hashlib
from rich.console import Console
from typing import List, Dict, Any, Tuple, Set
from collections import Counter

console = Console()

def detect_redundancy(command_history: List[str], new_command: str) -> bool:
    """
    Enhanced redundancy detection algorithm that considers:
    - Exact matches in recent history
    - Semantic similarity between commands
    - Pattern-based repetition
    - Command effectiveness analysis
    
    Args:
        command_history: List of previously executed commands
        new_command: The command being evaluated for redundancy
        
    Returns:
        bool: True if the command is redundant, False otherwise
    """
    if not command_history or not new_command:
        return False
        
    # Check for exact match in recent history
    recent_history = command_history[-10:] if len(command_history) > 10 else command_history
    if new_command in recent_history:
        console.print(f"[yellow]♻ Exact redundancy detected: {new_command}[/yellow]")
        return True
        
    # Check for normalized similarity (case insensitive, whitespace normalized)
    normalized_command = _normalize_command(new_command)
    normalized_history = [_normalize_command(cmd) for cmd in recent_history]
    if normalized_command in normalized_history:
        console.print(f"[yellow]♻ Normalized redundancy detected: {new_command}[/yellow]")
        return True
        
    # Check for semantic similarity with recent commands
    for cmd in recent_history:
        similarity = _calculate_similarity(new_command, cmd)
        if similarity > 0.85:  # High similarity threshold
            console.print(f"[yellow]♻ Semantic redundancy detected ({similarity:.2f}): {new_command}[/yellow]")
            return True
    
    # Check for pattern-based redundancy (cycling between same few commands)
    if _is_pattern_repeating(command_history + [new_command]):
        console.print(f"[yellow]♻ Pattern-based redundancy detected[/yellow]")
        return True
        
    return False

def detect_redundancy_batch(command_history: List[str]) -> List[int]:
    """
    Detect redundant commands across a batch of command history.
    Returns indices of redundant commands.
    """
    redundant_indices = []
    seen_normalized = set()
    
    for i, cmd in enumerate(command_history):
        normalized = _normalize_command(cmd)
        if normalized in seen_normalized:
            redundant_indices.append(i)
        else:
            seen_normalized.add(normalized)
            
    # Also detect pattern-based redundancy
    if len(command_history) >= 6:
        patterns = _find_repeating_patterns(command_history)
        for pattern_indices in patterns:
            redundant_indices.extend(pattern_indices[1:])  # Keep first occurrence, mark others as redundant
            
    return sorted(set(redundant_indices))  # Remove duplicates and sort

def detect_logical_redundancy(command_history: List[str], new_command: str) -> bool:
    """
    Detect logical redundancy using semantic similarity and phase context.
    """
    # Use semantic similarity and phase context (if available)
    if not command_history or not new_command:
        return False
    # Use existing redundancy checks
    if detect_redundancy(command_history, new_command):
        return True
    # Add: If command is a variant of a recent command (ignoring IP/port)
    norm_new = _normalize_command(new_command)
    for cmd in command_history[-10:]:
        norm_cmd = _normalize_command(cmd)
        if norm_new == norm_cmd:
            return True
        if _calculate_similarity(norm_new, norm_cmd) > 0.92:
            return True
    return False

def suggest_memory_pruning(command_history: List[str], reward_history: List[float], threshold: float = 0.0) -> List[int]:
    """
    Suggest indices for pruning memory based on redundancy and low reward.
    """
    redundant = detect_redundancy_batch(command_history)
    low_reward = [i for i, r in enumerate(reward_history) if r <= threshold]
    # Union of both
    return sorted(set(redundant + low_reward))

def _normalize_command(command: str) -> str:
    """Normalize command by removing extra whitespace and lowercase."""
    # Remove IP address placeholders to better detect repeated commands with different IPs
    normalized = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '{IP}', command)
    # Normalize ports
    normalized = re.sub(r'(?<!\d)\d{2,5}(?!\d)', '{PORT}', normalized)
    # Remove timestamp-like patterns
    normalized = re.sub(r'\d{2}:\d{2}(:\d{2})?', '{TIME}', normalized)
    # Basic normalization
    return ' '.join(normalized.lower().split())

def _calculate_similarity(cmd1: str, cmd2: str) -> float:
    """Calculate semantic similarity between two commands."""
    # Use difflib for string similarity
    seq_matcher = difflib.SequenceMatcher(None, cmd1, cmd2)
    return seq_matcher.ratio()

def _is_pattern_repeating(commands: List[str]) -> bool:
    """Detect if there's a repeating pattern in the command history."""
    if len(commands) < 6:
        return False
        
    # Check for A-B-A-B pattern
    last_4 = [_normalize_command(cmd) for cmd in commands[-4:]]
    if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
        return True
        
    # Check for A-B-C-A-B-C pattern
    last_6 = [_normalize_command(cmd) for cmd in commands[-6:]]
    if last_6[0] == last_6[3] and last_6[1] == last_6[4] and last_6[2] == last_6[5]:
        return True
        
    return False

def _find_repeating_patterns(commands: List[str]) -> List[List[int]]:
    """Find repeating patterns and return indices of redundant commands."""
    patterns = []
    normalized = [_normalize_command(cmd) for cmd in commands]
    
    # Build command fingerprint map
    cmd_indices = {}
    for i, cmd in enumerate(normalized):
        if cmd in cmd_indices:
            cmd_indices[cmd].append(i)
        else:
            cmd_indices[cmd] = [i]
    
    # Find repeated sequences
    for seq_len in range(2, 5):  # Check for patterns of length 2-4
        for i in range(len(commands) - seq_len * 2 + 1):
            pattern = tuple(normalized[i:i+seq_len])
            pattern_hash = hash(pattern)
            
            # Find potential matches elsewhere in history
            for j in range(i + seq_len, len(commands) - seq_len + 1):
                check_pattern = tuple(normalized[j:j+seq_len])
                if pattern == check_pattern:
                    # Found repeating pattern
                    patterns.append(list(range(j, j+seq_len)))
    
    return patterns

def suggest_alternative(command_history: List[str], redundant_command: str) -> str:
    """
    Suggest an alternative to a redundant command.
    
    Args:
        command_history: Previous command history
        redundant_command: The redundant command to replace
        
    Returns:
        str: Suggested alternative command
    """
    # Import GPT utility here to avoid circular imports
    try:
        # Query GPT for alternatives
        import subprocess
        
        prompt = f"""
        Command '{redundant_command}' is redundant in recent history.
        Suggest a more effective alternative command for the same task.
        Only respond with the alternative command.
        """
        
        try:
            result = subprocess.run(
                ["sgpt", "--model", "gpt-4o-mini", "--temperature", "0.2", "--role", "aria", prompt],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, text=True
            )
            suggestion = result.stdout.strip()
            if suggestion and len(suggestion) > 5:
                return suggestion
        except Exception as e:
            console.print(f"[yellow]⚠ GPT suggestion failed: {e}[/yellow]")
    except:
        pass
    
    # Fallback: Modify the command slightly
    if "nmap" in redundant_command:
        if "-p-" in redundant_command:
            return redundant_command.replace("-p-", "-p 1-10000")
        elif "-sC" in redundant_command:
            return redundant_command.replace("-sC", "--script=vuln")
        else:
            return redundant_command + " -Pn"
    
    return f"echo 'Alternative to: {redundant_command}'"

def detect_ineffective_commands(command_history: List[str], outputs: List[str]) -> List[int]:
    """
    Detect commands that produce consistently empty or error outputs.
    
    Args:
        command_history: List of commands
        outputs: Corresponding outputs for each command
        
    Returns:
        List[int]: Indices of potentially ineffective commands
    """
    ineffective = []
    
    if len(command_history) != len(outputs):
        return ineffective
        
    for i, (cmd, output) in enumerate(zip(command_history, outputs)):
        # Check for typical error patterns
        if (not output or 
            "command not found" in output.lower() or
            "error" in output.lower() or
            "host seems down" in output.lower() or
            "failed" in output.lower() or
            "[timeout]" in output.lower()):
            ineffective.append(i)
            
    return ineffective

# Diagnostic functions for visualization
def visualize_command_groups(command_history: List[str]) -> Dict[str, int]:
    """Group commands by type and count them."""
    groups = Counter()
    
    for cmd in command_history:
        cmd_type = cmd.split()[0] if cmd and ' ' in cmd else cmd
        groups[cmd_type] += 1
        
    return dict(groups.most_common())

# Testing hook
if __name__ == "__main__":
    test_history = [
        "nmap -sC -sV 10.10.10.10",
        "gobuster dir -u http://10.10.10.10",
        "nmap -p- 10.10.10.10",
        "gobuster dir -u http://10.10.10.10",  # Redundant
        "nmap -sC -sV -p- 10.10.10.10",  # Similar to past command
        "wget http://10.10.10.10/file.txt",
        "cat file.txt",
        "wget http://10.10.10.10/file.txt",  # Redundant
    ]
    
    for i, cmd in enumerate(test_history):
        if detect_redundancy(test_history[:i], cmd):
            print(f"Redundant: {cmd}")
            alt = suggest_alternative(test_history[:i], cmd)
            print(f"Alternative: {alt}")
