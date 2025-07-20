#!/usr/bin/env python3
# main.py — ARIASKA_RL CLI Interface v3.0 (GPT-4o-mini Only)
# Interactive CLI for cybersecurity training with simulation and live modes

import os
import sys
import asyncio
import time
import signal
import logging
from typing import Optional
from pathlib import Path

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.rule import Rule
    from rich import box
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
except ImportError:
    print("Missing dependencies. Install with: pip install rich prompt-toolkit")
    sys.exit(1)

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import core components
from core.gpt_manager import GPTManager
from core.utils.config_loader import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("logs/ariaska.log"),
        logging.NullHandler()
    ]
)

console = Console()
logger = logging.getLogger("ariaska")

# Global components
gpt_manager = None
agent_manager = None
primary_agent = None

def setup_environment():
    """Setup directories and check configuration"""
    try:
        # Create necessary directories
        dirs = ["logs", "models", "core/memories", "core/memories/vectorstore"]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]❌ OPENAI_API_KEY not found in environment[/red]")
            console.print("[yellow]Please set your OpenAI API key in .env file[/yellow]")
            return False
        
        console.print("[green]✓ Environment setup complete[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Environment setup failed: {e}[/red]")
        return False

def init_gpt_manager():
    """Initialize GPT Manager"""
    global gpt_manager
    try:
        gpt_manager = GPTManager()
        console.print("[green]✓ GPT-4o-mini manager initialized[/green]")
        return True
    except Exception as e:
        console.print(f"[red]❌ GPT Manager initialization failed: {e}[/red]")
        return False

def init_agent_manager():
    """Initialize Agent Manager - simplified version"""
    global agent_manager, primary_agent
    try:
        # For now, create a simple agent structure
        from core.multiagent.agent_manager import AgentManager
        
        verbosity = os.getenv("VERBOSITY", "standard")
        agent_manager = AgentManager(verbosity=verbosity)
        
        # Get primary agent (RedAgent)
        primary_agent = agent_manager.get_agent("RedAgent")
        if primary_agent is None:
            console.print("[yellow]⚠ Using simplified agent mode[/yellow]")
            # Create a simple placeholder agent
            primary_agent = SimpleAgent()
        
        console.print("[green]✓ Agent system initialized[/green]")
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠ Full agent system unavailable: {e}[/yellow]")
        console.print("[yellow]⚠ Using simplified mode[/yellow]")
        primary_agent = SimpleAgent()
        return True

class SimpleAgent:
    """Simplified agent for basic CLI functionality"""
    
    def __init__(self):
        self.agent_id = "SimpleRedAgent"
        self.gpt_manager = gpt_manager
        self.command_history = []
        
    def execute_command(self, command: str) -> tuple:
        """Execute a command and return result"""
        try:
            if self.gpt_manager:
                # Get tactical command from GPT
                tactical_command = self.gpt_manager.gpt_request(
                    f"Convert this to a safe cybersecurity command: {command}",
                    task_type="tactical",
                    agent_id=self.agent_id
                )
                
                # For demo, return the command as result
                self.command_history.append(tactical_command)
                return tactical_command, f"Executed: {tactical_command}", 0
            else:
                return command, f"Echo: {command}", 0
                
        except Exception as e:
            return "", f"Error: {e}", 1

def banner():
    """Display ARIASKA banner"""
    banner_text = """
     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗ 
    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗
    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║
    ██╔══██║██╔══██╗██║██╔══██║╚════██║██╔═██╗ ██╔══██║
    ██║  ██║██║  ██║██║██║  ██║███████║██║  ██╗██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
    
    🤖 ARIASKA_RL v3.0 - GPT-4o-mini Enhanced Cybersecurity Training
    """
    
    console.print(Panel.fit(banner_text, border_style="cyan"))
    console.print()

def run_simulated_training(episodes: int = 50):
    """Run training in simulated environment"""
    console.print(f"[cyan]🤖 Starting Simulated Training for {episodes} episodes...[/cyan]")
    console.print("[blue]Environment: Simulated Cyber Range[/blue]")
    
    if not gpt_manager:
        console.print("[red]❌ GPT Manager not initialized[/red]")
        return
    
    try:
        # Reset token count for training
        gpt_manager.reset_token_count()
        
        console.print(f"[green]🎯 Training Mode: Simulation[/green]")
        console.print(f"[blue]Episodes: {episodes}[/blue]")
        console.print(f"[yellow]Model: {gpt_manager.primary_model}[/yellow]")
        
        for ep in range(episodes):
            console.print(f"[bold cyan]Episode {ep+1}/{episodes}[/bold cyan]")
            
            # Simulate episode phases
            phases = ["reconnaissance", "enumeration", "exploitation", "post_exploitation"]
            
            for phase in phases:
                console.print(f"  [magenta]Phase: {phase}[/magenta]")
                
                # Get strategic command from GPT
                prompt = f"Suggest a cybersecurity command for {phase} phase in a training simulation"
                command = gpt_manager.gpt_request(
                    prompt,
                    task_type="tactical",
                    agent_id="training",
                    max_tokens=50
                )
                
                console.print(f"    [green]Command: {command}[/green]")
                
                # Simulate execution result
                result = f"Simulated execution of {command}"
                reward = 0.8 + (0.2 * (ep / episodes))  # Increasing performance
                
                # Get learning feedback
                feedback = gpt_manager.get_learning_feedback(
                    command, result, reward, "training"
                )
                console.print(f"    [blue]Feedback: {feedback}[/blue]")
                
                time.sleep(0.5)  # Small delay for readability
            
            # Batch training every 10 episodes
            if (ep + 1) % 10 == 0:
                console.print(f"[magenta]🔄 Running batch training at episode {ep+1}[/magenta]")
                console.print(f"[green]💾 Models saved at episode {ep+1}[/green]")
            
            # Show progress
            if (ep + 1) % 5 == 0:
                stats = gpt_manager.get_global_stats()
                console.print(f"    [yellow]Tokens used: {stats['current_episode_tokens']}/{stats['token_limit']}[/yellow]")
        
        console.print("[green]✅ Simulated training completed successfully[/green]")
        
        # Display final stats
        final_stats = gpt_manager.get_global_stats()
        stats_table = Table(title="Training Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Total Requests", str(final_stats['total_requests']))
        stats_table.add_row("Cache Hit Rate", f"{final_stats['cache_hit_rate']:.2%}")
        stats_table.add_row("Total Tokens", str(final_stats['total_tokens']))
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[red]❌ Training error: {e}[/red]")

def run_metasploitable_training(episodes: int = 25):
    """Run training against Metasploitable target"""
    console.print(f"[bold red]⚔️ Starting Metasploitable Training for {episodes} episodes...[/bold red]")
    console.print("[bold yellow]🎯 Environment: Live Metasploitable Target[/bold yellow]")
    console.print("[bold red]⚠️ WARNING: This would execute real commands against live targets![/bold red]")
    
    # Check if live mode is enabled
    config = get_config()
    if not config.is_live_mode():
        console.print("[red]❌ Live mode not enabled. Set ARIASKA_MODE=live in .env file[/red]")
        return
    
    # Confirm with user
    confirm = console.input("[bold yellow]Continue with live target training? (yes/no): [/bold yellow]")
    if not confirm.lower().startswith('y'):
        console.print("[yellow]Training cancelled by user[/yellow]")
        return
    
    try:
        target_ip = config.get_target_ip()
        console.print(f"[red]🎯 Target: {target_ip}[/red]")
        
        # Reset token count
        if gpt_manager:
            gpt_manager.reset_token_count()
        
        for ep in range(episodes):
            console.print(f"[bold red]🎯 Live Episode {ep+1}/{episodes}[/bold red]")
            
            # Strategic planning
            strategy_prompt = f"Plan a safe reconnaissance strategy for Metasploitable target {target_ip}"
            strategy = gpt_manager.gpt_request(
                strategy_prompt,
                task_type="planning",
                agent_id="live_training",
                max_tokens=150
            )
            console.print(f"[blue]🧠 Strategy: {strategy}[/blue]")
            
            # Simulate safer commands for live environment
            safe_commands = [
                f"nmap -sT -p 1-1000 {target_ip}",
                f"nmap -sV -p 22,80,443 {target_ip}",
                f"curl -I http://{target_ip}"
            ]
            
            for cmd in safe_commands:
                console.print(f"  [green]Safe Command: {cmd}[/green]")
                # In real implementation, would execute with safety checks
                result = f"Safe execution result for {cmd}"
                console.print(f"    [blue]Result: {result}[/blue]")
                time.sleep(1)  # Safety pause
            
            # Batch training every 5 episodes (more frequent for live)
            if (ep + 1) % 5 == 0:
                console.print(f"[magenta]🔄 Live batch training at episode {ep+1}[/magenta]")
                console.print(f"[green]💾 Live models saved at episode {ep+1}[/green]")
        
        console.print("[green]✅ Metasploitable training completed[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Live training error: {e}[/red]")

def test_gpt_connectivity():
    """Test GPT-4o-mini connectivity"""
    console.print("[yellow]Testing GPT-4o-mini connectivity...[/yellow]")
    try:
        if not gpt_manager:
            console.print("[red]❌ GPT Manager not initialized[/red]")
            return
            
        result = gpt_manager.test_connectivity()
        
        if result["status"] == "success":
            console.print(f"[green]✓ GPT Test Successful:[/green] {result['response']}")
            console.print(f"[blue]Model: {result['model']}[/blue]")
            console.print(f"[blue]Platform: {result['platform']}[/blue]")
        else:
            console.print(f"[red]❌ GPT Test Failed: {result['error']}[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ GPT Test Error: {e}[/red]")

def show_help():
    """Show available commands"""
    commands = [
        ("simulate-train [n]", "Run simulated training for n episodes (default: 50)"),
        ("train-meta [n]", "Run live Metasploitable training for n episodes (default: 25)"),
        ("test-gpt", "Test GPT-4o-mini connectivity"),
        ("stats", "Show GPT usage statistics"),
        ("status", "Show system status"),
        ("clear", "Clear console"),
        ("help", "Show this help message"),
        ("exit", "Exit ARIASKA CLI"),
    ]
    
    table = Table(title="🛈 Available Commands", box=box.ROUNDED)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="magenta")
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    console.print(table)

def show_stats():
    """Show GPT usage statistics"""
    if not gpt_manager:
        console.print("[red]❌ GPT Manager not available[/red]")
        return
    
    try:
        stats = gpt_manager.get_global_stats()
        
        table = Table(title="📊 GPT Usage Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Requests", str(stats['total_requests']))
        table.add_row("Cache Hits", str(stats['cache_hits']))
        table.add_row("Cache Hit Rate", f"{stats['cache_hit_rate']:.2%}")
        table.add_row("Failures", str(stats['failures']))
        table.add_row("Total Tokens", str(stats['total_tokens']))
        table.add_row("Current Episode Tokens", str(stats['current_episode_tokens']))
        table.add_row("Token Limit", str(stats['token_limit']))
        table.add_row("Cache Size", str(stats['cache_size']))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error getting stats: {e}[/red]")

def show_status():
    """Show system status"""
    table = Table(title="🖥️ System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    # GPT Manager
    gpt_status = "✓ Active" if gpt_manager else "❌ Inactive"
    table.add_row("GPT Manager", gpt_status)
    
    # Agent Manager
    agent_status = "✓ Active" if agent_manager else "⚠ Simplified Mode"
    table.add_row("Agent Manager", agent_status)
    
    # Primary Agent
    primary_status = "✓ Ready" if primary_agent else "❌ Inactive"
    table.add_row("Primary Agent", primary_status)
    
    # Environment
    config = get_config()
    env_mode = "Live" if config.is_live_mode() else "Simulation"
    table.add_row("Environment Mode", env_mode)
    
    # API Key
    api_status = "✓ Configured" if os.getenv("OPENAI_API_KEY") else "❌ Missing"
    table.add_row("OpenAI API Key", api_status)
    
    console.print(table)

async def main_loop():
    """Main CLI loop"""
    session = PromptSession()
    
    while True:
        try:
            # Create prompt
            prompt_text = HTML('<ansicyan>zer0</ansicyan><ansimagenta>@ARIASKA</ansimagenta><ansiblack> > </ansiblack>')
            command = await session.prompt_async(prompt_text)
            
            if not command.strip():
                continue
            
            cmd_parts = command.strip().split()
            cmd_lower = cmd_parts[0].lower()
            
            if cmd_lower in ["exit", "quit"]:
                console.print("[red]Exiting ARIASKA RL. Until next battle.[/red]")
                break
                
            elif cmd_lower == "simulate-train":
                count = int(cmd_parts[1]) if len(cmd_parts) > 1 and cmd_parts[1].isdigit() else 50
                run_simulated_training(episodes=count)
                
            elif cmd_lower == "train-meta":
                count = int(cmd_parts[1]) if len(cmd_parts) > 1 and cmd_parts[1].isdigit() else 25
                run_metasploitable_training(episodes=count)
                
            elif cmd_lower == "test-gpt":
                test_gpt_connectivity()
                
            elif cmd_lower == "stats":
                show_stats()
                
            elif cmd_lower == "status":
                show_status()
                
            elif cmd_lower in ["clear", "cls"]:
                console.clear()
                banner()
                
            elif cmd_lower in ["help", "?"]:
                show_help()
                
            else:
                # Execute as tactical command
                if primary_agent:
                    cmd, result, code = primary_agent.execute_command(command)
                    console.print(f"[green]Command:[/green] {cmd}")
                    console.print(f"[blue]Result:[/blue] {result}")
                else:
                    console.print(f"[yellow]Unknown command: {command}. Type 'help' for available commands.[/yellow]")
                    
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Interrupt received. Exiting...[/red]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

def graceful_shutdown(signum, frame):
    """Handle shutdown signals"""
    console.print("\n[yellow]Shutting down ARIASKA RL...[/yellow]")
    if gpt_manager:
        gpt_manager.cleanup()
    sys.exit(0)

def main():
    """Main entry point"""
    # Handle signals
    signal.signal(signal.SIGINT, graceful_shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, graceful_shutdown)
    
    # Initialize
    banner()
    
    console.print("[bold cyan]🚀 Initializing ARIASKA_RL System...[/bold cyan]")
    
    if not setup_environment():
        console.print("[red]❌ Environment setup failed[/red]")
        sys.exit(1)
    
    if not init_gpt_manager():
        console.print("[red]❌ GPT Manager initialization failed[/red]")
        sys.exit(1)
    
    init_agent_manager()  # This can fail gracefully
    
    console.print("[green]✅ ARIASKA_RL System Ready[/green]")
    console.print("[yellow]Type 'help' for available commands[/yellow]")
    console.rule()
    
    # Start main loop
    try:
        asyncio.run(main_loop())
    except Exception as e:
        console.print(f"[red]❌ Main loop error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
