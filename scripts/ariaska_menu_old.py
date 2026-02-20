#!/usr/bin/env python3
"""
Ariaska Interactive Command Center — Phase 43

Rich terminal-based interactive menu for launching Ariaska training,
managing GPU resources, viewing training history, and system diagnostics.

Usage:
    python scripts/ariaska_menu.py
    make ariaska
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── ASCII Banner ─────────────────────────────────────────────────────
BANNER = r"""[bold red]
     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗ 
    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗
    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║
    ██╔══██║██╔══██╗██║██╔══██║╚════██║██╔═██╗ ██╔══██║
    ██║  ██║██║  ██║██║██║  ██║███████║██║  ██╗██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
[/bold red][bold white]    ═══ Autonomous Multi-Agent RL Pentesting System ═══[/bold white]
[dim]    Phase 43 • GPU-Accelerated • 5 Agents • PPO Actor-Critic v3.0[/dim]
"""


@dataclass
class TrainingConfig:
    """Training configuration assembled from menu choices."""
    target_ip: str = "10.10.10.10"
    mode: str = "normal"           # normal, ctf, htb
    compute: str = "auto"          # cpu, gpu, cloud
    steps: int = 200
    seed: Optional[int] = None
    dry_run: bool = False
    ctf_flag: bool = False
    verbosity: str = "normal"      # quiet, normal, verbose, debug
    custom_flags: Dict[str, str] = field(default_factory=dict)


def detect_environment() -> Dict[str, Any]:
    """Detect the current execution environment."""
    env: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": "N/A",
        "gpu_vram_gb": 0.0,
        "local_llm_running": False,
        "local_llm_model": "N/A",
        "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "python_version": sys.version.split()[0],
        "cuda_available": False,
        "test_count": 0,
    }
    
    # GPU detection via torch
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu_available"] = True
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            env["cuda_available"] = True
    except ImportError:
        pass
    
    # Local LLM detection
    llm_port = int(os.environ.get("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"http://127.0.0.1:{llm_port}/v1/models", timeout=2)
        data = json.loads(resp.read())
        if data.get("data"):
            env["local_llm_running"] = True
            env["local_llm_model"] = data["data"][0].get("id", "unknown")
    except Exception:
        pass
    
    # Check for model files
    model_dirs = ["/models", "/root/models", str(Path.home() / "models"), "./models"]
    for d in model_dirs:
        p = Path(d)
        if p.exists():
            gguf_files = list(p.glob("*.gguf"))
            if gguf_files:
                env["local_llm_model"] = gguf_files[0].name
                break
    
    return env


def show_system_panel(env: Dict[str, Any]) -> Panel:
    """Create a system status panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    
    # Compute
    if env["gpu_available"]:
        gpu_text = f"[green]✓[/green] {env['gpu_name']} ({env['gpu_vram_gb']} GB VRAM)"
    else:
        gpu_text = "[yellow]✗[/yellow] CPU-only mode"
    table.add_row("GPU", gpu_text)
    
    # Local LLM
    if env["local_llm_running"]:
        llm_text = f"[green]✓ ONLINE[/green] — {env['local_llm_model']}"
    else:
        llm_text = "[dim]✗ Offline[/dim]"
    table.add_row("Local LLM", llm_text)
    
    # OpenAI
    if env["openai_key"]:
        table.add_row("OpenAI API", "[green]✓ Key configured[/green]")
    else:
        table.add_row("OpenAI API", "[red]✗ No key — offline mode[/red]")
    
    # Python
    table.add_row("Python", env["python_version"])
    
    # CUDA
    if env["cuda_available"]:
        table.add_row("CUDA", "[green]✓ Available[/green]")
    else:
        table.add_row("CUDA", "[dim]✗ Not available[/dim]")
    
    return Panel(table, title="[bold]System Status[/bold]", border_style="bright_blue", box=box.ROUNDED)


def show_main_menu() -> str:
    """Display the main menu and get user choice."""
    console.print()
    
    menu_table = Table(show_header=False, box=box.SIMPLE_HEAVY, border_style="bright_cyan")
    menu_table.add_column("Key", style="bold yellow", width=6, justify="center")
    menu_table.add_column("Action", style="white")
    menu_table.add_column("Description", style="dim")
    
    menu_table.add_row("1", "🎯 Smart Train", "Launch training with interactive configuration")
    menu_table.add_row("2", "⚡ Quick Train", "Start immediately — default settings, minimal prompts")
    menu_table.add_row("3", "🏴 CTF Mode", "Capture-the-flag optimized (flag hunting, fast exploitation)")
    menu_table.add_row("4", "🖥️  HTB Mode", "HackTheBox profile (realistic engagement, full kill chain)")
    menu_table.add_row("5", "🔁 Resume Last", "Resume from last checkpoint/trace")
    menu_table.add_row("", "", "")
    menu_table.add_row("6", "🔧 GPU Manager", "Start/stop local LLM, check GPU status, download models")
    menu_table.add_row("7", "📊 Training History", "View past engagements, compare metrics, cost tracker")
    menu_table.add_row("8", "🧪 Run Tests", "Execute test suite with coverage report")
    menu_table.add_row("9", "🔍 System Diagnostics", "Full system health check, dependency audit")
    menu_table.add_row("0", "🛠️  Advanced Config", "Feature flags, hyperparameters, model routing")
    menu_table.add_row("", "", "")
    menu_table.add_row("q", "Exit", "")
    
    console.print(Panel(menu_table, title="[bold]Command Center[/bold]", border_style="bright_cyan"))
    
    return Prompt.ask(
        "[bold yellow]Select[/bold yellow]",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q"],
        default="1"
    )


def configure_smart_train(env: Dict[str, Any]) -> Optional[TrainingConfig]:
    """Interactive training configuration."""
    config = TrainingConfig()
    
    console.print(Rule("[bold]Smart Train Configuration[/bold]", style="bright_cyan"))
    
    # Target IP
    config.target_ip = Prompt.ask(
        "🎯 Target IP",
        default="10.10.10.10"
    )
    
    # Mode selection
    console.print()
    mode_table = Table(show_header=False, box=None, padding=(0, 1))
    mode_table.add_column("Key", style="bold yellow", width=4)
    mode_table.add_column("Mode", style="white")
    mode_table.add_column("Description", style="dim")
    mode_table.add_row("1", "Normal", "Standard engagement — full kill chain, balanced exploration")
    mode_table.add_row("2", "CTF", "Flag-hunting — aggressive exploitation, speed-optimized")
    mode_table.add_row("3", "HTB", "HackTheBox — realistic engagement, careful enumeration")
    mode_table.add_row("4", "Stealth", "Minimal detection risk — shadow agent priority")
    console.print(Panel(mode_table, title="Mode", border_style="dim"))
    
    mode_choice = Prompt.ask("Mode", choices=["1", "2", "3", "4"], default="1")
    config.mode = {"1": "normal", "2": "ctf", "3": "htb", "4": "stealth"}[mode_choice]
    config.ctf_flag = config.mode in ("ctf", "htb")
    
    # Compute selection
    if env["gpu_available"]:
        console.print()
        compute_table = Table(show_header=False, box=None, padding=(0, 1))
        compute_table.add_column("Key", style="bold yellow", width=4)
        compute_table.add_column("Compute", style="white")
        compute_table.add_column("Info", style="dim")
        compute_table.add_row("1", "Auto", "GPU for nano/mini, OpenAI for codex/full")
        compute_table.add_row("2", "GPU Only", "All LLM calls via local GPU (no API cost)")
        compute_table.add_row("3", "OpenAI Only", "All calls via OpenAI (max quality)")
        compute_table.add_row("4", "CPU", "No GPU acceleration")
        console.print(Panel(compute_table, title="Compute", border_style="dim"))
        
        compute_choice = Prompt.ask("Compute", choices=["1", "2", "3", "4"], default="1")
        config.compute = {"1": "auto", "2": "gpu", "3": "cloud", "4": "cpu"}[compute_choice]
    else:
        config.compute = "cloud" if env["openai_key"] else "cpu"
        console.print(f"[dim]Compute: {config.compute} (auto-detected)[/dim]")
    
    # Steps
    config.steps = IntPrompt.ask("📏 Training steps", default=200)
    
    # Verbosity
    verbosity_choice = Prompt.ask(
        "📢 Verbosity",
        choices=["quiet", "normal", "verbose", "debug"],
        default="normal"
    )
    config.verbosity = verbosity_choice
    
    # Advanced options
    if Confirm.ask("🔧 Advanced options?", default=False):
        # Seed
        seed_str = Prompt.ask("🎲 Random seed (empty=random)", default="")
        if seed_str.strip():
            config.seed = int(seed_str)
        
        # Dry run
        config.dry_run = Confirm.ask("🔒 Dry run (no real commands)?", default=False)
        
        # Custom feature flags
        if Confirm.ask("🚩 Override feature flags?", default=False):
            flag_str = Prompt.ask(
                "Flags (comma-separated, e.g. FF_USE_MICRO_CHAIN=0,FF_STRICT_EXPLOIT_GATE=log)",
                default=""
            )
            if flag_str.strip():
                for pair in flag_str.split(","):
                    if "=" in pair:
                        k, v = pair.strip().split("=", 1)
                        config.custom_flags[k.strip()] = v.strip()
    
    # Confirmation
    console.print()
    summary = Table(title="Training Configuration", box=box.ROUNDED, border_style="green")
    summary.add_column("Parameter", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Target", config.target_ip)
    summary.add_row("Mode", config.mode)
    summary.add_row("Compute", config.compute)
    summary.add_row("Steps", str(config.steps))
    summary.add_row("Seed", str(config.seed) if config.seed else "random")
    summary.add_row("Dry Run", "Yes" if config.dry_run else "No")
    summary.add_row("Verbosity", config.verbosity)
    if config.custom_flags:
        summary.add_row("Flags", ", ".join(f"{k}={v}" for k, v in config.custom_flags.items()))
    console.print(summary)
    
    if not Confirm.ask("\n[bold]Launch training?[/bold]", default=True):
        return None
    
    return config


def build_command(config: TrainingConfig) -> List[str]:
    """Build the CLI command from config."""
    cmd = [
        sys.executable, "ariaska_cli.py", "smart-train",
        "--target", config.target_ip,
        "--steps", str(config.steps),
    ]
    
    if config.ctf_flag:
        cmd.append("--ctf")
    
    if config.seed is not None:
        cmd.extend(["--seed", str(config.seed)])
    
    return cmd


def build_env(config: TrainingConfig, env: Dict[str, Any]) -> Dict[str, str]:
    """Build environment variables from config."""
    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = str(_PROJECT_ROOT)
    
    if config.dry_run:
        run_env["ARIASKA_DRY_RUN"] = "1"
    
    # Compute mode env vars
    if config.compute == "gpu":
        run_env["FF_LOCAL_LLM"] = "1"
        run_env["FF_LOCAL_LLM_OFFLOAD_NANO"] = "1"
        run_env["FF_LOCAL_LLM_OFFLOAD_MINI"] = "1"
    elif config.compute == "cpu":
        run_env["FF_LOCAL_LLM"] = "0"
    elif config.compute == "auto":
        if env["gpu_available"] or env["local_llm_running"]:
            run_env["FF_LOCAL_LLM"] = "1"
    
    # Custom flags
    for k, v in config.custom_flags.items():
        run_env[k] = v
    
    # Mode-specific flags
    if config.mode == "stealth":
        run_env.setdefault("FF_SHADOW_PRIORITY", "1")
    
    return run_env


def launch_training(config: TrainingConfig, env: Dict[str, Any]) -> None:
    """Launch training with the given configuration."""
    cmd = build_command(config)
    run_env = build_env(config, env)
    
    console.print()
    console.print(Rule("[bold green]Launching Ariaska Training[/bold green]", style="green"))
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    console.print()
    
    try:
        os.chdir(str(_PROJECT_ROOT))
        os.execve(sys.executable, cmd, run_env)
    except Exception as e:
        console.print(f"[red]Launch failed: {e}[/red]")
        # Fallback to subprocess
        subprocess.run(cmd, env=run_env, cwd=str(_PROJECT_ROOT))


def show_gpu_manager(env: Dict[str, Any]) -> None:
    """GPU management submenu."""
    while True:
        console.print(Rule("[bold]GPU Manager[/bold]", style="bright_magenta"))
        
        # Current status
        console.print(show_system_panel(env))
        
        menu = Table(show_header=False, box=None, padding=(0, 1))
        menu.add_column("Key", style="bold yellow", width=4)
        menu.add_column("Action", style="white")
        menu.add_row("1", "Start Local LLM Server")
        menu.add_row("2", "Stop Local LLM Server")
        menu.add_row("3", "Check LLM Health")
        menu.add_row("4", "GPU Utilization (nvidia-smi)")
        menu.add_row("5", "Download/Update Model")
        menu.add_row("6", "View LLM Server Logs")
        menu.add_row("b", "Back to main menu")
        console.print(Panel(menu, border_style="bright_magenta"))
        
        choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5", "6", "b"], default="b")
        
        if choice == "b":
            break
        elif choice == "1":
            start_local_llm()
        elif choice == "2":
            stop_local_llm()
        elif choice == "3":
            check_llm_health()
        elif choice == "4":
            show_nvidia_smi()
        elif choice == "5":
            download_model()
        elif choice == "6":
            view_llm_logs()
        
        # Refresh env
        env.update(detect_environment())


def start_local_llm() -> None:
    """Start the local LLM server."""
    try:
        from core.llm.local_llm_provider import get_local_llm_provider
        provider = get_local_llm_provider()
        if provider.is_available():
            console.print("[yellow]LLM server already running[/yellow]")
            return
        console.print("[cyan]Starting local LLM server...[/cyan]")
        if provider.start_server():
            console.print("[green]✓ LLM server started successfully[/green]")
        else:
            console.print("[red]✗ Failed to start LLM server[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def stop_local_llm() -> None:
    """Stop the local LLM server."""
    try:
        from core.llm.local_llm_provider import get_local_llm_provider
        provider = get_local_llm_provider()
        provider.stop_server()
        console.print("[green]✓ LLM server stopped[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def check_llm_health() -> None:
    """Check LLM server health."""
    port = int(os.environ.get("ARIASKA_LOCAL_LLM_PORT", "8192"))
    try:
        import urllib.request
        
        # Check /v1/models
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        models = json.loads(resp.read())
        console.print(f"[green]✓ Server responding on port {port}[/green]")
        if models.get("data"):
            for m in models["data"]:
                console.print(f"  Model: [bold]{m.get('id', 'unknown')}[/bold]")
        
        # Quick inference test
        console.print("[cyan]Running inference test...[/cyan]")
        import urllib.request
        req_data = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": "Say 'Ariaska ready' in exactly 2 words"}],
            "max_tokens": 10
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=30)
        t1 = time.time()
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("total_tokens", "?")
        console.print(f"  Response: [bold]{content.strip()}[/bold]")
        console.print(f"  Latency: {(t1-t0)*1000:.0f}ms | Tokens: {tokens}")
        console.print("[green]✓ Inference test passed[/green]")
    except Exception as e:
        console.print(f"[red]✗ Health check failed: {e}[/red]")


def show_nvidia_smi() -> None:
    """Show nvidia-smi output."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            console.print(Panel(result.stdout, title="nvidia-smi", border_style="green"))
        else:
            console.print("[yellow]nvidia-smi not available[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]nvidia-smi not found — no NVIDIA GPU detected[/yellow]")


def download_model() -> None:
    """Download a model from HuggingFace."""
    console.print(Rule("Model Download", style="dim"))
    
    models = Table(show_header=True, box=box.SIMPLE)
    models.add_column("#", style="yellow")
    models.add_column("Model", style="white")
    models.add_column("Size", style="dim")
    models.add_column("VRAM", style="dim")
    models.add_row("1", "Qwen3-32B-Instruct Q4_K_M", "~18.5 GB", "~20 GB")
    models.add_row("2", "Qwen3-32B-Instruct Q5_K_M", "~22 GB", "~24 GB")
    models.add_row("3", "Qwen3-14B-Instruct Q5_K_M", "~10 GB", "~12 GB")
    models.add_row("4", "Qwen3-8B-Instruct Q8_0", "~8 GB", "~10 GB")
    console.print(models)
    
    choice = Prompt.ask("Select model", choices=["1", "2", "3", "4"], default="1")
    
    model_info = {
        "1": ("Qwen/Qwen3-32B-Instruct-GGUF", "Qwen3-32B-Instruct-Q4_K_M.gguf"),
        "2": ("Qwen/Qwen3-32B-Instruct-GGUF", "Qwen3-32B-Instruct-Q5_K_M.gguf"),
        "3": ("Qwen/Qwen3-14B-Instruct-GGUF", "Qwen3-14B-Instruct-Q5_K_M.gguf"),
        "4": ("Qwen/Qwen3-8B-Instruct-GGUF", "Qwen3-8B-Instruct-Q8_0.gguf"),
    }
    
    repo, filename = model_info[choice]
    model_dir = Path(os.environ.get("ARIASKA_MODEL_DIR", "/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    target = model_dir / filename
    if target.exists():
        console.print(f"[yellow]Model already exists: {target}[/yellow]")
        if not Confirm.ask("Re-download?", default=False):
            return
    
    console.print(f"[cyan]Downloading {filename}...[/cyan]")
    console.print("[dim]This may take 10-30 minutes depending on connection speed.[/dim]")
    
    subprocess.run([
        sys.executable, "-m", "huggingface_hub", "download",
        repo, filename,
        "--local-dir", str(model_dir),
        "--local-dir-use-symlinks", "False",
    ])


def view_llm_logs() -> None:
    """View LLM server logs."""
    log_path = Path("/var/log/ariaska_llm.log")
    if not log_path.exists():
        console.print("[yellow]No log file found at /var/log/ariaska_llm.log[/yellow]")
        return
    
    try:
        result = subprocess.run(["tail", "-50", str(log_path)], capture_output=True, text=True)
        console.print(Panel(result.stdout[-3000:] if result.stdout else "Empty", title="LLM Server Logs (last 50 lines)", border_style="dim"))
    except Exception as e:
        console.print(f"[red]Error reading logs: {e}[/red]")


def show_training_history() -> None:
    """Show training history from artifacts."""
    console.print(Rule("[bold]Training History[/bold]", style="bright_yellow"))
    
    artifacts_dir = _PROJECT_ROOT / "artifacts"
    if not artifacts_dir.exists():
        console.print("[dim]No artifacts directory found[/dim]")
        return
    
    # Find result files
    result_files = sorted(artifacts_dir.glob("*_results.json"), reverse=True)
    if not result_files:
        console.print("[dim]No training results found[/dim]")
        return
    
    # Show recent results
    table = Table(title=f"Recent Training Runs ({len(result_files)} total)", box=box.ROUNDED, border_style="yellow")
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Episodes", style="green")
    table.add_column("Avg Reward", style="yellow")
    table.add_column("Discoveries", style="magenta")
    table.add_column("Cost", style="red")
    
    for i, f in enumerate(result_files[:15], 1):
        try:
            data = json.loads(f.read_text())
            
            # Extract metrics (handle various formats)
            episodes = data.get("episodes", data.get("total_episodes", "?"))
            avg_reward = data.get("avg_reward", data.get("mean_reward", "?"))
            if isinstance(avg_reward, float):
                avg_reward = f"{avg_reward:.1f}"
            discoveries = data.get("total_discoveries", data.get("discoveries", "?"))
            cost = data.get("total_cost_usd", data.get("cost_usd", 0))
            if isinstance(cost, (int, float)):
                cost = f"${cost:.3f}"
            
            # Parse date from filename
            name = f.stem
            date_str = "—"
            if "engagement_" in name:
                parts = name.split("_")
                if len(parts) >= 2:
                    date_str = parts[1][:8] if len(parts[1]) >= 8 else parts[1]
            
            table.add_row(str(i), f.name, date_str, str(episodes), str(avg_reward), str(discoveries), str(cost))
        except Exception:
            table.add_row(str(i), f.name, "—", "?", "?", "?", "?")
    
    console.print(table)
    
    # Cost summary
    total_cost = 0.0
    for f in result_files:
        try:
            data = json.loads(f.read_text())
            total_cost += float(data.get("total_cost_usd", data.get("cost_usd", 0)))
        except Exception:
            pass
    
    if total_cost > 0:
        console.print(f"\n[bold]Total API cost across all runs: [red]${total_cost:.2f}[/red][/bold]")
    
    # Option to view details
    if result_files and Confirm.ask("\nView detailed results?", default=False):
        idx = IntPrompt.ask("Which # to view?", default=1)
        if 1 <= idx <= len(result_files):
            data = json.loads(result_files[idx - 1].read_text())
            console.print_json(json.dumps(data, indent=2, default=str)[:5000])


def run_tests() -> None:
    """Run the test suite."""
    console.print(Rule("[bold]Test Suite[/bold]", style="bright_green"))
    
    mode = Prompt.ask(
        "Test mode",
        choices=["quick", "full", "coverage", "specific"],
        default="quick"
    )
    
    base_env = dict(os.environ)
    base_env["ARIASKA_DRY_RUN"] = "1"
    base_env["PYTHONPATH"] = str(_PROJECT_ROOT)
    
    if mode == "quick":
        cmd = [str(_PROJECT_ROOT / ".venv/bin/pytest"), "tests/", "-x", "--tb=short", "-q", "--timeout=120",
               "--ignore=tests/test_online_makes_mentor_calls.py"]
    elif mode == "full":
        cmd = [str(_PROJECT_ROOT / ".venv/bin/pytest"), "tests/", "--tb=short", "-q", "--timeout=120",
               "--ignore=tests/test_online_makes_mentor_calls.py"]
    elif mode == "coverage":
        cmd = [str(_PROJECT_ROOT / ".venv/bin/pytest"), "tests/", "--tb=short", "-q", "--timeout=120",
               "--cov=core", "--cov-report=term-missing",
               "--ignore=tests/test_online_makes_mentor_calls.py"]
    else:
        test_path = Prompt.ask("Test file/pattern", default="tests/")
        cmd = [str(_PROJECT_ROOT / ".venv/bin/pytest"), test_path, "-v", "--tb=short", "--timeout=120"]
    
    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    subprocess.run(cmd, env=base_env, cwd=str(_PROJECT_ROOT))


def system_diagnostics() -> None:
    """Run comprehensive system diagnostics."""
    console.print(Rule("[bold]System Diagnostics[/bold]", style="bright_red"))
    
    env = detect_environment()
    console.print(show_system_panel(env))
    
    # Check all critical imports
    console.print("\n[bold]Import Health Check:[/bold]")
    critical_modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("rich", "Rich"),
        ("openai", "OpenAI SDK"),
        ("llama_cpp", "llama-cpp-python"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "SentenceTransformers"),
    ]
    
    for module, name in critical_modules:
        try:
            __import__(module)
            version = getattr(__import__(module), "__version__", "?")
            console.print(f"  [green]✓[/green] {name}: {version}")
        except ImportError:
            console.print(f"  [red]✗[/red] {name}: not installed")
    
    # Check core modules
    console.print("\n[bold]Core Module Health:[/bold]")
    core_modules = [
        "core.gpt_manager",
        "core.feature_flags",
        "core.runtime_flags",
        "core.llm.local_llm_provider",
        "core.llm.model_router",
        "core.llm.budget_manager",
        "core.llm.smart_mentor",
        "core.algorithms.ppo_agent",
        "core.models.state_encoder",
    ]
    
    for mod in core_modules:
        try:
            __import__(mod)
            console.print(f"  [green]✓[/green] {mod}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {mod}: {e}")
    
    # Disk usage
    console.print("\n[bold]Disk Usage:[/bold]")
    for d in ["data/", "models/", "artifacts/", "traces/", "postmortems/"]:
        p = _PROJECT_ROOT / d
        if p.exists():
            try:
                result = subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True, timeout=10)
                size = result.stdout.split()[0] if result.stdout else "?"
                console.print(f"  {d}: {size}")
            except Exception:
                console.print(f"  {d}: ?")


def show_advanced_config() -> None:
    """Advanced configuration submenu."""
    console.print(Rule("[bold]Advanced Configuration[/bold]", style="bright_cyan"))
    
    menu = Table(show_header=False, box=None, padding=(0, 1))
    menu.add_column("Key", style="bold yellow", width=4)
    menu.add_column("Action", style="white")
    menu.add_row("1", "View Current Feature Flags")
    menu.add_row("2", "Model Routing Table")
    menu.add_row("3", "Budget Manager Status")
    menu.add_row("4", "PPO Hyperparameters")
    menu.add_row("5", "Environment Variables")
    menu.add_row("b", "Back")
    console.print(Panel(menu, border_style="dim"))
    
    choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5", "b"], default="b")
    
    if choice == "1":
        try:
            from core.feature_flags import FeatureFlags
            flags = FeatureFlags()
            table = Table(title="Active Feature Flags", box=box.SIMPLE)
            table.add_column("Flag", style="cyan")
            table.add_column("Value", style="white")
            for attr in sorted(dir(flags)):
                if not attr.startswith("_") and attr != "profile":
                    val = getattr(flags, attr, None)
                    if isinstance(val, (bool, int, str)):
                        style = "green" if val else "red"
                        table.add_row(attr, Text(str(val), style=style))
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif choice == "2":
        try:
            from core.llm.model_router import ModelRouter
            router = ModelRouter.from_flags()
            table = Table(title="Model Routing Table", box=box.SIMPLE)
            table.add_column("Model", style="cyan")
            table.add_column("Tier", style="yellow")
            table.add_column("→ Provider", style="green")
            
            test_models = [
                "gpt-5.2-nano", "gpt-5.2-mini", "gpt-5.2", "gpt-5.2-codex",
                "gpt-5-nano", "gpt-5-mini",
            ]
            for m in test_models:
                decision = router.route(m, "tactical")
                table.add_row(m, decision.tier, f"{decision.provider} ({decision.reason})")
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif choice == "3":
        try:
            from core.llm.budget_manager import BudgetManagerV2
            bm = BudgetManagerV2(episode_id="diagnostic")
            stats = bm.get_stats()
            console.print_json(json.dumps(stats, indent=2, default=str))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif choice == "4":
        try:
            from core.algorithms.ppo_agent import PPOConfig
            config = PPOConfig()
            table = Table(title="PPO Hyperparameters", box=box.SIMPLE)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")
            for attr in sorted(dir(config)):
                if not attr.startswith("_"):
                    val = getattr(config, attr, None)
                    if isinstance(val, (bool, int, float, str, list, tuple)):
                        table.add_row(attr, str(val))
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif choice == "5":
        table = Table(title="Ariaska Environment Variables", box=box.SIMPLE)
        table.add_column("Variable", style="cyan")
        table.add_column("Value", style="white")
        ariaska_vars = sorted(
            (k, v) for k, v in os.environ.items()
            if k.startswith(("ARIASKA_", "FF_", "MC_", "OPENAI_", "MENTOR_"))
        )
        for k, v in ariaska_vars:
            display_v = v if "KEY" not in k else v[:8] + "..." if v else "(empty)"
            table.add_row(k, display_v)
        if not ariaska_vars:
            table.add_row("[dim]None set[/dim]", "")
        console.print(table)


def resume_last_training(env: Dict[str, Any]) -> None:
    """Resume from the last training trace."""
    traces_dir = _PROJECT_ROOT / "traces"
    if not traces_dir.exists():
        console.print("[yellow]No traces directory found[/yellow]")
        return
    
    trace_files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
    if not trace_files:
        console.print("[yellow]No trace files found[/yellow]")
        return
    
    console.print(f"[cyan]Found {len(trace_files)} trace files[/cyan]")
    console.print(f"Most recent: [bold]{trace_files[0].name}[/bold]")
    
    if Confirm.ask("Replay this trace?", default=True):
        cmd = [sys.executable, "ariaska_cli.py", "replay", str(trace_files[0]), "--verbose"]
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT))


def main() -> None:
    """Main menu loop."""
    console.clear()
    console.print(BANNER)
    
    env = detect_environment()
    console.print(show_system_panel(env))
    
    while True:
        try:
            choice = show_main_menu()
            
            if choice == "q":
                console.print("\n[bold red]Ariaska offline.[/bold red]\n")
                break
            
            elif choice == "1":
                config = configure_smart_train(env)
                if config:
                    launch_training(config, env)
            
            elif choice == "2":
                # Quick train — minimal config
                config = TrainingConfig(steps=200)
                config.target_ip = Prompt.ask("🎯 Target IP", default="10.10.10.10")
                if env["gpu_available"] or env["local_llm_running"]:
                    config.compute = "auto"
                launch_training(config, env)
            
            elif choice == "3":
                config = TrainingConfig(mode="ctf", ctf_flag=True, steps=150)
                config.target_ip = Prompt.ask("🎯 Target IP", default="10.10.10.10")
                config.compute = "auto" if env["gpu_available"] else "cloud"
                launch_training(config, env)
            
            elif choice == "4":
                config = TrainingConfig(mode="htb", ctf_flag=True, steps=300)
                config.target_ip = Prompt.ask("🎯 Target IP", default="10.10.10.10")
                config.compute = "auto" if env["gpu_available"] else "cloud"
                launch_training(config, env)
            
            elif choice == "5":
                resume_last_training(env)
            
            elif choice == "6":
                show_gpu_manager(env)
            
            elif choice == "7":
                show_training_history()
            
            elif choice == "8":
                run_tests()
            
            elif choice == "9":
                system_diagnostics()
            
            elif choice == "0":
                show_advanced_config()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — returning to menu[/yellow]")
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()
