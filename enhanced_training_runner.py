#!/usr/bin/env python3
"""
Enhanced Training System Runner for ARIASKA_RL

This script provides direct execution and CLI integration for the enhanced unified training system.
It can be called directly or through ariaska_cli.py for seamless training operations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the enhanced training system
from enhanced_unified_training import EnhancedUnifiedTrainingSystem, main
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_enhanced_training(
    episodes: int = 100,
    max_steps: int = 50,
    target_ip: str = "192.168.1.100",
    enable_gpu: bool = False,
    curriculum_learning: bool = True,
    save_interval: int = 10,
    verbose: bool = True
) -> dict:
    """
    Run the enhanced training system with specified parameters.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        target_ip: Target IP address for simulation
        enable_gpu: Enable GPU acceleration if available
        curriculum_learning: Enable adaptive curriculum learning
        save_interval: Save model checkpoint every N episodes
        verbose: Enable verbose logging and console output
    
    Returns:
        Dictionary containing training results and metrics
    """
    if verbose:
        console.print(Panel(
            f"[bold cyan]🧠 ARIASKA_RL Enhanced Training System[/bold cyan]\n\n"
            f"[white]Configuration:[/white]\n"
            f"• Episodes: [green]{episodes}[/green]\n"
            f"• Max Steps: [yellow]{max_steps}[/yellow]\n"
            f"• Target: [blue]{target_ip}[/blue]\n"
            f"• GPU: [magenta]{'Enabled' if enable_gpu else 'CPU Only'}[/magenta]\n"
            f"• Curriculum: [cyan]{'Adaptive' if curriculum_learning else 'Fixed'}[/cyan]\n"
            f"• Save Interval: [white]{save_interval} episodes[/white]",
            title="🚀 Starting Enhanced Training",
            border_style="green"
        ))
    
    # Initialize the enhanced training system
    training_system = EnhancedUnifiedTrainingSystem(
        episodes=episodes,
        max_steps_per_episode=max_steps,
        target_ip=target_ip,
        enable_gpu=enable_gpu,
        curriculum_learning=curriculum_learning,
        save_interval=save_interval
    )
    
    try:
        # Execute training
        results = training_system.run_training()
        
        if verbose:
            console.print(Panel(
                f"[bold green]✅ Training Completed Successfully![/bold green]\n\n"
                f"[white]Results Summary:[/white]\n"
                f"• Episodes Completed: [green]{results.get('episodes_completed', 0)}[/green]\n"
                f"• Total Training Time: [blue]{results.get('total_training_time', 0):.1f}s[/blue]\n"
                f"• Final Average Score: [yellow]{results.get('final_score', 0):.2f}[/yellow]\n"
                f"• Coordination Score: [cyan]{results.get('final_coordination', 0):.2f}[/cyan]\n"
                f"• GPU Tokens Used: [magenta]{results.get('total_gpt_tokens', 0):,}[/magenta]",
                title="🎯 Training Results",
                border_style="cyan"
            ))
        
        return results
        
    except KeyboardInterrupt:
        if verbose:
            console.print("\n[yellow]⚠️ Training interrupted by user. Progress saved.[/yellow]")
        return {"status": "interrupted", "message": "Training interrupted by user"}
        
    except Exception as e:
        if verbose:
            console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        return {"status": "error", "message": str(e)}


def run_quick_test(target_ip: str = "192.168.1.100") -> dict:
    """
    Run a quick training test with minimal episodes for validation.
    """
    console.print("[yellow]🧪 Running quick training test...[/yellow]")
    
    return run_enhanced_training(
        episodes=5,
        max_steps=10,
        target_ip=target_ip,
        enable_gpu=False,
        curriculum_learning=False,
        save_interval=2,
        verbose=True
    )


def run_full_training(target_ip: str = "192.168.1.100", enable_gpu: bool = False) -> dict:
    """
    Run full-scale training with recommended settings.
    """
    console.print("[green]🚀 Starting full-scale training...[/green]")
    
    return run_enhanced_training(
        episodes=200,
        max_steps=100,
        target_ip=target_ip,
        enable_gpu=enable_gpu,
        curriculum_learning=True,
        save_interval=20,
        verbose=True
    )


if __name__ == "__main__":
    # Direct execution
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ARIASKA_RL Enhanced Training System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_training_runner.py --quick-test
  python enhanced_training_runner.py --full-training --gpu
  python enhanced_training_runner.py --episodes 50 --target 10.0.0.1
        """
    )
    
    # Training mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick-test', action='store_true', 
                           help='Run quick test (5 episodes, 10 steps)')
    mode_group.add_argument('--full-training', action='store_true',
                           help='Run full training (200 episodes, 100 steps)')
    
    # Configuration options
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--max-steps', type=int, default=50,
                       help='Maximum steps per episode (default: 50)')
    parser.add_argument('--target', type=str, default='192.168.1.100',
                       help='Target IP address (default: 192.168.1.100)')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save interval in episodes (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal console output')
    
    args = parser.parse_args()
    
    # Execute based on mode
    try:
        if args.quick_test:
            results = run_quick_test(args.target)
        elif args.full_training:
            results = run_full_training(args.target, args.gpu)
        else:
            # Custom training
            results = run_enhanced_training(
                episodes=args.episodes,
                max_steps=args.max_steps,
                target_ip=args.target,
                enable_gpu=args.gpu,
                curriculum_learning=not args.no_curriculum,
                save_interval=args.save_interval,
                verbose=not args.quiet
            )
        
        # Exit with appropriate code
        if results.get("status") == "error":
            sys.exit(1)
        elif results.get("status") == "interrupted":
            sys.exit(130)  # SIGINT exit code
        else:
            sys.exit(0)
            
    except Exception as e:
        console.print(f"[red]❌ Runner failed: {e}[/red]")
        sys.exit(1)
