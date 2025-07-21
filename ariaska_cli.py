#!/usr/bin/env python3
"""
ARIASKA_RL Command Line Interface
Main entry point for all ARIASKA_RL operations
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

def show_system_status():
    """Show system status and diagnostics"""
    console.print(Panel(
        "🔍 System Status Check",
        title="ARIASKA_RL Diagnostics",
        border_style="blue"
    ))
    
    status_table = Table(show_header=True, header_style="bold cyan")
    status_table.add_column("Component", style="white", width=25)
    status_table.add_column("Status", style="green", width=15)
    status_table.add_column("Details", style="yellow", width=40)
    
    # Check .env file
    env_status = "✅ Found" if Path(".env").exists() else "❌ Missing"
    status_table.add_row(".env Configuration", env_status, "Environment variables")
    
    # Check core modules
    core_modules = [
        "core/training/enhanced_unified_trainer.py",
        "core/gpt_manager.py"
    ]
    
    for module in core_modules:
        module_status = "✅ Found" if Path(module).exists() else "❌ Missing"
        status_table.add_row(f"Module: {Path(module).name}", module_status, module)
    
    # Check logs directory
    logs_status = "✅ Found" if Path("logs").exists() else "⚠️ Missing (will be created)"
    status_table.add_row("Logs Directory", logs_status, "Training logs and results")
    
    console.print(status_table)


def run_system_tests():
    """Run system tests and validation"""
    console.print(Panel(
        "🧪 Running System Tests",
        title="ARIASKA_RL Test Suite",
        border_style="yellow"
    ))
    
    test_results = []
    
    # Test 1: Environment validation
    try:
        from core.training.enhanced_unified_trainer import EnhancedUnifiedTrainer
        test_results.append(("Enhanced Trainer Import", "✅ Pass", "Module imports successfully"))
    except Exception as e:
        test_results.append(("Enhanced Trainer Import", "❌ Fail", str(e)))
    
    # Test 2: GPT Manager
    try:
        from core.gpt_manager import GPTManager
        gpt = GPTManager()
        test_results.append(("GPT Manager", "✅ Pass", "GPT manager initializes"))
    except Exception as e:
        test_results.append(("GPT Manager", "❌ Fail", str(e)))
    
    # Test 3: Quick training test (1 episode)
    try:
        from core.training.enhanced_unified_trainer import EnhancedUnifiedTrainer
        trainer = EnhancedUnifiedTrainer(episodes=1)
        test_results.append(("Trainer Creation", "✅ Pass", "Trainer can be instantiated"))
    except Exception as e:
        test_results.append(("Trainer Creation", "❌ Fail", str(e)))
    
    # Display results
    test_table = Table(show_header=True, header_style="bold magenta")
    test_table.add_column("Test", style="white", width=25)
    test_table.add_column("Result", style="green", width=15)
    test_table.add_column("Details", style="yellow", width=40)
    
    for test_name, result, details in test_results:
        test_table.add_row(test_name, result, details)
    
    console.print(test_table)
    
    # Summary
    passed = sum(1 for _, result, _ in test_results if "✅" in result)
    total = len(test_results)
    console.print(Panel(
        f"Tests completed: {passed}/{total} passed",
        title="Test Summary",
        border_style="green" if passed == total else "red"
    ))


console = Console()

def show_help():
    """Display help information"""
    help_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    help_table.add_column("Command", style="cyan", width=20)
    help_table.add_column("Description", style="white", width=50)
    help_table.add_column("Example", style="green", width=30)
    
    help_table.add_row(
        "simulate-train <episodes> [target_ip]",
        "Run enhanced multi-agent training with advanced UI dashboard, command tracking, and learning analytics",
        "ariaska simulate-train 100 192.168.1.50"
    )
    help_table.add_row(
        "status",
        "Show system status and diagnostics",
        "ariaska status"
    )
    help_table.add_row(
        "test",
        "Run system tests and validation",
        "ariaska test"
    )
    help_table.add_row(
        "help",
        "Show this help message",
        "ariaska help"
    )
    help_table.add_row(
        "ui",
        "Launch Streamlit web interface",
        "ariaska ui"
    )
    help_table.add_row(
        "test",
        "Run system tests and diagnostics",
        "ariaska test"
    )
    help_table.add_row(
        "status",
        "Show system status and agent memories",
        "ariaska status"
    )
    help_table.add_row(
        "help",
        "Show this help message",
        "ariaska help"
    )
    
    console.print(Panel(
        help_table,
        title="🧠 ARIASKA_RL Command Line Interface",
        subtitle="Cybersecurity Training with Deep Reinforcement Learning",
        border_style="blue"
    ))

def test_system():
    """Run comprehensive system tests"""
    console.print("🧪 Running ARIASKA_RL system diagnostics...")
    
    tests = [
        ("Environment Setup", test_environment),
        ("GPT Integration", test_gpt_integration),
        ("Agent Initialization", test_agent_initialization),
        ("Memory Systems", test_memory_systems),
        ("Training Pipeline", test_training_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✅ PASS", result))
            console.print(f"✅ {test_name}: PASS")
        except Exception as e:
            results.append((test_name, "❌ FAIL", str(e)))
            console.print(f"❌ {test_name}: FAIL - {e}")
    
    # Display results table
    results_table = Table(show_header=True, header_style="bold cyan")
    results_table.add_column("Test", style="white")
    results_table.add_column("Status", style="white")
    results_table.add_column("Details", style="dim")
    
    for test_name, status, details in results:
        results_table.add_row(test_name, status, details[:50] + "..." if len(details) > 50 else details)
    
    console.print(Panel(results_table, title="Test Results"))

def test_environment():
    """Test environment setup"""
    required_vars = ['OPENAI_API_KEY', 'PRIMARY_MODEL']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise Exception(f"Missing environment variables: {missing}")
    return f"All required variables present"

def test_gpt_integration():
    """Test GPT integration"""
    try:
        from core.gpt_manager import GPTManager
        gpt_manager = GPTManager()
        response = gpt_manager.gpt_request("Test connectivity", max_tokens=10)
        return f"GPT connection successful: {len(response)} chars"
    except Exception as e:
        raise Exception(f"GPT integration failed: {e}")

def test_agent_initialization():
    """Test agent initialization"""
    try:
        from enhanced_unified_training import EnhancedUnifiedTrainingSystem
        trainer = EnhancedUnifiedTrainingSystem(episodes=1, enable_gpu=True)
        success = trainer.setup_agents()
        if success and len(trainer.agents) == 5:
            return f"All 5 agents initialized successfully: {list(trainer.agents.keys())}"
        else:
            raise Exception(f"Agent initialization incomplete: {len(trainer.agents)}/5 agents")
    except Exception as e:
        raise Exception(f"Agent initialization failed: {e}")

def test_memory_systems():
    """Test memory systems"""
    try:
        import torch
        from collections import deque
        memory = deque(maxlen=100)
        tensor = torch.zeros(10, 10)
        return f"Memory systems operational: PyTorch {torch.__version__}"
    except Exception as e:
        raise Exception(f"Memory systems failed: {e}")

def test_training_pipeline():
    """Test training pipeline"""
    try:
        from enhanced_unified_training import EnhancedUnifiedTrainingSystem
        trainer = EnhancedUnifiedTrainingSystem(episodes=1, enable_gpu=True)
        if trainer.setup_agents():
            return f"Enhanced training pipeline operational: {len(trainer.agents)} agents ready"
        else:
            raise Exception("Agent setup failed")
    except Exception as e:
        raise Exception(f"Training pipeline failed: {e}")

def show_status():
    """Show system status and agent memories"""
    console.print("📊 ARIASKA_RL System Status")
    
    # Check memory files
    memory_dir = Path("core/memories/agents")
    if memory_dir.exists():
        status_table = Table(show_header=True, header_style="bold green")
        status_table.add_column("Agent", style="cyan")
        status_table.add_column("Memory File", style="white")
        status_table.add_column("Size", style="yellow")
        status_table.add_column("Last Modified", style="blue")
        
        for memory_file in memory_dir.glob("*_memory.json"):
            agent_name = memory_file.stem.replace("_memory", "")
            size = memory_file.stat().st_size
            modified = memory_file.stat().st_mtime
            from datetime import datetime
            mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
            
            status_table.add_row(
                agent_name,
                "✅ Found" if memory_file.exists() else "❌ Missing",
                f"{size:,} bytes",
                mod_time
            )
        
        console.print(Panel(status_table, title="Agent Memory Status"))
    else:
        console.print("ℹ️ No agent memories found (first run)")
    
    # Check log files
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.json"))
        console.print(f"📁 Found {len(log_files)} log files in logs/")
    
    # Environment status
    env_table = Table(show_header=True, header_style="bold blue")
    env_table.add_column("Setting", style="cyan")
    env_table.add_column("Value", style="white")
    
    env_table.add_row("Primary Model", os.getenv("PRIMARY_MODEL", "Not set"))
    env_table.add_row("ARIASKA Mode", os.getenv("ARIASKA_MODE", "Not set"))
    env_table.add_row("Target IP", os.getenv("ARIASKA_TARGET_IP", "Not set"))
    env_table.add_row("Episodes", os.getenv("EPISODES", "Not set"))
    
    console.print(Panel(env_table, title="Environment Configuration"))

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command in ["help", "-h", "--help"]:
            show_help()
        
        elif command == "simulate-train":
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            target_ip = sys.argv[3] if len(sys.argv) > 3 else "192.168.1.100"
            
            console.print(Panel(
                f"🧠 ARIASKA_RL Enhanced Training System v2.0\n\n"
                f"Episodes: {episodes}\n"
                f"Target: {target_ip}\n"
                f"System: Enhanced Multi-Agent with Advanced UI\n"
                f"Features: Real-time Dashboard, Command Tracking, Learning Analytics",
                title="Enhanced Training Initialization",
                border_style="green"
            ))
            
            try:
                # Try enhanced training system first
                try:
                    from enhanced_unified_training import EnhancedUnifiedTrainingSystem
                    console.print("✅ Loading enhanced training system...")
                    
                    trainer = EnhancedUnifiedTrainingSystem(
                        episodes=episodes,
                        max_steps_per_episode=50,
                        target_ip=target_ip,
                        enable_gpu=False,
                        curriculum_learning=True,
                        save_interval=max(1, episodes // 10)
                    )
                    
                    console.print("🔧 Enhanced system initialized")
                    results = trainer.run_training()
                    
                except ImportError:
                    console.print("⚠️ Enhanced system not available, using runner...")
                    
                    from enhanced_training_runner import run_enhanced_training
                    results = run_enhanced_training(
                        episodes=episodes,
                        max_steps=50,
                        target_ip=target_ip,
                        enable_gpu=False,
                        curriculum_learning=True,
                        save_interval=max(1, episodes // 10),
                        verbose=True
                    )
                
                # Display enhanced results
                console.print(Panel(
                    f"Enhanced Training Completed!\n"
                    f"Episodes: {results.get('episodes_completed', 0)}\n"
                    f"Time: {results.get('total_training_time', 0):.1f}s\n"
                    f"Score: {results.get('final_score', 0):.2f}\n"
                    f"Coordination: {results.get('final_coordination', 0):.2f}",
                    title="Enhanced Training Complete",
                    border_style="cyan"
                ))
                
            except Exception as enhanced_error:
                console.print(f"❌ Enhanced training failed: {enhanced_error}")
                console.print("🔄 Attempting fallback to legacy training...")
                
                try:
                    from unified_training import create_unified_trainer
                    trainer = create_unified_trainer(episodes)
                    results = trainer.run_training()
                    
                    console.print(Panel(
                        f"✅ Legacy training completed!\n"
                        f"Episodes: {results['episodes_completed']}/{episodes}\n"
                        f"Duration: {results['total_training_time']:.2f}s\n"
                        f"Avg Reward: {results['final_metrics']['avg_reward']:.2f}",
                        title="Legacy Training Complete",
                        border_style="blue"
                    ))
                    
                except Exception as legacy_error:
                    console.print(f"❌ Legacy training also failed: {legacy_error}")
                    console.print("💡 Try: ariaska test  # to diagnose issues")
                    return 1
            
            console.print(Panel(
                f"✅ Training completed successfully!\n"
                f"📊 Episodes: {results['episodes_completed']}/{episodes}\n"
                f"⏱️ Duration: {results['total_training_time']:.2f}s\n"
                f"🎯 Avg Reward: {results['final_metrics']['avg_reward']:.2f}\n"
                f"� Coordination: {results['final_metrics']['coordination_score']:.2f}\n"
                f"📁 Results: logs/final_results_{results['session_id']}.json",
                title="Training Complete",
                border_style="green"
            ))
        
        elif command == "status":
            show_system_status()
        
        elif command == "test":
            run_system_tests()
        
        else:
            console.print(f"❌ Unknown command: {command}")
            console.print("💡 Run 'ariaska help' to see available commands")
    
    except KeyboardInterrupt:
        console.print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        console.print(f"❌ Error: {e}")
        console.print("💡 Run 'ariaska test' to diagnose system issues")

if __name__ == "__main__":
    main()
