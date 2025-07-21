# enhanced_training_with_dashboard.py - ARIASKA Enhanced Training with Real-time Dashboard
# 🎯 Visual Training System | 📊 Agent Monitoring | 🔄 Coordination Enhancement

import os
import sys
import time
import json
import threading
import numpy as np
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel

# Import our enhanced dashboard
from core.ui.enhanced_agent_dashboard import EnhancedAgentDashboard

# Import agents and training components
from core.agents.red_agent import RedAgent  
from core.agents.blue_agent import BlueAgent
from core.agents.orion_agent import OrionAgent
from core.agents.scout_agent import ScoutAgent
from core.agents.shadow_agent import ShadowAgent
from core.environment.cyber_environment import CyberEnvironment
from core.multiagent.agent_manager import AgentManager
from core.multiagent.memory_router import MemoryRouter
from core.utils.stats_monitor import StatsMonitor

console = Console()

class EnhancedTrainingWithDashboard:
    """Enhanced training system with real-time agent dashboard and improved coordination."""
    
    def __init__(self, verbosity: str = "standard", use_dashboard: bool = True):
        self.verbosity = verbosity
        self.use_dashboard = use_dashboard
        self.episode_counter = 0
        self.total_steps = 0
        
        console.print(Panel.fit(
            "🧠 ARIASKA Enhanced Training System v2.0\n"
            "✨ Real-time Agent Dashboard | 🤖 Improved Coordination | 📊 Enhanced Visibility",
            style="bold blue"
        ))
        
        # Initialize dashboard
        if self.use_dashboard:
            self.dashboard = EnhancedAgentDashboard(update_interval=1.0)
            console.print("[green]✓ Enhanced agent dashboard initialized[/green]")
        
        # Initialize core components
        self.memory_router = MemoryRouter()
        self.stats_monitor = StatsMonitor()
        
        # Initialize environment
        console.print("[cyan]Initializing cyber environment...[/cyan]")
        self.env = CyberEnvironment(
            agent_manager=None,  # Will be set after agent manager creation
            defer_reset=True
        )
        
        # Initialize agent manager
        console.print("[cyan]Initializing agent manager...[/cyan]")
        self.agent_manager = AgentManager(
            memory_router=self.memory_router,
            verbosity=verbosity
        )
        
        # Set environment reference in agent manager
        self.env.agent_manager = self.agent_manager
        
        # Initialize all agents
        self._initialize_agents()
        
        # Setup coordination enhancements
        self._setup_enhanced_coordination()
        
        # Training state
        self.is_training = False
        self.dashboard_thread = None
        
        console.print("[green]✓ Enhanced training system ready![/green]")
    
    def _initialize_agents(self):
        """Initialize all agents with enhanced configurations."""
        console.print("[cyan]Initializing enhanced agents...[/cyan]")
        
        # Initialize RedAgent with enhanced neural networks
        self.red_agent = RedAgent(
            agent_id="RedAgent",
            role="OffensiveOperator", 
            agent_manager=self.agent_manager,
            memory_router=self.memory_router,
            verbosity=self.verbosity
        )
        
        # Initialize BlueAgent with improved defense
        self.blue_agent = BlueAgent(
            agent_id="BlueAgent",
            role="DefensiveOperator",
            agent_manager=self.agent_manager,
            memory_router=self.memory_router,
            verbosity=self.verbosity
        )
        
        # Initialize OrionAgent as strategic overseer
        self.orion_agent = OrionAgent(
            agent_id="OrionAgent",
            role="StrategicOverseer",
            agent_manager=self.agent_manager,
            memory_router=self.memory_router,
            verbosity=self.verbosity
        )
        
        # Initialize ScoutAgent for reconnaissance
        self.scout_agent = ScoutAgent(
            agent_id="ScoutAgent", 
            role="ReconSpecialist",
            agent_manager=self.agent_manager,
            memory_router=self.memory_router,
            verbosity=self.verbosity
        )
        
        # Initialize ShadowAgent for stealth monitoring
        self.shadow_agent = ShadowAgent(
            agent_id="ShadowAgent",
            role="StealthMonitor", 
            agent_manager=self.agent_manager,
            memory_router=self.memory_router,
            verbosity=self.verbosity
        )
        
        # Register all agents with agent manager
        self.agent_manager.register_agent(self.red_agent)
        self.agent_manager.register_agent(self.blue_agent) 
        self.agent_manager.register_agent(self.orion_agent)
        self.agent_manager.register_agent(self.scout_agent)
        self.agent_manager.register_agent(self.shadow_agent)
        
        # Set environment references
        for agent in [self.red_agent, self.blue_agent, self.orion_agent, self.scout_agent, self.shadow_agent]:
            agent.env = self.env
        
        console.print("[green]✓ All agents initialized and registered[/green]")
    
    def _setup_enhanced_coordination(self):
        """Setup enhanced coordination between agents."""
        console.print("[cyan]Setting up enhanced agent coordination...[/cyan]")
        
        # Setup OrionAgent as overseer for all other agents
        self.orion_agent.register_subordinate(self.red_agent)
        self.orion_agent.register_subordinate(self.blue_agent)
        self.orion_agent.register_subordinate(self.scout_agent)
        self.orion_agent.register_subordinate(self.shadow_agent)
        
        # Setup inter-agent references for improved coordination
        self.red_agent.scout = self.scout_agent
        self.red_agent.shadow = self.shadow_agent
        self.red_agent.orion = self.orion_agent
        
        self.scout_agent.red_agent = self.red_agent
        self.scout_agent.shadow_agent = self.shadow_agent
        self.scout_agent.orion_agent = self.orion_agent
        
        self.shadow_agent.red_agent = self.red_agent
        self.shadow_agent.scout_agent = self.scout_agent
        self.shadow_agent.orion_agent = self.orion_agent
        
        console.print("[green]✓ Enhanced coordination established[/green]")
    
    def _update_dashboard(self):
        """Update dashboard with current agent states."""
        if not self.use_dashboard:
            return
        
        # Update each agent's status in dashboard
        agents = [
            ("RedAgent", self.red_agent),
            ("BlueAgent", self.blue_agent), 
            ("OrionAgent", self.orion_agent),
            ("ScoutAgent", self.scout_agent),
            ("ShadowAgent", self.shadow_agent)
        ]
        
        for agent_id, agent in agents:
            status_data = self._extract_agent_status(agent)
            self.dashboard.update_agent_status(agent_id, status_data)
        
        # Update coordination status
        coordination_data = self._extract_coordination_status()
        self.dashboard.update_coordination_status(coordination_data)
        
        # Update dashboard display
        self.dashboard.update_dashboard()
    
    def _extract_agent_status(self, agent) -> Dict[str, Any]:
        """Extract status information from an agent."""
        try:
            status = {
                "action": getattr(agent, "last_action", "No recent action"),
                "reasoning": getattr(agent, "last_reasoning", "No reasoning available"),
                "reward": getattr(agent, "last_reward", 0.0),
                "epsilon": getattr(agent, "epsilon", 0.0),
                "confidence": getattr(agent, "confidence", 0.0),
                "status": "ACTIVE" if self.is_training else "IDLE",
                "phase": getattr(agent, "current_phase", "unknown"),
                "step_count": getattr(agent, "step_counter", 0),
                "success_rate": getattr(agent, "success_rate", 0.0),
                "gpt_calls": getattr(agent, "gpt_calls", 0)
            }
            
            # Add neural state for agents with neural networks
            if hasattr(agent, "policy_net") and hasattr(agent, "target_net"):
                status["neural_state"] = {
                    "loss": getattr(agent, "last_loss", 0.0),
                    "learning_rate": getattr(agent, "learning_rate", 0.001)
                }
            
            return status
        except Exception as e:
            console.print(f"[yellow]Warning: Error extracting status for {agent.agent_id}: {e}[/yellow]")
            return {
                "action": "Error retrieving status",
                "reasoning": f"Status extraction failed: {e}",
                "reward": 0.0,
                "status": "ERROR"
            }
    
    def _extract_coordination_status(self) -> Dict[str, Any]:
        """Extract coordination status from OrionAgent."""
        try:
            return {
                "directives_active": len(getattr(self.orion_agent, "active_directives", [])),
                "global_strategy": getattr(self.orion_agent, "global_strategy", "balanced"),
                "coherence_score": getattr(self.orion_agent, "coherence_score", 0.5),
                "crisis_interventions": getattr(self.orion_agent, "crisis_interventions", 0),
                "orion_insights": getattr(self.orion_agent, "last_strategic_insight", "Monitoring systems..."),
                "team_efficiency": self._calculate_team_efficiency()
            }
        except Exception as e:
            console.print(f"[yellow]Warning: Error extracting coordination status: {e}[/yellow]")
            return {
                "directives_active": 0,
                "global_strategy": "unknown",
                "coherence_score": 0.0,
                "crisis_interventions": 0,
                "orion_insights": f"Error: {e}",
                "team_efficiency": 0.0
            }
    
    def _calculate_team_efficiency(self) -> float:
        """Calculate overall team efficiency."""
        try:
            agents = [self.red_agent, self.blue_agent, self.scout_agent, self.shadow_agent]
            efficiency_scores = []
            
            for agent in agents:
                if hasattr(agent, "success_rate"):
                    efficiency_scores.append(agent.success_rate)
                elif hasattr(agent, "last_reward"):
                    # Convert reward to efficiency score (normalize to 0-1)
                    efficiency_scores.append(max(0.0, min(1.0, agent.last_reward / 50.0)))
            
            return sum(efficiency_scores) / max(1, len(efficiency_scores))
        except Exception:
            return 0.0
    
    def _dashboard_update_loop(self):
        """Background thread for dashboard updates."""
        while self.is_training:
            try:
                self._update_dashboard()
                time.sleep(self.dashboard.update_interval)
            except Exception as e:
                console.print(f"[red]Dashboard update error: {e}[/red]")
                break
    
    def train_with_dashboard(self, num_episodes: int = 10, max_steps: int = 100):
        """Run training with real-time dashboard monitoring."""
        console.print(Panel(
            f"🚀 Starting Enhanced Training\n"
            f"Episodes: {num_episodes} | Max Steps: {max_steps}\n"
            f"Dashboard: {'Enabled' if self.use_dashboard else 'Disabled'}",
            style="bold green"
        ))
        
        self.is_training = True
        
        # Start dashboard
        if self.use_dashboard:
            self.dashboard.start_live_dashboard()
            # Start dashboard update thread
            self.dashboard_thread = threading.Thread(target=self._dashboard_update_loop, daemon=True)
            self.dashboard_thread.start()
        
        try:
            for episode in range(num_episodes):
                self.episode_counter = episode + 1
                console.print(f"\n[bold blue]🎯 Episode {self.episode_counter}/{num_episodes}[/bold blue]")
                
                # Reset environment and agents
                state = self.env.reset()
                self._reset_all_agents()
                
                episode_rewards = []
                episode_success = 0
                
                for step in range(max_steps):
                    self.total_steps += 1
                    
                    # OrionAgent strategic oversight
                    orion_result = self.orion_agent.simulate_step(
                        episode=self.episode_counter,
                        step=step + 1,
                        shared_context=state
                    )
                    
                    # RedAgent primary action
                    red_result = self.red_agent.simulate_step(
                        episode=self.episode_counter,
                        step=step + 1,
                        shared_context=state
                    )
                    
                    # BlueAgent defensive response
                    blue_result = self.blue_agent.simulate_step(
                        episode=self.episode_counter,
                        step=step + 1,
                        shared_context=state
                    )
                    
                    # ScoutAgent reconnaissance
                    scout_result = self.scout_agent.simulate_step(
                        episode=self.episode_counter,
                        step=step + 1,
                        shared_context=state
                    )
                    
                    # ShadowAgent stealth monitoring
                    shadow_result = self.shadow_agent.simulate_step(
                        episode=self.episode_counter,
                        step=step + 1,
                        shared_context=state
                    )
                    
                    # Process results and update environment
                    combined_reward = (
                        red_result.get("reward", 0.0) +
                        blue_result.get("reward", 0.0) +
                        orion_result.get("reward", 0.0) +
                        scout_result.get("reward", 0.0) +
                        shadow_result.get("reward", 0.0)
                    )
                    
                    episode_rewards.append(combined_reward)
                    
                    if combined_reward > 50:  # Success threshold
                        episode_success += 1
                    
                    # Update agent last action/reward for dashboard
                    self.red_agent.last_action = red_result.get("action", "No action")
                    self.red_agent.last_reward = red_result.get("reward", 0.0)
                    
                    # Update state for next iteration
                    state = self.env.get_global_state()
                    
                    # Brief pause to allow dashboard updates
                    time.sleep(0.1)
                
                # Episode summary
                total_reward = sum(episode_rewards)
                success_rate = episode_success / max_steps
                
                console.print(f"[green]Episode {self.episode_counter} complete: "
                            f"Total Reward: {total_reward:.2f}, Success Rate: {success_rate:.1%}[/green]")
                
                # Update statistics
                self.stats_monitor.log_episode_result(total_reward, max_steps, success_rate)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Training error: {e}[/red]")
        finally:
            self.is_training = False
            
            # Stop dashboard
            if self.use_dashboard:
                self.dashboard.stop_dashboard()
            
            console.print("[blue]Enhanced training session completed[/blue]")
    
    def _reset_all_agents(self):
        """Reset all agents for new episode."""
        for agent in [self.red_agent, self.blue_agent, self.orion_agent, self.scout_agent, self.shadow_agent]:
            if hasattr(agent, "reset"):
                agent.reset()

def main():
    """Main execution function."""
    try:
        # Create and run enhanced training system
        trainer = EnhancedTrainingWithDashboard(
            verbosity="standard",
            use_dashboard=True
        )
        
        # Run training with real-time dashboard
        trainer.train_with_dashboard(
            num_episodes=5,
            max_steps=50
        )
        
    except Exception as e:
        console.print(f"[red]Main execution error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
