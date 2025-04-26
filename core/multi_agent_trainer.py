# core/multiagent/multi_agent_trainer.py — ARIASKA MultiAgentTrainer v11.5 APEX PRIME
# 🎮 Global Orchestration Loop | 👁 Orion Live Strategy | ⚡ Dynamic Agent Sync | ♾️ Smart Cycles

import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich import box

from core.multiagent.agent_manager import AgentManager
from core.utils.stats_monitor import StatsMonitor
from core.logic.chainbuilder import build_and_store_chain_multiagent
from core.teach.teach import TeachModule
from core.visualization.training_visualizer import TrainingVisualizer

console = Console()

def display_live_training_dashboard(agents, episode, step, rewards_history, llm_usage):
    """
    Display a live dashboard with agent stats, reward graph, and LLM usage.
    """
    # Agent stats table
    stats_table = Table(title=f"Episode {episode} | Step {step}", box=box.ROUNDED)
    stats_table.add_column("Agent")
    stats_table.add_column("Epsilon", justify="right")
    stats_table.add_column("Last Reward", justify="right")
    stats_table.add_column("Total Reward", justify="right")
    stats_table.add_column("Phase", justify="center")
    stats_table.add_column("LLM Calls", justify="right")
    for agent in agents:
        eps = getattr(agent, "epsilon", 0.0)
        last_r = getattr(agent, "last_reward", 0.0)
        total_r = sum(getattr(agent.stats_monitor, "agent_stats", {}).get(agent.agent_id, {}).get("rewards", [])) if hasattr(agent, "stats_monitor") else 0.0
        phase = getattr(agent, "current_mode", "N/A")
        llm_calls = getattr(agent.stats_monitor, "agent_stats", {}).get(agent.agent_id, {}).get("gpt_calls", 0) if hasattr(agent, "stats_monitor") else 0
        stats_table.add_row(
            getattr(agent, "agent_id", "N/A"),
            f"{eps:.3f}",
            f"{last_r:+.2f}",
            f"{total_r:+.2f}",
            str(phase),
            str(llm_calls)
        )

    # Reward/performance sparkline
    reward_panel = Panel(
        "Episode rewards: " + " ".join(f"{r:+.1f}" for r in rewards_history[-10:]) +
        f"\nAverage reward: {sum(rewards_history)/len(rewards_history):.2f}" if rewards_history else "",
        title="Reward Trend",
        border_style="green"
    )

    # LLM usage summary
    llm_table = Table(title="LLM Usage", box=box.ROUNDED)
    llm_table.add_column("Model")
    llm_table.add_column("Calls", justify="right")
    llm_table.add_column("Tokens", justify="right")
    for model, usage in llm_usage.items():
        llm_table.add_row(model, str(usage.get("calls", 0)), str(usage.get("tokens", 0)))

    # Compose dashboard
    dashboard = Panel(
        stats_table,
        title="Live Agent Stats",
        border_style="cyan"
    )
    console.print(dashboard)
    console.print(reward_panel)
    console.print(llm_table)

def log_phase_transition(prev_phase, new_phase):
    """
    Print a summary panel when the system shifts phase.
    """
    if prev_phase != new_phase:
        console.print(
            Panel(
                f"Phase transition: [yellow]{prev_phase}[/yellow] → [green]{new_phase}[/green]",
                title="Phase Transition",
                border_style="magenta"
            )
        )

class MultiAgentTrainer:
    def __init__(self, agent_manager=None, stats_monitor=None, memory_router=None, verbosity="standard", optimize_mode=False, steps=40):
        self.verbosity = verbosity
        self.steps = steps
        self.agent_manager = agent_manager or AgentManager(verbosity=verbosity)
        self.stats_monitor = stats_monitor or StatsMonitor()
        self.memory_router = memory_router
        self.teach = TeachModule()
        self.orion = self.agent_manager.get_agent("OrionAgent")
        self.global_step = 0
        self.sync_interval = 5
        self.strategy_refresh = 10
        self.verbosity = verbosity
        self.optimize_mode = optimize_mode
        self.token_usage = {}
        # Advanced curriculum tracking
        self.curriculum_level = 1
        self.curriculum_progress = 0.0
        self.performance_history = []
        self.visualization_insights = ""
        # Initialize visualizer with proper agents
        self.visualizer = TrainingVisualizer.get_instance(
            agents=[a.agent_id for a in self.agent_manager.all_agents()],
            max_history=100
        )
        self.visualizer.start_live_display()
        self.visualizer_update_interval = 3  # More frequent updates
        console.print(
            Panel.fit(
                "[bold cyan]🎮 MultiAgentTrainer v11.5 APEX Initialized — Global Sync Online[/bold cyan]"
            )
        )

    # ─────────────────────────────────────────────
    # 🌐 Unified Simulation-Training Loop
    # ─────────────────────────────────────────────
    def run_global_cycle(self, total_steps=50):
        console.rule("[bold magenta]♾️ ARIASKA Global Orchestration Loop Started")
        agents = self.agent_manager.all_agents()
        red = self.agent_manager.get_agent("RedAgent")
        blue = self.agent_manager.get_agent("BlueAgent")
        scout = self.agent_manager.get_agent("ScoutAgent")
        shadow = self.agent_manager.get_agent("ShadowAgent")

        # Orion pre-briefing phase
        if hasattr(self.orion, "generate_strategic_chain"):
            strategic_insight = self.orion.generate_strategic_chain({}, verbosity=self.verbosity)
            self.visualizer.set_global_gpt_insight(strategic_insight)

        while self.global_step < total_steps:
            cycle_start_time = time.time()
            console.print(
                f"[green]🚀 Global Step {self.global_step + 1}/{total_steps} | Curriculum Level: {self.curriculum_level}[/green]"
            )

            # Offensive & Defensive Agents always act
            red_result = red.simulate_train(episodes=1)
            blue_result = blue.simulate_train(episodes=1)
            
            # Track agent performance for curriculum adaptation
            self._track_agent_performance(red_result, blue_result)

            # Dynamic difficulty scaling based on performance
            self._update_curriculum()

            # Sync Scout & Shadow periodically
            if self.global_step % self.sync_interval == 0:
                scout_insight = scout.advise_phase({}, all_agents=agents)
                shadow_optimization = shadow.optimize_memory(target_agent_id="RedAgent", all_agents=agents)
                # Update visualization with insights
                if scout_insight:
                    self.visualization_insights = scout_insight
                    self.visualizer.set_global_gpt_insight(scout_insight)

            # Orion live strategic adjustments with coherence calculation
            if self.global_step % self.strategy_refresh == 0:
                orion_result = self.orion.apply_orion_strategic_adjustments(agents)
                if orion_result and hasattr(orion_result, 'get') and orion_result.get('coherence'):
                    self.visualizer.set_coherence_score(orion_result.get('coherence'))

            # Batch train core agents
            red.train_on_batch()
            blue.train_on_batch()

            # Monitor token usage
            self._track_token_usage()

            # Dynamic episode termination with rich visualization
            terminated_early = self._check_dynamic_termination()
            if terminated_early:
                self.visualizer.push_alert("Episode terminated early due to high detection risk", "warning")
                break

            # Update visualizer with rich agent and environment data
            if self.global_step % self.visualizer_update_interval == 0:
                # Get detailed agent data for visualization
                for agent in agents:
                    agent_data = self._get_agent_visualization_data(agent)
                    self.visualizer.update(agent_data=agent_data)
                
                # Get environment state for visualization
                if hasattr(red, 'env') and hasattr(red.env, 'get_global_state'):
                    env_state = red.env.get_global_state()
                    self.visualizer.update(env_state=env_state)

            # Calculate step execution time
            cycle_time = time.time() - cycle_start_time
            if cycle_time < 1.0 and self.verbosity == "standard":  # Don't slow down if fast
                time.sleep(max(0, 0.5 - cycle_time))  # Shorter sleep for better UX

            self.global_step += 1

        console.print(
            "[bold green]🏁 Global Cycle Completed — Proceeding to Post-Processing[/bold green]"
        )
        self.post_cycle_operations()

    # ─────────────────────────────────────────────
    # 📊 Performance Tracking & Curriculum Adaptation
    # ─────────────────────────────────────────────
    def _track_agent_performance(self, red_result, blue_result):
        """
        Track agent performance metrics for curriculum adaptation.
        Now uses a multi-metric approach that considers rewards, success rate,
        and stealth metrics for a more holistic performance assessment.
        """
        # Track Red Team performance (primary curriculum driver)
        if red_result and isinstance(red_result, dict):
            # Base reward tracking
            reward = red_result.get('reward', 0)
            self.performance_history.append(reward)
            
            # Advanced metrics tracking
            if not hasattr(self, 'advanced_metrics'):
                self.advanced_metrics = {
                    'success_rate': [],  # Percentage of successful actions
                    'stealth_score': [], # Average stealth score (inverse of detection risk)
                    'objective_completion': [], # Percentage of objectives completed
                    'token_efficiency': [] # Objectives achieved per token spent
                }
                
            # Extract and track advanced metrics
            if 'success_rate' in red_result:
                self.advanced_metrics['success_rate'].append(red_result['success_rate'])
                
            if 'detection_risk' in red_result:
                # Convert detection risk to stealth score (inverse relationship)
                stealth = max(0, 10 - red_result.get('detection_risk', 0))
                self.advanced_metrics['stealth_score'].append(stealth)
                
            if 'objectives_completed' in red_result and 'total_objectives' in red_result:
                completion_rate = red_result['objectives_completed'] / max(1, red_result['total_objectives'])
                self.advanced_metrics['objective_completion'].append(completion_rate)
                
            if 'gpt_tokens' in red_result and 'objectives_completed' in red_result:
                # Track how many objectives completed per token (efficiency)
                token_efficiency = red_result['objectives_completed'] / max(1, red_result['gpt_tokens']) * 1000
                self.advanced_metrics['token_efficiency'].append(token_efficiency)
            
            # Keep limited history for trend analysis
            if len(self.performance_history) > 20:
                self.performance_history.pop(0)
                
            # Keep same length for all advanced metrics
            for metric_name in self.advanced_metrics:
                if len(self.advanced_metrics[metric_name]) > 20:
                    self.advanced_metrics[metric_name].pop(0)

    def _update_curriculum(self):
        """
        Dynamically update curriculum difficulty using a composite performance score
        based on multiple metrics and a weighted approach that adapts to the curriculum level.
        """
        if len(self.performance_history) < 10:  # Need enough data points
            return
            
        # Calculate traditional performance trend (rewards)
        recent_avg = sum(self.performance_history[-5:]) / 5
        older_avg = sum(self.performance_history[-10:-5]) / 5
        reward_trend = (recent_avg / older_avg) if older_avg != 0 else 1.0
        
        # Calculate advanced metrics trends if available
        metric_trends = {}
        composite_score = 0
        valid_metrics = 0
        
        # Weight factors - higher curriculum levels value different metrics
        weights = {
            'reward': 1.0,
            'success_rate': 0.5 + (self.curriculum_level * 0.1),  # More important at higher levels
            'stealth_score': 0.3 + (self.curriculum_level * 0.15),  # Much more important at higher levels
            'objective_completion': 0.7 + (self.curriculum_level * 0.05),  # Consistently important
            'token_efficiency': 0.2 + (self.curriculum_level * 0.1)  # Increasingly important
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        for k in weights:
            weights[k] /= total_weight
        
        # Add reward trend to composite score
        composite_score += reward_trend * weights['reward']
        valid_metrics += 1
        
        # Process all advanced metrics if we have enough history
        if hasattr(self, 'advanced_metrics'):
            for metric_name, values in self.advanced_metrics.items():
                if len(values) >= 10:  # Need enough history
                    recent_metric_avg = sum(values[-5:]) / 5
                    older_metric_avg = sum(values[-10:-5]) / 5
                    
                    # Calculate trend (ratio of recent to older)
                    if older_metric_avg > 0:
                        trend = recent_metric_avg / older_metric_avg
                        metric_trends[metric_name] = trend
                        
                        # Add to composite score with appropriate weight
                        weight = weights.get(metric_name, 0.5)  # Default weight if not specified
                        composite_score += trend * weight
                        valid_metrics += 1
        
        # Normalize composite score
        if valid_metrics > 0:
            composite_score /= valid_metrics
        
        # Define thresholds for curriculum adjustments - more stringent at higher levels
        improvement_threshold = 1.15 - (self.curriculum_level * 0.025)  # Gets harder to improve
        regression_threshold = 0.85 + (self.curriculum_level * 0.025)   # More sensitive to regression
        
        # Log composite performance for monitoring
        if self.verbosity != "quiet":
            console.print(f"[cyan]📈 Composite Performance Score: {composite_score:.3f} " +
                         f"(Thresholds: +{improvement_threshold:.2f}/-{1-regression_threshold:.2f})[/cyan]")
        
        # Apply curriculum adjustments based on composite score
        if composite_score > improvement_threshold:
            # Significant improvement detected
            self.curriculum_progress += 0.25
            if self.curriculum_progress >= 1.0:
                # Level up!
                self.curriculum_level += 1
                self.curriculum_progress = 0.0
                console.print(f"[bold green]⬆️ Curriculum advanced to level {self.curriculum_level}![/bold green]")
                self.visualizer.push_alert(f"Curriculum advanced to level {self.curriculum_level}", "green")
                
                # Apply curriculum adjustments to agents and environment
                self._apply_curriculum_adjustments()
                
        elif composite_score < regression_threshold:
            # Significant regression detected - only apply if we've seen consistent problems
            if self.curriculum_level > 1:
                self.curriculum_progress -= 0.25
                if self.curriculum_progress <= -1.0:
                    self.curriculum_level -= 1
                    self.curriculum_progress = 0.0
                    console.print(f"[bold yellow]⬇️ Curriculum adjusted to level {self.curriculum_level}[/bold yellow]")
                    self.visualizer.push_alert(f"Curriculum reduced to level {self.curriculum_level}", "yellow")
                    
                    # Apply curriculum adjustments to agents and environment
                    self._apply_curriculum_adjustments()
                    
        # Reset performance history if curriculum level changed significantly
        if self.curriculum_level >= 4 and len(self.performance_history) > 15:
            # At higher levels, use more recent data for better responsiveness
            self.performance_history = self.performance_history[-10:]
            if hasattr(self, 'advanced_metrics'):
                for metric_name in self.advanced_metrics:
                    self.advanced_metrics[metric_name] = self.advanced_metrics[metric_name][-10:]

    def _apply_curriculum_adjustments(self):
        """
        Apply curriculum-based adjustments to the environment and agents
        based on the current curriculum level.
        
        Each level increases complexity and challenges in different dimensions:
        - Level 1: Basic environment with standard services
        - Level 2: More varied services and basic defense mechanisms
        - Level 3: Advanced defense mechanisms, IDS, and stealth requirements
        - Level 4: Complex network segmentation and adaptive blue team
        - Level 5: Full enterprise environment with deception technology
        """
        # Log curriculum adjustment
        self.logger.info(f"Applying curriculum adjustments for level {self.curriculum_level}")
        
        # Environment complexity adjustments
        env_config = {
            "curriculum_level": self.curriculum_level,
            "detection_sensitivity": min(0.2 + (self.curriculum_level * 0.15), 0.9),
            "defense_complexity": min(0.1 + (self.curriculum_level * 0.2), 0.95),
            "network_segmentation": min((self.curriculum_level - 1) * 0.25, 0.9) if self.curriculum_level > 1 else 0,
            "service_diversity": min(0.3 + (self.curriculum_level * 0.15), 0.9),
            "deception_enabled": self.curriculum_level >= 4,
            "adaptive_defense": self.curriculum_level >= 3,
        }
        
        # Update environment configuration
        if hasattr(self.env, 'update_curriculum_config'):
            self.env.update_curriculum_config(env_config)
        
        # Agent-specific curriculum adjustments
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'adapt_to_curriculum'):
                agent_config = {
                    "curriculum_level": self.curriculum_level,
                }
                
                # Red team specific adjustments
                if agent_name == "RedAgent":
                    agent_config.update({
                        "stealth_requirement": min(0.1 + (self.curriculum_level * 0.2), 0.9),
                        "exploit_complexity": min(0.2 + (self.curriculum_level * 0.15), 0.85),
                        "reporting_detail": min(0.3 + (self.curriculum_level * 0.15), 0.9),
                        "token_budget": 1000 + (self.curriculum_level * 500),  # More tokens at higher levels
                        "enable_strategy_refinement": self.curriculum_level >= 3,
                    })
                
                # Blue team specific adjustments
                elif agent_name == "BlueAgent":
                    agent_config.update({
                        "detection_capability": min(0.2 + (self.curriculum_level * 0.15), 0.9),
                        "response_speed": min(0.2 + (self.curriculum_level * 0.1), 0.8),
                        "countermeasure_complexity": min(0.1 + (self.curriculum_level * 0.2), 0.9),
                        "token_budget": 800 + (self.curriculum_level * 400),  # More tokens at higher levels
                        "enable_threat_learning": self.curriculum_level >= 2,
                    })
                
                # Scout agent specific adjustments
                elif agent_name == "ScoutAgent":
                    agent_config.update({
                        "reconnaissance_depth": min(0.3 + (self.curriculum_level * 0.15), 0.9),
                        "stealth_requirement": min(0.2 + (self.curriculum_level * 0.15), 0.85),
                        "information_quality": min(0.3 + (self.curriculum_level * 0.1), 0.8),
                        "token_budget": 600 + (self.curriculum_level * 300),  # More tokens at higher levels
                    })
                
                # Shadow agent specific adjustments
                elif agent_name == "ShadowAgent":
                    agent_config.update({
                        "persistence_complexity": min(0.1 + (self.curriculum_level * 0.2), 0.9),
                        "evasion_capability": min(0.2 + (self.curriculum_level * 0.15), 0.8),
                        "token_budget": 700 + (self.curriculum_level * 350),  # More tokens at higher levels
                    })
                
                # Orion agent specific adjustments 
                elif agent_name == "OrionAgent":
                    agent_config.update({
                        "strategic_horizon": 1 + self.curriculum_level,  # Longer-term planning
                        "coordination_complexity": min(0.2 + (self.curriculum_level * 0.2), 0.9),
                        "adaptation_rate": min(0.3 + (self.curriculum_level * 0.1), 0.8),
                        "token_budget": 1200 + (self.curriculum_level * 600),  # More tokens for strategic oversight
                        "enable_strategy_evolution": self.curriculum_level >= 2,
                    })
                
                # Apply the curriculum adjustments to the agent
                agent.adapt_to_curriculum(agent_config)
                
                # Log agent-specific curriculum updates
                self.logger.info(f"Updated {agent_name} curriculum parameters: {agent_config}")
        
        # Update strategy optimizer if available
        if hasattr(self, 'strategy_optimizer'):
            optimizer_config = {
                "exploration_rate": max(0.8 - (self.curriculum_level * 0.12), 0.2),  # Decrease exploration over time
                "learning_rate": max(0.1 - (self.curriculum_level * 0.015), 0.04),   # More conservative learning at higher levels
                "reward_horizon": 2 + self.curriculum_level,  # Longer-term reward considerations
                "token_optimization_weight": min(0.1 + (self.curriculum_level * 0.1), 0.5),  # Increasing focus on token efficiency
            }
            
            self.strategy_optimizer.update_config(optimizer_config)
            self.logger.info(f"Updated strategy optimizer parameters: {optimizer_config}")
            
        # Trigger environment domain randomization if at curriculum level 3+
        if self.curriculum_level >= 3 and hasattr(self.env, 'randomize_domain'):
            console.print("[bold cyan]🎲 Triggering environment domain randomization...[/bold cyan]")
            self.env.randomize_domain(intensity=0.2 + (self.curriculum_level - 3) * 0.2)
            
        # Notify visualization system of curriculum change for dashboard updates
        if hasattr(self, 'visualizer') and hasattr(self.visualizer, 'update_curriculum_display'):
            self.visualizer.update_curriculum_display(self.curriculum_level, env_config)
            
        console.print(f"[bold blue]🔄 Applied curriculum level {self.curriculum_level} adjustments[/bold blue]")

    # ─────────────────────────────────────────────
    # 📊 Visualization & Monitoring
    # ─────────────────────────────────────────────
    def _track_token_usage(self):
        """Track token usage across all agents"""
        for agent in self.agent_manager.all_agents():
            if hasattr(agent, "gpt_calls"):
                self.token_usage.setdefault(agent.agent_id, {"4o-mini": 0, "4.1-full": 0, "tokens": 0})
                for k in ["4o-mini", "4.1-full"]:
                    self.token_usage[agent.agent_id][k] += agent.gpt_calls.get(k, 0)
                # Track total tokens if available
                if hasattr(agent, "gpt_tokens"):
                    self.token_usage[agent.agent_id]["tokens"] += agent.gpt_tokens

    def _check_dynamic_termination(self):
        """Check conditions for dynamic episode termination"""
        if hasattr(self.agent_manager.get_agent("RedAgent"), "env") and \
           hasattr(self.agent_manager.get_agent("RedAgent").env, "detection_risk"):
            risk = self.agent_manager.get_agent("RedAgent").env.detection_risk
            if risk > 7.0:
                console.log(f"🔚 Ending episode early due to high risk: {risk:.2f}/10.0")
                return True
        return False

    def _get_agent_visualization_data(self, agent):
        """Get rich agent data for visualization"""
        data = {
            "agent_id": agent.agent_id,
            "step": self.global_step,
            "reward": getattr(agent, "last_reward", 0.0),
            "phase": getattr(agent, "current_mode", "N/A"),
            "epsilon": getattr(agent, "epsilon", 0.5),
            "command": getattr(agent, "last_action", "N/A"),
            "gpt_calls": sum(getattr(agent, "gpt_calls", {}).values()),
            "gpt_tokens": getattr(agent, "gpt_tokens", 0),
        }
        
        # Add LLM model-specific usage if available
        for model in ["seneca", "lily", "gpt"]:
            if hasattr(agent, f"{model}_calls"):
                data[f"{model}_calls"] = getattr(agent, f"{model}_calls")
                
        # Add average response time if available
        if hasattr(agent, "response_times") and agent.response_times:
            data["llm_response_time"] = sum(agent.response_times) / len(agent.response_times)
            
        return data

    # ─────────────────────────────────────────────
    # 🧠 Post-Cycle Intelligence & GPT Sync
    # ─────────────────────────────────────────────
    def post_cycle_operations(self):
        console.rule("[bold cyan]🧠 Post-Cycle Intelligence: Sync | Analyze | Optimize")

        # 1. Consolidate GPT Cache & Global Insights
        if self.memory_router:
            self.memory_router.consolidate_gpt_cache()
            self.memory_router.sync_global_insights()

        # 2. Snapshot All Memories
        self.save_snapshots()

        # 3. Generate Updated Attack Chains
        self.generate_attack_chains()

        # 4. Orion Deep Strategic Review with enhanced feedback
        if hasattr(self.orion, "analyze_training"):
            strategic_analysis = self.orion.analyze_training(self.agent_manager.all_agents())
            if strategic_analysis:
                console.print(Panel(strategic_analysis, title="Orion Strategic Analysis", border_style="magenta"))
                self.visualizer.set_global_gpt_insight(strategic_analysis[:200] + "..." if len(strategic_analysis) > 200 else strategic_analysis)

        # 5. Apply Orion's Final Adjustments with coherence score
        if hasattr(self.orion, "apply_orion_strategic_adjustments"):
            result = self.orion.apply_orion_strategic_adjustments(self.agent_manager.all_agents())
            if result and hasattr(result, 'get') and result.get('coherence'):
                self.visualizer.set_coherence_score(result.get('coherence'))

        # 6. Memory optimization with advanced threshold
        self.repair_memories(threshold=15 - self.curriculum_level)  # Adaptive threshold

        # 7. Save final visualization snapshot and create training report
        self.visualizer.save_visualization_snapshot()
        self.visualizer.create_training_report(self.global_step)

        console.print(
            "[green]✔ Post-cycle operations completed. ARIASKA is optimized and aligned.[/green]"
        )

    # ─────────────────────────────────────────────
    # 💾 Snapshot & GPT Cache Management
    # ─────────────────────────────────────────────
    def save_snapshots(self):
        console.print(
            "[cyan]📸 Saving Memory Snapshots & Syncing GPT Intelligence...[/cyan]"
        )
        if self.memory_router:
            self.memory_router.snapshot_all_memories()
            self.memory_router.consolidate_gpt_cache()

    # ─────────────────────────────────────────────
    # ♾️ GPT-Enhanced Autopilot Mode
    # ─────────────────────────────────────────────
    def run_autopilot(self, cycles=3):
        console.rule(
            "[bold magenta]♾️ ARIASKA Autopilot — Adaptive Multi-Agent Execution"
        )
        for cycle in range(1, cycles + 1):
            console.print(f"[green]🚀 Starting Cycle {cycle}/{cycles}[/green]")

            # Dynamic GPT-driven adjustments before each cycle
            if hasattr(self.orion, "update_global_strategy"):
                strategy = self.orion.update_global_strategy(
                    self.agent_manager.all_agents(), environment="dynamic"
                )
                self.visualizer.set_global_gpt_insight(strategy[:200] + "..." if len(strategy) > 200 else strategy)

            # Execute Simulation & Training with dynamic steps based on cycle
            steps = 5 + (cycle * 5)  # Scale up steps per cycle
            self.run_global_cycle(total_steps=steps)
            self.orchestrate_batch_training(batches=2 + (cycle // 2))

            # Mid-Cycle Memory Optimization with adaptive threshold
            threshold = max(5, 15 - (cycle * 2))  # Lower threshold as cycles progress
            self.repair_memories(threshold=threshold)

            # Post-Cycle Intelligence Sync
            self.post_cycle_operations()
            
            # Save curriculum state
            console.print(f"[cyan]📊 Curriculum Level: {self.curriculum_level} | Progress: {self.curriculum_progress:.2f}[/cyan]")
            
            # Save checkpoint after each cycle
            self.agent_manager.save_all_models(prefix=f"cycle_{cycle}")

        console.print(
            "[bold green]🏁 Autopilot Complete — ARIASKA Optimized & Standing By[/bold green]"
        )
        
        # Display final performance analytics
        self.display_autopilot_analytics()

    def display_autopilot_analytics(self):
        """Display comprehensive analytics after autopilot run"""
        console.rule("[bold blue]📊 ARIASKA Performance Analytics[/bold blue]")
        
        # Token usage table
        token_table = Table(title="GPT Token Usage", box=box.ROUNDED)
        token_table.add_column("Agent", style="cyan")
        token_table.add_column("4o-mini Calls", style="magenta")
        token_table.add_column("4.1-full Calls", style="yellow")
        token_table.add_column("Total Tokens", style="green")
        
        total_tokens = 0
        for agent_id, usage in self.token_usage.items():
            token_table.add_row(
                agent_id,
                str(usage.get("4o-mini", 0)),
                str(usage.get("4.1-full", 0)),
                str(usage.get("tokens", 0))
            )
            total_tokens += usage.get("tokens", 0)
            
        console.print(token_table)
        console.print(f"[bold green]Total Token Usage: {total_tokens:,} tokens[/bold green]")
        
        # Final curriculum level reached
        console.print(Panel(
            f"Final Curriculum Level: {self.curriculum_level}\n"
            f"Progress to Next Level: {self.curriculum_progress:.2f}/1.0",
            title="Learning Progress",
            border_style="green"
        ))

    # ─────────────────────────────────────────────
    # 🔗 Chain Generation with Orion Oversight
    # ─────────────────────────────────────────────
    def generate_attack_chains(self):
        console.print("[magenta]🔗 Synthesizing Multi-Agent Attack Chains...[/magenta]")
        build_and_store_chain_multiagent(self.agent_manager)

    # ─────────────────────────────────────────────
    # 🎓 Memory Optimization with GPT
    # ─────────────────────────────────────────────
    def repair_memories(self, threshold=15):
        """Enhanced memory repair with adaptive threshold and detailed stats"""
        console.print(f"[cyan]🧠 Optimizing agent memories (threshold={threshold})...[/cyan]")
        if hasattr(self.memory_router, "optimize_memories"):
            stats = self.memory_router.optimize_memories(threshold=threshold)
            if stats:
                console.print(Panel(
                    f"✅ Optimized {stats.get('total_optimized', 0)} memories\n"
                    f"🔄 Deduplicated {stats.get('deduplicated', 0)} entries\n"
                    f"❌ Removed {stats.get('removed', 0)} low-value memories\n"
                    f"⭐ Enhanced {stats.get('enhanced', 0)} high-value memories",
                    title="Memory Optimization Results",
                    border_style="green"
                ))
                
                # Update visualization with optimization stats
                self.visualizer.push_alert(
                    f"Memory optimization: {stats.get('total_optimized', 0)} memories processed",
                    "green"
                )
        else:
            console.print("[yellow]⚠ Memory router missing optimize_memories method[/yellow]")

    # ─────────────────────────────────────────────
    # 🚀 Diagnostic Execution (Smart CLI Mode)
    # ─────────────────────────────────────────────
    def orchestrate_simulation(self, episodes=10):
        """Main simulation loop with advanced dashboard and error handling."""
        visualizer = self.visualizer
        try:
            for ep in range(episodes):
                console.print(f"[bold cyan]Episode {ep+1}/{episodes}[/bold cyan]")
                self.agent_manager.simulate_all_agents(episodes=1, max_steps=40)
                # Update visualization after each episode
                if hasattr(self.agent_manager.get_agent("RedAgent"), "env") and \
                   hasattr(self.agent_manager.get_agent("RedAgent").env, "get_global_state"):
                    env_state = self.agent_manager.get_agent("RedAgent").env.get_global_state()
                    visualizer.update(env_state=env_state)
                if (ep + 1) % 5 == 0:  # More frequent snapshots
                    visualizer.save_visualization_snapshot()
                    visualizer.create_training_report(ep + 1)
                self.agent_manager.display_full_status()
                if (ep + 1) % 10 == 0:
                    self.agent_manager.save_all_models()
        except Exception as e:
            console.print(f"[bold red]❌ Simulation error: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            try:
                self.agent_manager.display_full_status()
            except:
                console.print("[red]❌ Status display also failed.[/red]")

    def orchestrate_batch_training(self, batches=5):
        """Batch train all agents with adaptive batch size based on curriculum level"""
        # Adjust batch size based on curriculum level
        adjusted_batches = batches + (self.curriculum_level - 1)
        console.print(f"[cyan]🧠 Batch training agents ({adjusted_batches} batches)...[/cyan]")
        self.agent_manager.batch_train_all(batches=adjusted_batches)


if __name__ == "__main__":
    console.rule("[bold cyan]🧪 ARIASKA MultiAgentTrainer Diagnostic Mode[/bold cyan]")
    trainer = MultiAgentTrainer()
    trainer.run_autopilot(cycles=2)

    # After autopilot, generate strategic report
    trainer.orion.generate_strategic_report(
        trainer.agent_manager.all_agents(), environment="dynamic"
    )

    console.print(
        "[blue]📊 Diagnostic run complete. Review logs for detailed insights.[/blue]"
    )
