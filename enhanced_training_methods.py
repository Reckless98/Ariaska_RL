# Continuation of enhanced_unified_training.py methods

def _process_action_result(self, action_result: Dict[str, Any], agent_name: str, 
                          agent_state: Dict[str, Any], current_phase: TrainingPhase) -> Dict[str, Any]:
    """Process and enhance action result with additional context."""
    command = action_result.get('command', '')
    
    # Replace target placeholder with actual target
    command = command.replace('{target}', self.target_ip)
    command = command.replace('{agent}', agent_name.lower())
    command = command.replace('{phase}', current_phase.value)
    
    # Add command classification
    command_type = self._classify_command_type(command)
    
    # Enhance with context
    enhanced_result = action_result.copy()
    enhanced_result.update({
        'command': command,
        'target': self.target_ip,
        'command_type': command_type,
        'agent_role': self.agent_definitions[agent_name]['role'],
        'phase': current_phase.value,
        'timestamp': time.time()
    })
    
    return enhanced_result

def _create_enhanced_dashboard(self) -> Layout:
    """Create ultra-detailed real-time training dashboard."""
    layout = Layout()
    
    # Main layout structure
    layout.split_column(
        Layout(name="header", size=8),
        Layout(name="main"),
        Layout(name="footer", size=6)
    )
    
    # Header with comprehensive training status
    layout["header"].update(self._create_comprehensive_header())
    
    # Main content with detailed agent information
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="center", ratio=2),
        Layout(name="right", ratio=1)
    )
    
    # Left: Detailed agent performance
    layout["left"].split_column(
        Layout(name="agent_details", ratio=3),
        Layout(name="command_tracking", ratio=2)
    )
    
    # Center: Real-time actions and coordination
    layout["center"].split_column(
        Layout(name="live_actions", ratio=2),
        Layout(name="coordination_matrix", ratio=2),
        Layout(name="learning_curves", ratio=1)
    )
    
    # Right: System metrics and alerts
    layout["right"].split_column(
        Layout(name="system_health", ratio=1),
        Layout(name="performance_alerts", ratio=1),
        Layout(name="current_targets", ratio=1)
    )
    
    # Fill all sections with rich content
    layout["agent_details"].update(self._create_detailed_agent_panel())
    layout["command_tracking"].update(self._create_command_tracking_panel())
    layout["live_actions"].update(self._create_live_actions_panel())
    layout["coordination_matrix"].update(self._create_enhanced_coordination_panel())
    layout["learning_curves"].update(self._create_detailed_learning_panel())
    layout["system_health"].update(self._create_system_health_panel())
    layout["performance_alerts"].update(self._create_performance_alerts_panel())
    layout["current_targets"].update(self._create_targets_panel())
    layout["footer"].update(self._create_enhanced_footer())
    
    return layout

def _create_comprehensive_header(self) -> Panel:
    """Create comprehensive header with all training information."""
    # Calculate progress and timing
    progress_pct = (self.current_episode / self.episodes * 100) if self.episodes > 0 else 0
    elapsed_time = time.time() - (self.episode_start_time or time.time())
    
    # Calculate ETA
    if self.current_episode > 0:
        avg_episode_time = self.total_training_time / self.current_episode
        remaining_episodes = self.episodes - self.current_episode
        eta_seconds = remaining_episodes * avg_episode_time
        eta_str = f"{eta_seconds / 60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
    else:
        eta_str = "Calculating..."
    
    # Get current phase and metrics
    current_phase = self.dashboard_state['current_phase']
    active_agents = len([agent for agent, health in self.dashboard_state['system_health'].items() if health])
    
    # Calculate real-time performance
    recent_rewards = list(self.real_time_metrics['reward_velocity'])[-10:]
    avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
    
    coord_score = self.coordination_matrix.mean()
    
    header_content = f"""[bold cyan]🧠 ARIASKA_RL Enhanced Training System v2.0[/bold cyan]
[white]Session: {self.session_id}[/white] | [green]Episode: {self.current_episode + 1}/{self.episodes}[/green] | [yellow]Step: {self.current_step + 1}/{self.max_steps_per_episode}[/yellow] | [magenta]Phase: {current_phase.value.title()}[/magenta]

[blue]Progress: {progress_pct:.1f}%[/blue] | [cyan]ETA: {eta_str}[/cyan] | [red]Elapsed: {elapsed_time:.1f}s[/red] | [green]Agents: {active_agents}/{len(self.agents)}[/green]

[white]Performance:[/white] [green]Reward: {avg_reward:.2f}[/green] | [yellow]Coordination: {coord_score:.2f}[/yellow] | [blue]Target: {self.target_ip}[/blue] | [magenta]{'GPU' if self.enable_gpu else 'CPU'}[/magenta]"""
    
    return Panel(
        header_content,
        title="🎯 Enhanced Multi-Agent Training Dashboard",
        border_style="cyan",
        padding=(1, 2)
    )

def _create_detailed_agent_panel(self) -> Panel:
    """Create detailed agent performance panel with rich information."""
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Agent", style="cyan", width=10)
    table.add_column("Role", style="white", width=18)
    table.add_column("Last Command", style="yellow", width=25)
    table.add_column("Target", style="blue", width=12)
    table.add_column("Output", style="green", width=20)
    table.add_column("Reward", style="red", justify="right", width=8)
    table.add_column("Success", style="white", justify="center", width=7)
    
    for agent_name in self.agent_names:
        # Get latest action for this agent
        recent_actions = [
            action for action in self.action_history[-10:]
            if action.agent_id == agent_name
        ]
        
        if recent_actions:
            latest_action = recent_actions[-1]
            
            # Truncate long text
            command = latest_action.command[:23] + "..." if len(latest_action.command) > 23 else latest_action.command
            output = latest_action.output[:18] + "..." if len(latest_action.output) > 18 else latest_action.output
            target = latest_action.target
            
            # Color coding
            success_icon = "✅" if latest_action.success else "❌"
            reward_color = "green" if latest_action.reward > 0 else "red" if latest_action.reward < 0 else "yellow"
            
            table.add_row(
                agent_name,
                self.agent_definitions[agent_name]['role'][:16] + "...",
                f"[dim]{command}[/dim]",
                target,
                f"[dim]{output}[/dim]",
                f"[{reward_color}]{latest_action.reward:.2f}[/{reward_color}]",
                success_icon
            )
        else:
            table.add_row(
                agent_name,
                self.agent_definitions[agent_name]['role'][:16] + "...",
                "[dim]No actions yet[/dim]",
                self.target_ip,
                "[dim]—[/dim]",
                "[dim]0.00[/dim]",
                "⏳"
            )
    
    return Panel(
        table,
        title="🤖 Detailed Agent Performance & Actions",
        border_style="green",
        padding=(0, 1)
    )

def _create_command_tracking_panel(self) -> Panel:
    """Create command tracking panel showing what RedAgent specifically does."""
    content = "[bold]RedAgent Command Analysis[/bold]\n\n"
    
    # Get RedAgent's recent actions
    red_actions = [
        action for action in self.action_history[-5:]
        if action.agent_id == 'RedAgent'
    ]
    
    if red_actions:
        for i, action in enumerate(red_actions, 1):
            success_icon = "✅" if action.success else "❌"
            phase_color = self._get_phase_color(action.phase)
            
            content += f"[cyan]Command {i}:[/cyan] {success_icon}\n"
            content += f"  [{phase_color}]Phase:[/{phase_color}] {action.phase.title()}\n"
            content += f"  [yellow]Target:[/yellow] {action.target}\n"
            content += f"  [white]Command:[/white] {action.command[:40]}...\n"
            content += f"  [green]Output:[/green] {action.output[:35]}...\n"
            content += f"  [red]Reward:[/red] {action.reward:.2f} | [blue]GPT Tokens:[/blue] {action.gpt_tokens_used}\n"
            
            # Add learning information if available
            if action.learning_loss is not None:
                content += f"  [magenta]Learning Loss:[/magenta] {action.learning_loss:.4f}\n"
            
            content += "\n"
    else:
        content += "[dim]No RedAgent actions recorded yet...[/dim]\n"
    
    # Add RedAgent statistics
    if red_actions:
        avg_reward = np.mean([action.reward for action in red_actions])
        success_rate = np.mean([action.success for action in red_actions])
        total_tokens = sum([action.gpt_tokens_used for action in red_actions])
        
        content += f"[bold]Statistics:[/bold]\n"
        content += f"  [green]Avg Reward:[/green] {avg_reward:.2f}\n"
        content += f"  [yellow]Success Rate:[/yellow] {success_rate:.1%}\n"
        content += f"  [blue]Total GPT Tokens:[/blue] {total_tokens:,}\n"
    
    return Panel(
        content,
        title="🔍 RedAgent Command Tracking & Analysis",
        border_style="red",
        padding=(1, 1)
    )

def _get_phase_color(self, phase: str) -> str:
    """Get color coding for phases."""
    phase_colors = {
        'reconnaissance': 'blue',
        'enumeration': 'cyan',
        'exploitation': 'red',
        'persistence': 'magenta',
        'defense': 'green',
        'coordination': 'yellow'
    }
    return phase_colors.get(phase, 'white')

def _create_live_actions_panel(self) -> Panel:
    """Create live actions panel showing real-time agent activities."""
    content = "[bold]Live Agent Actions[/bold]\n\n"
    
    # Get most recent actions from all agents
    recent_actions = sorted(
        self.action_history[-8:],
        key=lambda x: x.timestamp,
        reverse=True
    )
    
    if recent_actions:
        for action in recent_actions:
            timestamp = datetime.fromtimestamp(action.timestamp).strftime("%H:%M:%S")
            success_icon = "✅" if action.success else "❌"
            agent_color = self._get_agent_color(action.agent_id)
            
            content += f"[dim]{timestamp}[/dim] [{agent_color}]{action.agent_id}[/{agent_color}] {success_icon}\n"
            content += f"  [white]{action.command[:45]}...[/white]\n"
            content += f"  [green]→[/green] {action.output[:40]}...\n"
            content += f"  [yellow]Reward: {action.reward:.2f}[/yellow]\n\n"
    else:
        content += "[dim]Waiting for agent actions...[/dim]\n"
    
    return Panel(
        content,
        title="⚡ Live Agent Activities",
        border_style="yellow",
        padding=(1, 1)
    )

def _get_agent_color(self, agent_id: str) -> str:
    """Get color coding for agents."""
    agent_colors = {
        'RedAgent': 'red',
        'BlueAgent': 'blue',
        'ScoutAgent': 'cyan',
        'ShadowAgent': 'magenta',
        'OrionAgent': 'yellow'
    }
    return agent_colors.get(agent_id, 'white')

def _create_enhanced_coordination_panel(self) -> Panel:
    """Create enhanced coordination matrix with detailed information."""
    content = "[bold]Multi-Agent Coordination Matrix[/bold]\n\n"
    
    # Create a visual coordination matrix
    content += "     "
    for agent in self.agent_names:
        content += f" {agent[:3]:>4}"
    content += "\n"
    
    for i, agent1 in enumerate(self.agent_names):
        content += f"{agent1[:3]:>3}  "
        for j, agent2 in enumerate(self.agent_names):
            if i == j:
                content += "  ■  "  # Self-coordination
            else:
                score = self.coordination_matrix[i][j]
                if score > 0.7:
                    content += f"[green]{score:.1f}[/green] "
                elif score > 0.4:
                    content += f"[yellow]{score:.1f}[/yellow] "
                elif score > 0.0:
                    content += f"[blue]{score:.1f}[/blue] "
                else:
                    content += f"[dim]{score:.1f}[/dim] "
        content += "\n"
    
    # Add coordination statistics
    avg_coordination = self.coordination_matrix.mean()
    max_coordination = self.coordination_matrix.max()
    coordination_trend = "↗️" if len(self.learning_analytics['coordination_evolution']) >= 2 and \
                               self.learning_analytics['coordination_evolution'][-1] > \
                               self.learning_analytics['coordination_evolution'][-2] else "→"
    
    content += f"\n[cyan]Average:[/cyan] {avg_coordination:.2f} | [green]Maximum:[/green] {max_coordination:.2f}\n"
    content += f"[yellow]Trend:[/yellow] {coordination_trend} | [blue]Phase:[/blue] {self.dashboard_state['current_phase'].value.title()}"
    
    return Panel(
        content,
        title="🔗 Enhanced Multi-Agent Coordination",
        border_style="blue",
        padding=(1, 1)
    )

def _create_detailed_learning_panel(self) -> Panel:
    """Create detailed learning analytics panel."""
    content = "[bold]Learning Analytics & Neural Networks[/bold]\n\n"
    
    # Neural network learning metrics
    total_params = 0
    learning_agents = 0
    
    for agent_name, agent in self.agents.items():
        if hasattr(agent, 'policy_net'):
            params = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
            total_params += params
            learning_agents += 1
            
            # Get recent learning metrics
            losses = self.learning_analytics['neural_losses'][agent_name]
            recent_loss = losses[-1] if losses else 0.0
            
            epsilon = getattr(agent, 'epsilon', 0.0)
            
            agent_color = self._get_agent_color(agent_name)
            content += f"[{agent_color}]{agent_name}:[/{agent_color}] "
            content += f"Loss: {recent_loss:.4f} | ε: {epsilon:.3f} | Params: {params:,}\n"
    
    content += f"\n[cyan]Total Parameters:[/cyan] {total_params:,}\n"
    content += f"[green]Learning Agents:[/green] {learning_agents}/{len(self.agents)}\n"
    
    # Learning trends
    if len(self.learning_analytics['coordination_evolution']) >= 5:
        recent_coord = self.learning_analytics['coordination_evolution'][-5:]
        coord_trend = np.mean(np.diff(recent_coord))
        trend_icon = "📈" if coord_trend > 0 else "📉" if coord_trend < 0 else "➡️"
        content += f"[yellow]Coordination Trend:[/yellow] {trend_icon} {coord_trend:+.3f}\n"
    
    # Memory utilization
    total_memories = sum(
        len(self.learning_analytics['episode_rewards'][agent])
        for agent in self.agent_names
    )
    content += f"[blue]Total Memories:[/blue] {total_memories:,} experiences"
    
    return Panel(
        content,
        title="🧠 Learning Analytics & Neural Networks",
        border_style="magenta",
        padding=(1, 1)
    )

def _create_system_health_panel(self) -> Panel:
    """Create system health monitoring panel."""
    content = "[bold]System Health Monitor[/bold]\n\n"
    
    # Agent health status
    for agent_name in self.agent_names:
        health = self.dashboard_state['system_health'].get(agent_name, False)
        health_icon = "🟢" if health else "🔴"
        content += f"{health_icon} {agent_name}\n"
    
    content += "\n[bold]Resources:[/bold]\n"
    
    # GPU/CPU status
    gpu_status = "🟢 GPU Active" if self.enable_gpu else "🟡 CPU Mode"
    content += f"{gpu_status}\n"
    
    # Memory router status
    memory_status = "🟢 Memory Router" if self.memory_router else "🔴 No Memory Router"
    content += f"{memory_status}\n"
    
    # Environment status
    env_status = "🟢 Cyber Environment" if self.environment else "🔴 Mock Environment"
    content += f"{env_status}\n"
    
    # Current performance
    recent_commands = len(self.action_history[-10:])
    content += f"\n[cyan]Commands/10 steps:[/cyan] {recent_commands}\n"
    
    if self.real_time_metrics['commands_per_second']:
        avg_cps = np.mean(list(self.real_time_metrics['commands_per_second'])[-5:])
        content += f"[green]Avg Commands/Step:[/green] {avg_cps:.1f}"
    
    return Panel(
        content,
        title="💻 System Health & Resources",
        border_style="green",
        padding=(1, 1)
    )

def _create_performance_alerts_panel(self) -> Panel:
    """Create performance alerts and warnings panel."""
    content = "[bold]Performance Alerts[/bold]\n\n"
    
    alerts = []
    
    # Check for low coordination
    coord_score = self.coordination_matrix.mean()
    if coord_score < 0.3:
        alerts.append("⚠️ Low coordination score")
    
    # Check for agent failures
    failed_agents = [
        agent for agent, health in self.dashboard_state['system_health'].items()
        if not health
    ]
    if failed_agents:
        alerts.append(f"🔴 Agent failures: {', '.join(failed_agents)}")
    
    # Check for learning stagnation
    for agent_name in self.agent_names:
        rewards = self.learning_analytics['episode_rewards'][agent_name]
        if len(rewards) >= 10:
            recent_rewards = rewards[-5:]
            if np.std(recent_rewards) < 0.1 and np.mean(recent_rewards) < 1.0:
                alerts.append(f"📉 {agent_name} learning stagnation")
    
    # Check for slow performance
    if self.real_time_metrics['commands_per_second']:
        recent_cps = list(self.real_time_metrics['commands_per_second'])[-3:]
        if recent_cps and np.mean(recent_cps) < 1.0:
            alerts.append("🐌 Slow command execution")
    
    # Display alerts
    if alerts:
        for alert in alerts[-5:]:  # Show last 5 alerts
            content += f"{alert}\n"
    else:
        content += "[green]✅ All systems nominal[/green]\n"
    
    # Performance recommendations
    content += "\n[bold]Recommendations:[/bold]\n"
    if coord_score < 0.5:
        content += "• Increase coordination training\n"
    if self.current_episode > 10:
        avg_reward = np.mean([
            sum(self.learning_analytics['episode_rewards'][agent][-5:])
            for agent in self.agent_names
            if self.learning_analytics['episode_rewards'][agent]
        ])
        if avg_reward < 2.0:
            content += "• Consider curriculum adjustment\n"
    
    return Panel(
        content,
        title="⚡ Performance Alerts & Recommendations",
        border_style="yellow",
        padding=(1, 1)
    )

def _create_targets_panel(self) -> Panel:
    """Create current targets and objectives panel."""
    content = "[bold]Current Targets & Objectives[/bold]\n\n"
    
    content += f"[cyan]Primary Target:[/cyan] {self.target_ip}\n"
    content += f"[yellow]Current Phase:[/yellow] {self.dashboard_state['current_phase'].value.title()}\n\n"
    
    # Phase-specific objectives
    phase = self.dashboard_state['current_phase']
    if phase == TrainingPhase.RECONNAISSANCE:
        content += "[blue]Objectives:[/blue]\n"
        content += "• Discover open ports\n"
        content += "• Identify services\n"
        content += "• Map network topology\n"
    elif phase == TrainingPhase.ENUMERATION:
        content += "[cyan]Objectives:[/cyan]\n"
        content += "• Enumerate services\n"
        content += "• Find directories/files\n"
        content += "• Identify vulnerabilities\n"
    elif phase == TrainingPhase.EXPLOITATION:
        content += "[red]Objectives:[/red]\n"
        content += "• Exploit vulnerabilities\n"
        content += "• Gain initial access\n"
        content += "• Execute payloads\n"
    elif phase == TrainingPhase.PERSISTENCE:
        content += "[magenta]Objectives:[/magenta]\n"
        content += "• Establish persistence\n"
        content += "• Create backdoors\n"
        content += "• Maintain access\n"
    elif phase == TrainingPhase.DEFENSE:
        content += "[green]Objectives:[/green]\n"
        content += "• Monitor threats\n"
        content += "• Block attacks\n"
        content += "• Respond to incidents\n"
    
    # Recent discoveries
    discoveries = self._get_recent_discoveries()
    if discoveries:
        content += f"\n[bold]Recent Discoveries:[/bold]\n"
        for discovery in discoveries[-3:]:
            content += f"• {discovery[:30]}...\n"
    
    return Panel(
        content,
        title="🎯 Current Targets & Objectives",
        border_style="cyan",
        padding=(1, 1)
    )

def _create_enhanced_footer(self) -> Panel:
    """Create enhanced footer with comprehensive system information."""
    # Calculate comprehensive statistics
    total_actions = len(self.action_history)
    successful_actions = sum(1 for action in self.action_history if action.success)
    success_rate = successful_actions / max(1, total_actions)
    
    total_tokens = sum(action.gpt_tokens_used for action in self.action_history)
    
    avg_reward = np.mean([action.reward for action in self.action_history]) if self.action_history else 0.0
    
    # Memory statistics
    memory_stats = ""
    if self.memory_router and hasattr(self.memory_router, 'get_stats'):
        stats = self.memory_router.get_stats()
        total_memories = stats.get('total_transitions', 0)
        memory_stats = f"Memory: {total_memories:,} transitions"
    else:
        memory_stats = f"Memory: {total_actions:,} actions"
    
    footer_content = f"""[dim]📊 Session Statistics: [green]{successful_actions}/{total_actions}[/green] actions ({success_rate:.1%} success) | [blue]Avg Reward: {avg_reward:.2f}[/blue] | [yellow]GPT Tokens: {total_tokens:,}[/yellow] | [cyan]{memory_stats}[/cyan]
🎮 Controls: [white]Ctrl+C[/white] to stop | 📁 Logs: [white]{self.log_dir}[/white] | 💾 Models: [white]{self.model_dir}[/white] | 🕒 Auto-save every [white]{self.save_interval}[/white] episodes[/dim]"""
    
    return Panel(
        footer_content,
        border_style="dim",
        padding=(0, 1)
    )

# Additional methods continue...
def _calculate_episode_metrics(self, episode_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive episode-level metrics."""
    metrics = {}
    
    # Basic metrics
    total_reward = sum(episode_results['agent_rewards'].values())
    total_actions = sum(len(actions) for actions in episode_results['agent_actions'].values())
    
    metrics.update({
        'total_reward': total_reward,
        'total_actions': total_actions,
        'average_reward_per_action': total_reward / max(1, total_actions)
    })
    
    # Agent performance metrics
    agent_performance = {}
    for agent_name in self.agent_names:
        if agent_name in episode_results['agent_rewards']:
            agent_performance[agent_name] = {
                'reward': episode_results['agent_rewards'][agent_name],
                'actions': len(episode_results['agent_actions'][agent_name]),
                'success_rate': np.mean([
                    action.success for action in self.action_history[-10:]
                    if action.agent_id == agent_name
                ]) if self.action_history else 0.0
            }
    
    metrics['agent_performance'] = agent_performance
    
    # Coordination metrics
    metrics['coordination_score'] = self.coordination_matrix.mean()
    
    # Learning efficiency
    neural_updates = episode_results.get('neural_updates', {})
    if neural_updates:
        avg_loss = np.mean([
            np.mean(losses) for losses in neural_updates.values() if losses
        ])
        metrics['learning_efficiency'] = 1.0 / (1.0 + avg_loss)
    else:
        metrics['learning_efficiency'] = 0.5
    
    # GPT usage
    gpt_usage = {}
    for agent_name in self.agent_names:
        recent_actions = [
            action for action in self.action_history[-10:]
            if action.agent_id == agent_name
        ]
        gpt_usage[agent_name] = sum(action.gpt_tokens_used for action in recent_actions)
    
    metrics['gpt_usage'] = gpt_usage
    
    return metrics
