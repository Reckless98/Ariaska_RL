# core/cyber_environment.py — ARIASKA Simulation Core v12.0 APEX PRIME
# 🌐 Unified Multi-Agent Arena | 🧠 Orion Strategic Oversight | ⚔️ Dynamic Red vs Blue Ops | 📊 Real-Time Adaptation

import random
import ipaddress
import subprocess
import json
from rich.console import Console

console = Console()


class CyberEnvironment:
    def __init__(self, scenario="dynamic", agent_manager=None):
        console.rule(
            "[bold cyan]🌐 Initializing CyberEnvironment v12.0 — Multi-Agent Combat Arena"
        )
        self.scenario = scenario
        self.default_services = [
            "ftp",
            "ssh",
            "http",
            "smb",
            "rdp",
            "smtp",
            "mysql",
            "postgres",
            "telnet",
            "dns",
            "ldap",
        ]
        self.phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]

        # Only set agent_manager if provided (prevents recursion)
        self.agent_manager = agent_manager
        self.stats_monitor = None
        self.orion_agent = None
        self.dynamic_profile = None
        self.max_difficulty = 20
        self.traceback_threshold = 75
        self.training_mode = "adaptive"
        self.blue_team_aggressiveness = 3

        # Do NOT call _initialize_dynamic_parameters here!
        self.reset_environment()

    def initialize_dynamic_parameters(self):
        # Delay the import of AgentManager to avoid circular imports
        from core.monitor.stats_monitor import StatsMonitor

        self.stats_monitor = StatsMonitor(
            agents_list=[agent.agent_id for agent in self.agent_manager.all_agents()]
        )
        self.orion_agent = self.agent_manager.orion_agent

        self.dynamic_profile = self.orion_agent.generate_dynamic_scenario(
            self.scenario, self.default_services
        )
        self.max_difficulty = self.dynamic_profile.get("difficulty", 20)
        self.traceback_threshold = self.dynamic_profile.get("traceback_threshold", 75)
        self.training_mode = self.dynamic_profile.get("training_mode", "adaptive")
        self.blue_team_aggressiveness = self.dynamic_profile.get(
            "blue_aggressiveness", 3
        )

    def reset_environment(self):
        console.print("[green]🔄 Resetting Environment State[/green]")
        # Dynamically adjust difficulty based on agent performance
        if self.agent_manager and hasattr(self.agent_manager, "red_agent"):
            avg_reward = self.agent_manager.red_agent.stats_monitor.get_average_reward()
            # Scale difficulty: higher reward → higher difficulty
            if avg_reward > 20:
                self.difficulty_level = min(self.difficulty_level + 1, self.max_difficulty)
            elif avg_reward < 5:
                self.difficulty_level = max(self.difficulty_level - 1, 1)
        else:
            self.difficulty_level = 1
        self.current_phase = "recon"
        self.open_ports = sorted(
            random.sample(range(20, 10000), k=random.randint(6, 12))
        )
        self.services = (
            self.dynamic_profile.get("services", random.sample(self.default_services, 5))
            if self.dynamic_profile else random.sample(self.default_services, 5)
        )
        self.target_ip = self._generate_random_ip()
        self.hostname = f"target-{random.randint(100,999)}"
        self.credentials_found = False
        self.privilege_level = "none"
        self.data_exfiltrated = False
        self.detection_risk = 0.0
        self.blue_team_alert = 0.0
        self.honeypots = []
        self.previous_actions = []
        self.done = False

    def reset(self):
        """Compatibility wrapper for RL agent code."""
        self.reset_environment()
        return self.get_global_state()

    def _generate_random_ip(self):
        while True:
            ip = ipaddress.IPv4Address(random.randint(1 << 24, (1 << 32) - 1))
            if not (
                ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_multicast
            ):
                return str(ip)

    def get_global_state(self):
        return {
            "phase": self.current_phase,
            "open_ports": self.open_ports,
            "services": self.services,
            "credentials_found": self.credentials_found,
            "privilege_level": self.privilege_level,
            "data_exfiltrated": self.data_exfiltrated,
            "detection_risk": round(self.detection_risk, 2),
            "blue_team_alert": round(self.blue_team_alert, 2),
            "target_ip": self.target_ip,
            "hostname": self.hostname,
            "scenario": self.scenario,
            "difficulty": self.difficulty_level,
            "honeypots": self.honeypots,
            "done": self.done,
        }

    # ─────────────────────────────────────────────
    # 🎮 Multi-Agent Step Execution
    # ─────────────────────────────────────────────
    def step(self, action=None):
        """
        Improved step method with better error handling, agent coordination, and visualization
        """
        if self.done:
            console.print(
                "[yellow]⚠ Environment already completed. Reset required.[/yellow]"
            )
            return self.get_global_state(), 0, True, {}
        
        # Check if we have an agent_manager
        if not self.agent_manager:
            # Standalone mode - simpler processing
            reward = 0
            self.current_phase = action if action in self.phases else self.current_phase
            self._adjust_risks(self.current_phase)
            if self.current_phase == "exfiltrate" and self.privilege_level == "root":
                self.data_exfiltrated = True
                self.done = True
                reward = 65.0
            
            # Enhanced visualization for step
            self._visualize_state_change("Phase change", {"phase": self.current_phase})
            
            return self.get_global_state(), reward, self.done, {}
        
        try:
            # --- RedAgent acts ---
            console.print("[bold red]🔴 RedAgent Action[/bold red]")
            # Expect a string command, not int
            red_action = action if isinstance(action, str) else str(action)
            red_result = self._process_red_action(self.current_phase)
            
            # Visualize red action result
            self._visualize_action_result("RedAgent", red_action, red_result)
            
            if hasattr(self, "stats_monitor") and self.stats_monitor:
                self.stats_monitor.log_step("RedAgent", red_result["reward"], command=str(red_action))

            # --- BlueAgent acts ---
            console.print("[bold blue]🔵 BlueAgent Response[/bold blue]")
            blue_result = {}
            if hasattr(self.agent_manager, "blue_agent") and self.agent_manager.blue_agent:
                if hasattr(self.agent_manager.blue_agent, "react_to_action"):
                    blue_result = self.agent_manager.blue_agent.react_to_action(
                        red_action, self.get_global_state()
                    )
                    self._apply_blue_defense(blue_result)
                    
                    # Visualize blue defense result
                    self._visualize_defense_result(blue_result)
                    
                    if hasattr(self, "stats_monitor") and self.stats_monitor:
                        self.stats_monitor.log_step(
                            "BlueAgent", 
                            blue_result.get("reward", 0), 
                            command=blue_result.get("action", "")
                        )

            # --- ScoutAgent phase analysis ---
            if hasattr(self.agent_manager, "scout_agent") and self.agent_manager.scout_agent:
                try:
                    # Pass all_agents parameter to advise_phase
                    scout_phase = self.agent_manager.scout_agent.advise_phase(
                        self.get_global_state(), 
                        self.agent_manager.all_agents()
                    )
                    
                    # Visualize scout phase advice
                    console.print(f"[bold cyan]🧭 ScoutAgent recommends phase: {scout_phase}[/bold cyan]")
                    
                    if hasattr(self, "stats_monitor") and self.stats_monitor:
                        self.stats_monitor.log_step("ScoutAgent", 0, command=f"Phase advice: {scout_phase}")
                except Exception as e:
                    console.print(f"[yellow]⚠ ScoutAgent phase advice failed: {e}[/yellow]")

            # --- End of step ---
            if hasattr(self, "_orion_overwatch"):
                self._orion_overwatch()
            
            # Final environment state visualization    
            self._visualize_environment_state()
                
            # Return results
            reward = red_result["reward"]
            if isinstance(reward, dict):
                reward = reward.get("reward", 0.0)
            try:
                reward = float(reward)
            except Exception:
                reward = 0.0
            return self.get_global_state(), reward, self.done, red_result
            
        except Exception as e:
            console.print(f"[red]❌ Error in environment step: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return self.get_global_state(), -1.0, False, {"error": str(e)}

    def _process_red_action(self, action):
        phase_rewards = {
            "recon": 2.0,
            "enumeration": 5.0 if random.random() < 0.3 else 1.0,
            "exploit": 14.0 if self.credentials_found else -5.0,
            "privesc": 28.0 if self.privilege_level == "user" else -6.0,
            "exfiltrate": 65.0 if self.privilege_level == "root" else -10.0,
        }

        if action in self.phases:
            self.current_phase = action
            reward = phase_rewards.get(action, -2.0)
            self._adjust_risks(action)
            if action == "exploit" and reward > 0:
                self.privilege_level = "user"
            if action == "privesc" and reward > 0:
                self.privilege_level = "root"
            if action == "exfiltrate" and reward > 0:
                self.data_exfiltrated = True
                self.done = True
        else:
            reward = -15.0
            self._increase_detection(0.5)
            console.print(f"[red]❌ Invalid action by RedAgent: {action}[/red]")

        return {
            "phase": self.current_phase,
            "reward": reward,
            "alert": self.blue_team_alert,
            "risk": self.detection_risk,
            "done": self.done,
        }

    def _apply_blue_defense(self, defense_result):
        if defense_result.get("honeypots_deployed"):
            self.honeypots += defense_result["honeypots"]
            self.services += defense_result["honeypots"]
            console.print("[yellow]🛡️ BlueAgent deployed honeypots![/yellow]")

        if defense_result.get("credentials_reset"):
            self.credentials_found = False
            self.privilege_level = "none"
            console.print("[yellow]🔐 BlueAgent reset credentials![/yellow]")

        self.blue_team_alert += defense_result.get("alert_increase", 0.0)
        self.detection_risk += defense_result.get("risk_increase", 0.0)
        self.blue_team_alert = min(self.blue_team_alert, 100.0)
        self.detection_risk = min(self.detection_risk, 10.0)

        if self.blue_team_alert >= self.traceback_threshold:
            console.print(
                "[red]🚨 TRACEBACK: BlueAgent has compromised RedAgent![/red]"
            )
            self.done = True

    # ─────────────────────────────────────────────
    # 👁️ Orion Strategic Oversight
    # ─────────────────────────────────────────────
    def _orion_overwatch(self):
        insight = self.agent_manager.orion_agent.evaluate_environment(
            self.get_global_state()
        )
        if insight:
            console.print(f"[blue]👁️ Orion Insight:[/blue] {insight}")
            self._adjust_strategy(insight)

    def _adjust_strategy(self, insight):
        if "increase stealth" in insight.lower():
            self.traceback_threshold += 5
            console.print(
                "[cyan]🔧 Environment adjusted for higher stealth tolerance.[/cyan]"
            )
        elif "prepare counter" in insight.lower():
            self.blue_team_alert += 5
            console.print("[magenta]⚠️ Blue Team readiness increased.[/magenta]")

    # ─────────────────────────────────────────────
    # ⚡ Risk & Alert Adjustments
    # ─────────────────────────────────────────────
    def _adjust_risks(self, phase):
        risk_map = {
            "recon": 0.2,
            "enumeration": 0.4,
            "exploit": 0.7,
            "privesc": 0.9,
            "exfiltrate": 1.0,
        }
        alert_map = {
            "recon": 1.0,
            "enumeration": 3.0,
            "exploit": 5.0,
            "privesc": 7.0,
            "exfiltrate": 10.0,
        }

        self._increase_detection(risk_map.get(phase, 0.5))
        self._increase_alert(alert_map.get(phase, 2.0))

    def _increase_detection(self, amount):
        scale = 1 + (self.difficulty_level / self.max_difficulty)
        self.detection_risk = min(self.detection_risk + amount * scale, 10.0)

    def _increase_alert(self, amount):
        aggressiveness = self.dynamic_profile.get("blue_aggressiveness", 2) / 3
        self.blue_team_alert = min(
            self.blue_team_alert + amount * aggressiveness, 100.0
        )

    # ─────────────────────────────────────────────
    # 🧠 GPT-Powered Output Generation
    # ─────────────────────────────────────────────
    def generate_output(self, command):
        if self.detection_risk > 9.5:
            return "⚠ ALERT: IDS detected malicious behavior."

        gpt_prompt = f"""You are a cyber range simulation AI.
Generate a realistic output for the following command in phase '{self.current_phase}':
Command: {command}
Services: {', '.join(self.services)}
Respond concisely."""

        try:
            result = subprocess.run(
                [
                    "sgpt",
                    "--model",
                    "gpt-4o-mini",
                    "--temperature",
                    "0.4",
                    "--role",
                    "aria",
                    gpt_prompt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            output = result.stdout.strip()
            return output if output else "Command executed."
        except:
            return "Command executed."

    def _visualize_state_change(self, title, changes):
        """Visualize state changes in the environment"""
        from rich.panel import Panel
        from rich.table import Table
        
        table = Table(title=title, show_header=False)
        table.add_column("Property")
        table.add_column("Value")
        
        for key, value in changes.items():
            table.add_row(key, str(value))
        
        console.print(Panel(table, border_style="blue"))
        
    def _visualize_action_result(self, agent, action, result):
        """Visualize the result of an agent's action"""
        from rich.panel import Panel
        from rich.table import Table
        
        color = "red" if agent == "RedAgent" else "blue"
        
        table = Table(title=f"{agent} Action Result", show_header=False)
        table.add_column("Property", style=color)
        table.add_column("Value")
        
        # Format action based on type
        if isinstance(action, int):
            action = f"Action ID: {action}"
            
        table.add_row("Action", str(action))
        table.add_row("Phase", result.get("phase", "N/A"))
        
        # Format reward with color
        reward = result.get("reward", 0)
        if reward > 20:
            reward_str = f"[bold green]{reward:.2f}[/bold green]"
        elif reward > 0:
            reward_str = f"[green]{reward:.2f}[/green]"
        elif reward < -10:
            reward_str = f"[bold red]{reward:.2f}[/bold red]"
        else:
            reward_str = f"[red]{reward:.2f}[/red]"
        table.add_row("Reward", reward_str)
        
        # Other results
        table.add_row("Alert Level", f"{result.get('alert', 0):.2f}")
        # Fix: risk may not be float, ensure conversion
        risk_val = result.get('risk', 0)
        try:
            risk_val = float(risk_val)
        except Exception:
            risk_val = 0.0
        table.add_row("Risk", f"{risk_val:.2f}")
        
        console.print(Panel(table, border_style=color))
        
    def _visualize_defense_result(self, defense):
        """Visualize defensive measures taken by BlueAgent"""
        from rich.panel import Panel
        from rich.table import Table
        
        if not defense:
            console.print("[blue]No defensive measures taken[/blue]")
            return
            
        table = Table(title="🛡️ Defensive Measures", show_header=False)
        table.add_column("Measure", style="blue")
        table.add_column("Value")
        
        for key, value in defense.items():
            if key in ["honeypots", "honeypots_deployed"] and value:
                table.add_row(key, f"[bold yellow]{', '.join(value)}[/bold yellow]")
            elif key in ["credentials_reset"] and value:
                table.add_row(key, "[bold red]True[/bold red]")
            elif key in ["alert_increase", "risk_increase"]:
                color = "yellow" if value > 5 else "green"
                table.add_row(key, f"[{color}]{value:.2f}[/{color}]")
            else:
                table.add_row(key, str(value))
                
        console.print(Panel(table, border_style="blue"))
    
    def _visualize_environment_state(self):
        """Visualize the current state of the environment"""
        from rich.panel import Panel
        from rich.table import Table
        from rich.columns import Columns
        
        # Create main system state table
        state_table = Table(title="🌐 System State")
        state_table.add_column("Property", style="cyan")
        state_table.add_column("Value", style="green")
        
        state_table.add_row("Phase", self.current_phase)
        state_table.add_row("Privilege", self.privilege_level)
        state_table.add_row("Target IP", self.target_ip)
        state_table.add_row("Hostname", self.hostname)
        state_table.add_row("Difficulty", str(self.difficulty_level))
        
        # Create security state table
        security_table = Table(title="🔒 Security State")
        security_table.add_column("Property", style="magenta")
        security_table.add_column("Value", style="yellow")
        
        # Format with colors based on values
        alert_color = "green" if self.blue_team_alert < 30 else "yellow" if self.blue_team_alert < 60 else "red"
        risk_color = "green" if self.detection_risk < 3 else "yellow" if self.detection_risk < 6 else "red"
        
        security_table.add_row("Blue Alert", f"[{alert_color}]{self.blue_team_alert:.2f}[/{alert_color}]")
        security_table.add_row("Detection Risk", f"[{risk_color}]{self.detection_risk:.2f}[/{risk_color}]")
        security_table.add_row("Credentials Found", str(self.credentials_found))
        security_table.add_row("Data Exfiltrated", str(self.data_exfiltrated))
        security_table.add_row("Honeypots", ", ".join(self.honeypots) if self.honeypots else "None")
        
        # Create service info table
        services_table = Table(title="🖥️ Network Services")
        services_table.add_column("Type", style="blue")
        services_table.add_column("Details", style="green")
        
        # Format port list nicely
        port_text = ", ".join(str(p) for p in self.open_ports[:8])
        if len(self.open_ports) > 8:
            port_text += f" + {len(self.open_ports) - 8} more"
            
        services_table.add_row("Open Ports", port_text)
        services_table.add_row("Services", ", ".join(self.services))
        
        # Combine tables in columns
        console.print(
            Panel(
                Columns([state_table, security_table, services_table]),
                title="🌍 Environment State",
                border_style="bright_blue"
            )
        )

    def adjust_difficulty(self, agent_performance):
        if agent_performance.get('reward_avg', 0) > 20:
            self.detection_risk += 0.1
            self.deploy_honeypot()

    def deploy_honeypot(self):
        if hasattr(self, "honeypots"):
            self.honeypots.append(f"honeypot_{random.randint(100,999)}")
            console.print("[yellow]🛡️ Honeypot deployed due to RedAgent aggression.[/yellow]")
