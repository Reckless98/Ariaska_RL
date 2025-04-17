# core/cyber_environment.py — v7.2 (State-Driven Realism + Training Compatible)

import random
import ipaddress
from rich.console import Console

console = Console()

class CyberEnvironment:
    def __init__(self, blue_team=True):
        self.max_difficulty = 20
        self.difficulty_level = 1
        self.phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
        self.current_phase = "recon"
        self.open_ports = []
        self.services = []
        self.credentials_found = False
        self.privilege_level = "none"
        self.data_exfiltrated = False
        self.detection_risk = 0.0
        self.blue_team_alert = 0.0
        self.blue_team_aggressiveness = 1
        self.traceback_threshold = 75
        self.target_ip = self.generate_random_ip()
        self.hostname = f"host-{random.randint(100, 999)}"
        self.honeypots = []
        self.previous_actions = []
        self.blue_team_enabled = blue_team

    def reset(self):
        self.current_phase = "recon"
        self.open_ports = sorted(random.sample(range(20, 10000), k=random.randint(8, 18)))
        self.services = random.choices(
            ["ftp", "ssh", "http", "smb", "rdp", "smtp", "mysql", "postgres", "telnet"],
            k=random.randint(3, 6)
        )
        self.credentials_found = False
        self.privilege_level = "none"
        self.data_exfiltrated = False
        self.detection_risk = 0.0
        self.blue_team_alert = 0.0
        self.blue_team_aggressiveness = 1
        self.previous_actions = []
        self.honeypots.clear()
        self.target_ip = self.generate_random_ip()
        self.hostname = f"host-{random.randint(100, 999)}"
        return self.get_state()

    def generate_random_ip(self):
        while True:
            ip = ipaddress.IPv4Address(random.randint(1 << 24, (1 << 32) - 1))
            if not (ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_multicast):
                return str(ip)

    def generate_output(self, command):
        command = command.lower()

        if self.detection_risk > 8.7 and random.random() < 0.3:
            return "⚠ Intrusion alert: Connection refused. Blue Team likely detecting your actions."

        if "nmap" in command:
            results = [f"{p}/tcp open {random.choice(self.services)}" for p in self.open_ports[:5]]
            if self.honeypots:
                results += [f"{random.randint(10000, 65535)}/tcp open {hp}" for hp in self.honeypots]
            return "\n".join(results)

        elif "masscan" in command:
            return "Discovered ports: " + ", ".join(str(p) for p in self.open_ports[:10])

        elif "enum4linux" in command or "ldapsearch" in command:
            if random.random() < 0.65:
                self.credentials_found = True
                return "Discovered user: admin\nFound password: admin123"
            return "[TIMEOUT] - No response from enumeration."

        elif "exploit-db" in command:
            if self.credentials_found:
                self.privilege_level = "user"
                return "Exploit launched: CVE-2021-12345\nAccess granted: user shell."
            return "Exploit failed. No valid credentials."

        elif "hydra" in command:
            if random.random() < 0.5:
                self.credentials_found = True
                return "Brute-force success: admin:admin123"
            return "Authentication failure."

        elif "sqlmap" in command:
            return "DB Found: users\nDumped: user1, user2, admin"

        elif "msfconsole" in command or "exploit" in command:
            if self.privilege_level == "user":
                self.privilege_level = "root"
                return "Exploit success. Root shell obtained via meterpreter."
            elif self.credentials_found:
                self.privilege_level = "user"
                return "Exploit success. User shell gained."
            return "Exploit failed."

        elif "linpeas" in command or "sudo -l" in command:
            if self.privilege_level == "user":
                self.privilege_level = "root"
                return "Privilege escalation possible: sudo/root vector found."
            return "Insufficient permissions for priv esc."

        elif "crackmapexec" in command:
            return "Shares available:\n- ADMIN$\n- C$\n- IPC$\nEnumeration successful."

        elif "evil-winrm" in command:
            if self.privilege_level == "user":
                return "Session opened. PowerShell WinRM ready."
            return "WinRM auth denied."

        elif any(x in command for x in ["scp", "curl", "wget", "exfil"]):
            if self.privilege_level == "root":
                self.data_exfiltrated = True
                return "Exfiltration successful. Data transferred."
            return "Exfiltration failed: insufficient privileges."

        return "Command executed. No actionable result."

    def step(self, action_id):
        actions_map = {
            0: self.recon_action,
            1: self.enumeration_action,
            2: self.exploit_action,
            3: self.privesc_action,
            4: self.exfiltrate_action
        }
        action_func = actions_map.get(action_id, self.invalid_action)
        reward, done = action_func()

        if self.blue_team_enabled:
            self.blue_team_logic()

        self.previous_actions.append(self.current_phase)
        state = self.get_state()
        info = {
            "msg": f"Action executed: {self.current_phase}",
            "phase": self.current_phase,
            "reward": reward,
            "done": done,
            "intel": {
                "ports": self.open_ports,
                "services": self.services,
                "creds": self.credentials_found,
                "privilege": self.privilege_level,
                "detection": self.detection_risk,
                "alert": self.blue_team_alert,
                "exfiltrated": self.data_exfiltrated
            }
        }

        if self.blue_team_alert >= self.traceback_threshold:
            done = True
            reward -= 100
            info["msg"] = "🚨 TRACEBACK: Blue Team detected the intrusion and neutralized the attack."

        return state, reward, done, info

    def get_state(self):
        return {
            "phase": self.current_phase,
            "open_ports": self.open_ports,
            "services": self.services,
            "credentials_found": self.credentials_found,
            "privilege_level": self.privilege_level,
            "data_exfiltrated": self.data_exfiltrated,
            "detection_risk": round(self.detection_risk, 3),
            "blue_team_alert": round(self.blue_team_alert, 3),
            "target_ip": self.target_ip,
            "hostname": self.hostname
        }

    def recon_action(self):
        self.current_phase = "recon"
        self.increase_detection(0.05)
        self.increase_alert(0.5 + random.uniform(0.1, 0.3))
        return 2.0, False

    def enumeration_action(self):
        self.current_phase = "enumeration"
        base_chance = 0.25 + 0.025 * self.difficulty_level
        self.credentials_found = random.random() < base_chance
        self.increase_detection(0.08)
        self.increase_alert(0.9)
        return (4.0 if self.credentials_found else 1.0), False

    def exploit_action(self):
        self.current_phase = "exploit"
        if not self.credentials_found:
            return -3.0, False
        success = random.random() < (0.2 + 0.04 * self.difficulty_level)
        if success:
            self.privilege_level = "user"
            return 12.0, False
        return -2.5, False

    def privesc_action(self):
        self.current_phase = "privesc"
        if self.privilege_level != "user":
            return -2.0, False
        success = random.random() < (0.2 + 0.04 * self.difficulty_level)
        if success:
            self.privilege_level = "root"
            self.increase_difficulty()
            return 25.0, False
        return -3.5, False

    def exfiltrate_action(self):
        self.current_phase = "exfiltrate"
        if self.privilege_level != "root":
            return -5.0, False
        success = random.random() < (0.35 + 0.04 * self.difficulty_level)
        if success:
            self.data_exfiltrated = True
            self.increase_difficulty()
            return 60.0, True
        return -6.0, False

    def invalid_action(self):
        self.increase_detection(0.5)
        self.increase_alert(5.5)
        return -12.0, False

    def increase_detection(self, amount):
        scale = 1 + (self.difficulty_level / self.max_difficulty)
        decay = 0.96 ** len(self.previous_actions)
        self.detection_risk = min(self.detection_risk + amount * scale * decay, 10.0)

    def increase_alert(self, amount):
        scale = 1 + (self.difficulty_level / self.max_difficulty)
        self.blue_team_alert = min(self.blue_team_alert + amount * scale, 100.0)

    def blue_team_logic(self):
        self.increase_alert(self.detection_risk * 1.25)
        if self.blue_team_alert > 25:
            self.lock_ports()
        if self.blue_team_alert > 50:
            self.reset_credentials()
            self.deploy_honeypots()
        if self.blue_team_alert > 75:
            self.traceback_player()
        self.blue_team_aggressiveness = min(10, 1 + (self.blue_team_alert // 10))

    def lock_ports(self):
        if len(self.open_ports) > 3:
            to_close = random.sample(self.open_ports, k=len(self.open_ports) // 2)
            for p in to_close:
                self.open_ports.remove(p)

    def reset_credentials(self):
        if self.credentials_found:
            self.credentials_found = False
            self.privilege_level = "none"

    def deploy_honeypots(self):
        if not self.honeypots:
            self.honeypots = ["fake_http", "fake_ssh", "fake_smb"]
            self.services += self.honeypots

    def traceback_player(self):
        if random.random() < 0.65:
            console.print("[red]⚠ TRACEBACK INITIATED: Host is being monitored.[/red]")

    def increase_difficulty(self):
        self.difficulty_level = min(self.max_difficulty, self.difficulty_level + 1)

    def decrease_difficulty(self):
        self.difficulty_level = max(1, self.difficulty_level - 1)
