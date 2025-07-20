#!/usr/bin/env python3
# core/utils/config_loader.py — ARIASKA Environment Configuration Loader
# 🌐 Unified Configuration System | 🔐 Environment Variable Management | 🧰 Runtime Configuration

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from dotenv import load_dotenv

console = Console()

class ConfigLoader:
    """
    Centralized configuration management system for ARIASKA_RL.
    Handles environment variables, configuration files, and runtime settings.
    """
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            env_file_path: Optional path to .env file. If None, tries standard locations.
        """
        self.logger = logging.getLogger("ariaska.config")
        
        # Load environment variables
        self._load_env_vars(env_file_path)
        
        # Store loaded configuration
        self.config = {}
        self._load_config()
        
    def _load_env_vars(self, env_file_path: Optional[str] = None) -> None:
        """
        Load environment variables from .env file.
        
        Args:
            env_file_path: Optional path to .env file
        """
        # Try project root .env first
        project_root = Path(__file__).parent.parent.parent.parent
        default_env_path = project_root / ".env"
        
        paths_to_try = [
            env_file_path if env_file_path else None,
            default_env_path,
            project_root / "Ariaska_RL" / ".env",
            Path.home() / ".ariaska" / ".env"
        ]
        
        env_loaded = False
        for path in paths_to_try:
            if path and path.exists():
                console.print(f"[green]✓ Loading environment from {path}[/green]")
                load_dotenv(path, override=True)
                env_loaded = True
                break
                
        if not env_loaded:
            console.print("[yellow]⚠ No .env file found. Using system environment variables.[/yellow]")
        
        # Validate critical environment variables
        self._validate_environment()
            
    def _validate_environment(self) -> None:
        """Validate that critical environment variables are set."""
        # Check live mode setting
        live_mode = os.environ.get("ARIASKA_LIVE_MODE", "false").lower() == "true"
        
        if live_mode:
            console.print("[bold green]🔥 LIVE MODE ENABLED[/bold green]")
            
            # Check for target IP
            target_ip = os.environ.get("ARIASKA_TARGET_IP")
            if not target_ip:
                console.print("[bold red]❌ ARIASKA_TARGET_IP must be set for live mode![/bold red]")
            else:
                console.print(f"[green]✓ Target IP: {target_ip}[/green]")
    
    def _load_config(self) -> None:
        """Load configuration from files and environment variables."""
        # Environment variables override everything
        self._load_env_config()
        
        # Print current configuration mode
        self._print_mode_info()
    
    def _load_env_config(self) -> None:
        """Load configuration from environment variables."""
        # Core settings
        self.config["live_mode"] = os.environ.get("ARIASKA_LIVE_MODE", "false").lower() == "true"
        self.config["target_ip"] = os.environ.get("ARIASKA_TARGET_IP", "192.168.1.119")
        self.config["port_range"] = os.environ.get("ARIASKA_PORT_RANGE", "1-1024")
        
        # Metasploit settings
        self.config["metasploit"] = {
            "rpc_port": os.environ.get("METASPLOIT_RPC_PORT", "55553"),
            "rpc_user": os.environ.get("METASPLOIT_RPC_USER", "msf"),
            "rpc_password": os.environ.get("METASPLOIT_RPC_PASS", "password"),
            "restart_rpc": os.environ.get("RESTART_METASPLOIT_RPC", "false").lower() == "true"
        }
        
        # Logging settings
        self.config["log_level"] = os.environ.get("LOG_LEVEL", "INFO")
    
    def _print_mode_info(self) -> None:
        """Print information about the current configuration mode."""
        if self.config["live_mode"]:
            console.print("[bold green]⚡ ARIASKA_RL running in LIVE MODE[/bold green]")
            console.print(f"[bold yellow]⚠ Live target: {self.config['target_ip']}[/bold yellow]")
            console.print("[bold red]❗ WARNING: Live mode will execute real commands![/bold red]")
        else:
            console.print("[blue]ℹ ARIASKA_RL running in SIMULATION MODE[/blue]")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.
        
        Returns:
            Dict containing all configuration values
        """
        return self.config.copy()

    def is_live_mode(self) -> bool:
        """
        Check if ARIASKA_RL is running in live mode.
        
        Returns:
            True if running in live mode, False otherwise
        """
        return self.config["live_mode"]
    
    def get_target_ip(self) -> str:
        """
        Get the target IP for live mode.
        
        Returns:
            Target IP address as string
        """
        return self.config["target_ip"]
    
    def dump_config(self) -> str:
        """
        Dump the current configuration as a JSON string.
        
        Returns:
            JSON string of current configuration
        """
        # Create a sanitized copy without sensitive data
        safe_config = self.config.copy()
        if "metasploit" in safe_config:
            safe_config["metasploit"] = safe_config["metasploit"].copy()
            if "rpc_password" in safe_config["metasploit"]:
                safe_config["metasploit"]["rpc_password"] = "********"
        
        return json.dumps(safe_config, indent=2)

# Singleton instance
_config_instance = None

def get_config(env_file_path: Optional[str] = None) -> ConfigLoader:
    """
    Get the global configuration instance.
    
    Args:
        env_file_path: Optional path to .env file
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(env_file_path)
    return _config_instance

if __name__ == "__main__":
    # If run directly, print the current configuration
    config = get_config()
    console.print("\n[bold]ARIASKA_RL Current Configuration:[/bold]")
    console.print(config.dump_config())
    
    if config.is_live_mode():
        console.print("\n[bold red]⚠ LIVE MODE IS ACTIVE - Commands will be executed on real targets! ⚠[/bold red]")
    else:
        console.print("\n[blue]ℹ Simulation mode active - No real commands will be executed[/blue]")