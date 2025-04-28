#!/usr/bin/env python3
"""
metasploitable_handler.py - Metasploitable Integration for Ariaska_RL
Author: Filip Volf
Date: April 27, 2025

This module provides automatic detection and integration with Metasploitable VMs
for live training mode with the Ariaska_RL framework.
"""

import os
import re
import json
import time
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import ipaddress
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load environment variables
load_dotenv()
console = Console()

class MetasploitableHandler:
    """
    Handles integration with Metasploitable VMs for live training.
    
    This class provides functionality to:
    1. Detect Metasploitable VMs on the local network
    2. Validate they are running and accessible
    3. Configure the environment for live training
    4. Provide target information to agents
    """
    
    def __init__(self, auto_detect: bool = True):
        """
        Initialize the Metasploitable handler.
        
        Args:
            auto_detect: Whether to automatically detect Metasploitable VMs at startup
        """
        self.targets = []
        self.active_target = None
        self.metasploitable_signatures = {
            "metasploitable2": {
                "ports": [22, 23, 21, 80, 139, 445, 3306],
                "banner_signatures": {
                    22: "OpenSSH 4.7p1",
                    23: "Ubuntu",
                    21: "vsftpd 2.3.4"
                }
            },
            "metasploitable3": {
                "ports": [22, 80, 8080, 3389],
                "banner_signatures": {
                    22: "OpenSSH",
                    80: "Apache",
                    8080: "Jetty"
                }
            }
        }
        self.live_mode = os.getenv("ARIASKA_MODE", "simulated").lower() == "live"
        self.target_ip = os.getenv("ARIASKA_TARGET_IP", None)
        self.port_range = os.getenv("ARIASKA_PORT_RANGE", "1-1024")
        
        # Cached scan results
        self.scan_results = {}
        self.last_scan_time = 0
        
        if auto_detect and self.live_mode:
            self.detect_metasploitable()
    
    def is_virtualbox_running(self) -> bool:
        """Check if VirtualBox is running on the system"""
        try:
            result = subprocess.run(
                ["pgrep", "VBoxHeadless"], 
                capture_output=True, 
                text=True
            )
            return bool(result.stdout.strip())
        except Exception:
            # Fall back to another method if pgrep fails
            try:
                result = subprocess.run(
                    ["ps", "aux"], 
                    capture_output=True, 
                    text=True
                )
                return "VBoxHeadless" in result.stdout
            except Exception:
                return False
    
    def get_local_ip_range(self) -> List[str]:
        """Get the local IP range for scanning"""
        local_ip_ranges = []
        
        try:
            # Get all network interfaces
            result = subprocess.run(
                ["ip", "addr", "show"], 
                capture_output=True, 
                text=True
            )
            
            # Extract IP addresses and netmasks
            for line in result.stdout.split("\n"):
                if "inet " in line and "scope global" in line:
                    # Extract IP/netmask
                    match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/(\d+)", line)
                    if match:
                        ip = match.group(1)
                        netmask = int(match.group(2))
                        
                        # Create network range
                        network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
                        local_ip_ranges.append(str(network))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not determine local IP range: {e}[/yellow]")
            # Fallback to common home network ranges
            local_ip_ranges = ["192.168.0.0/24", "192.168.1.0/24", "10.0.0.0/24", "10.0.2.0/24"]
            
        return local_ip_ranges
    
    def scan_for_hosts(self, ip_range: str = None) -> List[str]:
        """
        Scan for active hosts in the provided IP range
        
        Args:
            ip_range: IP range to scan (CIDR notation)
            
        Returns:
            List of active IP addresses
        """
        active_ips = []
        
        # Use provided range or get local network range
        if ip_range is None:
            ranges = self.get_local_ip_range()
            if not ranges:
                console.print("[red]Error: No valid IP range found for scanning[/red]")
                return []
            ip_range = ranges[0]
        
        try:
            console.print(f"[cyan]Scanning for hosts in {ip_range}...[/cyan]")
            # Run fast ping scan
            result = subprocess.run(
                ["nmap", "-sn", ip_range], 
                capture_output=True, 
                text=True
            )
            
            # Extract IP addresses from the nmap output
            for line in result.stdout.split("\n"):
                if "Nmap scan report for" in line:
                    ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", line)
                    if ip_match:
                        active_ips.append(ip_match.group(0))
            
            console.print(f"[green]Found {len(active_ips)} active hosts[/green]")
        except Exception as e:
            console.print(f"[red]Error scanning for hosts: {e}[/red]")
        
        return active_ips
    
    def check_ports(self, ip: str, ports: List[int]) -> List[int]:
        """Check if specific ports are open on the target IP"""
        open_ports = []
        
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                if result == 0:
                    open_ports.append(port)
                sock.close()
            except:
                pass
                
        return open_ports
    
    def get_banner(self, ip: str, port: int) -> str:
        """Get service banner from a specific port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((ip, port))
            sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
            banner = sock.recv(1024).decode('utf-8', errors='ignore')
            sock.close()
            return banner
        except:
            return ""
    
    def scan_target(self, ip: str) -> Dict[str, Any]:
        """
        Perform a detailed scan of the target to identify if it's Metasploitable
        
        Args:
            ip: Target IP address
            
        Returns:
            Dict with scan results
        """
        # Return cached result if available and recent
        current_time = time.time()
        if ip in self.scan_results and (current_time - self.last_scan_time) < 300:  # 5-minute cache
            return self.scan_results[ip]
            
        result = {
            "ip": ip,
            "is_metasploitable": False,
            "metasploitable_version": None,
            "open_ports": [],
            "service_banners": {},
            "confidence": 0
        }
        
        # Check common Metasploitable ports
        target_ports = []
        for meta_version, meta_info in self.metasploitable_signatures.items():
            target_ports.extend(meta_info["ports"])
        
        open_ports = self.check_ports(ip, list(set(target_ports)))
        result["open_ports"] = open_ports
        
        if not open_ports:
            return result
        
        # Check service banners for signature matches
        confidence_score = 0
        for port in open_ports:
            banner = self.get_banner(ip, port)
            result["service_banners"][port] = banner
            
            # Check against known signatures
            for meta_version, meta_info in self.metasploitable_signatures.items():
                if port in meta_info["banner_signatures"]:
                    signature = meta_info["banner_signatures"][port]
                    if signature in banner:
                        confidence_score += 1
        
        # Calculate confidence based on ports and banners
        for meta_version, meta_info in self.metasploitable_signatures.items():
            version_confidence = 0
            
            # Check how many expected ports are open
            expected_ports = set(meta_info["ports"])
            found_ports = set(open_ports)
            port_match_ratio = len(found_ports.intersection(expected_ports)) / len(expected_ports)
            version_confidence += port_match_ratio * 50  # 50% weight for port matches
            
            # Check banner matches
            banner_matches = 0
            for port, signature in meta_info["banner_signatures"].items():
                if port in result["service_banners"] and signature in result["service_banners"][port]:
                    banner_matches += 1
            
            if banner_matches > 0:
                banner_match_ratio = banner_matches / len(meta_info["banner_signatures"])
                version_confidence += banner_match_ratio * 50  # 50% weight for banner matches
            
            if version_confidence > confidence_score:
                confidence_score = version_confidence
                result["metasploitable_version"] = meta_version
        
        # Need at least 60% confidence to consider it Metasploitable
        result["confidence"] = confidence_score
        result["is_metasploitable"] = confidence_score >= 60
        
        # Cache the result
        self.scan_results[ip] = result
        self.last_scan_time = current_time
        
        return result
    
    def detect_metasploitable(self) -> List[Dict[str, Any]]:
        """
        Detect Metasploitable VMs on the local network
        
        Returns:
            List of detected Metasploitable targets with their details
        """
        console.print(Panel.fit("[bold cyan]🔍 Scanning for Metasploitable VMs...[/bold cyan]"))
        
        # First check if we have a manually specified target
        if self.target_ip:
            console.print(f"[cyan]Manually specified target IP: {self.target_ip}[/cyan]")
            scan_result = self.scan_target(self.target_ip)
            if scan_result["is_metasploitable"]:
                self.targets = [scan_result]
                self.active_target = scan_result
                console.print(f"[green]✅ Confirmed: {self.target_ip} is Metasploitable {scan_result['metasploitable_version']} (Confidence: {scan_result['confidence']}%)[/green]")
                return self.targets
            else:
                console.print(f"[yellow]⚠️ Warning: {self.target_ip} does not appear to be Metasploitable (Confidence: {scan_result['confidence']}%)[/yellow]")
                # Continue with auto-detection
        
        # If VirtualBox isn't running, we can't auto-detect
        if not self.is_virtualbox_running():
            console.print("[yellow]⚠️ VirtualBox not detected as running. Metasploitable VM may not be active.[/yellow]")
        
        # Scan local network
        detected_targets = []
        ip_ranges = self.get_local_ip_range()
        
        for ip_range in ip_ranges:
            active_hosts = self.scan_for_hosts(ip_range)
            
            for ip in active_hosts:
                scan_result = self.scan_target(ip)
                
                if scan_result["is_metasploitable"]:
                    console.print(f"[green]✅ Detected Metasploitable at {ip} (Version: {scan_result['metasploitable_version']}, Confidence: {scan_result['confidence']}%)[/green]")
                    detected_targets.append(scan_result)
        
        if detected_targets:
            # Sort by confidence score (descending)
            detected_targets.sort(key=lambda x: x["confidence"], reverse=True)
            self.targets = detected_targets
            self.active_target = detected_targets[0]  # Use the highest confidence target
            console.print(f"[green]✅ Selected {self.active_target['ip']} as active target[/green]")
        else:
            console.print("[yellow]⚠️ No Metasploitable VMs detected. Live training may not work correctly.[/yellow]")
            console.print("[yellow]Consider manually specifying target IP in .env file or starting the Metasploitable VM.[/yellow]")
        
        return detected_targets
    
    def get_target_info(self) -> Dict[str, Any]:
        """
        Get information about the current active target
        
        Returns:
            Dict with target details for agents to use
        """
        if not self.active_target:
            if self.live_mode and self.target_ip:
                # Try to use the manually specified target
                scan_result = self.scan_target(self.target_ip)
                self.active_target = scan_result
            else:
                return {
                    "available": False,
                    "reason": "No active target configured",
                    "mode": "simulated" 
                }
        
        return {
            "available": True,
            "ip": self.active_target["ip"],
            "is_metasploitable": self.active_target["is_metasploitable"],
            "version": self.active_target["metasploitable_version"],
            "open_ports": self.active_target["open_ports"],
            "mode": "live",
            "port_range": self.port_range
        }
    
    def switch_to_live_mode(self, target_ip: str = None) -> bool:
        """
        Switch the system to live mode with the specified target
        
        Args:
            target_ip: Target IP address (optional)
            
        Returns:
            True if switch succeeded, False otherwise
        """
        if target_ip:
            self.target_ip = target_ip
            scan_result = self.scan_target(target_ip)
            
            if not scan_result["is_metasploitable"]:
                console.print(f"[yellow]⚠️ Warning: {target_ip} may not be Metasploitable (Confidence: {scan_result['confidence']}%)[/yellow]")
                console.print("[yellow]Proceeding anyway with caution...[/yellow]")
            
            self.active_target = scan_result
            self.live_mode = True
            
            # Update environment variables in memory
            os.environ["ARIASKA_MODE"] = "live"
            os.environ["ARIASKA_TARGET_IP"] = target_ip
            return True
        elif self.targets:
            # Use the best detected target
            self.active_target = self.targets[0]
            self.target_ip = self.active_target["ip"]
            self.live_mode = True
            
            # Update environment variables in memory
            os.environ["ARIASKA_MODE"] = "live"
            os.environ["ARIASKA_TARGET_IP"] = self.target_ip
            return True
        else:
            console.print("[red]Error: No target specified and no Metasploitable VMs detected[/red]")
            return False
    
    def switch_to_simulated_mode(self) -> bool:
        """
        Switch the system to simulated mode
        
        Returns:
            True always (operation always succeeds)
        """
        self.live_mode = False
        os.environ["ARIASKA_MODE"] = "simulated"
        return True
    
    def print_target_summary(self):
        """Print a summary of the current target and mode"""
        table = Table(title="ARIASKA_RL Environment Configuration")
        
        # Add columns
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        # Mode information
        table.add_row("Mode", f"[bold]{'LIVE' if self.live_mode else 'SIMULATED'}[/bold]")
        
        if self.active_target:
            table.add_row("Target IP", self.active_target["ip"])
            table.add_row("Target Type", f"Metasploitable {self.active_target['metasploitable_version']}" 
                         if self.active_target["is_metasploitable"] else "Unknown")
            table.add_row("Confidence", f"{self.active_target['confidence']}%")
            table.add_row("Open Ports", ", ".join(map(str, self.active_target["open_ports"])))
            table.add_row("Port Range", self.port_range)
        elif self.live_mode:
            table.add_row("Target IP", "Not detected")
            table.add_row("Status", "[red]No valid target configured for live mode[/red]")
        else:
            table.add_row("Target", "N/A (Simulated Mode)")
        
        console.print(table)

def main():
    """Command-line interface for testing Metasploitable detection"""
    handler = MetasploitableHandler()
    handler.print_target_summary()

if __name__ == "__main__":
    main()