"""
core/execution/ssh_pool.py - Phase 40.1: Persistent SSH Session Pool

Maintains persistent SSH connections to target hosts, eliminating
per-command SSH handshake overhead (saves ~1-2s per remote command).
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger("ariaska.ssh_pool")

try:
    import paramiko
    _HAS_PARAMIKO = True
except ImportError:
    paramiko = None
    _HAS_PARAMIKO = False
    logger.info("[SSH-POOL] paramiko not installed - using sshpass fallback")


@dataclass
class SSHCredential:
    username: str
    password: str
    port: int = 22


@dataclass
class SSHSession:
    credential: SSHCredential
    host: str
    client: Optional[object] = None
    last_used: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    connected: bool = False
    connect_attempts: int = 0
    max_retries: int = 3

    def __post_init__(self) -> None:
        self.last_used = time.time()


class SSHSessionPool:
    """Thread-safe pool of persistent SSH connections."""

    def __init__(
        self,
        keepalive_interval: int = 30,
        connect_timeout: int = 10,
        command_timeout: int = 30,
    ):
        self._sessions: Dict[str, SSHSession] = {}
        self._credentials: Dict[str, SSHCredential] = {}
        self._lock = threading.Lock()
        self._keepalive_interval = keepalive_interval
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "reconnections": 0,
            "fallback_used": 0,
            "commands_executed": 0,
        }
        logger.info(
            f"[SSH-POOL] Initialized (paramiko={_HAS_PARAMIKO}, "
            f"keepalive={keepalive_interval}s, timeout={connect_timeout}s)"
        )

    def add_credentials(
        self, username: str, password: str, host: str, port: int = 22
    ) -> None:
        key = f"{username}@{host}:{port}"
        with self._lock:
            self._credentials[host] = SSHCredential(
                username=username, password=password, port=port
            )
        logger.info(f"[SSH-POOL] Credentials registered: {key}")

    def has_credentials(self, host: str) -> bool:
        with self._lock:
            return host in self._credentials

    def _get_or_create_session(self, host: str) -> Optional[SSHSession]:
        with self._lock:
            cred = self._credentials.get(host)
            if not cred:
                return None
            session = self._sessions.get(host)
            if session is None:
                session = SSHSession(credential=cred, host=host)
                self._sessions[host] = session
            return session

    def _connect(self, session: SSHSession) -> bool:
        if not _HAS_PARAMIKO or paramiko is None:
            return False
        with session.lock:
            if session.connected and session.client is not None:
                try:
                    transport = session.client.get_transport()
                    if transport and transport.is_active():
                        session.last_used = time.time()
                        self._stats["connections_reused"] += 1
                        return True
                except Exception:
                    pass
                session.connected = False
                self._stats["reconnections"] += 1
            if session.connect_attempts >= session.max_retries:
                return False
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    hostname=session.host,
                    port=session.credential.port,
                    username=session.credential.username,
                    password=session.credential.password,
                    timeout=self._connect_timeout,
                    allow_agent=False,
                    look_for_keys=False,
                )
                transport = client.get_transport()
                if transport:
                    transport.set_keepalive(self._keepalive_interval)
                session.client = client
                session.connected = True
                session.connect_attempts = 0
                session.last_used = time.time()
                self._stats["connections_created"] += 1
                logger.info(
                    f"[SSH-POOL] Connected to {session.credential.username}@"
                    f"{session.host}:{session.credential.port}"
                )
                return True
            except Exception as e:
                session.connect_attempts += 1
                logger.warning(f"[SSH-POOL] Connection failed: {e}")
                return False

    def execute(
        self,
        host: str,
        command: str,
        timeout: Optional[int] = None,
    ) -> Tuple[str, str, int]:
        self._stats["commands_executed"] += 1
        cmd_timeout = timeout or self._command_timeout

        session = self._get_or_create_session(host)
        if session is None:
            return ("", "No credentials registered for host", 1)

        if _HAS_PARAMIKO and self._connect(session):
            try:
                with session.lock:
                    if session.client is None:
                        raise RuntimeError("Client is None")
                    _, stdout_ch, stderr_ch = session.client.exec_command(
                        command, timeout=cmd_timeout
                    )
                    stdout = stdout_ch.read().decode("utf-8", errors="replace")
                    stderr = stderr_ch.read().decode("utf-8", errors="replace")
                    exit_code = stdout_ch.channel.recv_exit_status()
                    session.last_used = time.time()
                    return (stdout, stderr, exit_code)
            except Exception as e:
                logger.warning(f"[SSH-POOL] Paramiko exec failed: {e}")
                session.connected = False

        return self._execute_sshpass(session.credential, host, command, cmd_timeout)

    def _execute_sshpass(
        self,
        cred: SSHCredential,
        host: str,
        command: str,
        timeout: int,
    ) -> Tuple[str, str, int]:
        self._stats["fallback_used"] += 1
        ssh_cmd = [
            "sshpass", "-p", cred.password,
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-p", str(cred.port),
            f"{cred.username}@{host}",
            command,
        ]
        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout,
            )
            return (result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return ("", f"Command timed out after {timeout}s", 124)
        except FileNotFoundError:
            return ("", "sshpass not found", 127)
        except Exception as e:
            return ("", f"SSH fallback error: {e}", 1)

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def active_sessions(self) -> int:
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.connected)

    def close_all(self) -> None:
        with self._lock:
            for host, session in self._sessions.items():
                with session.lock:
                    if session.client is not None and session.connected:
                        try:
                            session.client.close()
                        except Exception:
                            pass
                    session.connected = False
                    session.client = None
            self._sessions.clear()
        logger.info("[SSH-POOL] All sessions closed")

    def reset_retries(self, host: str) -> None:
        with self._lock:
            session = self._sessions.get(host)
            if session:
                session.connect_attempts = 0
