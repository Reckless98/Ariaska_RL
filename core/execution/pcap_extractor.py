#!/usr/bin/env python3
"""
core/execution/pcap_extractor.py — PCAP credential extraction pipeline.

Two-stage extraction for binary PCAP files:
  Stage 1: tshark (structured, reliable) — extracts FTP USER/PASS, HTTP auth
  Stage 2: strings (heuristic fallback) — catches any credential patterns

Used by SmartOrchestrator when a PCAP file is downloaded (via ArtifactStore)
or when command output contains binary PCAP data.

Architecture:
    ArtifactStore captures .pcap → PcapExtractor.extract_credentials(path)
        Stage 1: tshark -r <path> -Y ftp.request.command ...
        Stage 2: strings <path> | grep -i "USER|PASS" ...
        → List[DiscoveredCredential]

The HTB Cap box attack chain:
    download_pcap → extract FTP creds (nathan:Buck3tH4TF0RM3!) → SSH reuse

Author: Filip Volf / Ariaska System
Phase: HTB Capability Upgrade — T0.2
"""

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.pcap_extractor")


@dataclass
class PcapCredential:
    """A credential extracted from PCAP analysis."""
    username: str
    password: str
    protocol: str    # "ftp", "http", "telnet", etc.
    source_file: str
    extraction_method: str  # "tshark" or "strings"

    def to_discovered_credential(self) -> "DiscoveredCredential":
        """Convert to DiscoveredCredential for the reuse engine."""
        from core.execution.cred_reuse import DiscoveredCredential
        return DiscoveredCredential(
            username=self.username,
            password=self.password,
            source="pcap",
            source_service=self.protocol,
        )


class PcapExtractor:
    """
    Extract credentials from PCAP files using tshark + strings fallback.

    Usage:
        extractor = PcapExtractor()
        creds = extractor.extract_credentials("/tmp/download.pcap")
        for cred in creds:
            print(f"{cred.username}:{cred.password} via {cred.protocol}")
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._tshark_available: Optional[bool] = None
        self._strings_available: Optional[bool] = None

    @property
    def has_tshark(self) -> bool:
        """Check if tshark is available on the system."""
        if self._tshark_available is None:
            self._tshark_available = shutil.which("tshark") is not None
        return self._tshark_available

    @property
    def has_strings(self) -> bool:
        """Check if strings is available on the system."""
        if self._strings_available is None:
            self._strings_available = shutil.which("strings") is not None
        return self._strings_available

    def extract_credentials(self, pcap_path: str) -> List[PcapCredential]:
        """
        Extract credentials from a PCAP file.

        Tries tshark first (structured), then strings (heuristic).
        Deduplicates results across both methods.

        Args:
            pcap_path: Path to the PCAP file.

        Returns:
            List of PcapCredential objects.
        """
        if not os.path.isfile(pcap_path):
            logger.warning(f"[PCAP] File not found: {pcap_path}")
            return []

        all_creds: List[PcapCredential] = []

        # Stage 1: tshark (preferred — structured extraction)
        if self.has_tshark:
            tshark_creds = self._extract_tshark(pcap_path)
            all_creds.extend(tshark_creds)
            if tshark_creds:
                logger.info(
                    f"[PCAP] tshark found {len(tshark_creds)} credentials "
                    f"in {pcap_path}"
                )

        # Stage 2: strings fallback
        if self.has_strings:
            strings_creds = self._extract_strings(pcap_path)
            # Only add creds not already found by tshark
            existing = {(c.username, c.password) for c in all_creds}
            new_strings = [c for c in strings_creds if (c.username, c.password) not in existing]
            all_creds.extend(new_strings)
            if new_strings:
                logger.info(
                    f"[PCAP] strings found {len(new_strings)} additional credentials"
                )

        # Final dedup
        seen: set = set()
        unique: List[PcapCredential] = []
        for cred in all_creds:
            key = (cred.username, cred.password, cred.protocol)
            if key not in seen:
                seen.add(key)
                unique.append(cred)

        if unique:
            logger.info(
                f"[PCAP] Total unique credentials from {pcap_path}: "
                f"{', '.join(f'{c.username}:{c.password[:3]}***' for c in unique)}"
            )
        else:
            logger.debug(f"[PCAP] No credentials found in {pcap_path}")

        return unique

    def extract_from_output(self, output: str) -> List[PcapCredential]:
        """
        Extract credentials from raw command output that may contain
        PCAP-derived text (e.g., from `strings` or `tshark` already run).

        This is for when we don't have the binary PCAP file, just the
        text output from a pipeline like `strings cap.pcap`.

        Args:
            output: Text output to parse.

        Returns:
            List of PcapCredential objects.
        """
        return self._parse_ftp_credentials(
            output, source_file="<output>", method="output_parse"
        )

    def _extract_tshark(self, pcap_path: str) -> List[PcapCredential]:
        """Extract credentials using tshark structured dissection."""
        creds: List[PcapCredential] = []

        # Extract FTP credentials
        ftp_creds = self._tshark_ftp(pcap_path)
        creds.extend(ftp_creds)

        # Extract HTTP Basic Auth
        http_creds = self._tshark_http_auth(pcap_path)
        creds.extend(http_creds)

        # Phase 19: Extract telnet credentials
        telnet_creds = self._tshark_telnet(pcap_path)
        creds.extend(telnet_creds)

        # Phase 19: Extract SMTP credentials
        smtp_creds = self._tshark_smtp(pcap_path)
        creds.extend(smtp_creds)

        return creds

    def _tshark_ftp(self, pcap_path: str) -> List[PcapCredential]:
        """Extract FTP USER/PASS pairs using tshark."""
        try:
            # Get FTP request commands (USER and PASS)
            result = subprocess.run(
                [
                    "tshark", "-r", pcap_path,
                    "-Y", "ftp.request.command == USER || ftp.request.command == PASS",
                    "-T", "fields",
                    "-e", "ftp.request.command",
                    "-e", "ftp.request.arg",
                ],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0:
                return []

            return self._parse_tshark_ftp_output(
                result.stdout, pcap_path
            )

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"[PCAP] tshark FTP extraction failed: {e}")
            return []

    def _parse_tshark_ftp_output(
        self, output: str, pcap_path: str
    ) -> List[PcapCredential]:
        """Parse tshark -T fields output for USER/PASS pairs."""
        creds: List[PcapCredential] = []
        current_user: Optional[str] = None

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            cmd, arg = parts[0].strip(), parts[1].strip()
            if cmd == "USER":
                current_user = arg
            elif cmd == "PASS" and current_user:
                # Filter out anonymous FTP
                if current_user.lower() not in ("anonymous", "ftp", "guest"):
                    creds.append(PcapCredential(
                        username=current_user,
                        password=arg,
                        protocol="ftp",
                        source_file=pcap_path,
                        extraction_method="tshark",
                    ))
                current_user = None

        return creds

    def _tshark_http_auth(self, pcap_path: str) -> List[PcapCredential]:
        """Extract HTTP Basic Auth credentials using tshark."""
        try:
            result = subprocess.run(
                [
                    "tshark", "-r", pcap_path,
                    "-Y", "http.authorization",
                    "-T", "fields",
                    "-e", "http.authorization",
                ],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []

            creds: List[PcapCredential] = []
            import base64

            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Basic Auth: "Basic base64(user:pass)"
                match = re.match(r"Basic\s+(\S+)", line)
                if match:
                    try:
                        decoded = base64.b64decode(match.group(1)).decode("utf-8", errors="replace")
                        if ":" in decoded:
                            user, passwd = decoded.split(":", 1)
                            creds.append(PcapCredential(
                                username=user,
                                password=passwd,
                                protocol="http",
                                source_file=pcap_path,
                                extraction_method="tshark",
                            ))
                    except Exception:
                        pass

            return creds

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"[PCAP] tshark HTTP auth extraction failed: {e}")
            return []

    def _extract_strings(self, pcap_path: str) -> List[PcapCredential]:
        """Fallback: extract credentials using `strings` + regex."""
        try:
            result = subprocess.run(
                ["strings", pcap_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0:
                return []

            return self._parse_ftp_credentials(
                result.stdout, pcap_path, "strings"
            )

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"[PCAP] strings extraction failed: {e}")
            return []

    def _parse_ftp_credentials(
        self, text: str, source_file: str, method: str
    ) -> List[PcapCredential]:
        """Parse FTP USER/PASS patterns from raw text."""
        creds: List[PcapCredential] = []

        users = re.findall(r'USER\s+(\S+)', text)
        passes = re.findall(r'PASS\s+(\S+)', text)

        if users and passes:
            for user, passwd in zip(users, passes):
                if user.lower() not in ("anonymous", "ftp", "guest"):
                    creds.append(PcapCredential(
                        username=user,
                        password=passwd,
                        protocol="ftp",
                        source_file=source_file,
                        extraction_method=method,
                    ))

        # Also look for generic credential patterns
        generic_patterns = [
            r'(?:login|username)[:\s=]+(\S+)\s+(?:password|passwd)[:\s=]+(\S+)',
            # Phase 19: Enhanced patterns
            r'(?:user|usr)[:\s=]+(\S+)\s+(?:pass|pwd)[:\s=]+(\S+)',
            r'(\w+@\w+\.\w+)[:\s]+(\S{4,})',  # email:password patterns
        ]
        for pattern in generic_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                user, passwd = match.group(1), match.group(2)
                creds.append(PcapCredential(
                    username=user,
                    password=passwd,
                    protocol="generic",
                    source_file=source_file,
                    extraction_method=method,
                ))

        # Phase 19: Telnet credential patterns from strings output
        telnet_user_pass = re.findall(
            r'login:\s*(\S+).*?Password:\s*(\S+)', text, re.IGNORECASE | re.DOTALL
        )
        for user, passwd in telnet_user_pass:
            if user.lower() not in ("", "last"):
                creds.append(PcapCredential(
                    username=user, password=passwd,
                    protocol="telnet", source_file=source_file,
                    extraction_method=method,
                ))

        return creds

    def _tshark_telnet(self, pcap_path: str) -> List[PcapCredential]:
        """Phase 19: Extract telnet credentials using tshark data stream reassembly."""
        try:
            result = subprocess.run(
                [
                    "tshark", "-r", pcap_path,
                    "-Y", "telnet",
                    "-T", "fields",
                    "-e", "telnet.data",
                ],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []

            # Telnet data comes as individual characters — reconstruct
            raw = result.stdout.replace("\n", "").replace("\t", "")
            # Look for login/password sequences in reconstructed data
            return self._parse_ftp_credentials(
                raw, pcap_path, "tshark_telnet"
            )

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"[PCAP] tshark telnet extraction failed: {e}")
            return []

    def _tshark_smtp(self, pcap_path: str) -> List[PcapCredential]:
        """Phase 19: Extract SMTP AUTH credentials using tshark."""
        try:
            result = subprocess.run(
                [
                    "tshark", "-r", pcap_path,
                    "-Y", "smtp.req.command == AUTH",
                    "-T", "fields",
                    "-e", "smtp.req.parameter",
                ],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []

            import base64
            creds: List[PcapCredential] = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # AUTH PLAIN base64(user\x00user\x00pass) or AUTH LOGIN sequences
                parts = line.split()
                for part in parts:
                    try:
                        decoded = base64.b64decode(part).decode("utf-8", errors="replace")
                        # PLAIN format: \x00username\x00password
                        segments = decoded.split("\x00")
                        segments = [s for s in segments if s]
                        if len(segments) >= 2:
                            creds.append(PcapCredential(
                                username=segments[0], password=segments[-1],
                                protocol="smtp", source_file=pcap_path,
                                extraction_method="tshark",
                            ))
                    except Exception:
                        pass

            return creds

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"[PCAP] tshark SMTP extraction failed: {e}")
            return []

    def analyze_pcap_summary(self, pcap_path: str) -> Dict[str, int]:
        """
        Get a protocol summary from a PCAP file.

        Returns:
            Dict mapping protocol names to packet counts.
        """
        if not self.has_tshark:
            return {}

        try:
            result = subprocess.run(
                [
                    "tshark", "-r", pcap_path, "-q", "-z", "io,phs",
                ],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0:
                return {}

            # Parse protocol hierarchy statistics
            protocols: Dict[str, int] = {}
            for line in result.stdout.split("\n"):
                match = re.match(r'\s+(\S+)\s+frames:(\d+)', line)
                if match:
                    protocols[match.group(1)] = int(match.group(2))

            return protocols

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return {}
