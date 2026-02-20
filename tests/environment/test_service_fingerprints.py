"""Phase 42 Stage 4: ServiceFingerprints unit tests."""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestServiceFingerprintDB:
    """Tests for the ServiceFingerprintDB lookup module."""

    def _make_db(self):
        from core.environment.service_fingerprints import ServiceFingerprintDB
        return ServiceFingerprintDB()

    def test_init_has_defaults(self):
        """DB loads default service profiles."""
        db = self._make_db()
        s = db.summary()
        assert s["total_profiles"] >= 10

    def test_lookup_by_service(self):
        """lookup_by_service finds known services."""
        db = self._make_db()
        ssh = db.lookup_by_service("ssh")
        assert ssh is not None
        assert ssh.service_name == "ssh"
        assert 22 in ssh.common_ports

    def test_lookup_by_service_case_insensitive(self):
        """lookup_by_service is case-insensitive."""
        db = self._make_db()
        assert db.lookup_by_service("SSH") is not None
        assert db.lookup_by_service("Http") is not None

    def test_lookup_by_service_unknown(self):
        """lookup_by_service returns None for unknown."""
        db = self._make_db()
        assert db.lookup_by_service("unknown_service_xyz") is None

    def test_lookup_by_port(self):
        """lookup_by_port finds profiles for common ports."""
        db = self._make_db()
        profiles = db.lookup_by_port(22)
        assert len(profiles) >= 1
        assert any(p.service_name == "ssh" for p in profiles)

    def test_lookup_by_port_445(self):
        """Port 445 maps to SMB."""
        db = self._make_db()
        profiles = db.lookup_by_port(445)
        assert any(p.service_name == "smb" for p in profiles)

    def test_get_default_creds(self):
        """get_default_creds returns credentials."""
        db = self._make_db()
        creds = db.get_default_creds("ftp")
        assert len(creds) >= 1
        assert any(u == "anonymous" for u, _ in creds)

    def test_get_exploit_paths(self):
        """get_exploit_paths returns paths."""
        db = self._make_db()
        paths = db.get_exploit_paths("smb")
        assert len(paths) >= 1

    def test_get_enum_commands(self):
        """get_enum_commands returns commands."""
        db = self._make_db()
        cmds = db.get_enum_commands("http")
        assert len(cmds) >= 1
        assert any("nikto" in c or "gobuster" in c for c in cmds)

    def test_match_version(self):
        """match_version finds CVE patterns."""
        db = self._make_db()
        matches = db.match_version("ftp", "vsftpd 2.3.4")
        assert len(matches) >= 1
        assert any("backdoor" in m for m in matches)

    def test_match_version_no_match(self):
        """match_version returns empty for no matches."""
        db = self._make_db()
        matches = db.match_version("ssh", "OpenSSH 9.0")
        assert len(matches) == 0

    def test_add_custom_profile(self):
        """add_profile registers custom profiles."""
        from core.environment.service_fingerprints import ServiceProfile
        db = self._make_db()
        db.add_profile(ServiceProfile(
            service_name="custom_svc",
            common_ports=[9999],
            default_creds=[("admin", "secret")],
        ))
        assert db.lookup_by_service("custom_svc") is not None
        profiles = db.lookup_by_port(9999)
        assert len(profiles) == 1

    def test_summary(self):
        """summary returns correct counts."""
        db = self._make_db()
        s = db.summary()
        assert "total_profiles" in s
        assert "indexed_ports" in s
        assert "services" in s
        assert "ssh" in s["services"]
