"""Smoke tests for LLM-primary output parser refactor."""
from core.execution.output_parser import OutputParser, ParsedOutput


def test_regex_fallback_nmap():
    """Without GPTManager, regex fallback parses nmap output correctly."""
    p = OutputParser()  # No gpt_manager — pure regex
    nmap_output = (
        "Starting Nmap 7.92\n"
        "22/tcp   open  ssh     OpenSSH 7.2p2 Ubuntu\n"
        "80/tcp   open  http    Apache httpd 2.4.18\n"
        "443/tcp  open  https   nginx 1.14.0\n"
        "OS details: Linux 4.15.0-20-generic\n"
        "Nmap done: 1 IP address\n"
    )
    result = p.parse("nmap -sV 10.10.10.10", nmap_output)
    assert result.open_ports == [22, 80, 443]
    assert result.success is True
    assert "Linux" in result.os_info
    assert result.phase == "recon"


def test_regex_fallback_hydra():
    """Without GPTManager, regex fallback parses hydra output."""
    p = OutputParser()
    hydra_output = "[22][ssh] host: 10.10.10.10 login: admin password: secret123"
    result = p.parse("hydra -l admin -P wordlist.txt ssh://10.10.10.10", hydra_output)
    assert len(result.credentials) == 1
    assert result.credentials[0]["username"] == "admin"
    assert result.credentials[0]["password"] == "secret123"


def test_json_to_parsed_output():
    """Static conversion from JSON dict to ParsedOutput."""
    data = {
        "open_ports": [22, 80, 443],
        "services": {"22": "ssh OpenSSH 7.2p2", "80": "http Apache 2.4.18"},
        "os_info": "Linux 4.15",
        "users": ["admin", "root"],
        "vulnerabilities": ["CVE-2021-12345"],
        "success": True,
    }
    result = OutputParser._json_to_parsed_output(data, "nmap -sV target", "nmap")
    assert result.open_ports == [22, 80, 443]
    assert result.services[22] == "ssh OpenSSH 7.2p2"
    assert result.os_info == "Linux 4.15"
    assert "admin" in result.users
    assert result.success is True
    assert result.discovery_count > 0


def test_json_to_parsed_output_empty():
    """Empty JSON produces empty ParsedOutput."""
    result = OutputParser._json_to_parsed_output({}, "cmd", "unknown")
    assert result.discovery_count == 0
    assert result.open_ports == []


def test_smart_output_parser_no_gpt():
    """SmartOutputParser without GPTManager still works (regex path)."""
    from core.execution.smart_output_parser import SmartOutputParser
    parser = SmartOutputParser(gpt_manager=None, enable_llm=False)
    nmap_output = (
        "22/tcp   open  ssh     OpenSSH 7.2p2\n"
        "80/tcp   open  http    Apache 2.4.18\n"
    )
    result = parser.parse(command="nmap -sV 10.10.10.10", output=nmap_output)
    assert "open_port" in result
    assert 22 in result["open_port"]
    assert 80 in result["open_port"]


if __name__ == "__main__":
    test_regex_fallback_nmap()
    test_regex_fallback_hydra()
    test_json_to_parsed_output()
    test_json_to_parsed_output_empty()
    test_smart_output_parser_no_gpt()
    print("All tests passed!")
