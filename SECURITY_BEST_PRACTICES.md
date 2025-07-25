# 🔒 Security Best Practices for ARIASKA_RL

## Overview

This document outlines essential security practices for developing, deploying, and maintaining the ARIASKA_RL platform. Given the cybersecurity focus of this project, security considerations are paramount.

## 🛡️ Core Security Principles

### 1. Defense in Depth
- Implement multiple layers of security controls
- Assume each layer may be compromised
- Design redundant security measures

### 2. Principle of Least Privilege
- Grant minimum necessary permissions
- Regularly review and rotate access credentials
- Use role-based access control (RBAC)

### 3. Secure by Default
- Default configurations should be secure
- Force users to explicitly enable risky features
- Provide secure templates and examples

## 🔐 API Security

### OpenAI API Key Management

**❌ Never do this:**
```python
# DON'T: Hardcode API keys
openai.api_key = "sk-1234567890abcdef..."

# DON'T: Commit keys to version control
API_KEY = "sk-1234567890abcdef..."
```

**✅ Best practices:**
```python
# DO: Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# DO: Validate key format
def validate_api_key(key: str) -> bool:
    if not key or not key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key format")
    if len(key) < 20:
        raise ValueError("API key too short")
    return True

# DO: Use secure credential storage
from cryptography.fernet import Fernet

class SecureCredentialStore:
    def __init__(self, key_file: str = ".key"):
        self.key_file = key_file
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.key_file, 0o600)  # Owner read/write only
            return key
    
    def store_credential(self, name: str, value: str):
        encrypted = self.cipher.encrypt(value.encode())
        with open(f".{name}.enc", 'wb') as f:
            f.write(encrypted)
        os.chmod(f".{name}.enc", 0o600)
    
    def load_credential(self, name: str) -> str:
        with open(f".{name}.enc", 'rb') as f:
            encrypted = f.read()
        return self.cipher.decrypt(encrypted).decode()
```

### Rate Limiting

```python
import time
from collections import defaultdict
from typing import Dict, Tuple

class APIRateLimiter:
    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: Dict[str, List[float]] = defaultdict(list)
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, float]:
        """Check if request is within rate limits"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old calls
        self.calls[identifier] = [
            call_time for call_time in self.calls[identifier] 
            if call_time > window_start
        ]
        
        # Check if under limit
        if len(self.calls[identifier]) < self.max_calls:
            self.calls[identifier].append(now)
            return True, 0.0
        
        # Calculate wait time
        oldest_call = min(self.calls[identifier])
        wait_time = self.window_seconds - (now - oldest_call)
        return False, wait_time

# Usage
rate_limiter = APIRateLimiter(max_calls=60, window_seconds=60)

def make_api_call(prompt: str) -> str:
    can_proceed, wait_time = rate_limiter.check_rate_limit("openai")
    if not can_proceed:
        time.sleep(wait_time)
    
    # Make actual API call
    return openai.ChatCompletion.create(...)
```

## 🛡️ Input Validation & Sanitization

### User Input Validation

```python
import re
import html
from typing import Any, Dict, List

class InputValidator:
    """Secure input validation utility"""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # XSS
        r'javascript:',              # JavaScript injection
        r'data:text/html',          # Data URL injection
        r'vbscript:',               # VBScript injection
        r'on\w+\s*=',               # Event handlers
        r'eval\s*\(',               # eval() calls
        r'exec\s*\(',               # exec() calls
        r'import\s+os',             # OS imports
        r'__import__',              # Dynamic imports
        r'subprocess',              # Subprocess calls
    ]
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Limit length
        if len(value) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")
        
        # HTML escape
        value = html.escape(value)
        
        # Check for dangerous patterns
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        
        return value
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address and check if it's in allowed ranges"""
        import ipaddress
        
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return False
        
        # Block dangerous IP ranges
        dangerous_ranges = [
            ipaddress.ip_network("127.0.0.0/8"),    # Loopback
            ipaddress.ip_network("169.254.0.0/16"), # Link-local
            ipaddress.ip_network("224.0.0.0/4"),    # Multicast
        ]
        
        for dangerous_range in dangerous_ranges:
            if ip_obj in dangerous_range:
                return False
        
        return True
    
    @staticmethod
    def validate_file_path(path: str, allowed_dirs: List[str]) -> bool:
        """Validate file path to prevent directory traversal"""
        import os.path
        
        # Normalize path
        norm_path = os.path.normpath(path)
        
        # Check for directory traversal attempts
        if ".." in norm_path or norm_path.startswith("/"):
            return False
        
        # Check if path is within allowed directories
        abs_path = os.path.abspath(norm_path)
        for allowed_dir in allowed_dirs:
            if abs_path.startswith(os.path.abspath(allowed_dir)):
                return True
        
        return False

# Usage examples
validator = InputValidator()

# Sanitize user prompt
user_prompt = validator.sanitize_string(user_input, max_length=500)

# Validate target IP
if not validator.validate_ip_address(target_ip):
    raise ValueError("Invalid or dangerous IP address")

# Validate file paths
if not validator.validate_file_path(file_path, ["data/", "logs/"]):
    raise ValueError("File path not allowed")
```

### Command Injection Prevention

```python
import subprocess
import shlex
from typing import List, Optional

class SecureCommandExecutor:
    """Secure command execution utility"""
    
    # Allowed commands (whitelist approach)
    ALLOWED_COMMANDS = {
        'nmap': ['-sS', '-sT', '-sU', '-p', '-O', '--script'],
        'ping': ['-c', '-W', '-i'],
        'traceroute': ['-m', '-w', '-q'],
    }
    
    @staticmethod
    def validate_command(command: str, args: List[str]) -> bool:
        """Validate command and arguments"""
        if command not in SecureCommandExecutor.ALLOWED_COMMANDS:
            return False
        
        allowed_args = SecureCommandExecutor.ALLOWED_COMMANDS[command]
        
        for arg in args:
            # Check if argument starts with allowed prefix
            if not any(arg.startswith(prefix) for prefix in allowed_args):
                # Allow IP addresses and hostnames
                if not re.match(r'^[\w\.-]+$', arg):
                    return False
        
        return True
    
    @staticmethod
    def execute_command(
        command: str, 
        args: List[str], 
        timeout: int = 30
    ) -> subprocess.CompletedProcess:
        """Safely execute system command"""
        
        # Validate command
        if not SecureCommandExecutor.validate_command(command, args):
            raise ValueError(f"Command not allowed: {command} {args}")
        
        # Build command list (no shell=True)
        cmd_list = [command] + args
        
        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise on non-zero exit
            )
            return result
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {command}")

# Usage
executor = SecureCommandExecutor()
result = executor.execute_command("nmap", ["-sS", "192.168.1.1"])
```

## 🔐 Secure Logging

```python
import logging
import re
from typing import Dict, Any

class SecureLogger:
    """Secure logging utility that prevents sensitive data leakage"""
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS = {
        'api_key': re.compile(r'sk-[a-zA-Z0-9]{32,}'),
        'password': re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
        'token': re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
        'secret': re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
        'credit_card': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
        'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    }
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages"""
        sanitized = message
        
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern_name == 'ip_address':
                # Partially mask IP addresses
                sanitized = pattern.sub(lambda m: f"{m.group()[:7]}***", sanitized)
            else:
                # Fully mask other sensitive data
                sanitized = pattern.sub("[REDACTED]", sanitized)
        
        return sanitized
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with sanitization"""
        sanitized = self.sanitize_message(message)
        self.logger.info(sanitized, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with sanitization"""
        sanitized = self.sanitize_message(message)
        self.logger.error(sanitized, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with sanitization"""
        sanitized = self.sanitize_message(message)
        self.logger.debug(sanitized, *args, **kwargs)

# Usage
secure_logger = SecureLogger("ariaska.security")
secure_logger.info("API call successful with key sk-abc123...")  # Logs: "API call successful with key [REDACTED]"
```

## 🛡️ Secure Configuration

### Environment-based Configuration

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class SecurityConfig:
    """Security configuration container"""
    safe_mode: bool = True
    max_api_calls_per_minute: int = 60
    allowed_ip_ranges: List[str] = None
    enable_audit_logging: bool = True
    validate_ssl_certs: bool = True
    max_file_size_mb: int = 100
    session_timeout_minutes: int = 30
    
    def __post_init__(self):
        if self.allowed_ip_ranges is None:
            self.allowed_ip_ranges = ["192.168.0.0/16", "10.0.0.0/8"]
    
    @classmethod
    def from_environment(cls) -> 'SecurityConfig':
        """Load security config from environment variables"""
        return cls(
            safe_mode=os.getenv("SAFE_MODE", "true").lower() == "true",
            max_api_calls_per_minute=int(os.getenv("MAX_API_CALLS_PER_MINUTE", "60")),
            allowed_ip_ranges=os.getenv("ALLOWED_IP_RANGES", "192.168.0.0/16,10.0.0.0/8").split(","),
            enable_audit_logging=os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true",
            validate_ssl_certs=os.getenv("VALIDATE_SSL_CERTS", "true").lower() == "true",
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),
        )
```

## 🔍 Audit Logging

```python
import json
import time
from typing import Dict, Any, Optional

class AuditLogger:
    """Audit logging for security-sensitive operations"""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("ariaska.audit")
        
        # Create audit log handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ):
        """Log security-relevant events"""
        audit_record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "success": success,
            "details": details or {}
        }
        
        self.logger.info(json.dumps(audit_record))
    
    def log_api_call(self, endpoint: str, user_id: str, success: bool):
        """Log API calls"""
        self.log_security_event(
            event_type="api_call",
            user_id=user_id,
            details={"endpoint": endpoint},
            success=success
        )
    
    def log_file_access(self, file_path: str, user_id: str, action: str):
        """Log file access attempts"""
        self.log_security_event(
            event_type="file_access",
            user_id=user_id,
            details={"file_path": file_path, "action": action},
            success=True
        )

# Usage
audit_logger = AuditLogger()
audit_logger.log_api_call("/api/train", "user123", True)
audit_logger.log_file_access("/data/models/agent.pt", "user123", "read")
```

## 🔒 Secure Deployment Checklist

### Development Environment

- [ ] Use virtual environments
- [ ] Never commit secrets to version control
- [ ] Use `.env` files for local development
- [ ] Enable all linting and security scanning tools
- [ ] Regular dependency vulnerability scans

### Production Environment

- [ ] Use environment variables for all secrets
- [ ] Enable SSL/TLS for all communications
- [ ] Implement proper authentication and authorization
- [ ] Set up comprehensive logging and monitoring
- [ ] Regular security updates and patches
- [ ] Network segmentation and firewalls
- [ ] Backup and disaster recovery procedures

### Container Security

```dockerfile
# Dockerfile security best practices
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash ariaska

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app
RUN chown ariaska:ariaska /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=ariaska:ariaska . .

# Switch to non-root user
USER ariaska

# Remove write permissions from application files
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "main.py"]
```

## 🛡️ Security Monitoring

```python
import time
import threading
from collections import defaultdict
from typing import Dict, List

class SecurityMonitor:
    """Monitor for suspicious activities"""
    
    def __init__(self):
        self.failed_attempts = defaultdict(list)
        self.suspicious_patterns = []
        self.monitoring = True
        self.alert_thresholds = {
            'failed_logins': 5,
            'rapid_requests': 100,
            'large_file_access': 10
        }
    
    def record_failed_attempt(self, identifier: str, attempt_type: str):
        """Record security failure"""
        now = time.time()
        self.failed_attempts[identifier].append((now, attempt_type))
        
        # Clean old attempts (last hour)
        cutoff = now - 3600
        self.failed_attempts[identifier] = [
            (timestamp, attempt) for timestamp, attempt in self.failed_attempts[identifier]
            if timestamp > cutoff
        ]
        
        # Check if threshold exceeded
        if len(self.failed_attempts[identifier]) >= self.alert_thresholds.get(attempt_type, 5):
            self.trigger_alert(f"Multiple {attempt_type} from {identifier}")
    
    def trigger_alert(self, message: str):
        """Trigger security alert"""
        alert = {
            'timestamp': time.time(),
            'message': message,
            'severity': 'HIGH'
        }
        
        # Log alert
        logging.getLogger("ariaska.security").critical(f"SECURITY ALERT: {message}")
        
        # Additional alerting mechanisms (email, webhook, etc.)
        self.send_security_notification(alert)
    
    def send_security_notification(self, alert: Dict):
        """Send security notification (implement based on your setup)"""
        # Could send email, Slack message, webhook, etc.
        pass

# Usage
security_monitor = SecurityMonitor()
security_monitor.record_failed_attempt("192.168.1.100", "failed_logins")
```

## 📋 Security Review Checklist

### Code Review Security Checklist

- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented for all user inputs
- [ ] Output encoding to prevent injection attacks
- [ ] Error handling doesn't leak sensitive information
- [ ] Authentication and authorization properly implemented
- [ ] Logging doesn't expose sensitive data
- [ ] Dependencies are up-to-date and vulnerability-free
- [ ] Secure communication protocols used
- [ ] Proper session management
- [ ] Rate limiting implemented for APIs

### Deployment Security Checklist

- [ ] All secrets stored in secure vaults
- [ ] Network access properly restricted
- [ ] SSL/TLS certificates properly configured
- [ ] Security headers implemented
- [ ] Monitoring and alerting configured
- [ ] Backup procedures in place
- [ ] Incident response plan documented
- [ ] Regular security assessments scheduled

---

## 🆘 Incident Response

### Security Incident Response Plan

1. **Immediate Response**
   - Isolate affected systems
   - Preserve evidence
   - Notify security team

2. **Assessment**
   - Determine scope of breach
   - Identify affected data
   - Assess potential impact

3. **Containment**
   - Stop ongoing attack
   - Prevent further damage
   - Secure systems

4. **Recovery**
   - Restore systems from clean backups
   - Apply security patches
   - Implement additional controls

5. **Lessons Learned**
   - Document incident
   - Update security procedures
   - Conduct security training

---

*Remember: Security is an ongoing process, not a one-time setup. Regularly review and update security measures as threats evolve.*