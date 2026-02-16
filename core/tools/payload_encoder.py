#!/usr/bin/env python3
"""
core/tools/payload_encoder.py — Phase 10.1F: Payload Encoding Engine

Context-aware payload encoding and obfuscation for exploit delivery.
Selects encoding transforms based on:
  • Target platform (Linux/Windows)
  • WAF/IDS detection risk
  • Attack phase (exploitation vs post-exploitation)
  • Delivery channel (web form, header, command injection, file upload)

Each transform is a pure function: bytes/str → bytes/str, composable
via chaining.  The engine tracks which transforms were applied for
telemetry and replay.

Architecture:
    EncodingTransform — individual encoding operation
    PayloadEncoder — select and apply transforms for a given context
    EncodingChain — ordered sequence of transforms applied

Usage:
    from core.tools.payload_encoder import PayloadEncoder
    encoder = PayloadEncoder()
    result = encoder.encode(payload, context={"channel": "web_form", "waf": True})
"""

import base64
import logging
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.payload_encoder")


# ============================================================================
# ENUMS
# ============================================================================

class EncodingType(Enum):
    """Types of encoding transforms."""
    BASE64 = "base64"
    URL_ENCODE = "url_encode"
    DOUBLE_URL = "double_url_encode"
    HTML_ENTITY = "html_entity"
    UNICODE_ESCAPE = "unicode_escape"
    HEX_ENCODE = "hex_encode"
    POWERSHELL_BASE64 = "powershell_base64"
    BASH_OCTAL = "bash_octal"
    BASH_HEX = "bash_hex"
    NULL_BYTE_INSERT = "null_byte_insert"
    CASE_TOGGLE = "case_toggle"
    WHITESPACE_BYPASS = "whitespace_bypass"
    COMMENT_INSERT = "comment_insert"


class DeliveryChannel(Enum):
    """How the payload will be delivered."""
    WEB_FORM = "web_form"
    URL_PARAM = "url_param"
    HTTP_HEADER = "http_header"
    COMMAND_INJECTION = "command_injection"
    FILE_UPLOAD = "file_upload"
    SQL_INJECTION = "sql_injection"
    SHELL_COMMAND = "shell_command"


class TargetPlatform(Enum):
    """Target OS platform."""
    LINUX = "linux"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


# ============================================================================
# DATA OBJECTS
# ============================================================================

@dataclass
class EncodingContext:
    """Context for payload encoding decisions.

    Attributes:
        channel: How the payload will be delivered
        platform: Target operating system
        waf_detected: Whether WAF/IDS was detected
        detection_risk: Current detection risk level (0.0-1.0)
        phase: Current attack phase
        double_encode: Whether to apply double encoding
        custom_transforms: Additional transforms to apply
    """
    channel: DeliveryChannel = DeliveryChannel.SHELL_COMMAND
    platform: TargetPlatform = TargetPlatform.LINUX
    waf_detected: bool = False
    detection_risk: float = 0.0
    phase: str = "EXPLOITATION"
    double_encode: bool = False
    custom_transforms: List[EncodingType] = field(default_factory=list)


@dataclass
class EncodingResult:
    """Result of payload encoding.

    Attributes:
        original: The original payload
        encoded: The encoded payload
        transforms_applied: Ordered list of transforms used
        encoding_chain: Human-readable description
        reversible: Whether the encoding is fully reversible
    """
    original: str
    encoded: str
    transforms_applied: List[EncodingType] = field(default_factory=list)
    encoding_chain: str = ""
    reversible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_length": len(self.original),
            "encoded_length": len(self.encoded),
            "transforms": [t.value for t in self.transforms_applied],
            "chain": self.encoding_chain,
            "reversible": self.reversible,
        }


@dataclass
class EncoderTelemetry:
    """Telemetry for payload encoding operations."""
    total_encodes: int = 0
    transforms_used: Dict[str, int] = field(default_factory=dict)
    waf_bypasses_attempted: int = 0
    channels_used: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_encodes": self.total_encodes,
            "transforms_used": dict(self.transforms_used),
            "waf_bypasses_attempted": self.waf_bypasses_attempted,
            "channels_used": dict(self.channels_used),
        }


# ============================================================================
# TRANSFORM IMPLEMENTATIONS
# ============================================================================

def _base64_encode(payload: str) -> str:
    """Standard base64 encoding."""
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


def _url_encode(payload: str) -> str:
    """URL percent encoding."""
    return urllib.parse.quote(payload, safe="")


def _double_url_encode(payload: str) -> str:
    """Double URL encoding (encode the percent signs too)."""
    first = urllib.parse.quote(payload, safe="")
    return urllib.parse.quote(first, safe="")


def _html_entity_encode(payload: str) -> str:
    """HTML entity encoding for special chars."""
    result = []
    for ch in payload:
        if ch.isalnum():
            result.append(ch)
        else:
            result.append(f"&#{ord(ch)};")
    return "".join(result)


def _unicode_escape(payload: str) -> str:
    """Unicode escape sequences."""
    return "".join(f"\\u{ord(c):04x}" for c in payload)


def _hex_encode(payload: str) -> str:
    """Hex encoding."""
    return "".join(f"\\x{ord(c):02x}" for c in payload)


def _powershell_base64(payload: str) -> str:
    """PowerShell-compatible UTF-16LE base64 encoding."""
    encoded = base64.b64encode(
        payload.encode("utf-16-le")
    ).decode("ascii")
    return f"powershell -enc {encoded}"


def _bash_octal(payload: str) -> str:
    """Bash $'\\NNN' octal encoding."""
    octals = "".join(f"\\{ord(c):03o}" for c in payload)
    return f"$'{octals}'"


def _bash_hex(payload: str) -> str:
    """Bash $'\\xNN' hex encoding."""
    hexes = "".join(f"\\x{ord(c):02x}" for c in payload)
    return f"$'{hexes}'"


def _null_byte_insert(payload: str) -> str:
    """Insert null bytes between characters (WAF bypass)."""
    return "%00".join(payload)


def _case_toggle(payload: str) -> str:
    """Alternate character case (WAF bypass for keyword filters)."""
    result = []
    for i, ch in enumerate(payload):
        if ch.isalpha():
            result.append(ch.upper() if i % 2 == 0 else ch.lower())
        else:
            result.append(ch)
    return "".join(result)


def _whitespace_bypass(payload: str) -> str:
    """Replace spaces with alternative whitespace (tabs, IFS)."""
    return payload.replace(" ", "${IFS}")


def _comment_insert_sql(payload: str) -> str:
    """Insert SQL comments between keywords (WAF bypass)."""
    # Insert /**/ between words
    words = payload.split()
    return "/**/".join(words)


# Transform registry
TRANSFORMS: Dict[EncodingType, Callable[[str], str]] = {
    EncodingType.BASE64: _base64_encode,
    EncodingType.URL_ENCODE: _url_encode,
    EncodingType.DOUBLE_URL: _double_url_encode,
    EncodingType.HTML_ENTITY: _html_entity_encode,
    EncodingType.UNICODE_ESCAPE: _unicode_escape,
    EncodingType.HEX_ENCODE: _hex_encode,
    EncodingType.POWERSHELL_BASE64: _powershell_base64,
    EncodingType.BASH_OCTAL: _bash_octal,
    EncodingType.BASH_HEX: _bash_hex,
    EncodingType.NULL_BYTE_INSERT: _null_byte_insert,
    EncodingType.CASE_TOGGLE: _case_toggle,
    EncodingType.WHITESPACE_BYPASS: _whitespace_bypass,
    EncodingType.COMMENT_INSERT: _comment_insert_sql,
}


# ============================================================================
# CHANNEL → TRANSFORM MAPPING
# ============================================================================

# Default transforms per delivery channel
CHANNEL_TRANSFORMS: Dict[DeliveryChannel, List[EncodingType]] = {
    DeliveryChannel.WEB_FORM: [EncodingType.URL_ENCODE],
    DeliveryChannel.URL_PARAM: [EncodingType.URL_ENCODE],
    DeliveryChannel.HTTP_HEADER: [EncodingType.BASE64],
    DeliveryChannel.COMMAND_INJECTION: [EncodingType.BASH_HEX],
    DeliveryChannel.FILE_UPLOAD: [EncodingType.BASE64],
    DeliveryChannel.SQL_INJECTION: [EncodingType.COMMENT_INSERT],
    DeliveryChannel.SHELL_COMMAND: [],  # No encoding by default
}

# Extra transforms when WAF is detected
WAF_EXTRA_TRANSFORMS: Dict[DeliveryChannel, List[EncodingType]] = {
    DeliveryChannel.WEB_FORM: [EncodingType.DOUBLE_URL, EncodingType.CASE_TOGGLE],
    DeliveryChannel.URL_PARAM: [EncodingType.DOUBLE_URL, EncodingType.NULL_BYTE_INSERT],
    DeliveryChannel.COMMAND_INJECTION: [
        EncodingType.WHITESPACE_BYPASS,
        EncodingType.BASH_OCTAL,
    ],
    DeliveryChannel.SQL_INJECTION: [
        EncodingType.COMMENT_INSERT,
        EncodingType.CASE_TOGGLE,
    ],
}

# Platform-specific transforms
PLATFORM_TRANSFORMS: Dict[TargetPlatform, List[EncodingType]] = {
    TargetPlatform.WINDOWS: [EncodingType.POWERSHELL_BASE64],
    TargetPlatform.LINUX: [EncodingType.BASH_HEX],
    TargetPlatform.UNKNOWN: [],
}


# ============================================================================
# PAYLOAD ENCODER
# ============================================================================

class PayloadEncoder:
    """Context-aware payload encoding engine.

    Selects and applies encoding transforms based on delivery channel,
    target platform, WAF presence, and detection risk level.
    """

    def __init__(self) -> None:
        self._telemetry = EncoderTelemetry()

    @property
    def telemetry(self) -> EncoderTelemetry:
        return self._telemetry

    def encode(
        self,
        payload: str,
        context: Optional[EncodingContext] = None,
    ) -> EncodingResult:
        """Encode a payload with context-appropriate transforms.

        Args:
            payload: Raw payload string to encode
            context: Encoding context (channel, platform, WAF, etc.)

        Returns:
            EncodingResult with encoded payload and metadata
        """
        from core.feature_flags import get_feature_flags
        if not get_feature_flags().payload_encoding:
            return EncodingResult(
                original=payload,
                encoded=payload,
                encoding_chain="passthrough (flag disabled)",
            )

        if context is None:
            context = EncodingContext()

        # Select transforms
        transforms = self._select_transforms(context)

        # Apply transforms in order
        result = self._apply_chain(payload, transforms)

        # Update telemetry
        self._telemetry.total_encodes += 1
        channel_key = context.channel.value
        self._telemetry.channels_used[channel_key] = (
            self._telemetry.channels_used.get(channel_key, 0) + 1
        )
        if context.waf_detected:
            self._telemetry.waf_bypasses_attempted += 1
        for t in transforms:
            t_key = t.value
            self._telemetry.transforms_used[t_key] = (
                self._telemetry.transforms_used.get(t_key, 0) + 1
            )

        return result

    def encode_raw(
        self,
        payload: str,
        transforms: List[EncodingType],
    ) -> EncodingResult:
        """Apply specific transforms without context selection.

        Useful for testing or manual transform specification.
        """
        return self._apply_chain(payload, transforms)

    def available_transforms(self) -> List[str]:
        """List all available encoding transforms."""
        return [t.value for t in EncodingType]

    def _select_transforms(
        self,
        context: EncodingContext,
    ) -> List[EncodingType]:
        """Select appropriate transforms for the given context."""
        transforms: List[EncodingType] = []

        # 1. Custom transforms first (user specified)
        if context.custom_transforms:
            transforms.extend(context.custom_transforms)
            return transforms

        # 2. Channel-based defaults
        channel_defaults = CHANNEL_TRANSFORMS.get(context.channel, [])
        transforms.extend(channel_defaults)

        # 3. WAF-aware extras
        if context.waf_detected:
            waf_extras = WAF_EXTRA_TRANSFORMS.get(context.channel, [])
            for t in waf_extras:
                if t not in transforms:
                    transforms.append(t)

        # 4. High detection risk → add obfuscation
        if context.detection_risk > 0.7:
            if EncodingType.CASE_TOGGLE not in transforms:
                transforms.append(EncodingType.CASE_TOGGLE)

        # 5. Double encode if requested
        if context.double_encode:
            if context.channel in (DeliveryChannel.URL_PARAM, DeliveryChannel.WEB_FORM):
                if EncodingType.DOUBLE_URL not in transforms:
                    transforms.append(EncodingType.DOUBLE_URL)
                # Remove single URL encode if double is present
                if EncodingType.URL_ENCODE in transforms:
                    transforms.remove(EncodingType.URL_ENCODE)

        return transforms

    def _apply_chain(
        self,
        payload: str,
        transforms: List[EncodingType],
    ) -> EncodingResult:
        """Apply a chain of transforms to a payload."""
        applied: List[EncodingType] = []
        current = payload

        for transform_type in transforms:
            func = TRANSFORMS.get(transform_type)
            if func is None:
                logger.warning("Unknown transform: %s", transform_type)
                continue
            try:
                current = func(current)
                applied.append(transform_type)
            except Exception as e:
                logger.warning("Transform %s failed: %s", transform_type, e)
                continue

        chain_desc = " → ".join(t.value for t in applied) if applied else "none"

        return EncodingResult(
            original=payload,
            encoded=current,
            transforms_applied=applied,
            encoding_chain=chain_desc,
            reversible=all(
                t not in (EncodingType.CASE_TOGGLE, EncodingType.COMMENT_INSERT)
                for t in applied
            ),
        )
