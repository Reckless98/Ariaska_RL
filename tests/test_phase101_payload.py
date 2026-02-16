#!/usr/bin/env python3
"""
tests/test_phase101_payload.py — Phase 10.1F: Payload Encoding Tests

Tests for PayloadEncoder transforms, context-aware selection, WAF
bypass chains, telemetry, and feature flag gating.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestIndividualTransforms:
    """Test each encoding transform independently."""

    def test_base64(self):
        from core.tools.payload_encoder import _base64_encode
        assert _base64_encode("hello") == "aGVsbG8="

    def test_url_encode(self):
        from core.tools.payload_encoder import _url_encode
        result = _url_encode("<script>alert(1)</script>")
        assert "%" in result
        assert "<" not in result

    def test_double_url_encode(self):
        from core.tools.payload_encoder import _double_url_encode
        result = _double_url_encode("<")
        # < → %3C → %253C
        assert "%25" in result

    def test_html_entity(self):
        from core.tools.payload_encoder import _html_entity_encode
        result = _html_entity_encode("<br>")
        assert "&#60;" in result  # <
        assert "&#62;" in result  # >

    def test_unicode_escape(self):
        from core.tools.payload_encoder import _unicode_escape
        result = _unicode_escape("AB")
        assert "\\u0041" in result
        assert "\\u0042" in result

    def test_hex_encode(self):
        from core.tools.payload_encoder import _hex_encode
        result = _hex_encode("A")
        assert "\\x41" in result

    def test_powershell_base64(self):
        from core.tools.payload_encoder import _powershell_base64
        result = _powershell_base64("whoami")
        assert result.startswith("powershell -enc ")
        # Should be valid base64 after the prefix
        import base64
        b64_part = result.split(" ")[-1]
        decoded = base64.b64decode(b64_part).decode("utf-16-le")
        assert decoded == "whoami"

    def test_bash_octal(self):
        from core.tools.payload_encoder import _bash_octal
        result = _bash_octal("ls")
        assert result.startswith("$'")
        assert "\\154" in result  # 'l' = 0o154

    def test_bash_hex(self):
        from core.tools.payload_encoder import _bash_hex
        result = _bash_hex("id")
        assert result.startswith("$'")
        assert "\\x69" in result  # 'i'
        assert "\\x64" in result  # 'd'

    def test_null_byte_insert(self):
        from core.tools.payload_encoder import _null_byte_insert
        result = _null_byte_insert("abc")
        assert "%00" in result
        assert result == "a%00b%00c"

    def test_case_toggle(self):
        from core.tools.payload_encoder import _case_toggle
        result = _case_toggle("select")
        # S, l, c uppercase; e, e, t lowercase
        assert result != "select"
        assert result.lower() == "select"

    def test_whitespace_bypass(self):
        from core.tools.payload_encoder import _whitespace_bypass
        result = _whitespace_bypass("cat /etc/passwd")
        assert " " not in result
        assert "${IFS}" in result

    def test_comment_insert(self):
        from core.tools.payload_encoder import _comment_insert_sql
        result = _comment_insert_sql("SELECT * FROM users")
        assert "/**/" in result


class TestPayloadEncoder:
    """Test the high-level PayloadEncoder."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("payload_encoding", True)

    def test_encode_passthrough_when_disabled(self):
        from core.tools.payload_encoder import PayloadEncoder
        from core.feature_flags import set_feature_flag
        set_feature_flag("payload_encoding", False)
        enc = PayloadEncoder()
        result = enc.encode("whoami")
        assert result.encoded == "whoami"
        assert "passthrough" in result.encoding_chain

    def test_encode_shell_no_transform(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.SHELL_COMMAND)
        result = enc.encode("whoami", ctx)
        # Shell command channel has no default transforms
        assert result.encoded == "whoami"

    def test_encode_web_form(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.WEB_FORM)
        result = enc.encode("<script>", ctx)
        assert "%" in result.encoded
        assert len(result.transforms_applied) >= 1

    def test_encode_url_param(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.URL_PARAM)
        result = enc.encode("id=1 OR 1=1", ctx)
        assert "%" in result.encoded

    def test_encode_sql_injection(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.SQL_INJECTION)
        result = enc.encode("UNION SELECT 1", ctx)
        assert "/**/" in result.encoded

    def test_encode_command_injection(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.COMMAND_INJECTION)
        result = enc.encode("cat /etc/passwd", ctx)
        assert "\\x" in result.encoded

    def test_waf_adds_extra_transforms(self):
        from core.tools.payload_encoder import (
            PayloadEncoder, EncodingContext, DeliveryChannel, EncodingType,
        )
        enc = PayloadEncoder()
        ctx = EncodingContext(
            channel=DeliveryChannel.WEB_FORM,
            waf_detected=True,
        )
        result = enc.encode("test", ctx)
        # Should include WAF extras (double_url and/or case_toggle)
        assert len(result.transforms_applied) > 1

    def test_high_detection_risk_adds_case_toggle(self):
        from core.tools.payload_encoder import (
            PayloadEncoder, EncodingContext, DeliveryChannel, EncodingType,
        )
        enc = PayloadEncoder()
        ctx = EncodingContext(
            channel=DeliveryChannel.WEB_FORM,
            detection_risk=0.9,
        )
        result = enc.encode("select", ctx)
        assert EncodingType.CASE_TOGGLE in result.transforms_applied

    def test_double_encode_replaces_single(self):
        from core.tools.payload_encoder import (
            PayloadEncoder, EncodingContext, DeliveryChannel, EncodingType,
        )
        enc = PayloadEncoder()
        ctx = EncodingContext(
            channel=DeliveryChannel.URL_PARAM,
            double_encode=True,
        )
        result = enc.encode("<", ctx)
        assert EncodingType.DOUBLE_URL in result.transforms_applied
        assert EncodingType.URL_ENCODE not in result.transforms_applied

    def test_custom_transforms(self):
        from core.tools.payload_encoder import (
            PayloadEncoder, EncodingContext, EncodingType,
        )
        enc = PayloadEncoder()
        ctx = EncodingContext(
            custom_transforms=[EncodingType.BASE64],
        )
        result = enc.encode("test", ctx)
        assert result.transforms_applied == [EncodingType.BASE64]
        assert result.encoded == "dGVzdA=="

    def test_encode_raw(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingType
        enc = PayloadEncoder()
        result = enc.encode_raw("hello", [EncodingType.BASE64])
        assert result.encoded == "aGVsbG8="

    def test_available_transforms(self):
        from core.tools.payload_encoder import PayloadEncoder
        enc = PayloadEncoder()
        transforms = enc.available_transforms()
        assert "base64" in transforms
        assert "url_encode" in transforms
        assert len(transforms) >= 10

    def test_encoding_chain_description(self):
        from core.tools.payload_encoder import (
            PayloadEncoder, EncodingContext, DeliveryChannel,
        )
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.WEB_FORM)
        result = enc.encode("test", ctx)
        assert result.encoding_chain  # non-empty

    def test_result_to_dict(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.WEB_FORM)
        result = enc.encode("test<>", ctx)
        d = result.to_dict()
        assert "original_length" in d
        assert "encoded_length" in d
        assert "transforms" in d
        assert isinstance(d["transforms"], list)

    def test_reversible_flag(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingType
        enc = PayloadEncoder()
        # base64 is reversible
        r1 = enc.encode_raw("test", [EncodingType.BASE64])
        assert r1.reversible is True
        # case_toggle is not
        r2 = enc.encode_raw("test", [EncodingType.CASE_TOGGLE])
        assert r2.reversible is False

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestEncoderTelemetry:
    """Test encoder telemetry tracking."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("payload_encoding", True)

    def test_telemetry_counts(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(channel=DeliveryChannel.WEB_FORM)
        enc.encode("a", ctx)
        enc.encode("b", ctx)
        assert enc.telemetry.total_encodes == 2
        assert enc.telemetry.channels_used.get("web_form", 0) == 2

    def test_waf_bypass_counter(self):
        from core.tools.payload_encoder import PayloadEncoder, EncodingContext, DeliveryChannel
        enc = PayloadEncoder()
        ctx = EncodingContext(
            channel=DeliveryChannel.WEB_FORM,
            waf_detected=True,
        )
        enc.encode("payload", ctx)
        assert enc.telemetry.waf_bypasses_attempted == 1

    def test_telemetry_to_dict(self):
        from core.tools.payload_encoder import EncoderTelemetry
        t = EncoderTelemetry(total_encodes=5, waf_bypasses_attempted=2)
        d = t.to_dict()
        assert d["total_encodes"] == 5
        assert d["waf_bypasses_attempted"] == 2

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
