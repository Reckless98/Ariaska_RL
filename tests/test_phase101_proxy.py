#!/usr/bin/env python3
"""
tests/test_phase101_proxy.py — Phase 10.1E: Web Proxy Capture Tests

Tests for WebProxyLayer HAR ingestion, endpoint extraction, parameter
classification, technology fingerprinting, and replay templates.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


def _make_har(entries: list) -> dict:
    """Build minimal HAR structure."""
    return {
        "log": {
            "version": "1.2",
            "entries": entries,
        }
    }


def _make_entry(
    url: str,
    method: str = "GET",
    status: int = 200,
    query_params: list = None,
    post_data: str = "",
    post_mime: str = "",
    req_headers: list = None,
    resp_headers: list = None,
    resp_body: str = "",
    content_type: str = "text/html",
    cookies: list = None,
) -> dict:
    """Build a single HAR entry."""
    entry = {
        "request": {
            "method": method,
            "url": url,
            "headers": req_headers or [],
            "queryString": query_params or [],
            "cookies": cookies or [],
        },
        "response": {
            "status": status,
            "headers": resp_headers or [
                {"name": "Content-Type", "value": content_type},
            ],
            "content": {
                "text": resp_body,
                "size": len(resp_body),
                "mimeType": content_type,
            },
        },
    }
    if post_data:
        entry["request"]["postData"] = {
            "mimeType": post_mime or "application/x-www-form-urlencoded",
            "text": post_data,
        }
    return entry


class TestWebProxyLayer:
    """Test core HAR ingestion."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)

    def test_ingest_empty_har(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        result = layer.ingest_har({"log": {"entries": []}})
        assert result == []

    def test_ingest_json_string(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry("http://10.10.10.1/index.html")])
        result = layer.ingest_har(json.dumps(har))
        assert len(result) > 0

    def test_ingest_basic_endpoint(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/page", status=200),
        ])
        discoveries = layer.ingest_har(har)
        assert any(d["type"] == "web_path" for d in discoveries)
        assert layer.telemetry.endpoints_discovered == 1

    def test_filter_by_target_ip(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/a"),
            _make_entry("http://8.8.8.8/b"),
        ])
        discoveries = layer.ingest_har(har, target_ip="10.10.10.1")
        urls = [d["value"] for d in discoveries if d["type"] == "web_path"]
        assert "/a" in urls
        assert "/b" not in urls

    def test_dedup_same_url(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/page"),
            _make_entry("http://10.10.10.1/page"),
        ])
        layer.ingest_har(har)
        assert layer.telemetry.endpoints_discovered == 1

    def test_disabled_flag(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        from core.feature_flags import set_feature_flag
        set_feature_flag("proxy_capture", False)
        layer = WebProxyLayer()
        har = _make_har([_make_entry("http://10.10.10.1/")])
        result = layer.ingest_har(har)
        assert result == []

    def test_invalid_json_string(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        result = layer.ingest_har("not json at all")
        assert result == []

    def test_reset(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry("http://10.10.10.1/")])
        layer.ingest_har(har)
        assert layer.telemetry.endpoints_discovered > 0
        layer.reset()
        assert layer.telemetry.endpoints_discovered == 0
        assert len(layer.endpoints) == 0

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestEndpointClassification:
    """Test endpoint type detection."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)

    def test_api_endpoint(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry(
                "http://10.10.10.1/api/v1/users",
                content_type="application/json",
            ),
        ])
        layer.ingest_har(har)
        assert layer.telemetry.api_endpoints >= 1
        assert layer.endpoints[0].is_api

    def test_auth_endpoint(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/login", method="POST"),
        ])
        discoveries = layer.ingest_har(har)
        assert any(d["type"] == "auth_endpoint" for d in discoveries)
        assert layer.endpoints[0].is_auth

    def test_admin_endpoint(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/admin/dashboard"),
        ])
        discoveries = layer.ingest_har(har)
        assert any(d["type"] == "admin_panel" for d in discoveries)
        assert layer.endpoints[0].is_admin

    def test_upload_endpoint(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/upload", method="POST"),
        ])
        discoveries = layer.ingest_har(har)
        assert any(d["type"] == "upload_endpoint" for d in discoveries)

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestParamExtraction:
    """Test parameter extraction and classification."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)

    def test_query_params(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/search",
            query_params=[
                {"name": "q", "value": "test"},
                {"name": "page", "value": "1"},
            ],
        )])
        layer.ingest_har(har)
        assert layer.telemetry.params_extracted >= 2

    def test_injectable_param_detection(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/item",
            query_params=[{"name": "id", "value": "42"}],
        )])
        discoveries = layer.ingest_har(har)
        injectable = [d for d in discoveries if d["type"] == "injectable_param"]
        assert len(injectable) >= 1
        assert layer.telemetry.injectable_params >= 1

    def test_auth_param_detection(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/api",
            query_params=[{"name": "token", "value": "abc123xyz"}],
        )])
        discoveries = layer.ingest_har(har)
        auth_tokens = [d for d in discoveries if d["type"] == "auth_token"]
        assert len(auth_tokens) >= 1
        assert layer.telemetry.auth_tokens_found >= 1

    def test_post_form_params(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/login",
            method="POST",
            post_data="username=admin&password=secret",
            post_mime="application/x-www-form-urlencoded",
        )])
        layer.ingest_har(har)
        assert layer.telemetry.params_extracted >= 2

    def test_post_json_params(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/api/action",
            method="POST",
            post_data='{"user": "admin", "cmd": "ls"}',
            post_mime="application/json",
        )])
        discoveries = layer.ingest_har(har)
        injectable = [d for d in discoveries if d["type"] == "injectable_param"]
        # "cmd" and "user" are both in INJECTABLE_PARAM_NAMES
        assert len(injectable) >= 1

    def test_hidden_fields(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/form",
            resp_body='<form><input type="hidden" name="csrf_token" value="abc123"></form>',
            content_type="text/html",
        )])
        layer.ingest_har(har)
        params = layer.endpoints[0].params
        hidden = [p for p in params if p.hidden_field]
        assert len(hidden) >= 1
        assert hidden[0].name == "csrf_token"

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestCookieExtraction:
    """Test cookie and session discovery."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)

    def test_session_cookie(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/page",
            cookies=[{"name": "PHPSESSID", "value": "abc123"}],
        )])
        discoveries = layer.ingest_har(har)
        session = [d for d in discoveries if d["type"] == "session_cookie"]
        assert len(session) >= 1
        assert layer.telemetry.cookies_extracted >= 1

    def test_non_session_cookie(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/page",
            cookies=[{"name": "theme", "value": "dark"}],
        )])
        discoveries = layer.ingest_har(har)
        session = [d for d in discoveries if d["type"] == "session_cookie"]
        assert len(session) == 0

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestTechFingerprinting:
    """Test technology fingerprint extraction."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)

    def test_server_header(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/",
            resp_headers=[
                {"name": "Server", "value": "Apache/2.4.41"},
                {"name": "Content-Type", "value": "text/html"},
            ],
        )])
        discoveries = layer.ingest_har(har)
        tech = [d for d in discoveries if d["type"] == "technology"]
        assert any(d["value"] == "Apache" for d in tech)

    def test_x_powered_by(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/",
            resp_headers=[
                {"name": "X-Powered-By", "value": "PHP/7.4.3"},
                {"name": "Content-Type", "value": "text/html"},
            ],
        )])
        discoveries = layer.ingest_har(har)
        tech = [d for d in discoveries if d["type"] == "technology"]
        assert any(d["value"] == "PHP" for d in tech)

    def test_no_duplicate_tech(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry(
                "http://10.10.10.1/a",
                resp_headers=[
                    {"name": "Server", "value": "Apache"},
                    {"name": "Content-Type", "value": "text/html"},
                ],
            ),
            _make_entry(
                "http://10.10.10.1/b",
                resp_headers=[
                    {"name": "Server", "value": "Apache"},
                    {"name": "Content-Type", "value": "text/html"},
                ],
            ),
        ])
        discoveries = layer.ingest_har(har)
        tech = [d for d in discoveries if d["type"] == "technology"]
        apache_count = sum(1 for d in tech if d["value"] == "Apache")
        assert apache_count == 1

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestReplayTemplates:
    """Test RequestReplayTemplate generation."""

    def setup_method(self):
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)

    def test_replay_for_injectable(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([_make_entry(
            "http://10.10.10.1/search",
            query_params=[{"name": "id", "value": "1"}],
        )])
        layer.ingest_har(har)
        assert len(layer.replay_templates) >= 1
        assert "id" in layer.replay_templates[0].fuzz_params

    def test_replay_curl_render(self):
        from core.tools.web_proxy_layer import RequestReplayTemplate
        t = RequestReplayTemplate(
            name="test",
            url="http://10.10.10.1/api",
            method="POST",
            headers={"Content-Type": "application/json"},
            body='{"id": 1}',
        )
        curl = t.to_curl()
        assert "curl" in curl
        assert "-X POST" in curl
        assert "10.10.10.1" in curl

    def test_replay_to_dict(self):
        from core.tools.web_proxy_layer import RequestReplayTemplate
        t = RequestReplayTemplate(
            name="test",
            url="http://target/path",
            fuzz_params=["id"],
        )
        d = t.to_dict()
        assert d["name"] == "test"
        assert "id" in d["fuzz_params"]

    def test_no_replay_for_static(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        layer = WebProxyLayer()
        har = _make_har([
            _make_entry("http://10.10.10.1/style.css"),
            _make_entry("http://10.10.10.1/app.js"),
            _make_entry("http://10.10.10.1/logo.png"),
        ])
        layer.ingest_har(har)
        assert len(layer.replay_templates) == 0

    def teardown_method(self):
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()


class TestProxyTelemetry:
    """Test telemetry tracking."""

    def test_telemetry_to_dict(self):
        from core.tools.web_proxy_layer import ProxyTelemetry
        t = ProxyTelemetry(
            entries_ingested=10,
            endpoints_discovered=5,
            params_extracted=20,
            injectable_params=3,
        )
        d = t.to_dict()
        assert d["entries_ingested"] == 10
        assert d["injectable_params"] == 3

    def test_telemetry_accumulates(self):
        from core.tools.web_proxy_layer import WebProxyLayer
        from core.feature_flags import set_feature_flag, reset_feature_flags
        reset_feature_flags()
        set_feature_flag("proxy_capture", True)
        layer = WebProxyLayer()

        har1 = _make_har([_make_entry("http://10.10.10.1/a")])
        har2 = _make_har([_make_entry("http://10.10.10.1/b")])
        layer.ingest_har(har1)
        layer.ingest_har(har2)
        assert layer.telemetry.entries_ingested == 2
        assert layer.telemetry.endpoints_discovered == 2
        reset_feature_flags()
