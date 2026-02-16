#!/usr/bin/env python3
"""
core/tools/web_proxy_layer.py — Phase 10.1E: Burp/Proxy Capture Integration

Ingests HAR-format HTTP request/response data (from Burp Suite, ZAP,
mitmproxy, browser DevTools) and extracts structured intelligence:
  • Endpoints (URLs, methods, status codes)
  • Parameters (query, body, headers)
  • Cookies and auth tokens
  • Hidden form fields and CSRF tokens
  • API patterns and content-type fingerprints

All discoveries are emitted as DiscoveryEvent objects for the reward
pipeline.  RequestReplayTemplate wraps individual requests as re-playable
CommandTemplate-compatible objects.

Architecture:
    HarEntry — single HTTP transaction (request/response pair)
    WebProxyLayer — ingests HAR JSON, extracts intelligence
    RequestReplayTemplate — re-playable request template for registry

Usage:
    from core.tools.web_proxy_layer import WebProxyLayer
    layer = WebProxyLayer()
    discoveries = layer.ingest_har(har_json, target_ip="10.10.10.1")
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("ariaska.web_proxy_layer")


# ============================================================================
# DATA OBJECTS
# ============================================================================

class ParamLocation(Enum):
    """Where a parameter was found."""
    QUERY = "query"
    BODY = "body"
    HEADER = "header"
    COOKIE = "cookie"
    PATH = "path"


@dataclass
class ExtractedParam:
    """A parameter extracted from HTTP traffic."""
    name: str
    value: str
    location: ParamLocation
    url: str = ""
    injectable: bool = False        # heuristic: looks fuzzable
    auth_related: bool = False      # token/session/auth/api_key
    hidden_field: bool = False      # type=hidden form field


@dataclass
class ExtractedEndpoint:
    """A discovered HTTP endpoint."""
    url: str
    method: str = "GET"
    status_code: int = 0
    content_type: str = ""
    params: List[ExtractedParam] = field(default_factory=list)
    response_size: int = 0
    is_api: bool = False            # /api/ path or JSON response
    is_auth: bool = False           # login/auth/session endpoint
    is_upload: bool = False         # file upload endpoint
    is_admin: bool = False          # admin/dashboard/manage
    technologies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "method": self.method,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "params": [{"name": p.name, "location": p.location.value} for p in self.params],
            "is_api": self.is_api,
            "is_auth": self.is_auth,
            "is_upload": self.is_upload,
            "is_admin": self.is_admin,
        }


@dataclass
class HarEntry:
    """A single HTTP request/response pair from HAR data."""
    url: str
    method: str
    status_code: int
    request_headers: Dict[str, str] = field(default_factory=dict)
    response_headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    post_data: str = ""
    post_mime: str = ""
    response_body: str = ""
    response_size: int = 0
    cookies: List[Dict[str, str]] = field(default_factory=list)
    content_type: str = ""


@dataclass
class RequestReplayTemplate:
    """A replayable request template for the command registry.

    Can be rendered as a curl command for execution through the
    CommandRegistry pipeline.
    """
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    description: str = ""
    fuzz_params: List[str] = field(default_factory=list)

    def to_curl(self) -> str:
        """Render as a curl command."""
        parts = ["curl", "-s", "-k"]

        if self.method != "GET":
            parts.extend(["-X", self.method])

        for key, val in self.headers.items():
            # Skip host header (curl handles it)
            if key.lower() == "host":
                continue
            parts.extend(["-H", f"'{key}: {val}'"])

        if self.body:
            parts.extend(["-d", f"'{self.body}'"])

        parts.append(f"'{self.url}'")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "body": self.body,
            "fuzz_params": self.fuzz_params,
        }


@dataclass
class ProxyTelemetry:
    """Telemetry for proxy capture operations."""
    entries_ingested: int = 0
    endpoints_discovered: int = 0
    params_extracted: int = 0
    cookies_extracted: int = 0
    auth_tokens_found: int = 0
    injectable_params: int = 0
    replay_templates_created: int = 0
    api_endpoints: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_ingested": self.entries_ingested,
            "endpoints_discovered": self.endpoints_discovered,
            "params_extracted": self.params_extracted,
            "cookies_extracted": self.cookies_extracted,
            "auth_tokens_found": self.auth_tokens_found,
            "injectable_params": self.injectable_params,
            "replay_templates_created": self.replay_templates_created,
            "api_endpoints": self.api_endpoints,
        }


# ============================================================================
# AUTH / INJECTION HEURISTICS
# ============================================================================

# Parameter names that suggest auth significance
AUTH_PARAM_NAMES = {
    "token", "session", "sessionid", "session_id", "csrf", "csrftoken",
    "csrf_token", "_token", "api_key", "apikey", "api-key", "auth",
    "authorization", "bearer", "jwt", "access_token", "refresh_token",
    "x-auth-token", "x-api-key", "x-csrf-token", "password", "passwd",
    "secret", "key", "nonce", "otp",
}

# Parameter names that suggest injectability
INJECTABLE_PARAM_NAMES = {
    "id", "user", "username", "name", "search", "query", "q", "page",
    "file", "path", "url", "redirect", "next", "return", "callback",
    "cmd", "exec", "command", "action", "type", "sort", "order",
    "filter", "cat", "category", "item", "product", "view", "template",
    "include", "require", "dir", "lang", "language", "debug",
}

# URL path segments suggesting admin areas
ADMIN_PATH_KEYWORDS = {
    "admin", "administrator", "dashboard", "manage", "manager",
    "control", "panel", "console", "backend", "config", "settings",
    "setup", "install", "maintenance",
}

# URL path segments suggesting auth flows
AUTH_PATH_KEYWORDS = {
    "login", "signin", "sign-in", "auth", "authenticate", "oauth",
    "logout", "signout", "register", "signup", "sign-up", "session",
    "token", "sso", "saml", "callback", "forgot", "reset-password",
}

# URL path segments suggesting file upload
UPLOAD_PATH_KEYWORDS = {
    "upload", "file", "attach", "media", "import", "document",
}

# Technology fingerprints from headers/content
TECH_FINGERPRINTS = {
    "x-powered-by": {
        "php": "PHP", "asp.net": "ASP.NET", "express": "Express.js",
        "next.js": "Next.js", "flask": "Flask", "django": "Django",
    },
    "server": {
        "apache": "Apache", "nginx": "Nginx", "iis": "IIS",
        "tomcat": "Tomcat", "gunicorn": "Gunicorn", "jetty": "Jetty",
        "lighttpd": "Lighttpd", "caddy": "Caddy",
    },
}


# ============================================================================
# WEB PROXY LAYER
# ============================================================================

class WebProxyLayer:
    """Ingests HAR data and extracts structured web intelligence.

    Supports HAR 1.2 format (standard from Burp, ZAP, Chrome DevTools).
    Extracts endpoints, parameters, cookies, auth tokens, and creates
    replay templates for promising injection targets.
    """

    def __init__(self) -> None:
        self._telemetry = ProxyTelemetry()
        self._seen_urls: Set[str] = set()
        self._endpoints: List[ExtractedEndpoint] = []
        self._params: List[ExtractedParam] = []
        self._replay_templates: List[RequestReplayTemplate] = []
        self._technologies: Set[str] = set()

    @property
    def telemetry(self) -> ProxyTelemetry:
        return self._telemetry

    @property
    def endpoints(self) -> List[ExtractedEndpoint]:
        return list(self._endpoints)

    @property
    def replay_templates(self) -> List[RequestReplayTemplate]:
        return list(self._replay_templates)

    @property
    def technologies(self) -> Set[str]:
        return set(self._technologies)

    def ingest_har(
        self,
        har_data: Any,
        target_ip: str = "",
    ) -> List[Dict[str, Any]]:
        """Ingest HAR data and extract discoveries.

        Args:
            har_data: Parsed HAR JSON (dict) or raw JSON string
            target_ip: Target IP to filter entries (empty = accept all)

        Returns:
            List of discovery dicts compatible with DiscoveryEvent creation
        """
        from core.feature_flags import get_feature_flags
        if not get_feature_flags().proxy_capture:
            logger.debug("proxy_capture feature flag disabled, skipping HAR ingest")
            return []

        # Parse if string
        if isinstance(har_data, str):
            try:
                har_data = json.loads(har_data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse HAR JSON")
                return []

        if not isinstance(har_data, dict):
            logger.warning("HAR data must be a dict")
            return []

        # Extract entries from HAR
        log = har_data.get("log", har_data)
        raw_entries = log.get("entries", [])
        if not raw_entries:
            logger.info("No entries in HAR data")
            return []

        # Parse entries
        entries = self._parse_har_entries(raw_entries, target_ip)
        self._telemetry.entries_ingested += len(entries)

        # Extract intelligence
        discoveries: List[Dict[str, Any]] = []
        for entry in entries:
            discoveries.extend(self._process_entry(entry))

        # Create replay templates for interesting endpoints
        self._create_replay_templates()

        logger.info(
            "HAR ingest complete: %d entries → %d endpoints, %d params, %d replays",
            len(entries),
            len(self._endpoints),
            self._telemetry.params_extracted,
            len(self._replay_templates),
        )

        return discoveries

    def _parse_har_entries(
        self,
        raw_entries: List[Dict[str, Any]],
        target_ip: str,
    ) -> List[HarEntry]:
        """Parse raw HAR entries into HarEntry objects."""
        entries: List[HarEntry] = []

        for raw in raw_entries:
            try:
                request = raw.get("request", {})
                response = raw.get("response", {})

                url = request.get("url", "")
                if not url:
                    continue

                # Filter by target IP if specified
                if target_ip:
                    parsed = urlparse(url)
                    if target_ip not in (parsed.hostname or ""):
                        continue

                # Extract headers
                req_headers = {}
                for h in request.get("headers", []):
                    req_headers[h.get("name", "")] = h.get("value", "")

                resp_headers = {}
                for h in response.get("headers", []):
                    resp_headers[h.get("name", "")] = h.get("value", "")

                # Extract query params
                query_params = {}
                for p in request.get("queryString", []):
                    query_params[p.get("name", "")] = p.get("value", "")

                # Post data
                post_data_obj = request.get("postData", {})
                post_text = post_data_obj.get("text", "") if post_data_obj else ""
                post_mime = post_data_obj.get("mimeType", "") if post_data_obj else ""

                # Response content
                content = response.get("content", {})
                resp_body = content.get("text", "") if content else ""
                resp_size = content.get("size", 0) if content else 0

                # Cookies
                cookies = request.get("cookies", [])

                # Content type from response headers
                content_type = resp_headers.get(
                    "Content-Type",
                    resp_headers.get("content-type", ""),
                )

                entry = HarEntry(
                    url=url,
                    method=request.get("method", "GET"),
                    status_code=response.get("status", 0),
                    request_headers=req_headers,
                    response_headers=resp_headers,
                    query_params=query_params,
                    post_data=post_text,
                    post_mime=post_mime,
                    response_body=resp_body,
                    response_size=resp_size,
                    cookies=cookies,
                    content_type=content_type,
                )
                entries.append(entry)

            except Exception as e:
                logger.debug("Failed to parse HAR entry: %s", e)
                continue

        return entries

    def _process_entry(self, entry: HarEntry) -> List[Dict[str, Any]]:
        """Process a single HAR entry, extract discoveries."""
        discoveries: List[Dict[str, Any]] = []
        parsed = urlparse(entry.url)
        path = parsed.path or "/"

        # Skip if we've seen this URL+method combo
        url_key = f"{entry.method}:{entry.url}"
        if url_key in self._seen_urls:
            return discoveries
        self._seen_urls.add(url_key)

        # Create endpoint
        endpoint = ExtractedEndpoint(
            url=entry.url,
            method=entry.method,
            status_code=entry.status_code,
            content_type=entry.content_type,
            response_size=entry.response_size,
        )

        # Classify endpoint
        path_lower = path.lower()
        path_parts = set(path_lower.strip("/").split("/"))

        endpoint.is_api = (
            "/api/" in path_lower
            or "application/json" in entry.content_type.lower()
        )
        endpoint.is_auth = bool(path_parts & AUTH_PATH_KEYWORDS)
        endpoint.is_upload = bool(path_parts & UPLOAD_PATH_KEYWORDS)
        endpoint.is_admin = bool(path_parts & ADMIN_PATH_KEYWORDS)

        if endpoint.is_api:
            self._telemetry.api_endpoints += 1

        # Extract technologies from headers
        self._extract_technologies(entry, endpoint)

        # Extract parameters
        params = self._extract_params(entry, path)
        endpoint.params = params

        self._endpoints.append(endpoint)
        self._telemetry.endpoints_discovered += 1

        # Generate discoveries
        # 1. Web path discovery
        discoveries.append({
            "type": "web_path",
            "value": path,
            "method": entry.method,
            "status_code": entry.status_code,
            "confidence": 0.9,
        })

        # 2. Auth endpoint discovery
        if endpoint.is_auth:
            discoveries.append({
                "type": "auth_endpoint",
                "value": entry.url,
                "confidence": 0.8,
            })

        # 3. Admin panel discovery
        if endpoint.is_admin:
            discoveries.append({
                "type": "admin_panel",
                "value": entry.url,
                "confidence": 0.8,
            })

        # 4. Upload endpoint discovery
        if endpoint.is_upload:
            discoveries.append({
                "type": "upload_endpoint",
                "value": entry.url,
                "confidence": 0.8,
            })

        # 5. Injectable parameter discoveries
        for param in params:
            if param.injectable:
                discoveries.append({
                    "type": "injectable_param",
                    "value": f"{param.name}@{entry.url}",
                    "param_name": param.name,
                    "location": param.location.value,
                    "confidence": 0.6,
                })

        # 6. Auth token discoveries
        for param in params:
            if param.auth_related:
                discoveries.append({
                    "type": "auth_token",
                    "value": f"{param.name}={param.value[:20]}...",
                    "param_name": param.name,
                    "confidence": 0.7,
                })

        # 7. Cookie discoveries
        for cookie in entry.cookies:
            cookie_name = cookie.get("name", "")
            if cookie_name:
                self._telemetry.cookies_extracted += 1
                is_session = cookie_name.lower() in {
                    "sessionid", "session_id", "phpsessid", "jsessionid",
                    "asp.net_sessionid", "connect.sid", "sid",
                }
                if is_session:
                    discoveries.append({
                        "type": "session_cookie",
                        "value": cookie_name,
                        "confidence": 0.9,
                    })

        # 8. Technology fingerprints
        for tech in endpoint.technologies:
            if tech not in self._technologies:
                discoveries.append({
                    "type": "technology",
                    "value": tech,
                    "confidence": 0.8,
                })
                self._technologies.add(tech)

        return discoveries

    def _extract_params(
        self,
        entry: HarEntry,
        path: str,
    ) -> List[ExtractedParam]:
        """Extract all parameters from a request."""
        params: List[ExtractedParam] = []

        # Query string params
        for name, value in entry.query_params.items():
            if not name:
                continue
            param = ExtractedParam(
                name=name,
                value=value,
                location=ParamLocation.QUERY,
                url=entry.url,
                injectable=name.lower() in INJECTABLE_PARAM_NAMES,
                auth_related=name.lower() in AUTH_PARAM_NAMES,
            )
            params.append(param)
            self._telemetry.params_extracted += 1
            if param.injectable:
                self._telemetry.injectable_params += 1
            if param.auth_related:
                self._telemetry.auth_tokens_found += 1

        # POST body params (form-encoded)
        if entry.post_data and "form" in entry.post_mime.lower():
            try:
                body_params = parse_qs(entry.post_data)
                for name, values in body_params.items():
                    val = values[0] if values else ""
                    param = ExtractedParam(
                        name=name,
                        value=val,
                        location=ParamLocation.BODY,
                        url=entry.url,
                        injectable=name.lower() in INJECTABLE_PARAM_NAMES,
                        auth_related=name.lower() in AUTH_PARAM_NAMES,
                    )
                    params.append(param)
                    self._telemetry.params_extracted += 1
                    if param.injectable:
                        self._telemetry.injectable_params += 1
                    if param.auth_related:
                        self._telemetry.auth_tokens_found += 1
            except Exception:
                pass

        # POST body params (JSON)
        if entry.post_data and "json" in entry.post_mime.lower():
            try:
                body_json = json.loads(entry.post_data)
                if isinstance(body_json, dict):
                    for name, val in body_json.items():
                        param = ExtractedParam(
                            name=name,
                            value=str(val),
                            location=ParamLocation.BODY,
                            url=entry.url,
                            injectable=name.lower() in INJECTABLE_PARAM_NAMES,
                            auth_related=name.lower() in AUTH_PARAM_NAMES,
                        )
                        params.append(param)
                        self._telemetry.params_extracted += 1
                        if param.injectable:
                            self._telemetry.injectable_params += 1
                        if param.auth_related:
                            self._telemetry.auth_tokens_found += 1
            except (json.JSONDecodeError, Exception):
                pass

        # Auth-related headers
        for header_name, header_val in entry.request_headers.items():
            h_lower = header_name.lower()
            if h_lower in AUTH_PARAM_NAMES or h_lower.startswith("x-"):
                param = ExtractedParam(
                    name=header_name,
                    value=header_val,
                    location=ParamLocation.HEADER,
                    url=entry.url,
                    auth_related=h_lower in AUTH_PARAM_NAMES,
                )
                params.append(param)
                self._telemetry.params_extracted += 1
                if param.auth_related:
                    self._telemetry.auth_tokens_found += 1

        # Hidden fields from response body (HTML forms)
        if "html" in entry.content_type.lower() and entry.response_body:
            hidden_fields = re.findall(
                r'<input[^>]+type=["\']?hidden["\']?[^>]+name=["\']?([^"\'>\s]+)',
                entry.response_body,
                re.IGNORECASE,
            )
            # Also match reversed order: name before type
            hidden_fields += re.findall(
                r'<input[^>]+name=["\']?([^"\'>\s]+)[^>]+type=["\']?hidden',
                entry.response_body,
                re.IGNORECASE,
            )
            for field_name in set(hidden_fields):
                # Try to get value
                val_match = re.search(
                    rf'<input[^>]+name=["\']?{re.escape(field_name)}[^>]+value=["\']?([^"\'>\s]*)',
                    entry.response_body,
                    re.IGNORECASE,
                )
                param = ExtractedParam(
                    name=field_name,
                    value=val_match.group(1) if val_match else "",
                    location=ParamLocation.BODY,
                    url=entry.url,
                    hidden_field=True,
                    auth_related=field_name.lower() in AUTH_PARAM_NAMES,
                )
                params.append(param)
                self._telemetry.params_extracted += 1

        return params

    def _extract_technologies(
        self,
        entry: HarEntry,
        endpoint: ExtractedEndpoint,
    ) -> None:
        """Extract technology fingerprints from headers."""
        for header_name, fingerprints in TECH_FINGERPRINTS.items():
            header_val = ""
            # Case-insensitive header lookup
            for h, v in entry.response_headers.items():
                if h.lower() == header_name:
                    header_val = v.lower()
                    break
            if not header_val:
                continue
            for pattern, tech_name in fingerprints.items():
                if pattern in header_val:
                    endpoint.technologies.append(tech_name)

    def _create_replay_templates(self) -> None:
        """Create RequestReplayTemplates for interesting endpoints."""
        for ep in self._endpoints:
            # Skip static assets
            if any(
                ep.url.endswith(ext)
                for ext in (".css", ".js", ".png", ".jpg", ".gif", ".ico", ".svg", ".woff")
            ):
                continue

            # Create replays for injectable endpoints
            injectable = [p for p in ep.params if p.injectable]
            if injectable:
                template = RequestReplayTemplate(
                    name=f"replay_{ep.method}_{urlparse(ep.url).path.replace('/', '_').strip('_')}",
                    url=ep.url,
                    method=ep.method,
                    description=f"Replay {ep.method} {urlparse(ep.url).path} with injectable params: {', '.join(p.name for p in injectable)}",
                    fuzz_params=[p.name for p in injectable],
                )
                self._replay_templates.append(template)
                self._telemetry.replay_templates_created += 1

            # Create replays for auth endpoints
            elif ep.is_auth:
                template = RequestReplayTemplate(
                    name=f"replay_auth_{urlparse(ep.url).path.replace('/', '_').strip('_')}",
                    url=ep.url,
                    method=ep.method,
                    description=f"Auth endpoint replay: {ep.method} {urlparse(ep.url).path}",
                )
                self._replay_templates.append(template)
                self._telemetry.replay_templates_created += 1

            # Create replays for upload endpoints
            elif ep.is_upload:
                template = RequestReplayTemplate(
                    name=f"replay_upload_{urlparse(ep.url).path.replace('/', '_').strip('_')}",
                    url=ep.url,
                    method=ep.method,
                    description=f"Upload endpoint replay: {ep.method} {urlparse(ep.url).path}",
                )
                self._replay_templates.append(template)
                self._telemetry.replay_templates_created += 1

    def reset(self) -> None:
        """Reset state for new episode."""
        self._seen_urls.clear()
        self._endpoints.clear()
        self._params.clear()
        self._replay_templates.clear()
        self._technologies.clear()
        self._telemetry = ProxyTelemetry()
