import json

import pytest
from fasthtml.common import FastHTML
from starlette.testclient import TestClient

from retrain_pipelines.dag_engine.web_console.views import server


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("RP_WEB_SERVER_LOGS", str(tmp_path))
    fasthtml_app, rt = FastHTML(), None
    fasthtml_app, rt = fasthtml_app, fasthtml_app.route
    server.register(fasthtml_app, rt)
    return fasthtml_app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_heartbeat(client):
    resp = client.get("/web_server/heartbeat")
    assert resp.status_code == 200


def test_heartbeat_prefixed(monkeypatch, tmp_path):
    monkeypatch.setenv("RP_WEB_SERVER_LOGS", str(tmp_path))
    fasthtml_app = FastHTML()
    server.register(fasthtml_app, fasthtml_app.route, prefix="/sub")
    client = TestClient(fasthtml_app)
    resp = client.get("/sub/web_server/heartbeat")
    assert resp.status_code == 200


def test_load_logs_default_count(client, monkeypatch):
    captured = {}

    def fake_read(log_dir, base_filename, n, regex_filter):
        captured["args"] = (log_dir, base_filename, n, regex_filter)
        return ["<div>entry</div>"]

    monkeypatch.setattr(server, "read_last_access_logs", fake_read)

    resp = client.post("/web_server/load_logs", data={})
    assert resp.status_code == 200
    assert captured["args"][2] == 50
    assert captured["args"][3] is None


def test_load_logs_valid_count_and_regex(client, monkeypatch):
    captured = {}

    def fake_read(log_dir, base_filename, n, regex_filter):
        captured["args"] = (log_dir, base_filename, n, regex_filter)
        return ["<div>entry</div>"]

    monkeypatch.setattr(server, "read_last_access_logs", fake_read)

    resp = client.post(
        "/web_server/load_logs", data={"count": "5", "regex_filter": "GET"}
    )
    assert resp.status_code == 200
    assert captured["args"][2] == 5
    assert captured["args"][3] == "GET"


def test_load_logs_invalid_count_falls_back_to_default(client, monkeypatch):
    captured = {}

    def fake_read(log_dir, base_filename, n, regex_filter):
        captured["args"] = (log_dir, base_filename, n, regex_filter)
        return []

    monkeypatch.setattr(server, "read_last_access_logs", fake_read)

    resp = client.post("/web_server/load_logs", data={"count": "not-a-number"})
    assert resp.status_code == 200
    assert captured["args"][2] == 50


def test_server_dashboard_default_state(client):
    resp = client.get("/web_server")
    assert resp.status_code == 200
    html = resp.text
    # default lines option (100) is the one marked selected
    import re

    options = re.findall(r"<option[^>]*>(\d+)</option>", html)
    selected = re.findall(r"<option[^>]*selected[^>]*>(\d+)</option>", html)
    assert options == ["50", "100", "1000"]
    assert selected == ["100"]
    # autoscroll defaults to true -> checkbox checked
    m = re.search(r'<input[^>]*id="autoscroll"[^>]*>', html)
    assert m is not None
    assert "checked" in m.group(0)
    # empty regex/logic filters by default
    assert 'id="regex-filter"' in html


def test_server_dashboard_with_cookies(client):
    client.cookies.set("server_dashboard:lines", "1000")
    client.cookies.set("server_dashboard:autoscroll", "false")
    client.cookies.set(
        "server_dashboard:filter_values",
        json.dumps(['AND("a","b")', "^.*a.*$"]),
    )

    resp = client.get("/web_server")
    assert resp.status_code == 200
    html = resp.text
    import re

    selected = re.findall(r"<option[^>]*selected[^>]*>(\d+)</option>", html)
    assert selected == ["1000"]
    assert (
        'value=\'AND("a","b")\'' in html
        or 'value="AND(&quot;a&quot;,&quot;b&quot;)"' in html
    )
    assert "^.*a.*$" in html


def test_server_dashboard_autoscroll_false_unchecked(client):
    client.cookies.set("server_dashboard:autoscroll", "false")
    resp = client.get("/web_server")
    assert resp.status_code == 200
    # When unchecked, fasthtml omits the "checked" attribute on the input
    import re

    m = re.search(r'<input[^>]*id="autoscroll"[^>]*>', resp.text)
    assert m is not None
    assert "checked" not in m.group(0)


def test_server_dashboard_invalid_filter_values_cookie_falls_back(client):
    client.cookies.set("server_dashboard:filter_values", "not-json")
    resp = client.get("/web_server")
    assert resp.status_code == 200
    # Falls back to empty logic_filter / regex_filter without raising
    assert 'id="regex-filter"' in resp.text


def test_server_dashboard_prefixed(monkeypatch, tmp_path):
    monkeypatch.setenv("RP_WEB_SERVER_LOGS", str(tmp_path))
    fasthtml_app = FastHTML()
    server.register(fasthtml_app, fasthtml_app.route, prefix="/sub")
    client = TestClient(fasthtml_app)
    resp = client.get("/sub/web_server")
    assert resp.status_code == 200
    assert "/sub/web_server/stream_logs" in resp.text
    assert "/sub/web_server/load_logs" in resp.text
