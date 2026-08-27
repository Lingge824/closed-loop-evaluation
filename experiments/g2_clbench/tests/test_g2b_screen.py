from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen = _load("g2b_provider_screen", "g2b_provider_screen.py")
audit = _load("g2_operational_audit", "g2_operational_audit.py")


class FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.parsed = None
        self.tool_calls = None


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]
        self.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
        self.id = "resp_test"
        self.system_fingerprint = "fp_test"


def test_candidate_order_is_frozen():
    assert [candidate.model for candidate in screen.CANDIDATES] == [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]
    assert [candidate.reasoning_effort for candidate in screen.CANDIDATES] == [
        "minimal",
        "minimal",
    ]


def test_probe_has_24_calls_and_exact_synthetic_lengths():
    assert screen.EXPECTED_CALLS == 24
    for session in screen.SESSIONS:
        block = screen.synthetic_block(session.name, 1, session.characters_per_call)
        assert len(block) == session.characters_per_call
        assert "SYNTHETIC TRANSPORT PAYLOAD" in block


def test_completion_kwargs_pin_controls():
    candidate = screen.CANDIDATES[0]
    kwargs = screen.completion_kwargs(candidate, [{"role": "user", "content": "x"}])
    assert kwargs["model"] == candidate.model
    assert kwargs["temperature"] == 1.0
    assert kwargs["seed"] == 20260827
    assert kwargs["timeout"] == 90.0
    assert kwargs["max_output_tokens"] == 128
    assert kwargs["thinking_level"] == "MINIMAL"
    assert kwargs["response_schema"] == screen.ScreenAction.model_json_schema()
    assert screen.MAX_INPUT_TOKENS_PER_CANDIDATE == 1_200_000
    assert screen.MAX_ESTIMATED_COST_PER_CANDIDATE_USD == 2.00


def test_native_gemini_payload_preserves_multiturn_roles():
    assert screen._gemini_contents(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    ) == [
        {"role": "user", "parts": [{"text": "question"}]},
        {"role": "model", "parts": [{"text": "answer"}]},
    ]


def test_native_gemini_payload_replays_thought_signature_exactly():
    provider_content = {
        "role": "model",
        "parts": [
            {
                "text": '{"action":"ANSWER","content":"ok"}',
                "thoughtSignature": "opaque-signature",
            }
        ],
    }
    response = {"candidates": [{"content": provider_content}]}
    action = screen.ScreenAction(action="ANSWER", content="ok")
    history_message = screen._gemini_history_message(response, action)

    assert history_message["gemini_content"] is provider_content
    assert screen._gemini_contents(
        [
            {"role": "user", "content": "question"},
            history_message,
            {"role": "user", "content": "next question"},
        ]
    ) == [
        {"role": "user", "parts": [{"text": "question"}]},
        provider_content,
        {"role": "user", "parts": [{"text": "next question"}]},
    ]


def test_native_gemini_response_is_parsed_and_metered():
    response = {
        "candidates": [
            {"content": {"parts": [{"text": '{"action":"ANSWER","content":"ok"}'}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 20,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 0,
        },
        "responseId": "gemini_response",
        "modelVersion": "gemini-3.6-flash",
    }
    assert screen._extract_action(response).content == "ok"
    assert screen._usage_payload(response) == {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "output_tokens": 10,
    }


def test_native_completion_sends_frozen_controls(monkeypatch):
    captured = {}

    class HTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"candidates":[],"usageMetadata":{}}'

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return HTTPResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setattr(screen.urllib.request, "urlopen", urlopen)
    screen.native_completion(
        **screen.completion_kwargs(
            screen.CANDIDATES[0],
            [{"role": "user", "content": "synthetic"}],
        )
    )
    assert captured["url"].endswith("/v1beta/models/gemini-3.6-flash:generateContent")
    assert captured["timeout"] == 90.0
    config = captured["payload"]["generationConfig"]
    assert config["temperature"] == 1.0
    assert config["seed"] == 20260827
    assert config["maxOutputTokens"] == 128
    assert config["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == screen.ScreenAction.model_json_schema()


def test_first_passing_candidate_stops_screen(monkeypatch):
    calls = []

    def fake_run(candidate, **_kwargs):
        calls.append(candidate.model)
        return {
            "candidate": screen.asdict(candidate),
            "passed": True,
            "failure": None,
            "records": [],
        }

    monkeypatch.setattr(screen, "run_candidate", fake_run)
    result = screen.run_screen(progress=lambda _message: None)
    assert result["status"] == "passed"
    assert calls == [screen.CANDIDATES[0].model]
    assert result["selected_candidate"]["model"] == screen.CANDIDATES[0].model


def test_account_failure_does_not_trigger_fallback(monkeypatch):
    calls = []

    def fake_run(candidate, **_kwargs):
        calls.append(candidate.model)
        return {
            "candidate": screen.asdict(candidate),
            "passed": False,
            "failure": {"category": "account_failure", "message": "billing"},
            "records": [],
        }

    monkeypatch.setattr(screen, "run_candidate", fake_run)
    result = screen.run_screen(progress=lambda _message: None)
    assert result["status"] == "account_not_ready"
    assert calls == [screen.CANDIDATES[0].model]


def test_model_failure_triggers_frozen_fallback(monkeypatch):
    calls = []

    def fake_run(candidate, **_kwargs):
        calls.append(candidate.model)
        passed = candidate.rank == 2
        return {
            "candidate": screen.asdict(candidate),
            "passed": passed,
            "failure": None
            if passed
            else {"category": "model_failure", "message": "unsupported"},
            "records": [],
        }

    monkeypatch.setattr(screen, "run_candidate", fake_run)
    result = screen.run_screen(progress=lambda _message: None)
    assert result["status"] == "passed"
    assert calls == [candidate.model for candidate in screen.CANDIDATES]
    assert result["selected_candidate"]["rank"] == 2


def test_run_candidate_accepts_all_synthetic_calls(monkeypatch):
    monkeypatch.setattr(
        screen,
        "SESSIONS",
        (screen.SessionSpec("tiny", 2, 80),),
    )
    monkeypatch.setattr(screen, "EXPECTED_CALLS", 2)
    now = {"value": 0.0}

    def monotonic():
        now["value"] += 20.0
        return now["value"]

    def completion(**kwargs):
        content = kwargs["messages"][-1]["content"]
        nonce = content.split("equal to '", 1)[1].split("'", 1)[0]
        return FakeResponse(json.dumps({"action": "ANSWER", "content": nonce}))

    result = screen.run_candidate(
        screen.CANDIDATES[0],
        completion=completion,
        sleep=lambda _seconds: None,
        monotonic=monotonic,
        progress=lambda _message: None,
    )
    assert result["passed"] is True
    assert result["calls_completed"] == 2
    assert result["token_totals"] == {
        "input_tokens": 200,
        "cached_input_tokens": 40,
        "output_tokens": 20,
    }


def test_schema_or_nonce_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(
        screen,
        "SESSIONS",
        (screen.SessionSpec("tiny", 1, 80),),
    )
    monkeypatch.setattr(screen, "EXPECTED_CALLS", 1)
    calls = {"count": 0}

    def completion(**_kwargs):
        calls["count"] += 1
        return FakeResponse('{"action":"ANSWER","content":"wrong"}')

    result = screen.run_candidate(
        screen.CANDIDATES[0],
        completion=completion,
        sleep=lambda _seconds: None,
        monotonic=lambda: 100.0,
        progress=lambda _message: None,
    )
    assert result["passed"] is False
    assert result["failure"]["category"] == "model_failure"
    assert calls["count"] == 1


def test_private_provider_error_is_recorded_but_not_published(monkeypatch):
    monkeypatch.setattr(
        screen,
        "SESSIONS",
        (screen.SessionSpec("tiny", 1, 80),),
    )
    monkeypatch.setattr(screen, "EXPECTED_CALLS", 1)

    def completion(**_kwargs):
        raise screen.GeminiAPIError(
            "model is unavailable to this project",
            status_code=404,
        )

    candidate_result = screen.run_candidate(
        screen.CANDIDATES[0],
        completion=completion,
        sleep=lambda _seconds: None,
        monotonic=lambda: 100.0,
        progress=lambda _message: None,
    )
    assert candidate_result["failure_details"] == {
        "exception_type": "GeminiAPIError",
        "status_code": 404,
        "provider_message": "model is unavailable to this project",
    }

    private_screen = {
        "candidate_results": [candidate_result],
        "status": "no_candidate_passed",
    }
    public = screen.public_summary(private_screen)
    assert "records" not in public["candidate_results"][0]
    assert "failure_details" not in public["candidate_results"][0]


def test_cost_guard_stops_candidate(monkeypatch):
    monkeypatch.setattr(
        screen,
        "SESSIONS",
        (screen.SessionSpec("tiny", 2, 80),),
    )
    monkeypatch.setattr(screen, "EXPECTED_CALLS", 2)
    monkeypatch.setattr(screen, "MAX_INPUT_TOKENS_PER_CANDIDATE", 50)

    def completion(**kwargs):
        content = kwargs["messages"][-1]["content"]
        nonce = content.split("equal to '", 1)[1].split("'", 1)[0]
        return FakeResponse(json.dumps({"action": "ANSWER", "content": nonce}))

    result = screen.run_candidate(
        screen.CANDIDATES[0],
        completion=completion,
        sleep=lambda _seconds: None,
        monotonic=lambda: 100.0,
        progress=lambda _message: None,
    )
    assert result["passed"] is False
    assert result["failure"]["category"] == "account_failure"
    assert result["calls_completed"] == 1


def test_audit_reports_no_outcome_with_failure_evidence(tmp_path):
    private = tmp_path / "private"
    private.mkdir(parents=True)
    (private / "attempt.log").write_text(
        "tool_use_failed\njson_validate_failed\n429 Too Many Requests\n",
        encoding="utf-8",
    )
    report = audit.build_audit(tmp_path)
    assert report["log_count"] == 1
    assert report["no_research_outcome"] is True
    assert report["operational_status"] == "infeasible_no_g2_outcome"
    assert report["aggregate_signal_counts"]["tool_use_failed"] == 1


def test_audit_requires_manual_review_if_clean_cache_exists(tmp_path):
    private = tmp_path / "private" / "seed_7"
    private.mkdir(parents=True)
    (tmp_path / "private" / "attempt.log").write_text(
        "LLM returned empty content\n", encoding="utf-8"
    )
    (private / "clean_run_0.json").write_text("{}", encoding="utf-8")
    report = audit.build_audit(tmp_path)
    assert report["no_research_outcome"] is False
    assert report["operational_status"] == "manual_review_required"
