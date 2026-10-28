"""Offline checks: protocol integrity and recoverable provider failures."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as lab
import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (240, 120), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(lab, "RUNS", tmp_path)
    monkeypatch.setattr(lab, "JOBS", {})
    monkeypatch.setattr(lab, "CANCEL", {})
    monkeypatch.setattr(lab, "wait_for_slot", lambda *args: True)
    # Keep tests deterministic; production runs use one worker per candidate.
    monkeypatch.setattr(lab.POOL, "submit", lambda function, *args: function(*args))
    return TestClient(lab.app)


def answer(text):
    return {
        "text": text,
        "seconds": 1,
        "usage": {"input": 10, "output": 20},
        "finish_reason": "stop",
        "truncated": False,
    }


def test_one_image_request_per_model_with_exact_shared_instruction(
    client, monkeypatch, image_bytes
):
    calls = []

    def fake(provider, system, text, image, limit):
        calls.append((provider, system, text, image, limit))
        return answer(f"{provider}: reply")

    monkeypatch.setattr(lab, "call_model", fake)
    response = client.post(
        "/api/runs",
        files={"image": ("test.png", image_bytes)},
        data={
            "system_prompt": "Hãy tự do nhận xét ảnh này, giữ màu ấm.",
            "intent": "OLD CONTEXT MUST NOT BE SENT",
            "plan_prompt": "OLD TOOL RULES MUST NOT BE SENT",
        },
    )
    assert response.status_code == 200
    run_id = response.json()["id"]
    job = client.get(f"/api/runs/{run_id}").json()
    assert job["status"] == "done"
    assert len(calls) == 3
    visual = [call for call in calls if call[3]]
    assert len(visual) == 3
    assert all(call[3] == visual[0][3] for call in visual)
    for provider, system, text, image, limit in calls:
        assert text is None
        assert image is not None
        assert system == "Hãy tự do nhận xét ảnh này, giữ màu ấm."
        assert limit == 1000
        assert "plan" not in job["results"][provider]
    exported = client.get(f"/api/runs/{run_id}/download").json()
    assert exported["system_prompt"] == "Hãy tự do nhận xét ảnh này, giữ màu ấm."
    assert exported["protocol_version"] == "vision-free-v1"
    assert exported["stages"] == ["analysis"]
    assert "intent" not in exported and "prompts" not in exported


def test_provider_failure_preserves_other_and_retry(client, monkeypatch, image_bytes):
    def fake(provider, system, text, image, limit):
        if provider == "google":
            raise ValueError("HTTP 503: busy")
        return answer("Reply")

    monkeypatch.setattr(lab, "call_model", fake)
    run_id = client.post(
        "/api/runs", files={"image": ("test.png", image_bytes)}
    ).json()["id"]
    job = client.get(f"/api/runs/{run_id}").json()
    assert job["results"]["google"]["analysis"]["status"] == "error"
    assert "plan" not in job["results"]["google"]
    assert job["results"]["groq"]["analysis"]["text"] == "Reply"
    assert job["results"]["gemini"]["analysis"]["text"] == "Reply"
    # Restoring a saved completed run after a server restart keeps retry usable.
    lab.JOBS.clear()
    lab.CANCEL.clear()
    job = client.get(f"/api/runs/{run_id}").json()
    assert job["results"]["google"]["analysis"]["status"] == "error"
    monkeypatch.setattr(lab, "call_model", lambda *args: answer("Recovered"))
    assert client.post(f"/api/runs/{run_id}/retry/google/analysis").status_code == 200
    job = client.get(f"/api/runs/{run_id}").json()
    assert job["results"]["google"]["analysis"]["text"] == "Recovered"
    assert job["results"]["groq"]["analysis"]["text"] == "Reply"
    assert client.post(f"/api/runs/{run_id}/retry/google/plan").status_code == 404


def test_upload_validation_and_secret_boundaries(client, monkeypatch, image_bytes):
    monkeypatch.setattr(
        lab,
        "credentials",
        lambda: {
            "google": "secret-google",
            "groq": "secret-groq",
            "gemini": "secret-google",
        },
    )
    config = client.get("/api/config")
    assert config.json()["models"]["groq"]["ready"]
    assert config.json()["models"]["gemini"]["model"] == "gemini-3.5-flash-lite"
    assert "secret-google" not in config.text and "secret-groq" not in config.text
    assert "secret-google" not in lab.safe_error("failure secret-google")
    assert (
        client.post("/api/runs", files={"image": ("fake.png", b"invalid")}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/runs",
            files={"image": ("x.png", image_bytes)},
            data={"max_output_tokens": 99999},
        ).status_code
        == 422
    )
    assert client.get("/api/runs/../../secrets").status_code == 404


def test_native_provider_instructions_and_hidden_thoughts(monkeypatch, image_bytes):
    monkeypatch.setattr(
        lab, "credentials", lambda: {name: "fake" for name in lab.MODELS}
    )
    requests = []

    def fake(url, headers, json, timeout):
        requests.append((url, json))
        if "googleapis" in url:
            value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "private", "thought": True},
                                {"text": "visible"},
                            ]
                        },
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            }
        else:
            value = {
                "choices": [
                    {
                        "message": {"content": "visible", "reasoning": "private"},
                        "finish_reason": "stop",
                    }
                ]
            }
        return httpx.Response(200, json=value)

    monkeypatch.setattr(lab.httpx, "post", fake)
    google = lab.call_model("google", "instruction", None, image_bytes, 768)
    groq = lab.call_model("groq", "instruction", None, image_bytes, 768)
    gemini = lab.call_model("gemini", "instruction", None, image_bytes, 768)
    assert google["text"] == groq["text"] == gemini["text"] == "visible"
    assert google["truncated"] is True
    assert requests[0][1]["systemInstruction"]["parts"][0]["text"] == "instruction"
    assert requests[1][1]["messages"][0] == {"role": "system", "content": "instruction"}
    assert requests[1][1]["reasoning_format"] == "hidden"
    assert "gemini-3.5-flash-lite:generateContent" in requests[2][0]
    assert requests[2][1]["systemInstruction"]["parts"][0]["text"] == "instruction"
    assert "messages" not in requests[2][1]
    assert gemini["truncated"] is True
    for index in (0, 2):
        parts = requests[index][1]["contents"][0]["parts"]
        assert len(parts) == 1 and "inlineData" in parts[0]
    content = requests[1][1]["messages"][1]["content"]
    assert len(content) == 1 and content[0]["type"] == "image_url"
    assert all("tools" not in payload for _, payload in requests)


def test_legacy_results_stay_readable_without_reusing_tool_prompt(
    client, monkeypatch, image_bytes
):
    monkeypatch.setattr(lab, "call_model", lambda *args: answer("old reply"))
    run_id = client.post("/api/runs", files={"image": ("x.png", image_bytes)}).json()[
        "id"
    ]
    job = lab.JOBS[run_id]
    job.pop("protocol_version")
    job.pop("stages")
    job.pop("system_prompt")
    job["prompts"] = {"analysis": "old diagnosis", "plan": "old tool bounds"}
    for result in job["results"].values():
        result["plan"] = {"status": "done", "text": "old constrained plan"}
    lab.persist(job)
    lab.JOBS.clear()
    lab.CANCEL.clear()
    saved = client.get(f"/api/runs/{run_id}").json()
    assert saved["prompts"]["plan"] == "old tool bounds"
    assert client.post(f"/api/runs/{run_id}/retry/google/analysis").status_code == 409
    assert "system_prompt" not in saved


def test_retry_new_candidate_not_in_legacy_run_returns_404(
    client, monkeypatch, image_bytes
):
    monkeypatch.setattr(lab, "call_model", lambda *args: answer("ok"))
    run_id = client.post("/api/runs", files={"image": ("x.png", image_bytes)}).json()[
        "id"
    ]
    del lab.JOBS[run_id]["results"]["gemini"]
    assert client.post(f"/api/runs/{run_id}/retry/gemini/analysis").status_code == 404


def test_image_preparation_does_not_upscale_and_handles_alpha():
    buffer = io.BytesIO()
    Image.new("RGBA", (40, 20), (0, 0, 0, 0)).save(buffer, format="PNG")
    data, meta = lab.prepare_image(buffer.getvalue(), 1280)
    assert meta["sent_size"] == [40, 20]
    with Image.open(io.BytesIO(data)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
