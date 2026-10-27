"""
Kiểm thử cơ chế gọi lại của call_gemini_json khi server Gemini lỗi tạm thời (500/503).
Gemini được mock hoàn toàn, sleep được patch → chạy offline và tức thì.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors as genai_errors

from src.agent.vlm_diagnostician import (
    GEMINI_MAX_ATTEMPTS,
    GEMINI_RETRY_BASE_DELAY,
    GEMINI_TIMEOUT_MS,
    call_gemini_json,
)


def _server_error(code: int) -> genai_errors.ServerError:
    """Tạo ServerError giống phản hồi thật của google-genai."""
    status = {500: "INTERNAL", 503: "UNAVAILABLE"}[code]
    return genai_errors.ServerError(code, {"error": {"code": code, "status": status}})


def _ok_response(payload: str = '{"ok": true}') -> MagicMock:
    """Phản hồi hợp lệ có thuộc tính .text."""
    response = MagicMock()
    response.text = payload
    return response


def _call(mock_genai: MagicMock, side_effect: list) -> Dict[str, Any]:
    """Gọi call_gemini_json với generate_content trả lần lượt theo side_effect."""
    mock_genai.Client.return_value.models.generate_content.side_effect = side_effect
    return call_gemini_json("fake-key", "system", ["prompt"], {"type": "object"})


@patch("src.agent.vlm_diagnostician.time.sleep")
@patch("src.agent.vlm_diagnostician.genai")
@pytest.mark.parametrize("code", [500, 503])
def test_retries_transient_error_then_succeeds(
    mock_genai: MagicMock, mock_sleep: MagicMock, code: int
) -> None:
    result = _call(mock_genai, [_server_error(code), _ok_response()])

    assert result == {"ok": True}
    assert mock_genai.Client.return_value.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(GEMINI_RETRY_BASE_DELAY)


@patch("src.agent.vlm_diagnostician.time.sleep")
@patch("src.agent.vlm_diagnostician.genai")
def test_backoff_doubles_between_attempts(mock_genai: MagicMock, mock_sleep: MagicMock) -> None:
    result = _call(mock_genai, [_server_error(503), _server_error(500), _ok_response()])

    assert result == {"ok": True}
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    assert delays == [GEMINI_RETRY_BASE_DELAY, GEMINI_RETRY_BASE_DELAY * 2]


@patch("src.agent.vlm_diagnostician.time.sleep")
@patch("src.agent.vlm_diagnostician.genai")
def test_gives_up_after_max_attempts(mock_genai: MagicMock, mock_sleep: MagicMock) -> None:
    errors = [_server_error(503) for _ in range(GEMINI_MAX_ATTEMPTS)]

    with pytest.raises(genai_errors.ServerError):
        _call(mock_genai, errors)

    generate = mock_genai.Client.return_value.models.generate_content
    assert generate.call_count == GEMINI_MAX_ATTEMPTS
    # Không chờ sau lần thử cuối
    assert mock_sleep.call_count == GEMINI_MAX_ATTEMPTS - 1


@patch("src.agent.vlm_diagnostician.time.sleep")
@patch("src.agent.vlm_diagnostician.genai")
def test_retries_timeout_then_succeeds(mock_genai: MagicMock, mock_sleep: MagicMock) -> None:
    result = _call(mock_genai, [httpx.ReadTimeout("timed out"), _ok_response()])

    assert result == {"ok": True}
    assert mock_genai.Client.return_value.models.generate_content.call_count == 2


@patch("src.agent.vlm_diagnostician.genai")
def test_client_has_request_timeout(mock_genai: MagicMock) -> None:
    # Không có timeout thì một request treo làm pipeline chờ vô hạn
    _call(mock_genai, [_ok_response()])

    http_options = mock_genai.Client.call_args.kwargs["http_options"]
    assert http_options.timeout == GEMINI_TIMEOUT_MS


@patch("src.agent.vlm_diagnostician.time.sleep")
@patch("src.agent.vlm_diagnostician.genai")
def test_does_not_retry_client_error(mock_genai: MagicMock, mock_sleep: MagicMock) -> None:
    client_error = genai_errors.ClientError(400, {"error": {"code": 400}})

    with pytest.raises(genai_errors.ClientError):
        _call(mock_genai, [client_error, _ok_response()])

    assert mock_genai.Client.return_value.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


@patch("src.agent.vlm_diagnostician.time.sleep")
@patch("src.agent.vlm_diagnostician.genai")
def test_does_not_retry_invalid_json(mock_genai: MagicMock, mock_sleep: MagicMock) -> None:
    # Lỗi parse không phải lỗi server → ném ngay để bên gọi dùng plan rule-based
    with pytest.raises(ValueError):
        _call(mock_genai, [_ok_response("not json"), _ok_response()])

    assert mock_genai.Client.return_value.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()
