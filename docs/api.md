[English](api.md) | [Tiếng Việt](api.vi.md)

# API integration

Source baseline: `02dd6f9`. Run `uvicorn src.api.main:app --port 8000`. FastAPI exposes interactive schemas at http://localhost:8000/docs and OpenAPI at http://localhost:8000/openapi.json. Consult schemas for complete fields and export query parameters.

## Authentication

`POST /api/v1/auth/register` and `/login` accept JSON `username` and `password`; responses contain user information and set an HTTP-only session cookie. Keep cookies between requests. `/me` reads the user; `/logout` ends the login. Gemini credentials stay on the server.

For PowerShell use `curl.exe` rather than the curl alias. Register first through the UI. Create `scratch/login.json` containing your local `username` and `password`; do not commit it or cookie files. Replace the image path with an existing file:

```powershell
curl.exe -c scratch/cookies.txt -H 'Content-Type: application/json' --data-binary '@scratch/login.json' http://localhost:8000/api/v1/auth/login
curl.exe -b scratch/cookies.txt -F 'file=@C:/images/input.jpg' -F 'title=Demo' http://localhost:8000/api/v1/workspace/sessions
curl.exe -b scratch/cookies.txt http://localhost:8000/api/v1/workspace/sessions
```

Create returns session detail including `id`; list returns session summaries. Use OpenAPI for full response examples rather than assuming the detail object is just an ID.

## Workspace lifecycle

| Endpoint under `/api/v1/workspace` | Request/purpose |
|---|---|
| `POST /sessions` | Multipart: exactly one file or sample; optional title, user_prompt, ground_truth, max_iterations |
| `GET /sessions/{id}` | Read owned session detail |
| `POST /sessions/{id}/chat` | JSON message and current revision; prepare a proposal |
| `POST /sessions/{id}/execute` | JSON proposal revision and idempotency_key; render the displayed proposal |
| `GET /sessions/{id}/export` | Export according to OpenAPI query parameters |
| `DELETE /sessions/{id}` | Delete session and artifacts |

Read `conversation.revision` from the latest response. Stale execute revisions return 409. Repeating successful execute with the same idempotency key does not duplicate treatments. See [API conversation tests](../tests/unit/test_api_conversation.py). OpenAPI also describes clarification, notes, refinement and rating routes.

## Errors, configuration and limitations

New routes use `{"error":{"code":"…","message":"…"}}`, optionally field errors. Messages are currently Vietnamese. Typical codes include unauthenticated, validation, session_busy and thread_expired. Statuses include 401 authentication, 404 missing resource, 409 invalid state, 413 oversized upload, 415 unsupported format and 422 invalid fields. Legacy routes can use older FastAPI errors.

Defaults in [config](../src/api/config.py): upload 20 MB (`MAX_UPLOAD_MB`), 40,000,000 pixels (`MAX_IMAGE_PIXELS`), shortest side 16 (`MIN_IMAGE_SIDE`), one concurrent pipeline (`PIPELINE_CONCURRENCY`), token TTL 30 days (`AUTH_TOKEN_TTL_DAYS`). Upload applies EXIF orientation. Ground truth must match decoded dimensions.

`APP_DATA_DIR` defaults to `data/app` for SQLite and images. `CORS_ORIGINS` explicitly enables cross-origin clients; same-origin needs no CORS setting. `COOKIE_SECURE` supports HTTPS; `ALLOW_REGISTRATION` controls signups. See source for parsing/clamping.

Legacy `/api/v1/diagnose`, `/process`, `/render`, `/refine`, `/sessions`, `/memory` coexist without workspace authentication. Review them before public deployment. Processing requests are synchronous; live percentage progress and cancellation are not promised. Examples are source-verified; network execution status is recorded in [verification](documentation-verification.md).
