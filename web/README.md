[English](README.md) | [Tiếng Việt](README.vi.md)

# Web application

React/Vite/TypeScript uses authenticated FastAPI workspace routes. Node 20+ is required. Run backend and Vite in separate terminals; open http://localhost:5173. The proxy uses IID_BACKEND_URL, default http://127.0.0.1:8000. For one server, build with `npm --prefix web run build`, then run uvicorn and open http://localhost:8000. WEB_DIST_DIR defaults to web/dist.

Run from the repository root:

```powershell
uvicorn src.api.main:app --reload --port 8000
npm --prefix web ci
npm --prefix web run dev
```

GEMINI_API_KEY is server-only; empty selects rule-based behavior. CASE_MEMORY_PATH set to an empty string disables memory. See [quickstart](../README.md), [API configuration](../docs/api.md), [user guide](../docs/user-guide.md).

The [previous detailed README](README.reference.md) is historical Vietnamese reference material. Its timings and older multi-version UI descriptions belong to that snapshot.
