[English](README.md) | [Tiếng Việt](README.vi.md)

# AI Image Doctor

A closed-loop image-processing project for a university Image Processing course. Gemini and LangGraph diagnose and orchestrate; OpenCV, scikit-image and NumPy perform classical pixel operations. Generative pixel synthesis is outside the project scope.

## Quickstart

Python 3.10+, Node.js 20+, and npm are required for the web application. Run from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
npm --prefix web ci
npm --prefix web run build
uvicorn src.api.main:app --port 8000
```

Open http://localhost:8000, register a local account, and upload an image. Copy `.env.example` to `.env` to configure Gemini; leave `GEMINI_API_KEY` empty for rule-based operation. A configured key can still fall back after a failed call. Inspect the reported source of each diagnosis/plan.

For development, run the backend with `--reload` and `npm --prefix web run dev` in another terminal; open http://localhost:5173. Gradio is a separate legacy demo: `python frontend/app.py`.

## Documentation

Start at the [documentation index](docs/README.md). Read [architecture](docs/architecture.md), [algorithms](docs/algorithms.md), [evaluation](docs/evaluation.md), [user guide](docs/user-guide.md), and [API guide](docs/api.md).

See [web setup](web/README.md), [contracts](docs/interfaces.md), [contributing](CONTRIBUTING.md) and [decisions](docs/decisions.md). Historical plans and reports are indexed separately.

## Data and checks

`data/app/` contains local accounts, sessions and images. `data/memory/cases.sqlite` stores case memory; `CASE_MEMORY_PATH=""` disables it. Sample images are not guaranteed to be present in a clone; see [data setup](data/README.md).

```powershell
ruff check src/ tests/
pytest tests/unit/ -v
npm --prefix web run typecheck
npm --prefix web test -- --run
```

The server includes legacy routes without the workspace authentication boundary. Use a local server for demos; review the [API limitations](docs/api.md) before network exposure.
