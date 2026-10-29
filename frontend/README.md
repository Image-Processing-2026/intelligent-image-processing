[English](README.md) | [Tiếng Việt](README.vi.md)

# Legacy Gradio demo

Gradio calls the pipeline directly, outside workspace authentication/conversation. Install root Python requirements, then open http://localhost:7860.

Run from the repository root:

```powershell
python frontend/app.py
```

GEMINI_API_KEY is server-only; empty selects rule-based behavior. CASE_MEMORY_PATH set to an empty string disables memory. See [quickstart](../README.md), [API configuration](../docs/api.md), [user guide](../docs/user-guide.md).

The [previous detailed README](README.reference.md) is historical Vietnamese reference material. Its timings and older multi-version UI descriptions belong to that snapshot.
