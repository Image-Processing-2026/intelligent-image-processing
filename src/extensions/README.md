# Deferred / Future Extensions (OUT OF MVP SCOPE)

> [!WARNING]
> **DO NOT IMPORT OR IMPLEMENT CODE IN THIS DIRECTORY UNTIL THE MVP IS FULLY TESTED AND SHIPPED.**

---

## 1. Purpose of this Directory
This directory isolates deferred features mentioned in the course project specification that are explicitly **out of scope for the MVP**:
1. **`deblur.py`:** Wiener deconvolution, Richardson–Lucy restoration.
2. **`inpaint.py`:** Telea, Navier–Stokes scratch and watermark removal.
3. **`super_res.py`:** SRCNN, Real-ESRGAN or deep learning super-resolution.

---

## 2. Guardrail Rule
- **Nothing in `src/agent/`, `src/processing_engine/`, or `src/api/` may import from `src/extensions/`.**
- AI agents working on MVP tasks should ignore files in this directory.
