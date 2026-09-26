# Lumexa Backend (v8.24)

AI-generated mechanical CAD parts with engineering validation before handoff to
Ansys/SolidWorks: generate a design from a prompt, analyse it on SimScale (cloud FEA),
feed any failure back to the LLM, and repeat until it passes. Exports STEP/STL/DXF.

## Stack

FastAPI + **build123d** (OpenCASCADE) for geometry, **SimScale** for FEA (with the
CalculiX analysis service and an analytical model as automatic fallbacks),
trimesh/scipy for mesh analysis, ezdxf for 2D drawings. LLM: Nemotron 3 Ultra through
NVIDIA NIM (`AI_PROVIDER=nvidia`); Groq, Cerebras, OpenRouter, Claude, Gemini and
Lovable are still supported.

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `AI_PROVIDER` | yes | `nvidia` (set explicitly) |
| `NVIDIA_API_KEY` | yes | from build.nvidia.com |
| `NVIDIA_MODEL` | no | default `nvidia/nemotron-3-ultra-550b-a55b` |
| `NVIDIA_TIMEOUT_S` / `NVIDIA_GEN_MAX_TOKENS` | no | default 180 / 12000 |
| `SIMSCALE_API_KEY` | for SimScale | needs a SimScale plan with API access |
| `SIMSCALE_TEMPLATE_PROJECT_ID`, `_SIMULATION_ID`, `_MESH_OPERATION_ID` | for SimScale | one-time template, see `SETUP.md` |
| `SIMSCALE_TEMPLATE_MATERIAL` | no | MATERIALS key the template uses (default `aluminum_6061`) |
| `SIMSCALE_TIMEOUT_S` | no | per-analysis budget, default 480 |
| `USE_PYVISTA` / `PYVISTA_RENDER` | no | default `0`/`0` — read SimScale results with PyVista / also return a stress picture. See below |
| `PYVISTA_GL_BACKEND` | no | `osmesa` (default), `egl` or `auto` |
| `KB_ENABLED` / `KB_EMBEDDINGS` | no | default `1` / `0` — build123d API knowledge base; semantic embeddings only if `1` |
| `ALLOWED_ORIGINS` | recommended | comma-separated frontend origins for CORS |
| `ANALYSIS_SERVICE_URL` | no | CalculiX fallback service |

## Deploy (Render)

Render dashboard -> **New -> Blueprint** -> connect this repo, fill in the prompted
secrets. Or create a Docker web service manually and enter the variables above.
Free instances have 512 MB RAM and sleep after ~15 min idle (first request is slow).

After deploying, open `/cad-selftest` and `/simscale-selftest`.

## Endpoints

- `GET /`, `/materials`, `/part-types`, `/cad-selftest`, `/simscale-selftest`, `/pyvista-selftest`
- `GET /kb/status`, `/kb/search?q=...`, `POST /kb/lint`, `/kb/rebuild` — the API knowledge base
- `POST /generate-part`, `/generate-and-analyze`, `/generate-from-prompt`
- `POST /generate-validate-refine` and `/generate-validate-refine-async` (poll `GET /job/{job_id}`) — generate -> SimScale -> feedback -> refine
- `POST /engineering-agent` — tool-calling agent on the same machinery
- `POST /refine-from-external-fea`, `/edit-design-region`, `/export-step`, `/export-drawing-dxf`
- `POST /analyze-part`, `/analyze-part-deep`, `/analyze-composite`, `/analyze-rainflow`, `/analyze-assembly`, `/compare-designs`, `/image-to-params`

## Hole/bore verification

The LLM declares each hole with a comment (`# FEATURE: hole dia=6 count=4`). After the script
runs, the finished solid is measured for cylindrical cavities; a missing or wrong-size hole
triggers a `missing_features` refinement round before any SimScale run. Results are in
`feature_verification`.

## build123d API knowledge base (on by default)

At startup the backend reads the *installed* build123d (every class/function/enum signature and docstring the
sandbox exposes — never the file-I/O functions) plus a set of idiom examples that are machine-checked against
that version. For every generation it retrieves the relevant entries (BM25 + exact identifiers, optionally
NVIDIA embeddings, then the NVIDIA reranker) and puts them in the prompt, so the model stops inventing
parameters. A script that fails to run also gets a static API check (unknown names/keywords/enum members with
"did you mean") so one refinement round fixes all API mistakes. Everything degrades gracefully; `KB_ENABLED=0`
turns it off. Check `GET /kb/status`. Semantic embeddings are off by default (`KB_EMBEDDINGS=1` to enable) because
Render's free disk is ephemeral and the index would be re-embedded on every wake-up.

## PyVista (optional)

Off by default because VTK needs a lot of RAM. On the 512 MB free instance, open
`/pyvista-selftest` first: it loads PyVista and reports `rss_mb` (before/after/peak) so you can see
whether it fits next to OpenCascade. Then:
1. `USE_PYVISTA=1` — SimScale result files are read with PyVista (more formats, handles cell-data-only
   results); meshio stays as the fallback.
2. `/pyvista-selftest?render=true`, and if it passes, `PYVISTA_RENDER=1` — each SimScale run also returns
   `simscale_stress_image_base64` (PNG of the von Mises field with the hotspot marked). If the render is blank
   or errors, try `PYVISTA_GL_BACKEND=egl` or `auto`.

## State of things

The sandbox, SimScale orchestration and feedback logic were tested against stubs/mocks.
The build123d calls and the live SimScale API have not been run end-to-end by the
author of this change — use the two selftest endpoints before relying on it.
