# Lumexa Backend (v8.6)

AI-generated mechanical CAD parts with real engineering validation before
handoff to Ansys/SolidWorks — generate a design from a prompt, run real FEA
(solid tetrahedral via CalculiX+Gmsh) and DFM checks on it, self-correct
against failures, export STEP/DXF for downstream tools.

## Stack

FastAPI + CadQuery (OpenCASCADE) for geometry, CalculiX for FEM, Gmsh for
volume meshing, trimesh/scipy for analysis, ezdxf for 2D drawing export.
AI generation routes through one of three providers — Google's Gemini API
directly, direct Anthropic Claude API, or the Lovable/Gemini gateway —
whichever env var is set (see below).

## Setup

```
pip install -r requirements.txt --break-system-packages
```
Needs system binaries too: `ccx` (CalculiX) and the libs in `Dockerfile` —
use the Dockerfile rather than a bare `pip install` unless you already have
those installed locally.

### Environment variables

| Variable | Required | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | one of these three | Direct Gemini API — current setup |
| `ANTHROPIC_API_KEY` | " | Direct Claude API |
| `LOVABLE_API_KEY` | " | Gemini via Lovable's gateway |
| `AI_PROVIDER` | no | Force `claude`/`gemini`/`lovable`; auto-detects from whichever key above is set otherwise |
| `ALLOWED_ORIGINS` | recommended | Comma-separated frontend origins for CORS; defaults to `*` with credentials off if unset |

## Deploy (Render)

`render.yaml` is a Blueprint — Render dashboard → **New → Blueprint** →
connect this repo → confirm → paste in `GOOGLE_API_KEY` when prompted.
See `Dockerfile` for what it installs; free tier works for testing but
real FEA jobs (solid tet meshing) are memory/CPU-heavy — expect to need a
paid tier once this is handling real traffic, not just test calls.

## Endpoints

- `GET /` — capabilities, active AI provider, changelog
- `GET /materials`, `GET /part-types` — reference data
- `POST /generate-part`, `/generate-and-analyze`, `/generate-from-prompt` — generation, increasing levels of analysis
- `POST /generate-validate-refine` — self-healing loop: generate → analyze → auto-fix until it passes quality checks
- `POST /refine-from-external-fea` — feed in a real Ansys stress-hotspot CSV export, get a corrected design through the same refinement engine
- `POST /edit-design-region` — draw a 3D bounding box, AI regenerates only what's inside it (cut+union — everything outside is geometrically guaranteed unchanged)
- `POST /export-step` — STEP export at any point in a design's lifecycle (for manual editing in FreeCAD)
- `POST /export-drawing-dxf` — 2D manufacturing drawing (convex-hull orthographic views + dimensions + title block)
- `POST /analyze-part`, `/analyze-part-deep` — FEA/DFM analysis on an uploaded mesh (deep = background CalculiX job, poll via `/job/{job_id}`)
- `POST /analyze-composite`, `/analyze-rainflow`, `/analyze-assembly` — specialized analyses (CLT/Tsai-Wu, ASTM E1049 fatigue, multi-part assemblies)
- `POST /compare-designs`, `/image-to-params` — design comparison, image-to-parameter estimation

## Honest state of things

Built and tested where this environment allowed: sandbox validation for
AI-generated CadQuery scripts (AST-based, checked against real sandbox-escape
techniques), CSV parsing for the Ansys-feedback endpoint, the DXF geometry
math, the boundary-region script-merging logic — all unit-tested against
realistic inputs during development.

**Not executable-tested end to end**: the actual CalculiX/Gmsh solid-tet FEM
pipeline and CadQuery boolean operations, since those binaries/libraries
aren't available in the dev sandbox this was built in. Test a real generation
+ FEA call on your actual deployment before depending on it for anything —
the shell-FEM and analytical fallbacks stay in place if the solid-tet path
fails on a given part, so a failure there degrades gracefully rather than
breaking the request.
