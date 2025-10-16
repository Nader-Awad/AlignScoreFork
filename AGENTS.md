# AlignScore Service Plan

## Repository Purpose
- Fork of `yuh-zha/AlignScore` dedicated to running AlignScore as a standalone scoring service.
- Exposes a stable HTTP API that downstream projects (e.g., `Honours-Thesis/Code`) can call for factual consistency metrics without inheriting AlignScore’s legacy dependency stack.

## Key Requirements
- Keep AlignScore’s Python/Torch stack pinned to the upstream versions (PyTorch 1.12.x, Python 3.10/3.11, spaCy `en_core_web_sm`).
- Publish a lightweight REST service (FastAPI or Flask) that
  - Loads the AlignScore checkpoint once at startup.
  - Accepts batched `(context, claim)` payloads.
  - Returns per-example scores plus aggregate metadata (model name, evaluation mode, inference time).
- Provide containerized deployment (Docker) with clear build/run instructions.
- Maintain reproducibility: document all environment variables, seed handling, model paths, and checkpoints.

## Immediate To‑Dos
1. **Environment Scaffolding**
   - ✅ Added `service/` package with FastAPI app, Pydantic schemas, and AlignScore runner (`service/app.py`, `service/schemas.py`, `service/alignscore_runner.py`, `service/config.py`).
   - Introduce `pyproject.toml` or `requirements-service.txt` separate from upstream training requirements.
   - Add unit tests for request validation and mocked scoring in `tests/`.
2. **Model Management**
   - Decide where to store checkpoints (`/models/alignscore` inside the image vs mounted volume).
   - Support configuration via `ALIGNSCORE_MODEL_PATH`, `ALIGNSCORE_DEVICE`, `ALIGNSCORE_BATCH_SIZE`.
3. **Dockerization**
   - Write `Dockerfile` targeting Python 3.11 slim.
   - Install system deps (`build-essential`, `libomp`), Torch 1.12.x CPU wheel, AlignScore (editable or package), spaCy model.
   - Provide `docker-compose.yml` for local dev (optional).
4. **API Contract**
   - Implemented `/healthz` and `/score` endpoints with JSON schema enforcement (Pydantic models); add documentation & curls to README.
   - Include example curl / `httpx` client snippets.
   - Consider `/metadata` if extra runtime info is helpful.
5. **Integration with Honours-Thesis**
   - Provide a small Python client (`clients/honours_client.py`) or documented curl commands.
   - Wire the honours CLI to call `http://127.0.0.1:9000/score` (service verified locally via curl; pending honours-side integration).
6. **CI / QA**
   - Add GitHub Actions workflow to build the Docker image and run unit tests.
   - Set up linting (ruff/black) for the new service code.

## Deliverables
- ✅ `service/app.py` (FastAPI), `service/alignscore_runner.py`, `service/schemas.py`, `service/config.py`.
- Dockerfile + README section with build/run instructions.
- Automated tests (`pytest`) covering schema validation and mocked scoring.
- API reference (`docs/service.md`) detailing endpoints and expected payloads.
- Simple client example for the Honours project.

## References
- AlignScore paper: Zha et al., “AlignScore: Evaluating Factual Consistency with a Unified Alignment Function,” ACL 2023.
- AlignScore upstream README for checkpoint URLs and evaluation modes.
