# Project-GPTea

## Dependencies

Install from `requirements.txt`.

For development tooling (tests, lint, type checks), install `requirements-dev.txt`.

## Layout

- `app.py` - app bootstrap and FastAPI/Gradio mounting
- `config.py` - environment-based configuration
- `model_manager.py` - one model instance + adapter toggle logic
- `api.py` - `/generate`, `/compare`, `/health`
- `ui.py` - Gradio interface mounted in FastAPI

## Setup (.venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Adapter setup

By default, the service expects adapter files in `./adapter` (project root).

If you do not pass custom adapter flags/env vars, place the adapter folder exactly here:

```text
Project-GPTea/
  adapter/
    adapter_config.json
    adapter_model.safetensors
    ...
```

Download adapter zip: [Google Drive](https://drive.google.com/file/d/1WRj8HkNxus5qiB9pW7XJPcoR82C9Q2tK/view?usp=sharing)

Then unzip it in the project root so the path is `./adapter`:

```bash
unzip adapter.zip -d .
```

## Endpoints

- `GET /health`
- `POST /generate`
- `POST /compare`

Gradio UI is available at `http://localhost:8000/ui`.

## Test & run service

```bash
./test.sh # Run all checks (format + lint + mypy + pytest)
./run.sh # Start service in blocking mode
```

## Configuration

Environment variables (optional):

- `BASE_MODEL_ID` (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `ADAPTER_LOCAL_PATH` (default: `adapter`)
- `ADAPTER_REPO_ID` (used only if local adapter path is missing)
- `ADAPTER_NAME` (default: `rs_lora`)
- `HF_CACHE_DIR`
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `UI_MOUNT_PATH` (default: `/ui`)
- `DEFAULT_TEMPERATURE` (default: `0.7`)
- `DEFAULT_TOP_P` (default: `0.9`)
- `DEFAULT_MAX_NEW_TOKENS` (default: `128`)
- `MAX_NEW_TOKENS_CAP` (hard-capped at `256`)
