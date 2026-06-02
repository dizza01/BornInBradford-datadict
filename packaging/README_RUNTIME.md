# Born in Bradford Assistant Runtime

This is a runtime-only bundle for internal testing. It includes the assistant UI,
the prebuilt Chroma index, the data dictionary docs, and the quantized GGUF model.

It does not include evaluation scripts, notebooks, training code, quantization
tools, or the full non-quantized model.

## Start on macOS

Requires Python 3.11 or newer. If needed, install it with Homebrew:

```bash
brew install python@3.11
```

From this folder:

```bash
./start_mac.sh
```

Then open:

```text
http://127.0.0.1:5050/assistant
```

The first run creates `.venv` and installs Python dependencies. On Apple Silicon,
the script installs `llama-cpp-python` with Metal support.

If the script says the existing `.venv` was created with an older Python, delete
`.venv`, install Python 3.11+, and run `./start_mac.sh` again.

## Start on Windows

Open PowerShell in this folder and run:

```powershell
.\start_windows.ps1
```

Then open:

```text
http://127.0.0.1:5050/assistant
```

The Windows script installs the CPU build of `llama-cpp-python` by default. This
is simpler for internal testing, but may be slower than Apple Silicon Metal or a
separate CUDA build.

If PowerShell blocks the script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\start_windows.ps1
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_windows.ps1
```

Requires Python 3.11 or newer. If the script says Python was not found, install Python 3.11 from
https://www.python.org/downloads/windows/ and tick `Add python.exe to PATH`.
Then reopen PowerShell and run the launcher again.

## Runtime Settings

You can tune the local model/RAG settings with environment variables before
starting:

```text
LLAMA_N_CTX=4096
LLAMA_N_GPU_LAYERS=-1
RAG_N_RESULTS=3
RAG_CONTEXT_MAX_CHARS=3500
```

On Windows CPU, `LLAMA_N_GPU_LAYERS` defaults to `0`.

## Included Runtime Data

- `llm_poc/.chroma_db/`: prebuilt Chroma vector database.
- `docs/`: data dictionary HTML and CSV files used by the UI and registry.
- `papers/RCADS25-Youth-English-2018.pdf`: source PDF used for direct RCADS-25 item wording answers.
- `models/bib-llama-3.1-8b.Q4_K_M.gguf`: main quantized local model.

## Rebuilding the Bundle

Run this from the full development repository:

```bash
packaging/make_runtime_bundle.sh
```

The output is written to:

```text
dist/bib-assistant-runtime
```
