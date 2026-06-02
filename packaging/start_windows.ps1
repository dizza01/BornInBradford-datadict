$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path (Join-Path $ScriptDir "llm_poc\requirements_runtime.txt")) {
    $RootDir = $ScriptDir
    $RequirementsFile = Join-Path $RootDir "llm_poc\requirements_runtime.txt"
}
elseif ((Test-Path (Join-Path $ScriptDir "runtime_requirements.txt")) -and (Test-Path (Join-Path $ScriptDir "..\llm_poc"))) {
    $RootDir = Resolve-Path (Join-Path $ScriptDir "..")
    $RequirementsFile = Join-Path $ScriptDir "runtime_requirements.txt"
}
else {
    Write-Error "Could not find runtime files. Run this from dist\bib-assistant-runtime\start_windows.ps1, or rebuild with packaging\make_runtime_bundle.sh."
}

$VenvDir = Join-Path $RootDir ".venv"
$ModelPath = Join-Path $RootDir "models\bib-llama-3.1-8b.Q4_K_M.gguf"

Set-Location $RootDir
$env:BIB_RUNTIME_LOCKED = "1"

function Get-PythonCommand {
    $candidates = @(
        @("py", "-3.11"),
        @("py", "-3"),
        @("python"),
        @("python3")
    )

    foreach ($candidate in $candidates) {
        $cmd = $candidate[0]
        $extraArgs = @()
        if ($candidate.Count -gt 1) {
            $extraArgs = $candidate[1..($candidate.Count - 1)]
        }

        try {
            $versionText = & $cmd @extraArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($LASTEXITCODE -eq 0) {
                $parts = "$versionText".Trim().Split(".")
                $major = [int]$parts[0]
                $minor = [int]$parts[1]
                if (($major -gt 3) -or (($major -eq 3) -and ($minor -ge 11))) {
                    return @{ Command = $cmd; Args = $extraArgs }
                }
            }
        }
        catch {
            continue
        }
    }

    Write-Error "Python 3.11 or newer was not found. Install Python 3.11 from https://www.python.org/downloads/windows/ and tick 'Add python.exe to PATH', then reopen PowerShell."
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating local Python environment..."
    $PythonCommand = Get-PythonCommand
    & $PythonCommand.Command @($PythonCommand.Args) -m venv $VenvDir
}

$Python = Join-Path $VenvDir "Scripts\python.exe"

$VenvVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$VenvParts = "$VenvVersion".Trim().Split(".")
if (([int]$VenvParts[0] -lt 3) -or (([int]$VenvParts[0] -eq 3) -and ([int]$VenvParts[1] -lt 11))) {
    Write-Error "The existing .venv was created with Python $VenvVersion. Delete $VenvDir, install Python 3.11+, then rerun this script."
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r $RequirementsFile

& $Python -c "import llama_cpp" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing llama-cpp-python CPU build..."
    & $Python -m pip install -U llama-cpp-python --no-cache-dir
}

$ChromaPath = Join-Path $RootDir "llm_poc\.chroma_db\chroma.sqlite3"
if (-not (Test-Path $ChromaPath)) {
    Write-Error "Missing Chroma DB at llm_poc\.chroma_db\chroma.sqlite3. This runtime bundle expects a prebuilt index."
}

if (-not (Test-Path $ModelPath)) {
    Write-Error "Missing GGUF model at $ModelPath"
}

$LlamaNCtx = if ($env:LLAMA_N_CTX) { $env:LLAMA_N_CTX } else { "4096" }
$LlamaNGpuLayers = if ($env:LLAMA_N_GPU_LAYERS) { $env:LLAMA_N_GPU_LAYERS } else { "0" }
$RagNResults = if ($env:RAG_N_RESULTS) { $env:RAG_N_RESULTS } else { "3" }
$RagContextMaxChars = if ($env:RAG_CONTEXT_MAX_CHARS) { $env:RAG_CONTEXT_MAX_CHARS } else { "3500" }

Write-Host "Starting Born in Bradford assistant..."
Write-Host "Open http://127.0.0.1:5050/assistant"

& $Python (Join-Path $RootDir "llm_poc\server.py") `
    --llm-backend llama_cpp `
    --gguf-model-path $ModelPath `
    --llama-n-ctx $LlamaNCtx `
    --llama-n-gpu-layers $LlamaNGpuLayers `
    --rag-n-results $RagNResults `
    --rag-context-max-chars $RagContextMaxChars `
    @args
