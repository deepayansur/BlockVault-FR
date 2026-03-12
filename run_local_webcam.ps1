# Run local face_service (no Docker) + webcam test
# Usage: powershell -ExecutionPolicy Bypass -File .\run_local_webcam.ps1

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$VenvPath = Join-Path $RepoRoot '.venv'
$PythonExe = Join-Path $VenvPath 'Scripts\python.exe'
$ClientRequirements = Join-Path $RepoRoot 'requirements.txt'
$ServiceRequirements = Join-Path $RepoRoot 'face_service\requirements.txt'
$FaceDbDir = Join-Path $RepoRoot 'face_service\face_db\authorized'
$EmbeddingsPath = Join-Path $RepoRoot 'face_service\face_db\embeddings.json'
$WebcamScript = Join-Path $RepoRoot 'webcam\webcam_test.py'
$OrtPackage = if ($env:FACE_SERVICE_ORT_PACKAGE) { $env:FACE_SERVICE_ORT_PACKAGE } else { 'onnxruntime' }
$ServerWorkers = if ($env:FACE_SERVER_WORKERS) { $env:FACE_SERVER_WORKERS } else { '2' }

if (-not (Test-Path $PythonExe)) {
  python -m venv $VenvPath
}

& $PythonExe -m pip install -r $ClientRequirements
& $PythonExe -m pip install -r $ServiceRequirements
& $PythonExe -m pip install $OrtPackage

New-Item -ItemType Directory -Force -Path $FaceDbDir | Out-Null
$env:FACE_DB_DIR = [System.IO.Path]::GetFullPath($FaceDbDir)
$env:EMBEDDINGS_PATH = [System.IO.Path]::GetFullPath($EmbeddingsPath)
$env:ARC_THRESHOLD = '0.38'
$env:DET_SIZE = '448'
$env:FACE_USE_GPU = '0'
$env:FACE_GPU_DEVICE_ID = '0'
$env:FACE_RUNTIME_PROFILE = 'cpu-balanced'
$env:FACE_SERVER_WORKERS = $ServerWorkers
$env:FACE_CPU_TARGET = 'concurrent_throughput'
$env:OMP_NUM_THREADS = '6'
$env:OMP_WAIT_POLICY = 'PASSIVE'
$env:OMP_PROC_BIND = 'TRUE'
$env:OMP_PLACES = 'cores'

Start-Process -WindowStyle Minimized -WorkingDirectory $RepoRoot -FilePath $PythonExe -ArgumentList '-m','uvicorn','face_service.app:app','--host','127.0.0.1','--port','8001','--workers',$ServerWorkers

Start-Sleep -Seconds 3
& $PythonExe $WebcamScript --face-url http://127.0.0.1:8001 --camera 0
