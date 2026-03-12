# BVS Face Recognition Evaluation + Live Demo

This repo contains:
- A Dockerized `face_service` built on InsightFace (SCRFD + ArcFace)
- Local webcam demos via OpenCV and Streamlit
- Offline evaluation and CPU benchmark tooling for VGGFace2

Datasets are not included. Download them separately and point the evaluation scripts at your local dataset paths.

## Repo location
Run commands from this repo directory:

```powershell
cd "C:\Users\deepa\Desktop\BlockVault\BlockVault FR"
```

## CPU face_service (default compose profile)
The compose file is now CPU-first for the target web-server deployment.

```bash
docker compose up --build
```

Health check: `http://localhost:8001/health`

## Authorized identities
Put 1-5 images per person under:

```text
face_service/face_db/authorized/<person_id>/*.jpg
```

Then rebuild embeddings:

```bash
curl -X POST http://localhost:8001/reload
```

## Live webcam test
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -r face_service/requirements.txt
pip install onnxruntime

python webcam/webcam_test.py --face-url http://localhost:8001 --camera 0
```

On Windows you can also use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_local_webcam.ps1
```

## Streamlit live UI
```bash
streamlit run streamlit_app.py
```

## Offline evaluation and generated reports
See `eval/README_EVAL.md` for dataset layout expectations and commands.

The main CPU report command is:

```bash
python eval/generate_results.py --max-images 1000 --num-workers 4
```

That writes:
- `results.json`
- `results.txt`

See `docs/architecture.md` for architecture details.


docker compose build --no-cache face_service
docker compose up