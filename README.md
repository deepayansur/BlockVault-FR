# BlockVault FR Deploy

This branch is trimmed for one purpose: run the face recognition API.

## Start the API

Run from this repo directory:

```powershell
cd "C:\Users\deepa\Desktop\BlockVault\BlockVault FR"
docker compose up --build
```

The API is exposed on `http://localhost:8001`.

## Endpoints

- `GET /health`
- `POST /reload`
- `POST /verify`

## Authorized identities

Add 1-5 images per person under:

```text
face_service/face_db/authorized/<person_id>/*.jpg
```

Then rebuild embeddings:

```bash
curl -X POST http://localhost:8001/reload
```

## Local run without Docker

```bash
pip install -r face_service/requirements.txt
pip install onnxruntime
uvicorn face_service.app:app --host 0.0.0.0 --port 8001
```

## Notes

- `face_service/face_db/authorized/.gitkeep` keeps the directory in git.
- Authorized face images and generated embeddings are intentionally ignored by git.
