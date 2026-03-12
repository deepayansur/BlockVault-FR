# Software Architecture (Face Recognition + Evaluation)

## Components
1) **Client / UI**
   - `webcam/webcam_test.py` (OpenCV) or `streamlit_app.py`
   - Captures frames and periodically sends a JPEG snapshot to the face service.
   - Displays detections + recognition decision overlay.

2) **Face Service** (`face_service`)
   - Dockerized FastAPI microservice.
   - Uses InsightFace bundle `buffalo_l`:
     - **SCRFD** for face detection
     - **ArcFace** for embeddings
   - Endpoints:
     - `GET /health`
     - `POST /reload` rebuild embeddings DB from `face_db/authorized`
     - `POST /verify` returns boxes + best match + decision

3) **Local Face Engine (No Server)**
   - `FaceEngine` in `face_service/ml.py` can run directly without Docker.
   - `app.FaceClient(mode="local")` uses the same engine for local testing.
   - Used by local scripts:
     - `local_fr_test.py` (image/webcam verification)
     - `eval/vggface2_verify.py` (VGGFace2 verification)
     - `eval/det_size_sweep.py` (detector size sweep)
     - `eval/generate_results.py` (generated CPU metrics + latency report)
   - `eval/qmul_verify_pairs.py` remains available for separate experiments but is not part of the main generated metrics flow.

4) **Authorized Face DB**
   - Filesystem-based DB for demo:
      - Images in `face_service/face_db/authorized/<person_id>/*.jpg`
      - Derived embeddings in `face_service/face_db/embeddings.json`
   - Startup loads existing `embeddings.json`; if missing, builds from images.
   - Default DB paths are resolved from `face_service/ml.py`, so the repo can sit inside a parent `BlockVault` workspace without changing code paths.

5) **Datasets (Local Evaluation)**
   - Stored under `datasets/` (not checked into repo).
   - Main generated-report dataset:
      - `datasets/VGGface2_HQ_1/VGGface2_None_norm_512_true_bygfpgan`

## Data Flow (Service Mode)
Frame -> JPEG -> `/verify` -> `{faces, bestMatch, decision}` -> overlay + logging.

## Data Flow (Local Eval Mode)
Dataset image -> `FaceEngine.embed_largest_face` -> embeddings -> verification metrics + CPU latency report.

## Integration into your incident system (Axis cameras later)
- Axis camera frames/snapshots still become JPEG bytes (same as webcam).
- Your API Gateway should call `face_service /verify` only on incident snapshots (motion gate).
- You record `frDecision/facesDetected/matchScore` into incident metadata and (optionally) ledger.

## Local tooling note
- `run_local_webcam.ps1` resolves all repo paths from the script location instead of the caller's current directory.
