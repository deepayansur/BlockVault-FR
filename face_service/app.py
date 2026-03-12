import os
import sys
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from fastapi import FastAPI
from pydantic import BaseModel

from face_service.ml import FaceEngine

app = FastAPI(title="Face Service (InsightFace SCRFD + ArcFace)")
engine = FaceEngine()


def service_runtime_profile() -> Dict[str, Any]:
    return {
        "profile": os.getenv("FACE_RUNTIME_PROFILE", "default"),
        "server_workers": int(os.getenv("FACE_SERVER_WORKERS", "1")),
        "cpu_target": os.getenv("FACE_CPU_TARGET", "concurrent_throughput"),
    }


@app.on_event("startup")
def startup() -> None:
    engine.init_models()
    if not engine.load_db():
        engine.rebuild_db()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "db_size": engine.db_size(),
        "db_embeddings": engine.db_embedding_count(),
        "execution": engine.runtime_info(),
        "service": service_runtime_profile(),
    }


@app.post("/reload")
def reload_db() -> Dict[str, Any]:
    db_size = engine.rebuild_db()
    return {
        "ok": True,
        "db_size": db_size,
        "db_embeddings": engine.db_embedding_count(),
        "execution": engine.runtime_info(),
        "service": service_runtime_profile(),
    }


class VerifyReq(BaseModel):
    imageBase64: str
    cameraId: Optional[str] = None


@app.post("/verify")
def verify(req: VerifyReq) -> Dict[str, Any]:
    return engine.verify_dataurl(req.imageBase64)
