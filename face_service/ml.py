import base64
import glob
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - runtime dependency may be absent during static inspection
    ort = None

FACE_SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_FACE_DB_DIR = os.getenv("FACE_DB_DIR", str(FACE_SERVICE_DIR / "face_db" / "authorized"))
DEFAULT_EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", str(FACE_SERVICE_DIR / "face_db" / "embeddings.json"))
DEFAULT_ARC_THRESHOLD = float(os.getenv("ARC_THRESHOLD", "0.38"))
DEFAULT_DET_SIZE = int(os.getenv("DET_SIZE", "640"))
DEFAULT_USE_GPU = os.getenv("FACE_USE_GPU", "1").strip().lower() not in {"0", "false", "no", "off"}
DEFAULT_GPU_DEVICE_ID = int(os.getenv("FACE_GPU_DEVICE_ID", "0"))
MODEL_NAME = "insightface(buffalo_l):SCRFD+ArcFace"


class FaceEngine:
    def __init__(
        self,
        face_db_dir: str = DEFAULT_FACE_DB_DIR,
        embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
        arc_threshold: float = DEFAULT_ARC_THRESHOLD,
        det_size: int = DEFAULT_DET_SIZE,
        providers: Optional[List[str]] = None,
        use_gpu: bool = DEFAULT_USE_GPU,
        gpu_device_id: int = DEFAULT_GPU_DEVICE_ID,
    ) -> None:
        self.face_db_dir = face_db_dir
        self.embeddings_path = embeddings_path
        self.arc_threshold = arc_threshold
        self.det_size = det_size
        self.gpu_device_id = gpu_device_id
        self.use_gpu = use_gpu

        self.available_providers = self._available_providers()
        self.providers, self.execution_device, self.ctx_id = self._resolve_execution(providers)

        self.face_app: Optional[FaceAnalysis] = None
        self.db: Dict[str, np.ndarray] = {}
        self.db_matrix = np.zeros((0, 512), dtype=np.float32)
        self.db_labels: List[str] = []
        self.infer_lock = threading.Lock()
        self.db_lock = threading.Lock()

    def _available_providers(self) -> List[str]:
        if ort is None:
            return []
        try:
            return list(ort.get_available_providers())
        except Exception:
            return []

    def _resolve_execution(self, providers: Optional[Sequence[str]]) -> Tuple[List[str], str, int]:
        if providers:
            provider_list = list(providers)
        elif self.use_gpu and "CUDAExecutionProvider" in self.available_providers:
            provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            provider_list = ["CPUExecutionProvider"]

        uses_cuda = any(provider == "CUDAExecutionProvider" for provider in provider_list)
        execution_device = f"cuda:{self.gpu_device_id}" if uses_cuda else "cpu"
        ctx_id = self.gpu_device_id if uses_cuda else -1
        return provider_list, execution_device, ctx_id

    def runtime_info(self) -> Dict[str, Any]:
        return {
            "execution_device": self.execution_device,
            "providers": self.providers,
            "available_providers": self.available_providers,
            "gpu_device_id": self.gpu_device_id,
            "det_size": self.det_size,
            "arc_threshold": self.arc_threshold,
            "runtime_profile": os.getenv("FACE_RUNTIME_PROFILE", "default"),
            "server_workers": int(os.getenv("FACE_SERVER_WORKERS", "1")),
            "cpu_target": os.getenv("FACE_CPU_TARGET", "concurrent_throughput"),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
            "omp_wait_policy": os.getenv("OMP_WAIT_POLICY", ""),
            "omp_proc_bind": os.getenv("OMP_PROC_BIND", ""),
            "omp_places": os.getenv("OMP_PLACES", ""),
        }

    def db_size(self) -> int:
        with self.db_lock:
            return len(self.db)

    def db_embedding_count(self) -> int:
        with self.db_lock:
            return int(self.db_matrix.shape[0])

    def _norm(self, vector: np.ndarray) -> np.ndarray:
        vector = vector.astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)

    def _norm_rows(self, embs: np.ndarray) -> np.ndarray:
        embs = embs.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        return embs / norms

    def _to_embedding_matrix(self, value: Any) -> Optional[np.ndarray]:
        arr = np.array(value, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim != 2:
            return None
        return self._norm_rows(arr)

    def _build_index_snapshot(self, db_state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        mats: List[np.ndarray] = []
        labels: List[str] = []
        for person_id, embs in db_state.items():
            if embs.size == 0:
                continue
            mats.append(embs)
            labels.extend([person_id] * int(embs.shape[0]))

        if mats:
            return np.concatenate(mats, axis=0).astype(np.float32), labels
        return np.zeros((0, 512), dtype=np.float32), []

    def _swap_db_state(self, db_state: Dict[str, np.ndarray], db_matrix: np.ndarray, db_labels: List[str]) -> None:
        with self.db_lock:
            self.db = db_state
            self.db_matrix = db_matrix
            self.db_labels = db_labels

    def _snapshot_match_index(self) -> Tuple[np.ndarray, List[str]]:
        with self.db_lock:
            return self.db_matrix, self.db_labels

    def _write_db_snapshot(self, db_state: Dict[str, np.ndarray]) -> None:
        serial = {key: value.tolist() for key, value in db_state.items()}
        directory = os.path.dirname(self.embeddings_path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix="embeddings_", suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(serial, handle)
            os.replace(tmp_path, self.embeddings_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _timings(self, decode_ms: float = 0.0, infer_ms: float = 0.0, match_ms: float = 0.0) -> Dict[str, float]:
        total_ms = decode_ms + infer_ms + match_ms
        return {
            "decode": round(float(decode_ms), 2),
            "infer": round(float(infer_ms), 2),
            "match": round(float(match_ms), 2),
            "total": round(float(total_ms), 2),
        }

    def init_models(self) -> None:
        if self.face_app is not None:
            return
        self.face_app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.face_app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size))

    def decode_dataurl(self, dataurl: str) -> np.ndarray:
        if "," in dataurl:
            dataurl = dataurl.split(",", 1)[1]
        raw = base64.b64decode(dataurl)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    def encode_jpeg_dataurl(self, img_bgr: np.ndarray, q: int = 80) -> str:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise ValueError("Failed to encode JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return "data:image/jpeg;base64," + b64

    def load_db(self) -> bool:
        if not os.path.exists(self.embeddings_path):
            return False
        with open(self.embeddings_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        loaded: Dict[str, np.ndarray] = {}
        for person_id, value in payload.items():
            embs = self._to_embedding_matrix(value)
            if embs is None or embs.shape[0] == 0:
                continue
            loaded[person_id] = embs

        db_matrix, db_labels = self._build_index_snapshot(loaded)
        self._swap_db_state(loaded, db_matrix, db_labels)
        return True

    def save_db(self) -> None:
        with self.db_lock:
            snapshot = {key: value.copy() for key, value in self.db.items()}
        self._write_db_snapshot(snapshot)

    def _detect_faces_unlocked(self, img_bgr: np.ndarray):
        if self.face_app is None:
            raise RuntimeError("FaceAnalysis is not initialized")
        return self.face_app.get(img_bgr)

    def detect_faces(self, img_bgr: np.ndarray):
        self.init_models()
        with self.infer_lock:
            faces = self._detect_faces_unlocked(img_bgr)
        return list(faces)

    def rebuild_db(self) -> int:
        self.init_models()
        os.makedirs(self.face_db_dir, exist_ok=True)

        new_db: Dict[str, List[np.ndarray]] = {}
        subdirs = [path for path in glob.glob(os.path.join(self.face_db_dir, "*")) if os.path.isdir(path)]

        if subdirs:
            image_sets = []
            for person_dir in subdirs:
                person_id = os.path.basename(person_dir)
                image_sets.append((person_id, glob.glob(os.path.join(person_dir, "*.*"))))
        else:
            image_sets = [
                (
                    os.path.splitext(os.path.basename(image_path))[0],
                    [image_path],
                )
                for image_path in glob.glob(os.path.join(self.face_db_dir, "*.*"))
            ]

        for person_id, images in image_sets:
            for image_path in images:
                if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(image_path)
                if img is None:
                    continue
                faces = self.detect_faces(img)
                if not faces:
                    continue
                faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
                emb = self._norm(faces[0].embedding)
                new_db.setdefault(person_id, []).append(emb)

        rebuilt_db: Dict[str, np.ndarray] = {}
        for person_id, embs in new_db.items():
            if embs:
                rebuilt_db[person_id] = self._norm_rows(np.stack(embs, axis=0))

        db_matrix, db_labels = self._build_index_snapshot(rebuilt_db)
        self._write_db_snapshot(rebuilt_db)
        self._swap_db_state(rebuilt_db, db_matrix, db_labels)
        return self.db_size()

    def embed_largest_face(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self.detect_faces(img_bgr)
        if not faces:
            return None
        faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
        return self._norm(faces[0].embedding)

    def _best_match(self, emb: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        db_matrix, db_labels = self._snapshot_match_index()
        if db_matrix.shape[0] == 0:
            return None, None
        scores = db_matrix @ emb
        best_idx = int(np.argmax(scores))
        return db_labels[best_idx], float(scores[best_idx])

    def verify_bgr(self, img_bgr: np.ndarray, decode_ms: float = 0.0) -> Dict[str, Any]:
        try:
            infer_start = time.perf_counter()
            faces = self.detect_faces(img_bgr)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            if not faces:
                return {
                    "ok": True,
                    "facesDetected": 0,
                    "frDecision": "NO_FACE",
                    "bestMatch": {"id": None, "score": None},
                    "model": MODEL_NAME,
                    "threshold": self.arc_threshold,
                    "execution": self.runtime_info(),
                    "timingsMs": self._timings(decode_ms=decode_ms, infer_ms=infer_ms),
                    "faces": [],
                }

            faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
            face_out = []
            best_overall = {"id": None, "score": None}
            best_score = -1.0

            match_start = time.perf_counter()
            for face in faces:
                emb = self._norm(face.embedding)
                person_id, score = self._best_match(emb)
                if score is not None and score > best_score:
                    best_score = score
                    best_overall = {"id": person_id, "score": score}
                face_out.append(
                    {
                        "bbox": [float(x) for x in face.bbox.tolist()],
                        "matchId": person_id,
                        "matchScore": score,
                    }
                )
            match_ms = (time.perf_counter() - match_start) * 1000.0

            decision = "ALLOW" if (best_overall["score"] is not None and best_overall["score"] >= self.arc_threshold) else "UNKNOWN"
            return {
                "ok": True,
                "facesDetected": len(faces),
                "frDecision": decision,
                "bestMatch": best_overall,
                "model": MODEL_NAME,
                "threshold": self.arc_threshold,
                "execution": self.runtime_info(),
                "timingsMs": self._timings(decode_ms=decode_ms, infer_ms=infer_ms, match_ms=match_ms),
                "faces": face_out,
            }
        except Exception as exc:
            return {
                "ok": False,
                "facesDetected": 0,
                "frDecision": "ERROR",
                "bestMatch": {"id": None, "score": None},
                "model": MODEL_NAME,
                "threshold": self.arc_threshold,
                "execution": self.runtime_info(),
                "timingsMs": self._timings(decode_ms=decode_ms),
                "error": str(exc),
                "faces": [],
            }

    def verify_dataurl(self, dataurl: str) -> Dict[str, Any]:
        try:
            decode_start = time.perf_counter()
            img = self.decode_dataurl(dataurl)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
        except Exception as exc:
            return {
                "ok": False,
                "facesDetected": 0,
                "frDecision": "ERROR",
                "bestMatch": {"id": None, "score": None},
                "model": MODEL_NAME,
                "threshold": self.arc_threshold,
                "execution": self.runtime_info(),
                "timingsMs": self._timings(),
                "error": str(exc),
                "faces": [],
            }
        return self.verify_bgr(img, decode_ms=decode_ms)
