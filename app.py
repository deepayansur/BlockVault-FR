import base64
import os
from typing import Any, Dict, Optional

import cv2
import requests


def resize_max_side(img_bgr, max_side: Optional[int]):
    if max_side is None or max_side <= 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return img_bgr

    scale = float(max_side) / float(current_max)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def encode_jpeg_dataurl(img_bgr, q: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise ValueError("Failed to encode JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


class FaceClient:
    def __init__(
        self,
        mode: Optional[str] = None,
        base_url: Optional[str] = None,
        engine: Optional[object] = None,
        max_side: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> None:
        self.mode = (mode or os.getenv("FACE_MODE", "remote")).lower()
        self.base_url = (base_url or os.getenv("FACE_URL", "http://localhost:8001")).rstrip("/")
        self.engine = engine
        self.max_side = int(max_side) if max_side is not None else int(os.getenv("FACE_MAX_SIDE", "960"))
        self.jpeg_quality = int(jpeg_quality) if jpeg_quality is not None else int(os.getenv("FACE_JPEG_QUALITY", "75"))
        self.session = requests.Session() if self.mode != "local" else None

        if self.mode == "local":
            from face_service.ml import FaceEngine
            if self.engine is None:
                self.engine = FaceEngine()
            self.engine.init_models()
            if not self.engine.load_db():
                self.engine.rebuild_db()

    def reload_embeddings(self) -> Dict[str, Any]:
        if self.mode == "local":
            assert self.engine is not None
            db_size = self.engine.rebuild_db()
            return {"ok": True, "db_size": db_size, "db_embeddings": self.engine.db_embedding_count()}

        assert self.session is not None
        r = self.session.post(self.base_url + "/reload", timeout=20)
        return r.json()

    def verify_dataurl(self, dataurl: str) -> Dict[str, Any]:
        if self.mode == "local":
            assert self.engine is not None
            return self.engine.verify_dataurl(dataurl)

        assert self.session is not None
        r = self.session.post(self.base_url + "/verify", json={"imageBase64": dataurl}, timeout=5)
        return r.json()

    def verify_bgr(self, img_bgr) -> Dict[str, Any]:
        if self.mode == "local":
            assert self.engine is not None
            return self.engine.verify_bgr(img_bgr)

        img_bgr = resize_max_side(img_bgr, self.max_side)
        dataurl = encode_jpeg_dataurl(img_bgr, q=self.jpeg_quality)
        return self.verify_dataurl(dataurl)
