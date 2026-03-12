import os
import time

import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, WebRtcMode, webrtc_streamer

from app import FaceClient

FACE_URL_DEFAULT = os.getenv("FACE_URL", "http://localhost:8001")
FACE_MODE_DEFAULT = os.getenv("FACE_MODE", "remote")
FACE_MAX_SIDE_DEFAULT = int(os.getenv("FACE_MAX_SIDE", "960"))
FACE_JPEG_QUALITY_DEFAULT = int(os.getenv("FACE_JPEG_QUALITY", "75"))

st.set_page_config(page_title="BVS Face Live Demo", layout="wide")
st.title("Live Face Recognition Demo (SCRFD + ArcFace)")

face_url = st.text_input("face_service URL", FACE_URL_DEFAULT)
face_mode = st.selectbox("Mode", ["remote", "local"], index=0 if FACE_MODE_DEFAULT == "remote" else 1)

col1, col2 = st.columns([2, 1])
with col2:
    st.markdown("### Controls")
    fps = st.slider("Verify FPS (calls/sec)", 1, 10, 5)
    max_input_side = st.slider("Max input side (px)", 480, 1440, FACE_MAX_SIDE_DEFAULT, step=80)
    jpeg_quality = st.slider("JPEG quality", 50, 95, FACE_JPEG_QUALITY_DEFAULT)
    show_boxes = st.checkbox("Show boxes", True)
    if st.button("Reload embeddings"):
        try:
            r = FaceClient(mode=face_mode, base_url=face_url).reload_embeddings()
            st.success(r)
        except Exception as e:
            st.error(str(e))


class Transformer(VideoTransformerBase):
    def __init__(self):
        self.last = 0.0
        self.last_res = {"ok": True, "frDecision": "-", "faces": [], "bestMatch": {"id": None, "score": None}}
        self.interval = 0.2
        self.client = None
        self.client_config = None
        self.init_error = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        self.interval = 1.0 / max(int(fps), 1)

        desired_config = (face_mode, face_url, int(max_input_side), int(jpeg_quality))
        if desired_config != self.client_config:
            self.client = None
            self.client_config = desired_config
            self.init_error = None

        if self.client is None and self.init_error is None:
            try:
                self.client = FaceClient(
                    mode=face_mode,
                    base_url=face_url,
                    max_side=max_input_side,
                    jpeg_quality=jpeg_quality,
                )
            except Exception as e:
                self.init_error = str(e)

        if now - self.last >= self.interval:
            self.last = now
            if self.client is None:
                self.last_res = {"ok": False, "error": self.init_error or "Client init failed"}
            else:
                try:
                    self.last_res = self.client.verify_bgr(img)
                except Exception as e:
                    self.last_res = {"ok": False, "error": str(e)}

        res = self.last_res
        if res.get("ok"):
            decision = res.get("frDecision", "UNKNOWN")
            bm = res.get("bestMatch") or {}
            pid = bm.get("id")
            score = bm.get("score")
            total_ms = (res.get("timingsMs") or {}).get("total")
            txt = f"FR: {decision}"
            if pid is not None and score is not None:
                txt += f" ({pid}, {score:.3f})"
            if total_ms is not None:
                txt += f" [{float(total_ms):.0f}ms]"
            cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if show_boxes:
                for f in res.get("faces", []):
                    x1, y1, x2, y2 = [int(v) for v in f.get("bbox", [0, 0, 0, 0])]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    mid = f.get("matchId")
                    ms = f.get("matchScore")
                    if mid is not None and ms is not None:
                        cv2.putText(img, f"{mid} {ms:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            err = res.get("error", "FR ERROR")
            cv2.putText(img, "FR ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, str(err)[:60], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img


with col1:
    webrtc_streamer(
        key="bvs-face",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=Transformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.markdown("### Setup")
st.write("Add authorized images under `face_service/face_db/authorized/<person_id>/*.jpg`, then click Reload.")
