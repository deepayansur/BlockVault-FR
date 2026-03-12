import os
import sys
import argparse
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app import FaceClient


def draw_overlay(frame, result):
    if not result.get("ok"):
        cv2.putText(frame, "FR ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    decision = result.get("frDecision", "UNKNOWN")
    bm = result.get("bestMatch") or {}
    pid = bm.get("id")
    score = bm.get("score")
    total_ms = (result.get("timingsMs") or {}).get("total")

    txt = f"FR: {decision}"
    if pid is not None and score is not None:
        txt += f" ({pid}, {score:.3f})"
    if total_ms is not None:
        txt += f" [{float(total_ms):.0f}ms]"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    for f in result.get("faces", []):
        x1, y1, x2, y2 = [int(v) for v in f.get("bbox", [0, 0, 0, 0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        mid = f.get("matchId")
        ms = f.get("matchScore")
        if mid is not None and ms is not None:
            cv2.putText(frame, f"{mid} {ms:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-url", default="http://localhost:8001")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--mode", choices=["remote", "local"], default="remote")
    ap.add_argument("--max-side", type=int, default=int(os.getenv("FACE_MAX_SIDE", "960")))
    ap.add_argument("--jpeg-quality", type=int, default=int(os.getenv("FACE_JPEG_QUALITY", "75")))
    args = ap.parse_args()

    client = FaceClient(
        mode=args.mode,
        base_url=args.face_url,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    interval = 1.0 / max(args.fps, 0.1)
    last = 0.0
    last_res = {"ok": True, "frDecision": "-", "faces": []}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if now - last >= interval:
            last = now
            try:
                last_res = client.verify_bgr(frame)
            except Exception as e:
                last_res = {"ok": False, "error": str(e)}

        frame = draw_overlay(frame, last_res)
        cv2.imshow("BVS Face Webcam Test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
