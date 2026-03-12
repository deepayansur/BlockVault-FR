# Local face recognition test (no server)
# Usage:
#   python .\local_fr_test.py --test-image .\path\to\image.jpg
#   python .\local_fr_test.py --rebuild-only

import argparse, os, sys, time
import cv2

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app import FaceClient


def run_camera(client: FaceClient, camera: int, fps: float) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    interval = 1.0 / max(fps, 0.1)
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
            print(last_res)

        # Draw high-contrast overlay so it is always visible
        decision = last_res.get("frDecision", "UNKNOWN")
        label = f"FR: {decision}"
        cv2.rectangle(frame, (6, 6), (360, 42), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Draw boxes if present
        for f in last_res.get("faces", []):
            x1,y1,x2,y2 = [int(v) for v in f.get("bbox", [0,0,0,0])]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            mid = f.get("matchId"); ms = f.get("matchScore")
            if mid is not None and ms is not None:
                cv2.putText(frame, f"{mid} {ms:.2f}", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Local FR Test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def prompt_menu() -> str:
    print("Choose mode:")
    print("1) Image file")
    print("2) Camera")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice. Enter 1 or 2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-only", action="store_true", help="Only rebuild embeddings.json and exit")
    args = ap.parse_args()

    client = FaceClient(mode="local")

    if args.rebuild_only:
        r = client.reload_embeddings()
        print(r)
        return

    choice = prompt_menu()

    if choice == "1":
        path = input("Path to image (jpg/png): ").strip().strip('"')
        if not path:
            print("No image path provided.")
            return
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        print(client.verify_bgr(img))
        return

    cam_in = input("Camera index (default 0): ").strip()
    fps_in = input("FPS (default 5): ").strip()
    camera = int(cam_in) if cam_in else 0
    fps = float(fps_in) if fps_in else 5.0
    run_camera(client, camera, fps)


if __name__ == "__main__":
    main()
