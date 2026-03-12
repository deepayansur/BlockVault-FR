import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="SurvFace root (adapt if needed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Generic walker: root/person_id/*.jpg
    lines = []
    for pid in sorted(os.listdir(args.root)):
        pdir = os.path.join(args.root, pid)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith((".jpg",".jpeg",".png")):
                lines.append(f"{os.path.join(pid, fn)} {pid}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {len(lines)} lines to {args.out} (you may need to adapt for your SurvFace distribution).")

if __name__ == "__main__":
    main()
