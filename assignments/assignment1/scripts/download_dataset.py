import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def count_images(root: Path):
    """Count image files in target folder."""
    return sum(1 for p in root.glob("*.png"))

def main():
    p = argparse.ArgumentParser(description="Download + materialize dataset to imagefolder.")
    p.add_argument("--repo-id", default="Norod78/cartoon-blip-captions",
                   help="Hugging Face dataset repo.")
    p.add_argument("--target", default="assignments/assignment1/data/cartoon-blip-captions/imagefolder",
                   help="Output folder with PNG + TXT captions.")
    p.add_argument("--min-files", type=int, default=500,
                   help="Skip if at least this many images already exist.")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all samples; otherwise cap the number of examples.")
    p.add_argument("--force", action="store_true",
                   help="Force re-materialization even if enough images exist.")
    args = p.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    existing = count_images(target)
    if existing >= args.min_files and not args.force:
        print(f"✅ Dataset already materialized: {existing} images at {target.resolve()}")
        return

    print(f"⬇️ Loading HF dataset: {args.repo_id} (split=train)")
    ds = load_dataset(args.repo_id, split="train")

    n_total = len(ds)
    n = n_total if args.limit == 0 else min(args.limit, n_total)
    print(f"🧪 Materializing {n} samples → {target.resolve()}")

    written = 0
    for i, ex in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        img = ex.get("image")
        if img is None:
            continue
        cap = (ex.get("text") or ex.get("caption") or "").strip()
        stem = f"{i:06d}"
        img_p = target / f"{stem}.png"
        txt_p = target / f"{stem}.txt"

        if img_p.exists() and txt_p.exists() and not args.force:
            written += 1
            continue

        img.convert("RGB").save(img_p)
        txt_p.write_text(cap, encoding="utf-8")
        written += 1
        if written % 200 == 0:
            print(f"  wrote {written}/{n} …")

    print(f"✅ Done. Wrote {written} images at {target.resolve()}")

if __name__ == "__main__":
    main()
