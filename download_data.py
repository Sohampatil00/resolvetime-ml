"""Download only the official UCI incident CSV when the user runs this file."""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SOURCE = "https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log"
URL = "https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/incident_event_log.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"Already exists: {args.output}. Use the existing file or pass --overwrite.")
    print("Downloading the official UCI incident dataset...", flush=True)
    with urlopen(Request(URL, headers={"User-Agent": "ResolveTime-Educational-Project/1.0"}), timeout=90) as response:
        archive = response.read(30_000_001)
    if len(archive) > 30_000_000:
        raise ValueError("Download exceeds the expected archive size.")
    with ZipFile(io.BytesIO(archive)) as bundle:
        candidates = [item for item in bundle.infolist()
                      if Path(item.filename).name == "incident_event_log.csv" and not item.is_dir()]
        if len(candidates) != 1 or candidates[0].file_size > 150_000_000:
            raise ValueError("The archive does not contain the expected single incident CSV.")
        data = bundle.read(candidates[0])
    header = data.splitlines()[0].decode("utf-8-sig")
    if not all(name in header for name in ("number", "opened_at", "resolved_at", "sys_mod_count")):
        raise ValueError("Unexpected dataset columns. No file was written.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    provenance = {"source": SOURCE, "download_url": URL,
                  "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                  "csv_sha256": hashlib.sha256(data).hexdigest(),
                  "archive_sha256": hashlib.sha256(archive).hexdigest(), "csv_bytes": len(data)}
    args.output.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {args.output}\nNext: python train.py", flush=True)


if __name__ == "__main__":
    main()
