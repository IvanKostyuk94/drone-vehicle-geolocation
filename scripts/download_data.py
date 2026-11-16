"""Download and unpack the Songdo Vision and Songdo Traffic files.

Usage (from the repository root):
    python scripts/download_data.py                  # about 5.3 GB
    python scripts/download_data.py --dry-run        # only list the files
    python scripts/download_data.py --include-train  # adds train.zip, 12.9 GB
"""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOWNLOADS = DATA / "downloads"

# Concept record ids: the Zenodo API resolves them to the latest version.
VISION_CONCEPT = "13828407"
TRAFFIC_CONCEPT = "13828383"

# (concept id, file name in the record, folder to unpack into)
FILES = [
    (VISION_CONCEPT, "test.zip", DATA / "songdo_vision"),
    (VISION_CONCEPT, "README.md", DATA / "songdo_vision"),
    (TRAFFIC_CONCEPT, "master_frames.zip", DATA / "songdo_traffic"),
    (TRAFFIC_CONCEPT, "orthophotos.zip", DATA / "songdo_traffic"),
    (TRAFFIC_CONCEPT, "segmentations.zip", DATA / "songdo_traffic"),
    (TRAFFIC_CONCEPT, "README.md", DATA / "songdo_traffic"),
]
TRAIN = (VISION_CONCEPT, "train.zip", DATA / "songdo_vision")


def record_files(concept_id: str) -> tuple[str, dict]:
    """Latest version of a Zenodo record.

    Returns (version, {file name: file info}).
    """
    r = requests.get(
        f"https://zenodo.org/api/records/{concept_id}", timeout=60
    )
    r.raise_for_status()
    rec = r.json()
    return rec["metadata"].get("version", "?"), {
        f["key"]: f for f in rec["files"]
    }


def md5_of(path: Path) -> str:
    """MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, size: int, md5: str) -> None:
    """Download url to target (resumes partial files), check the MD5."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.exists()
        and target.stat().st_size == size
        and md5_of(target) == md5
    ):
        print(f"  already complete: {target.name}")
        return
    have = target.stat().st_size if target.exists() else 0
    if have > size:
        target.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        # server ignored the range: start over
        if have and r.status_code != 206:
            have = 0
        mode = "ab" if have else "wb"
        with open(target, mode) as f, tqdm(
            total=size,
            initial=have,
            unit="B",
            unit_scale=True,
            desc=target.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if md5_of(target) != md5:
        sys.exit(
            f"Checksum mismatch for {target.name}; delete it and run again."
        )


def main() -> None:
    """Download, verify and unpack all files."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list files and sizes, download nothing",
    )
    ap.add_argument(
        "--include-train",
        action="store_true",
        help="also download train.zip (12.9 GB)",
    )
    ap.add_argument(
        "--keep-zips",
        action="store_true",
        help="keep zip files after unpacking",
    )
    args = ap.parse_args()

    wanted = FILES + ([TRAIN] if args.include_train else [])
    records = {}
    for concept in {c for c, _, _ in wanted}:
        version, files = record_files(concept)
        records[concept] = files
        print(f"Zenodo concept {concept}: latest version {version}")

    total = sum(records[c][name]["size"] for c, name, _ in wanted)
    print(f"\n{len(wanted)} files, {total / 1e9:.2f} GB in total:")
    for c, name, dest in wanted:
        size_mb = records[c][name]["size"] / 1e6
        print(
            f"  {name:<20} {size_mb:>10.1f} MB  ->  {dest.relative_to(ROOT)}"
        )
    if args.dry_run:
        return
    free = shutil.disk_usage(DATA if DATA.exists() else ROOT).free
    if free < 2.2 * total:
        sys.exit(
            f"Need about {2.2 * total / 1e9:.1f} GB free (zip plus unpacked), "
            f"have {free / 1e9:.1f} GB."
        )

    for c, name, dest in wanted:
        info = records[c][name]
        md5 = info["checksum"].split(":", 1)[1]
        is_zip = name.endswith(".zip")
        target = DOWNLOADS / f"{c}_{name}" if is_zip else dest / name
        print(f"\n{name}")
        download(info["links"]["self"], target, info["size"], md5)
        if is_zip:
            print(f"  unpacking into {dest.relative_to(ROOT)}")
            with zipfile.ZipFile(target) as z:
                z.extractall(dest)
            if not args.keep_zips:
                target.unlink()

    print(
        "\nDone. Check: ls data/songdo_vision/test/images | wc -l "
        "(expect 1084)"
    )


if __name__ == "__main__":
    main()
