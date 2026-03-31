"""Download and extract the Vimeo90K Triplet dataset for VFI experiments.

Dataset: http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip (~30GB)
Reference: T. Xue et al., "Video Enhancement with Task-Oriented Flow", IJCV 2019.

Usage:
    pixi run python anvil_exp01/data/download_vimeo90k.py
    pixi run python anvil_exp01/data/download_vimeo90k.py --output-dir ./data/vimeo_triplet
    pixi run python anvil_exp01/data/download_vimeo90k.py --skip-download
    pixi run python anvil_exp01/data/download_vimeo90k.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

FULL_DATASET_URL = "http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip"
TEST_ONLY_URL = "http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip"
EXPECTED_TRAIN_TRIPLETS = 51313
EXPECTED_TEST_TRIPLETS = 3782
EXPECTED_TOTAL_TRIPLETS = EXPECTED_TRAIN_TRIPLETS + EXPECTED_TEST_TRIPLETS

# Each triplet folder has im1.png, im2.png, im3.png
EXPECTED_IMAGES_PER_TRIPLET = 3


class DownloadProgressBar(tqdm):
    """tqdm wrapper for urllib reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1) -> None:
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file to avoid leaving partial downloads behind.
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    print(f"Downloading {url}")
    print(f"  -> {dest}")

    try:
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=dest.name,
        ) as pbar:
            urllib.request.urlretrieve(url, tmp_dest, reporthook=pbar.update_to)
    except (urllib.error.URLError, OSError) as exc:
        # Clean up partial download on failure.
        tmp_dest.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {exc}") from exc

    tmp_dest.rename(dest)
    print(f"Download complete: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract *zip_path* into *output_dir* with a progress bar."""
    print(f"Extracting {zip_path} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, output_dir)

    print("Extraction complete.")


def parse_triplet_list(list_file: Path) -> list[str]:
    """Parse a tri_trainlist.txt / tri_testlist.txt file.

    Each non-empty line is a triplet identifier like "00001/0001".
    Returns the list of identifiers.
    """
    if not list_file.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")

    triplets: list[str] = []
    text = list_file.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            triplets.append(stripped)
    return triplets


def verify_dataset(dataset_dir: Path, *, test_only: bool = False) -> bool:
    """Verify the extracted Vimeo90K triplet dataset.

    Checks:
      1. tri_trainlist.txt and tri_testlist.txt exist and have expected counts.
      2. Every listed triplet directory exists with im1.png, im2.png, im3.png.

    If *test_only* is True, only verify the test split.

    Returns True if verification passes, False otherwise.
    """
    sequences_dir = dataset_dir / "sequences"
    train_list_file = dataset_dir / "tri_trainlist.txt"
    test_list_file = dataset_dir / "tri_testlist.txt"

    ok = True

    # --- Check list files ---
    checks = [("test", test_list_file, EXPECTED_TEST_TRIPLETS)]
    if not test_only:
        checks.insert(0, ("train", train_list_file, EXPECTED_TRAIN_TRIPLETS))

    for name, list_file, expected_count in checks:
        if not list_file.exists():
            print(f"FAIL: {list_file} does not exist.")
            ok = False
            continue

        triplets = parse_triplet_list(list_file)
        actual = len(triplets)
        if actual != expected_count:
            print(f"FAIL: {name} split has {actual} triplets, expected {expected_count}.")
            ok = False
        else:
            print(f"OK:   {name} split has {actual} triplets.")

    if not ok:
        return False

    # --- Check that sequence directories and images exist ---
    all_triplets: list[str] = []
    if not test_only and train_list_file.exists():
        all_triplets.extend(parse_triplet_list(train_list_file))
    if test_list_file.exists():
        all_triplets.extend(parse_triplet_list(test_list_file))

    print(f"Verifying {len(all_triplets)} triplet directories ...")
    missing_dirs = 0
    missing_images = 0

    for triplet_id in tqdm(all_triplets, desc="Verifying", unit="triplet"):
        triplet_dir = sequences_dir / triplet_id
        if not triplet_dir.is_dir():
            missing_dirs += 1
            if missing_dirs <= 5:
                print(f"  Missing directory: {triplet_dir}")
            continue

        for img_name in ("im1.png", "im2.png", "im3.png"):
            img_path = triplet_dir / img_name
            if not img_path.is_file():
                missing_images += 1
                if missing_images <= 5:
                    print(f"  Missing image: {img_path}")

    if missing_dirs > 0:
        print(f"FAIL: {missing_dirs} triplet directories missing.")
        ok = False
    else:
        print(f"OK:   All {len(all_triplets)} triplet directories present.")

    if missing_images > 0:
        print(f"FAIL: {missing_images} images missing.")
        ok = False
    else:
        expected_total_images = len(all_triplets) * EXPECTED_IMAGES_PER_TRIPLET
        print(f"OK:   All {expected_total_images} images present.")

    return ok


def find_dataset_root(output_dir: Path) -> Path:
    """Resolve the actual dataset root after extraction.

    The zip file contains a top-level ``vimeo_triplet/`` directory, so after
    extracting to *output_dir* the real root is ``output_dir / vimeo_triplet``.
    However, if the user already points *output_dir* at the right place (i.e.
    it already contains ``sequences/``), use it directly.
    """
    if (output_dir / "sequences").is_dir():
        return output_dir
    nested = output_dir / "vimeo_triplet"
    if (nested / "sequences").is_dir():
        return nested
    return output_dir


def relocate_extracted_contents(output_dir: Path) -> Path:
    """After extraction, move contents from nested vimeo_triplet/ up if needed.

    The zip extracts as ``<output_dir>/vimeo_triplet/sequences/...``.
    If the user wants ``<output_dir>`` to *be* the dataset root, we relocate
    the inner contents up one level so the final layout is::

        <output_dir>/sequences/...
        <output_dir>/tri_trainlist.txt
        <output_dir>/tri_testlist.txt

    Returns the final dataset root.
    """
    nested = output_dir / "vimeo_triplet"
    if not nested.is_dir():
        # Nothing to relocate -- zip may have been structured differently
        # or output_dir was already the correct root.
        return output_dir

    # Already has sequences at top level -- nested is something else, leave it.
    if (output_dir / "sequences").is_dir():
        return output_dir

    print(f"Relocating extracted contents from {nested} -> {output_dir} ...")
    for child in nested.iterdir():
        target = output_dir / child.name
        if target.exists():
            # Safety: don't overwrite existing files/dirs in the output dir.
            print(f"  Skipping {child.name} (already exists at destination).")
            continue
        child.rename(target)

    # Remove the now-empty nested directory.
    if nested.exists() and not any(nested.iterdir()):
        nested.rmdir()

    return output_dir


def reorganize_test_only(output_dir: Path) -> Path:
    """Reorganize test-only zip extraction to match full dataset layout.

    The test-only zip extracts as::

        <output_dir>/vimeo_interp_test/
        ├── input/<seq>/<trip>/im1.png, im3.png
        ├── target/<seq>/<trip>/im1.png, im2.png, im3.png
        ├── tri_testlist.txt
        └── readme.txt

    We reorganize to::

        <output_dir>/
        ├── sequences/<seq>/<trip>/im1.png, im2.png, im3.png
        └── tri_testlist.txt

    The ``target/`` dir already has all 3 frames, so we rename it to
    ``sequences/`` and move ``tri_testlist.txt`` up.

    Returns the final dataset root.
    """
    nested = output_dir / "vimeo_interp_test"
    if not nested.is_dir():
        # Already reorganized or different structure
        return output_dir

    # Already has sequences at top level
    if (output_dir / "sequences").is_dir():
        return output_dir

    print(f"Reorganizing test-only data from {nested} ...")

    # target/ -> sequences/ (has all 3 frames per triplet)
    target_dir = nested / "target"
    sequences_dir = output_dir / "sequences"
    if target_dir.is_dir():
        target_dir.rename(sequences_dir)
        print(f"  {target_dir} -> {sequences_dir}")

    # Move tri_testlist.txt up
    test_list = nested / "tri_testlist.txt"
    if test_list.is_file():
        test_list.rename(output_dir / "tri_testlist.txt")
        print(f"  Moved tri_testlist.txt")

    # Clean up the nested directory (input/ is redundant, target/ is moved)
    if nested.exists():
        shutil.rmtree(nested)
        print(f"  Removed {nested}")

    return output_dir


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    default_output = project_root / "data" / "vimeo_triplet"

    parser = argparse.ArgumentParser(
        description="Download and extract the Vimeo90K Triplet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory to extract the dataset into (default: {default_output})",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Download only the test set (~1.7GB) instead of the full dataset (~30GB).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if the zip file already exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing dataset; do not download or extract.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    test_only: bool = args.test_only

    if test_only:
        dataset_url = TEST_ONLY_URL
        zip_name = "vimeo_interp_test.zip"
    else:
        dataset_url = FULL_DATASET_URL
        zip_name = "vimeo_triplet.zip"

    zip_path = output_dir.parent / zip_name

    # ---- Verify-only mode ----
    if args.verify_only:
        dataset_root = find_dataset_root(output_dir)
        print(f"Verifying dataset at {dataset_root} ...")
        if verify_dataset(dataset_root, test_only=test_only):
            print("\nDataset verification PASSED.")
        else:
            print("\nDataset verification FAILED.")
            sys.exit(1)
        return

    # ---- Download ----
    if args.skip_download and zip_path.exists():
        print(f"Skipping download, using existing {zip_path}")
    elif args.skip_download and not zip_path.exists():
        raise SystemExit(
            f"--skip-download specified but zip not found at {zip_path}"
        )
    else:
        if zip_path.exists():
            print(f"Zip file already exists at {zip_path}, re-downloading ...")
        download_file(dataset_url, zip_path)

    # ---- Extract ----
    extract_zip(zip_path, output_dir)

    # ---- Relocate / reorganize ----
    if test_only:
        dataset_root = reorganize_test_only(output_dir)
    else:
        dataset_root = relocate_extracted_contents(output_dir)

    # ---- Verify ----
    print()
    print(f"Verifying dataset at {dataset_root} ...")
    if verify_dataset(dataset_root, test_only=test_only):
        print("\nDataset verification PASSED.")
        print(f"Dataset root: {dataset_root}")
        print(f"You can safely delete the zip file to reclaim space:")
        print(f"  rm {zip_path}")
    else:
        print("\nDataset verification FAILED.")
        print("The download or extraction may be incomplete. Try again or inspect manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
