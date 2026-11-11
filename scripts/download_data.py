"""Download and verify the NASA C-MAPSS turbofan data into data/raw/.

Safe to re-run: if every expected file is already present and passes the
sanity checks, nothing is downloaded.

Usage (from the repo root):  python scripts/download_data.py
"""
import io
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Tried in order. Both are NASA-published links to the same C-MAPSS data.
URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP_PATH = RAW_DIR / "CMAPSSData.zip"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
EXPECTED_FILES = (
    [f"train_{s}.txt" for s in SUBSETS]
    + [f"test_{s}.txt" for s in SUBSETS]
    + [f"RUL_{s}.txt" for s in SUBSETS]
)
# Engine counts from the dataset documentation: (train, test)
EXPECTED_ENGINES = {
    "FD001": (100, 100),
    "FD002": (260, 259),
    "FD003": (100, 100),
    "FD004": (249, 248),
}
N_COLS = 26  # unit, cycle, 3 operational settings, 21 sensors


def ssl_context() -> ssl.SSLContext:
    """Verified TLS using certifi's CA bundle (fixes python.org Python on macOS)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def all_present() -> bool:
    return all((RAW_DIR / f).exists() for f in EXPECTED_FILES)


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        print(f"Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "cmapss-rul-project"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=ssl_context()) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  failed: {e}")
            continue
        ZIP_PATH.write_bytes(data)
        print(f"Saved {ZIP_PATH.name} ({len(data) / 1e6:.1f} MB)")
        return
    raise SystemExit("All download URLs failed. See the error messages above.")


def _extract_zip(zf: zipfile.ZipFile) -> None:
    for member in zf.namelist():
        name = Path(member).name  # flatten any folder structure
        if not name:
            continue
        data = zf.read(member)
        if name.lower().endswith(".zip"):  # nested zip: unpack it too
            _extract_zip(zipfile.ZipFile(io.BytesIO(data)))
        else:
            (RAW_DIR / name).write_bytes(data)


def extract() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        _extract_zip(zf)
    print(f"Extracted into {RAW_DIR}")


def verify() -> bool:
    ok = True
    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"MISSING files: {missing}")
        return False
    print(f"{'subset':<7}{'train rows':>11}{'train eng':>10}{'test rows':>10}{'test eng':>9}{'RUL rows':>9}")
    for s in SUBSETS:
        tr = [l.split() for l in (RAW_DIR / f"train_{s}.txt").read_text().splitlines() if l.strip()]
        te = [l.split() for l in (RAW_DIR / f"test_{s}.txt").read_text().splitlines() if l.strip()]
        rul = [l for l in (RAW_DIR / f"RUL_{s}.txt").read_text().splitlines() if l.strip()]
        tr_eng, te_eng = len({r[0] for r in tr}), len({r[0] for r in te})
        print(f"{s:<7}{len(tr):>11}{tr_eng:>10}{len(te):>10}{te_eng:>9}{len(rul):>9}")
        exp_tr, exp_te = EXPECTED_ENGINES[s]
        if any(len(r) != N_COLS for r in tr + te):
            print(f"  ERROR: {s} has rows without {N_COLS} columns")
            ok = False
        if (tr_eng, te_eng) != (exp_tr, exp_te):
            print(f"  ERROR: {s} engine counts expected {(exp_tr, exp_te)}")
            ok = False
        if len(rul) != te_eng:
            print(f"  ERROR: {s} RUL file rows ({len(rul)}) != test engines ({te_eng})")
            ok = False
    return ok


def main() -> int:
    if not all_present():
        download()
        extract()
    else:
        print("All files already in data/raw/, skipping download.")
    ok = verify()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
