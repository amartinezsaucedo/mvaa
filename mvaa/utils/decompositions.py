from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Any


SERVICE_HEADER_RE      = re.compile(r"^\s*Service\s+(\d+)\s*$", re.IGNORECASE)
FILENAME_RE = re.compile(
    r"^.+?_(?P<run_id>\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{6})_K(?P<k>\d+)_R(?P<r>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
RESOLUTION_ONLY_RE = re.compile(r"_R(?P<r>\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_membership_from_text(text: str, *, on_duplicate: str = "error") -> Dict[str, int]:
    membership: Dict[str, int] = {}
    current_service: int | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SERVICE_HEADER_RE.match(line)
        if m:
            current_service = int(m.group(1))
            continue
        if current_service is None:
            continue

        key = f"class:{line}"
        if key in membership and membership[key] != current_service:
            if on_duplicate == "error":
                raise ValueError(
                    f"Class assigned to multiple services: {key} "
                    f"(was in {membership[key]}, now in {current_service})"
                )
            elif on_duplicate == "keep_first":
                continue
            elif on_duplicate == "overwrite":
                membership[key] = current_service
            else:
                raise ValueError(f"invalid on_duplicate value: {on_duplicate}")
        else:
            membership[key] = current_service

    return membership


def parse_filename(path: Path) -> Dict[str, str]:
    m = FILENAME_RE.search(path.stem)
    if m:
        run_id = m.group("run_id")
        k      = m.group("k")
        r      = m.group("r")
        dec_id = run_id
        return {"dec_id": dec_id, "k": k, "resolution": r, "run_id": run_id}

    raise ValueError(f"Could not extract metadata from: {path.name}")

def build_dict_from_directory(
        directory: str | Path,
        *,
        glob_pattern: str = "*_K*_R*",
        encoding: str = "utf-8",
        on_duplicate: str = "error",
        deduplicate: bool = True,
) -> Dict[str, Dict[str, Any]]:
    directory = Path(directory)
    out:       Dict[str, Dict[str, Any]] = {}
    seen_sigs: set = set()
    skipped  = 0

    for path in sorted(directory.glob(glob_pattern)):
        if not path.is_file():
            continue

        try:
            meta = parse_filename(path)
        except ValueError as e:
            print(f"[WARN] Skipping {path.name}: {e}")
            continue

        text       = path.read_text(encoding=encoding, errors="replace")
        membership = parse_membership_from_text(text, on_duplicate=on_duplicate)

        if not membership:
            print(f"[WARN] Empty membership in {path.name}, skipping.")
            continue

        if deduplicate:
            sig = tuple(sorted(membership.items()))
            if sig in seen_sigs:
                skipped += 1
                continue
            seen_sigs.add(sig)

        dec_id = meta["dec_id"]

        if dec_id in out:
            dec_id = f"{dec_id}_{path.stem[-6:]}"

        out[dec_id] = {
            "membership": membership,
            "k":          meta["k"],
            "resolution": meta["resolution"],
            "run_id":     meta["run_id"],
            "source":     path.name,
        }

    return out


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])

    parser = argparse.ArgumentParser(description="Build decompositions pickle from mid-results services")
    parser.add_argument("--app", default="daytrader",
                        choices=["cargo", "jpetstore", "daytrader"],
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    services_dir = f"monoliths/{project}/services/{project}"
    output_pkl   = f"monoliths/{project}/decompositions_{project}.pkl"

    result = build_dict_from_directory(f"monoliths/{project}/mid_results/services")

    print(result)

    with open(output_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved to {output_pkl}")
