import argparse
import glob

import pandas as pd

from mvaa.utils.decompositions import build_dict_from_directory

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    parser = argparse.ArgumentParser(description="Decomposition evaluation summary")
    parser.add_argument("--apps", nargs="+", default=["cargo", "jpetstore", "daytrader"],
                        metavar="APP", help="Systems to include (default: all three)")
    args = parser.parse_args()

    print(f"{'System':<12} {'Classes':>8} {'BCs':>5} {'Decomp':>10} {'Duplicates':>12}")
    print("-" * 55)

    rows = []
    for project in args.apps:
        decompositions = build_dict_from_directory(f"monoliths/{project}/mid_results/services")
        all_classes = set()
        for payload in decompositions.values():
            all_classes.update(payload["membership"].keys())

        csv_files = glob.glob(f"monoliths/{project}/mid_results/metrics/{project}_K*.csv")
        df = pd.concat(
            [pd.read_csv(f, names=['RESOLUTION', 'CHM', 'CHD', 'IFN', 'IRN', 'OPN',
                                   'SMQ', 'SCOH', 'SCOP', 'CMQ', 'CCOH', 'CCOP', 'SERVICES'])
             for f in csv_files],
            ignore_index=True,
        )
        total_csv = len(df)
        unique    = len(decompositions)
        dupes     = total_csv - unique

        print(f"{project:<12} {len(all_classes):>8} {'4':>5} {unique:>10} {dupes:>12}")
        rows.append({"system": project, "classes": len(all_classes), "bcs": 4,
                      "decompositions": unique, "duplicates": dupes})

    pd.DataFrame(rows).to_csv("results/RQ1/decomposition_space.csv", index=False)
