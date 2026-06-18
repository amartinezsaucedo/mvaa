import argparse

import pandas as pd

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    parser = argparse.ArgumentParser(description="Discriminative power analysis (NMI range and std)")
    parser.add_argument("--apps", nargs="+", default=["cargo", "jpetstore", "daytrader"],
                        metavar="APP", help="Systems to include (default: all three)")
    args = parser.parse_args()

    print(f"{'System':<12} {'NMI min':>9} {'NMI max':>9} {'NMI std':>9} {'NMI range':>11} "
          f"{'NMIf min':>9} {'NMIf max':>9} {'NMIf std':>9} {'NMIf range':>11} {'pen_swm std':>12}")
    print("-" * 100)

    rows = []
    for name in args.apps:
        df = pd.read_csv(f"results/RQ1/metrics_{name}.csv")
        nmi = df["NMI"].dropna()
        nmif = df["NMI_filtered"].dropna()
        pen = df["penalty_swm"].dropna()
        print(f"{name:<12} {nmi.min():>9.4f} {nmi.max():>9.4f} {nmi.std():>9.4f} "
              f"{(nmi.max()-nmi.min()):>11.4f} "
              f"{nmif.min():>9.4f} {nmif.max():>9.4f} {nmif.std():>9.4f} "
              f"{(nmif.max()-nmif.min()):>11.4f} {pen.std():>12.4f}")
        rows.append({"system": name, "nmi_min": nmi.min(), "nmi_max": nmi.max(),
                      "nmi_std": nmi.std(), "nmi_range": nmi.max() - nmi.min(),
                      "nmi_filtered_min": nmif.min(), "nmi_filtered_max": nmif.max(),
                      "nmi_filtered_std": nmif.std(), "nmi_filtered_range": nmif.max() - nmif.min(),
                      "pen_swm_std": pen.std()})

    pd.DataFrame(rows).to_csv("results/RQ1/discriminative_power.csv", index=False)
