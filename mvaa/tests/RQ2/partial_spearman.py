import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
from numpy.linalg import lstsq

from mvaa.tests.RQ1.bc_penalty_analysis import STRUCTURAL_METRICS


def partial_spearman_df(df: pd.DataFrame, target: str, control: str = "SERVICES") -> pd.DataFrame:
    df = df.dropna(subset=[target, control]).copy()
    r_target  = rankdata(df[target].values)
    r_control = rankdata(df[control].values)
    X = np.column_stack([r_control, np.ones(len(r_control))])
    res_target = r_target - X @ lstsq(X, r_target, rcond=None)[0]

    rows = []
    for metric in STRUCTURAL_METRICS:
        if metric == control:
            continue
        col = df[metric].astype(float)
        if col.nunique() <= 1:
            rows.append({"metric": metric, "rho_partial": float("nan"), "p_value": float("nan")})
            continue
        r_metric   = rankdata(col.values)
        res_metric = r_metric - X @ lstsq(X, r_metric, rcond=None)[0]
        rho, p     = spearmanr(res_target, res_metric)
        rows.append({"metric": metric, "rho_partial": rho, "p_value": p})
    return pd.DataFrame(rows)


def print_comparative_partial(projects: list[dict]):
    partials = {}
    for proj in projects:
        pdf = partial_spearman_df(proj["df"], "NMI", control="SERVICES")
        partials[proj["name"]] = pdf.set_index("metric")
        pdf.to_csv(f"results/RQ2/partial_spearman_{proj['name']}.csv", index=False)

    names = [p["name"] for p in projects]

    col_w = 14
    header = f"{'Metric':<10}" + "".join(f"{n:>{col_w}}" for n in names) + f"  {'pattern'}"
    print(f"\nPartial Spearman NMI (controlling for SERVICES):")
    print(header)
    print("-" * (10 + col_w * len(names) + 10))

    pattern_rows = []
    for metric in STRUCTURAL_METRICS:
        if metric == "SERVICES":
            continue

        row_str = f"{metric:<10}"
        rhos = []
        ps   = []
        row_data = {"metric": metric}

        for name in names:
            pdf = partials[name]
            if metric not in pdf.index or pd.isna(pdf.loc[metric, "rho_partial"]):
                row_str += f"{'nan':>{col_w}}"
                rhos.append(float("nan"))
                ps.append(float("nan"))
                row_data[f"rho_{name}"] = float("nan")
            else:
                rho = pdf.loc[metric, "rho_partial"]
                p   = pdf.loc[metric, "p_value"]
                sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
                row_str += f"{rho:>8.3f}{sig:<4}".rjust(col_w)
                rhos.append(rho)
                ps.append(p)
                row_data[f"rho_{name}"] = rho

        valid_rhos = [r for r in rhos if not pd.isna(r)]
        if len(valid_rhos) < 2:
            pattern = "—"
        elif all(r > 0 for r in valid_rhos):
            pattern = "✓ +"
        elif all(r < 0 for r in valid_rhos):
            pattern = "✓ -"
        elif all(abs(r) < 0.10 for r in valid_rhos):
            pattern = "✓ ~0"
        else:
            pattern = "✗"

        row_data["pattern"] = pattern
        pattern_rows.append(row_data)
        print(row_str + f"  {pattern}")

    pd.DataFrame(pattern_rows).to_csv("results/RQ2/partial_spearman_pattern.csv", index=False)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    import argparse
    parser = argparse.ArgumentParser(description="Partial Spearman correlation (controlling for #services)")
    parser.add_argument("--apps", nargs="+", default=["cargo", "jpetstore", "daytrader"],
                        metavar="APP", help="Systems to include (default: all three)")
    args = parser.parse_args()

    print_comparative_partial([
        {"name": p, "df": pd.read_csv(f"results/RQ1/metrics_{p}.csv")}
        for p in args.apps
    ])

