import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2 as chi2_dist

os.chdir(Path(__file__).resolve().parents[3])

# Rankings per evaluator [A, B, C, D, E, F]
evaluators = {
    'E1': [4, 6, 3, 1, 2, 5],
    'E2': [4, 2, 3, 6, 5, 5],
    'E3': [2, 6, 1, 5, 4, 6],
    'E4': [6, 5, 4, 3, 2, 1],
    'E5': [3, 4, 2, 6, 6, 6],
    'E6': [3, 6, 1, 2, 5, 4],
    'E7': [5, 4, 1, 6, 2, 3],
    'E8': [4, 5, 2, 3, 6, 1],
    'E9': [4, 5, 3, 2, 6, 1],
    'E10': [4, 3, 1, 6, 5, 2],
}

# Ranking per NMI: A=1, B=2, C=3, D=4, E=5, F=6
nmi_ranking = [1, 2, 3, 4, 5, 6]

print("=== SPEARMAN vs NMI RANKING ===")
spearman_rows = []
for name, ranks in evaluators.items():
    rho, p = stats.spearmanr(nmi_ranking, ranks)
    print(f"{name}: rho={rho:.3f}, p={p:.3f} {'**' if p < 0.05 else ''}")
    spearman_rows.append({"evaluator": name, "rho": rho, "p_value": p, "significant": p < 0.05})

# Mean and std of rho
rhos = [row["rho"] for row in spearman_rows]
print(f"\nMean rho: {np.mean(rhos):.3f}")
print(f"Std rho:  {np.std(rhos):.3f}")
print(f"Positive correlations: {sum(1 for r in rhos if r > 0)}/{len(rhos)}")

pd.DataFrame(spearman_rows).to_csv("results/RQ3/spearman_vs_nmi.csv", index=False)

print("\n=== CONSENSUS RANKING (mean rank per decomposition) ===")
decomps = ['A', 'B', 'C', 'D', 'E', 'F']
all_ranks = np.array(list(evaluators.values()))
mean_ranks = all_ranks.mean(axis=0)
std_ranks = all_ranks.std(axis=0)
for i, d in enumerate(decomps):
    print(f"{d}: mean={mean_ranks[i]:.2f}, std={std_ranks[i]:.2f}")

consensus_order = np.argsort(mean_ranks)
print(f"\nConsensus order (best to worst): {[decomps[i] for i in consensus_order]}")

pd.DataFrame({
    "decomposition": decomps,
    "mean_rank": mean_ranks,
    "std_rank": std_ranks,
    "consensus_position": stats.rankdata(mean_ranks),
}).to_csv("results/RQ3/consensus_ranking.csv", index=False)

# Spearman consensus vs NMI
rho_consensus, p_consensus = stats.spearmanr(nmi_ranking, mean_ranks)
print(f"\nConsensus vs NMI: rho={rho_consensus:.3f}, p={p_consensus:.3f} {'**' if p_consensus < 0.05 else ''}")

print("\n=== KRIPPENDORFF ALPHA (ordinal) ===")
# Simple approximation using pairwise agreement using Kendall's W as proxy for inter-rater agreement

n_raters = len(evaluators)
n_items = 6
rankings_matrix = np.array(list(evaluators.values()))

# Kendall's W
rank_sums = rankings_matrix.sum(axis=0)
mean_rank_sum = rank_sums.mean()
S = sum((r - mean_rank_sum)**2 for r in rank_sums)
W = 12 * S / (n_raters**2 * (n_items**3 - n_items))
print(f"Kendall's W: {W:.3f}")
chi2 = n_raters * (n_items - 1) * W

p_W = 1 - chi2_dist.cdf(chi2, n_items - 1)
print(f"Chi2={chi2:.3f}, p={p_W:.4f} {'**' if p_W < 0.05 else ''}")

pd.DataFrame([{
    "kendalls_w": W,
    "chi2": chi2,
    "chi2_p": p_W,
    "consensus_vs_nmi_rho": rho_consensus,
    "consensus_vs_nmi_p": p_consensus,
}]).to_csv("results/RQ3/inter_rater_agreement.csv", index=False)

print("\n=== DATA OWNERSHIP (ALPHA vs BETA) ===")
responses = ['ALPHA', 'BETA', 'ALPHA', 'ALPHA', 'No meaningful difference', 'BETA', 'ALPHA', 'ALPHA', 'ALPHA', 'ALPHA']
counts = Counter(responses)
for k, v in counts.items():
    print(f"{k}: {v}/{len(responses)} ({100*v/len(responses):.0f}%)")

pd.DataFrame([
    {"preference": k, "count": v, "percentage": 100 * v / len(responses)}
    for k, v in counts.items()
]).to_csv("results/RQ3/data_ownership_preference.csv", index=False)
