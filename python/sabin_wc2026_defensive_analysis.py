"""
FIFA World Cup 2026: does defensive activity relate to goals conceded?

HIT140 Objective 1 analytic task.

Analytic question
------------------
Do teams with higher defensive activity concede fewer goals per 90 minutes
in the FIFA World Cup 2026?

Input: sabin_wc2026_defensive_data.csv (merged FBref + FIFA team-level stats, one
row per team, columns cited to their source in FBref_Source / FIFA_Source).

Population and sample
----------------------
Population = all 48 teams that competed at the FIFA World Cup 2026.
We do not analyse the full population directly. Instead we draw a simple
random sample (without replacement, fixed seed for reproducibility) of teams
from that population and base every descriptive/inferential statistic below
on that sample, then generalise back to the population of WC2026 teams via
the confidence interval and t-test.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05
RANDOM_SEED = 42
SAMPLE_SIZE = 30  # simple random sample drawn from the 48-team population

# ---------------------------------------------------------------
# 1. DATA WRANGLING: load and inspect the new verified dataset
# ---------------------------------------------------------------
df = pd.read_csv("sabin_wc2026_defensive_data.csv")

print(f"Rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\nMissing values per column:")
print(df.isnull().sum().to_string())

dup_rows = df.duplicated().sum()
dup_teams = df["Team"].duplicated().sum()
print(f"\nFully duplicated rows: {dup_rows}")
print(f"Duplicated team names: {dup_teams}")

# ---------------------------------------------------------------
# 2. DATA PREPARATION: per-90 rates using FBref's "90s" exposure column
# ---------------------------------------------------------------
# "90s" is the number of full 90-minute equivalents each team played, so it
# already accounts for extra time in knockout matches - a more precise
# denominator than raw games played.
df["TacklesWon_per90"] = df["Tackles_Won"] / df["90s"]
df["Interceptions_per90"] = df["Interceptions"] / df["90s"]
df["ForcedTurnovers_per90"] = df["Forced_Turnovers"] / df["90s"]
df["DefensivePressures_per90"] = df["Defensive_Pressures_Applied"] / df["90s"]
df["GoalsConceded_per90"] = df["Goals_Conceded"] / df["90s"]

ACTIVITY_VARS = [
    "TacklesWon_per90",
    "Interceptions_per90",
    "ForcedTurnovers_per90",
    "DefensivePressures_per90",
]

print("\nDescriptive statistics for the full population (all 48 teams), for reference:")
print(df[ACTIVITY_VARS + ["GoalsConceded_per90"]].describe().round(3).to_string())

# ---------------------------------------------------------------
# 3. SAMPLING: simple random sample of teams from the population
# ---------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
sample_index = rng.choice(df.index, size=SAMPLE_SIZE, replace=False)
sample_df = df.loc[sample_index].reset_index(drop=True)

print(f"\nPopulation size (all WC2026 teams): {len(df)}")
print(f"Sample size (simple random sample, seed={RANDOM_SEED}): {len(sample_df)}")
print("Sampled teams:", ", ".join(sorted(sample_df["Team"])))

# All remaining analysis is performed on sample_df, not the full population.
df = sample_df

# ---------------------------------------------------------------
# STANDARDISE the main defensive activity variables (z-scores)
# ---------------------------------------------------------------
z_cols = []
for col in ACTIVITY_VARS:
    z = f"{col}_z"
    df[z] = (df[col] - df[col].mean()) / df[col].std(ddof=1)
    z_cols.append(z)

df["Defensive_Score"] = df[z_cols].mean(axis=1)

median_score = df["Defensive_Score"].median()
df["Activity_Group"] = np.where(df["Defensive_Score"] >= median_score, "High", "Low")

print("\nSampled teams by defensive activity score:")
print(
    df[["Team", "90s"] + ACTIVITY_VARS + ["Defensive_Score", "Activity_Group", "GoalsConceded_per90"]]
    .sort_values("Defensive_Score", ascending=False)
    .round(3)
    .to_string(index=False)
)

# ---------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------
desc = df.groupby("Activity_Group")["GoalsConceded_per90"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
print("\nGoals conceded per 90 minutes by group (sample):")
print(desc.round(3).to_string())

# ---------------------------------------------------------------
# 5. CONFIDENCE INTERVAL (95%)
# ---------------------------------------------------------------
ci_results = {}
for group in ["High", "Low"]:
    v = df.loc[df["Activity_Group"] == group, "GoalsConceded_per90"]
    lo, hi = stats.t.interval(0.95, df=len(v) - 1, loc=v.mean(), scale=stats.sem(v))
    ci_results[group] = {"mean": v.mean(), "lo": lo, "hi": hi}
    print(f"\n{group} activity (n = {len(v)})")
    print(f"  Sample mean goals conceded per 90: {v.mean():.3f}")
    print(f"  95% CI for the population mean: ({lo:.3f}, {hi:.3f})")

# ---------------------------------------------------------------
# ASSUMPTION CHECKS
# ---------------------------------------------------------------
high = df.loc[df["Activity_Group"] == "High", "GoalsConceded_per90"]
low = df.loc[df["Activity_Group"] == "Low", "GoalsConceded_per90"]

print("\nAssumption checks:")
for label, v in [("High", high), ("Low", low)]:
    w, p = stats.shapiro(v)
    print(f"  Shapiro-Wilk {label}: W = {w:.3f}, p = {p:.4f}")
lev_f, lev_p = stats.levene(high, low)
print(f"  Levene equal variance: F = {lev_f:.3f}, p = {lev_p:.4f}")

# ---------------------------------------------------------------
# 6. TWO-SAMPLE INDEPENDENT T-TEST
# ---------------------------------------------------------------
t_stat, p_val = stats.ttest_ind(high, low, equal_var=False)

print("\nWelch's two-sample t-test (High vs Low defensive activity):")
print(f"  t = {t_stat:.4f}")
print(f"  p = {p_val:.4f}")
print(f"  Mean difference = {high.mean() - low.mean():+.3f} goals conceded per 90")
print("  Reject the null." if p_val < ALPHA else "  Do not reject the null.")

# ---------------------------------------------------------------
# 7. VISUALISATION
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column="GoalsConceded_per90", by="Activity_Group", ax=ax, grid=False)
ax.set_title("Goals conceded per 90 by defensive activity group (sample)")
ax.set_xlabel("Defensive activity group")
ax.set_ylabel("Goals conceded per 90 minutes")
fig.suptitle("")
fig.tight_layout()
fig.savefig("sabin_wc2026_boxplot.png", dpi=150)
plt.close(fig)

# Bar chart of the group means with 95% CI error bars, straight from the
# confidence-interval results printed above.
fig, ax = plt.subplots(figsize=(6, 5))
groups = ["High", "Low"]
means = [ci_results[g]["mean"] for g in groups]
lower_err = [ci_results[g]["mean"] - ci_results[g]["lo"] for g in groups]
upper_err = [ci_results[g]["hi"] - ci_results[g]["mean"] for g in groups]
bars = ax.bar(groups, means, yerr=[lower_err, upper_err], capsize=8,
              color=["#d62728", "#1f77b4"], alpha=0.85)
for bar, m in zip(bars, means):
    ax.annotate(f"{m:.2f}", (bar.get_x() + bar.get_width() / 2, m),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
ax.set_title("Mean goals conceded per 90 by activity group\n(error bars = 95% CI, sample n=30)")
ax.set_xlabel("Defensive activity group")
ax.set_ylabel("Goals conceded per 90 minutes")
fig.tight_layout()
fig.savefig("sabin_wc2026_ci_barplot.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# 8. PLAIN-ENGLISH INTERPRETATION
# ---------------------------------------------------------------
direction = "fewer" if high.mean() < low.mean() else "more"
significance = "a statistically significant" if p_val < ALPHA else "no statistically significant"
print("\nInterpretation:")
print(
    f"In this random sample of {len(df)} of the 48 World Cup 2026 teams, teams classified as "
    f"'High' defensive activity (based on tackles won, interceptions, forced turnovers and "
    f"defensive pressures, all per 90 minutes) conceded {direction} goals per 90 minutes on "
    f"average ({high.mean():.2f}) than 'Low' activity teams ({low.mean():.2f}). "
    f"A Welch two-sample t-test found {significance} difference between the two groups "
    f"(t = {t_stat:.2f}, p = {p_val:.4f}, 95% CI reported above for each group's population mean). "
    "Defensive_Score measures the volume of defensive actions rather than their success rate, "
    "so this result speaks to how often a team defends, not necessarily how well."
)

# ---------------------------------------------------------------
# SAVE PROCESSED (SAMPLE) DATASET
# ---------------------------------------------------------------
df.to_csv("sabin_wc2026_defensive_processed_sample.csv", index=False)
print("\nSaved: sabin_wc2026_defensive_processed_sample.csv, sabin_wc2026_boxplot.png, sabin_wc2026_ci_barplot.png")
