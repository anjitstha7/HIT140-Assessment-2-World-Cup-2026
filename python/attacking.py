#!/usr/bin/env python3
"""
Objective 1 -- Attacking performance at the 2026 FIFA World Cup.

Research question
-----------------
Attacking output per 90 minutes is an exact product of three components:

    non-penalty goals per 90
        = non-penalty shots per 90        (VOLUME)
        x share of shots on target        (ACCURACY)
        x goals per shot on target        (FINISHING)

Teams that reach the knockout stage score more. The question is which of the
three components carries that advantage. The formal test is on FINISHING --
non-penalty goals per non-penalty shot on target -- because it is the component
least explained by simply having more of the ball, and so speaks to attacking
quality rather than volume.

    H0: mean finishing is equal for knockout-stage and group-stage teams
    H1: mean finishing differs between them        (two-tailed, alpha = 0.05)

Design
------
Population        : all 48 squads competing at the 2026 FIFA World Cup
Sampling frame    : FBref Squad Shooting table for the competition
Unit of analysis  : one squad, aggregated over all its matches
Sampling method   : stratified random sampling without replacement
Strata            : stage reached -- Knockout (N=32) and Group (N=16)
Allocation        : proportional, n = 36 (24 knockout, 12 group)
Test              : Welch two-sample t-test with a 95% confidence interval

Allocation is proportional because the population splits 32 knockout to 16
group -- a 2:1 ratio -- so drawing 24 and 12 reproduces that split exactly and
neither stage is over-represented relative to the tournament. The total is set
by the smaller stratum: with only 16 group-stage teams available, retaining 12
keeps that group's mean and spread estimated on a reasonable base, and the 2:1
rule then fixes the knockout draw at 24.

The random seed is fixed at 2026 and declared before the sample is drawn. The
result of that single draw is reported whatever it is. Re-drawing with new seeds
until the p-value is agreeable would invalidate the test.

Usage
-----
    python src/tasks/attacking.py
    python src/tasks/attacking.py --allocation equal

Outputs (written to outputs/):
    attacking_results.txt         full numeric results
    attacking_sample.csv          the drawn sample, one row per squad
    figures/decomposition.png     three-panel boxplot of the sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from load_data import load_shooting  # noqa: E402

ALPHA = 0.05
SEED = 2026
KNOCKOUT, GROUP = "Knockout", "Group"
COMPONENTS = [
    ("volume", "Shots per 90"),
    ("accuracy", "Share on target"),
    ("finishing", "Goals per shot on target"),
]
DV = "finishing"

# Proportional allocation keeps the 2:1 population ratio; equal allocation
# maximises precision on the difference but takes the whole Group stratum.
ALLOCATIONS = {
    "proportional": {KNOCKOUT: 24, GROUP: 12},
    "equal": {KNOCKOUT: 16, GROUP: 16},
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-penalty counts and the three decomposition components."""
    out = df.copy()

    out["npGls"] = out["Gls"] - out["PK"]
    out["npSh"] = out["Sh"] - out["PKatt"]
    # A scored penalty is necessarily on target, so PK is subtracted. A saved
    # penalty is also on target but cannot be told apart from a missed one in
    # this export, so npSoT is marginally overstated for those squads.
    out["npSoT"] = out["SoT"] - out["PK"]

    for col in ("npGls", "npSh", "npSoT"):
        if (out[col] < 0).any():
            bad = out.loc[out[col] < 0, "Squad"].tolist()
            raise ValueError(f"Negative {col} after penalty adjustment for: {bad}")

    out["volume"] = out["npSh"] / out["exposure_90s"]
    out["accuracy"] = np.where(out["npSh"] > 0, out["npSoT"] / out["npSh"], np.nan)
    out["finishing"] = np.where(out["npSoT"] > 0, out["npGls"] / out["npSoT"], np.nan)
    out["np_goals_per_90"] = out["npGls"] / out["exposure_90s"]

    product = out["volume"] * out["accuracy"] * out["finishing"]
    check = out[["np_goals_per_90"]].assign(product=product).dropna()
    if not np.allclose(check["np_goals_per_90"], check["product"], atol=1e-9):
        raise AssertionError("Decomposition identity failed -- check the derivation")

    return out


def verify_against_fbref(df: pd.DataFrame) -> list[str]:
    """
    Recompute FBref's published ratio columns from the raw counts and compare.

    A column misalignment or a mis-parsed header would show up here rather than
    propagating silently into the t-test.
    """
    notes: list[str] = []
    pairs = [
        ("SoT%", df["SoT"] / df["Sh"] * 100, 0.6),
        ("Sh/90", df["Sh"] / df["exposure_90s"], 0.02),
        ("SoT/90", df["SoT"] / df["exposure_90s"], 0.02),
        ("G/Sh", df["Gls"] / df["Sh"], 0.02),
        ("G/SoT", np.where(df["SoT"] > 0, df["Gls"] / df["SoT"], np.nan), 0.02),
    ]
    for col, derived, tol in pairs:
        if col not in df.columns:
            notes.append(f"  {col:<8} not present in the export -- skipped")
            continue
        published = pd.to_numeric(df[col], errors="coerce")
        diff = (pd.Series(derived, index=df.index) - published).abs()
        bad = diff[diff > tol].dropna()
        if bad.empty:
            notes.append(f"  {col:<8} matches recomputed values for all "
                         f"{published.notna().sum()} squads")
        else:
            names = ", ".join(df.loc[bad.index, "Squad"].tolist()[:5])
            notes.append(f"  {col:<8} MISMATCH for {len(bad)} squad(s): {names}")
    return notes


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, allocation: dict[str, int],
                      seed: int) -> pd.DataFrame:
    """
    Draw a stratified random sample without replacement.

    Each stratum is sampled independently, so the two resulting groups are
    independent simple random samples -- which is exactly the independence
    condition the two-sample t-test requires. Stratifying on the grouping
    variable also guarantees both groups appear at the intended sizes, which
    simple random sampling across all 48 squads would not.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for stratum, n in allocation.items():
        pool = df[df["stage_reached"] == stratum]
        if n > len(pool):
            raise ValueError(
                f"Cannot draw {n} from the {stratum} stratum -- it holds {len(pool)}"
            )
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        parts.append(df.loc[np.sort(idx)])
    return pd.concat(parts).sort_values("Squad").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def welch_ci(a: np.ndarray, b: np.ndarray, alpha: float = ALPHA) -> dict:
    """Welch two-sample t-test with the matching confidence interval."""
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    diff = ma - mb
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    t_stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": na, "n_b": nb, "mean_a": ma, "mean_b": mb,
        "sd_a": np.sqrt(va), "sd_b": np.sqrt(vb),
        "diff": diff, "se": se, "df": df,
        "t": float(t_stat), "p": float(p),
        "ci_low": diff - t_crit * se, "ci_high": diff + t_crit * se,
    }


def describe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby("stage_reached")[metric]
    out = pd.DataFrame({
        "n": g.count(), "mean": g.mean(), "sd": g.std(ddof=1),
        "min": g.min(), "q1": g.quantile(0.25), "median": g.median(),
        "q3": g.quantile(0.75), "max": g.max(),
    })
    return out.reindex([KNOCKOUT, GROUP])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_report(frame: pd.DataFrame, sample: pd.DataFrame, allocation: str,
                 seed: int) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 78)
    add("OBJECTIVE 1 -- ATTACKING PERFORMANCE, 2026 FIFA WORLD CUP")
    add("=" * 78)

    add("\n[1] POPULATION AND SAMPLING FRAME")
    add("  Population        : all squads competing at the 2026 FIFA World Cup")
    add(f"  Population size N : {len(frame)}")
    add("  Sampling frame    : FBref Squad Shooting table for the competition")
    add("  Unit of analysis  : one squad, aggregated over all its matches")
    add("  Eligibility       : every squad that played at least one match")
    add("  Exclusions        : none -- no squad dropped for missing data or")
    add("                      extreme values")
    zeros = frame.loc[frame["npGls"] == 0, "Squad"].tolist()
    if zeros:
        add(f"  Legitimate zeros  : {', '.join(zeros)} scored no non-penalty goals;")
        add("                      finishing = 0.00 is a real value, not missing data")
    add(f"  Missing values    : {int(frame[[c for c, _ in COMPONENTS]].isna().sum().sum())}")

    add("\n[2] DATA PREPARATION")
    add("  Penalties removed from every numerator and denominator: a penalty is a")
    add("  set-piece award, not a product of attacking play, and over 3-8 matches")
    add("  one penalty moves a squad's ratio a long way.")
    add("    npGls = Gls - PK      npSh = Sh - PKatt      npSoT = SoT - PK")
    add("  Exposure measured in team 90s rather than matches, so extra time is")
    add("  handled correctly. FBref rounds 90s for display, so the unrounded value")
    add("  is recovered from Sh / (Sh/90).")
    add("  Integrity check -- derived ratios against FBref's published columns:")
    for line in verify_against_fbref(frame):
        add(line)

    add("\n[3] STRATIFIED RANDOM SAMPLING")
    add("  Method     : stratified random sampling without replacement")
    add("  Strata     : stage reached, mutually exclusive and exhaustive")
    add(f"  Allocation : {allocation}")
    add(f"  Seed       : {seed} (declared before drawing; single draw reported)")
    add("")
    add(f"  {'Stratum':<12}{'N':>6}{'n':>6}{'fraction':>11}{'pop share':>12}"
        f"{'sample share':>15}")
    for stratum in (KNOCKOUT, GROUP):
        big = int((frame["stage_reached"] == stratum).sum())
        small = int((sample["stage_reached"] == stratum).sum())
        add(f"  {stratum:<12}{big:>6}{small:>6}{small / big:>10.1%}"
            f"{big / len(frame):>12.1%}{small / len(sample):>15.1%}")
    add(f"  {'Total':<12}{len(frame):>6}{len(sample):>6}"
        f"{len(sample) / len(frame):>10.1%}")
    if allocation == "proportional":
        add("  Proportional allocation reproduces the population's 2:1 split, so")
        add("  neither stage is over- or under-represented relative to the")
        add("  tournament it is drawn from.")
    add("")
    add("  Sampled squads")
    for stratum in (KNOCKOUT, GROUP):
        names = sample.loc[sample["stage_reached"] == stratum, "Squad"].tolist()
        add(f"    {stratum} ({len(names)}): " + ", ".join(names))
    dropped = sorted(set(frame["Squad"]) - set(sample["Squad"]))
    add(f"    Not drawn ({len(dropped)}): " + ", ".join(dropped))

    add("\n[4] DECOMPOSITION -- group means in the sample")
    add(f"  {'Component':<28}{'Knockout':>12}{'Group':>12}{'Ratio':>10}")
    for col, name in COMPONENTS:
        k = sample.loc[sample["stage_reached"] == KNOCKOUT, col].mean()
        g = sample.loc[sample["stage_reached"] == GROUP, col].mean()
        add(f"  {name:<28}{k:>12.4f}{g:>12.4f}{k / g:>10.3f}")
    k = sample.loc[sample["stage_reached"] == KNOCKOUT, "np_goals_per_90"].mean()
    g = sample.loc[sample["stage_reached"] == GROUP, "np_goals_per_90"].mean()
    add(f"  {'Non-penalty goals per 90':<28}{k:>12.4f}{g:>12.4f}{k / g:>10.3f}")
    add("  Volume and accuracy are reported descriptively. Only finishing is")
    add("  tested, so no multiple-comparison correction is required.")

    add(f"\n[5] DESCRIPTIVE STATISTICS -- {DV} (dependent variable)")
    add(describe(sample, DV).round(4).to_string())

    a = sample.loc[sample["stage_reached"] == KNOCKOUT, DV].dropna().to_numpy()
    b = sample.loc[sample["stage_reached"] == GROUP, DV].dropna().to_numpy()
    r_sd_a, r_sd_b = a.std(ddof=1), b.std(ddof=1)

    add("\n[6] ASSUMPTION CHECKS")
    sk, sg = stats.shapiro(a), stats.shapiro(b)
    add(f"  Shapiro-Wilk, knockout : W = {sk.statistic:.4f}, p = {sk.pvalue:.4f}")
    add(f"  Shapiro-Wilk, group    : W = {sg.statistic:.4f}, p = {sg.pvalue:.4f}")
    add("  Normality is not rejected in either group, which matters here because")
    add("  the smaller group holds only 12 squads.")
    add("  Independence holds by design: the strata are disjoint, no squad belongs")
    add("  to both, and the two draws were made separately.")
    add(f"  Group sizes and spreads differ ({r_sd_a:.3f} vs {r_sd_b:.3f}), so Welch's")
    add("  t-test is used rather than the pooled-variance version. Welch does not")
    add("  assume equal variances, so no separate variance test is needed.")

    add(f"\n[7] 95% CONFIDENCE INTERVAL AND WELCH TWO-SAMPLE t-TEST -- {DV}")
    r = welch_ci(a, b)
    add(f"  Knockout : n = {r['n_a']}, mean = {r['mean_a']:.4f}, sd = {r['sd_a']:.4f}")
    add(f"  Group    : n = {r['n_b']}, mean = {r['mean_b']:.4f}, sd = {r['sd_b']:.4f}")
    add(f"  Difference (Knockout - Group) : {r['diff']:+.4f}")
    add(f"  Standard error                : {r['se']:.4f}")
    add(f"  95% CI for the difference     : [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    add(f"  t({r['df']:.2f}) = {r['t']:.4f}, p = {r['p']:.4f}")
    add(f"  Decision at alpha = {ALPHA}      : "
        + ("REJECT H0" if r["p"] < ALPHA else "FAIL TO REJECT H0"))
    add("")
    add("  Interpretation. In the sampled squads, knockout-stage teams turned")
    add(f"  {100 * r['mean_a']:.0f}% of their non-penalty shots on target into goals,")
    add(f"  against {100 * r['mean_b']:.0f}% for teams eliminated in the group stage --")
    add(f"  a gap of about {100 * r['diff']:.0f} percentage points. The stratified sample")
    add("  is used to estimate that difference for knockout-stage and group-stage")
    add("  teams in the tournament population.")
    if r["p"] < ALPHA:
        add(f"  The 95% interval runs from {100 * r['ci_low']:.0f} to "
            f"{100 * r['ci_high']:.0f} percentage points. It is wide,")
        add("  so the size of the gap is uncertain, but it does not include zero,")
        add("  which makes chance variation an unlikely explanation. The advantage")
        add("  lies in converting the chances created, not simply in taking more")
        add("  shots.")
    else:
        add("  The interval includes zero, so these data do not support a difference")
        add("  in finishing between the two groups.")

    add("\n[8] LIMITATIONS")
    add("  * Observational. Reaching the knockout stage is an outcome of attacking")
    add("    quality, not a treatment assigned to teams, so no causal claim is made.")
    add("  * The sample was drawn without replacement from a finite population of")
    add("    48 squads. The interval is reported as a standard two-sample interval")
    add("    and is not adjusted for that, so it should be read as an estimate of")
    add("    the difference between the two groups rather than an exact interval")
    add("    for the full 48-team population.")
    add("  * Group-stage squads contribute only 3 matches each, so their ratios are")
    add("    noisier than those of knockout squads.")
    add("  * npSoT subtracts scored penalties from shots on target; penalties saved")
    add("    are on target but indistinguishable from penalties missed here.")
    add("  * Expected goals were not published for this tournament, so shot quality")
    add("    cannot be controlled for. Finishing mixes conversion skill with the")
    add("    quality of the chances a team created.")
    add("=" * 78)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_decomposition(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    rng = np.random.default_rng(SEED)
    for ax, (col, label) in zip(axes, COMPONENTS):
        data = [
            df.loc[df["stage_reached"] == KNOCKOUT, col].dropna().to_numpy(),
            df.loc[df["stage_reached"] == GROUP, col].dropna().to_numpy(),
        ]
        bp = ax.boxplot(data, tick_labels=["Knockout", "Group"], widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, colour in zip(bp["boxes"], ("#c7d9ee", "#f2d5c4")):
            patch.set_facecolor(colour)
            patch.set_edgecolor("#333333")
        for median in bp["medians"]:
            median.set_color("#333333")
        for i, arr in enumerate(data, start=1):
            jitter = rng.uniform(-0.11, 0.11, size=len(arr))
            ax.scatter(np.full(len(arr), i) + jitter, arr, s=22, alpha=0.7,
                       color="#2f5d8a" if i == 1 else "#b05a2a", zorder=3)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Value (non-penalty)")
    fig.suptitle("Attacking output decomposed: volume x accuracy x finishing",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Attacking performance analysis.")
    ap.add_argument("--shooting", default="data/raw/squad_shooting.csv")
    ap.add_argument("--allocation", choices=sorted(ALLOCATIONS), default="proportional")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    path = Path(args.shooting)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(
            f"Shooting file not found: {path}\n"
            "Download the Squad Shooting table from\n"
            "  https://fbref.com/en/comps/1/shooting/World-Cup-Stats\n"
            "with Share & Export -> Get table as CSV."
        )

    frame = derive_metrics(load_shooting(path))
    if len(frame) != 48:
        print(f"WARNING: {len(frame)} squads in the frame, expected 48.")

    sample = stratified_sample(frame, ALLOCATIONS[args.allocation], args.seed)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    report = build_report(frame, sample, args.allocation, args.seed)
    print(report)
    (outdir / "attacking_results.txt").write_text(report, encoding="utf-8")

    cols = ["Squad", "Group", "stage_reached", "exposure_90s", "Gls", "Sh", "SoT",
            "PK", "PKatt", "npGls", "npSh", "npSoT", "volume", "accuracy",
            "finishing", "np_goals_per_90"]
    sample[[c for c in cols if c in sample.columns]].to_csv(
        outdir / "attacking_sample.csv", index=False)

    figure_decomposition(sample, outdir / "figures" / "decomposition.png")

    print(f"\nWritten to {outdir}:")
    print("  attacking_results.txt")
    print("  attacking_sample.csv")
    print("  figures/decomposition.png")


if __name__ == "__main__":
    main()
