"""
Loader for FBref shooting exports (2026 FIFA World Cup).

Handles the awkward bits of an FBref CSV so the analysis script never has to:

  * two-row headers -- FBref stacks a group row ("Standard", "Expected") above
    the real column names, and the group row is blank for the leading columns.
  * repeated header rows -- FBref reprints the header every ~25 rows in long
    tables, and those rows survive a naive read_csv.
  * blank separator rows between sections.
  * country-code prefixes on squad names ("br Brazil", "kr Korea Republic"),
    which appear at the start of the string; the "vs" opponent tables use
    "br vs Brazil" and are dropped.
  * player-level exports -- if the file has a Player column, rows are summed to
    squad level so both the squad table and the player table work as input.

Accepts either the Squad Shooting table or the Player Shooting table:
    https://fbref.com/en/comps/1/shooting/World-Cup-Stats
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from stage_lookup import canonical_team, group_letter, stage_reached

# Columns we count on. xG/npxG are optional: they were not published for this
# tournament (D-009), so nothing downstream may depend on them.
REQUIRED_COUNTS = ["Gls", "Sh", "SoT", "PK", "PKatt"]
OPTIONAL_COUNTS = ["xG", "npxG"]
# FBref's own published ratios. Not summable, so they are carried through only
# for squad-level input, where they are used to verify our derivations.
RATIO_COLS = ["SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT"]


def _sniff_header_rows(path: Path) -> int:
    """Return 0 or 1 -- the number of FBref group-header rows above the real header."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [row for _, row in zip(range(2), csv.reader(fh))]
    if not rows:
        raise ValueError(f"{path} is empty")
    first = [c.strip() for c in rows[0]]
    # The real header row always contains 'Squad' or 'Player'.
    if any(c in ("Squad", "Player") for c in first):
        return 0
    if len(rows) > 1 and any(c.strip() in ("Squad", "Player") for c in rows[1]):
        return 1
    raise ValueError(
        f"{path}: could not find a header row containing 'Squad' or 'Player'. "
        "Export the table from FBref with 'Get table as CSV'."
    )


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a two-level FBref header, keeping the lower level as the name."""
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).strip() for c in df.columns]
        return df
    flat = []
    for upper, lower in df.columns:
        lower = str(lower).strip()
        upper = str(upper).strip()
        # FBref pads the group row with 'Unnamed: n_level_0' for leading columns.
        name = lower if lower and not lower.startswith("Unnamed") else upper
        flat.append(name)
    df.columns = flat
    return df


def _strip_country_code(raw: str) -> str:
    """'br Brazil' -> 'Brazil'; 'eng England' -> 'England'; 'Brazil' -> 'Brazil'."""
    text = str(raw).strip()
    if not text:
        return text
    parts = text.split()
    if len(parts) > 1 and parts[0].isalpha() and parts[0].islower() and len(parts[0]) <= 3:
        return " ".join(parts[1:]).strip()
    return text


def load_shooting(
    path: str | Path,
    *,
    matches_played: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Read an FBref shooting export and return one tidy row per squad.

    Returns columns:
        Squad, Group, stage_reached, Pl, 90s, Gls, Sh, SoT, PK, PKatt
        (+ xG, npxG if present in the source file)
    """
    path = Path(path)
    skip = _sniff_header_rows(path)
    header = [0, 1] if skip else [0]
    df = pd.read_csv(path, header=header, encoding="utf-8-sig")
    df = _flatten_columns(df)

    # Drop FBref's repeated header rows and blank separators.
    key_col = "Player" if "Player" in df.columns else "Squad"
    if key_col not in df.columns:
        raise ValueError(f"{path}: no 'Squad' or 'Player' column after header flattening")
    df = df[df[key_col].notna()]
    df = df[df[key_col].astype(str).str.strip() != ""]
    df = df[df[key_col].astype(str).str.strip() != key_col]

    if "Squad" not in df.columns:
        raise ValueError(f"{path}: a 'Squad' column is required (player exports include one)")

    # Drop the 'vs Opponent' half of FBref's squad tables -- those are conceded,
    # not created, and belong to Sabin's defensive task, not this one.
    squad_raw = df["Squad"].astype(str).str.strip()
    is_opponent = squad_raw.str.contains(r"\bvs\b", case=False, regex=True)
    if is_opponent.all() and len(df):
        raise ValueError(
            f"{path}: every row is an opponent row ('xx vs Country'). You exported "
            "the OPPONENT STATS tab, which is shots conceded -- that is the "
            "defensive task, not attacking. On the FBref page click the 'Squad "
            "Stats' tab (left of 'Opponent Stats') before Share & Export."
        )
    df = df[~is_opponent]

    df["Squad"] = df["Squad"].map(_strip_country_code)

    numeric_cols = [c for c in REQUIRED_COUNTS + OPTIONAL_COUNTS + ["90s"] if c in df.columns]
    missing = [c for c in REQUIRED_COUNTS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}. "
            "Use FBref's Shooting table, not Standard Stats."
        )
    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).fillna(0.0)

    # Player-level input -> aggregate to squad.
    is_player_level = "Player" in df.columns
    if is_player_level:
        agg = {c: "sum" for c in numeric_cols}
        grouped = df.groupby("Squad", as_index=False).agg(agg)
        grouped["Pl"] = df.groupby("Squad")["Player"].nunique().values
    else:
        grouped = df.groupby("Squad", as_index=False)[numeric_cols].sum()
        for col in RATIO_COLS:
            if col in df.columns:
                grouped[col] = pd.to_numeric(
                    df.groupby("Squad")[col].first().values, errors="coerce")
        pl_col = next((c for c in ("# Pl", "Pl", "#Pl") if c in df.columns), None)
        grouped["Pl"] = (
            pd.to_numeric(df.groupby("Squad")[pl_col].max().values, errors="coerce")
            if pl_col
            else np.nan
        )

    # Attach the grouping variable, failing loudly on any unmatched squad.
    canon, unmatched = [], []
    for name in grouped["Squad"]:
        c = canonical_team(name)
        if c is None:
            unmatched.append(name)
        canon.append(c)
    if unmatched:
        raise ValueError(
            "Squad name(s) not recognised: "
            + ", ".join(repr(u) for u in sorted(set(unmatched)))
            + ". Add the spelling to _TEAM_ALIASES in src/stage_lookup.py."
        )

    grouped["Squad"] = canon
    grouped = grouped.groupby("Squad", as_index=False).sum(numeric_only=True)
    grouped["Group"] = grouped["Squad"].map(group_letter)
    grouped["stage_reached"] = grouped["Squad"].map(stage_reached)

    # ------------------------------------------------------------------
    # Exposure (D-012). Volume measures must be normalised by playing time.
    #
    # In FBref's SQUAD table, '90s' is the team's own 90s: 3.0 for a squad that
    # played three group matches, 9.0 for a finalist whose run included extra
    # time. In the PLAYER table, '90s' is per player, so a squad total is
    # roughly 11x that. Both are handled; 'exposure_90s' is always team-level.
    #
    # Extra time is why exposure_90s, not a match count, is the right
    # denominator: a squad that played 30 extra minutes had 30 more minutes to
    # shoot in, and per-match rates would silently reward that.
    # ------------------------------------------------------------------
    if "90s" not in grouped.columns or not grouped["90s"].gt(0).all():
        raise ValueError(f"{path}: a positive '90s' column is required for exposure")

    grouped["exposure_90s"] = (
        grouped["90s"] / 11.0 if is_player_level else grouped["90s"]
    )

    # FBref rounds the displayed 90s to one decimal, but publishes Sh/90 to two.
    # Sh / (Sh/90) therefore recovers the unrounded exposure: Germany shows
    # 90s = 4.3 but is really 4.3326, and using 4.3 overstates shots per 90 by
    # about 0.8%. Refine where the arithmetic is consistent with the rounding,
    # and leave the displayed value alone where it is not.
    if not is_player_level and "Sh/90" in grouped.columns:
        sh90 = pd.to_numeric(grouped["Sh/90"], errors="coerce")
        recovered = grouped["Sh"] / sh90.where(sh90 > 0)
        plausible = (recovered - grouped["exposure_90s"]).abs() <= 0.06
        n_fixed = int((plausible & recovered.notna()).sum())
        grouped["exposure_90s"] = recovered.where(plausible & recovered.notna(),
                                                  grouped["exposure_90s"])
        grouped.attrs["exposure_refined"] = n_fixed

    if not grouped["exposure_90s"].between(2.5, 9.5).all():
        odd = grouped.loc[
            ~grouped["exposure_90s"].between(2.5, 9.5), ["Squad", "exposure_90s"]
        ]
        raise ValueError(
            "Implausible exposure -- every squad played 3 to 8 matches, so "
            "team 90s should sit between 3.0 and roughly 9.0:\n"
            + odd.to_string(index=False)
        )

    if matches_played:
        mapped = {}
        for k, v in matches_played.items():
            c = canonical_team(k)
            if c is None:
                raise ValueError(f"matches_played: unrecognised team {k!r}")
            mapped[c] = int(v)
        grouped["MP"] = grouped["Squad"].map(mapped)
        if grouped["MP"].isna().any():
            gaps = grouped.loc[grouped["MP"].isna(), "Squad"].tolist()
            raise ValueError(f"matches_played missing entries for: {gaps}")
    else:
        # Reported in descriptives only. Nothing is divided by it.
        grouped["MP"] = grouped["exposure_90s"].round().clip(3, 8).astype(int)

    ordered = ["Squad", "Group", "stage_reached", "exposure_90s", "MP", "Pl", "90s"] + [
        c for c in REQUIRED_COUNTS + OPTIONAL_COUNTS + RATIO_COLS if c in grouped.columns
    ]
    ordered = [c for c in ordered if c in grouped.columns]
    return grouped[ordered].sort_values("Squad").reset_index(drop=True)


def load_matches_played(path: str | Path) -> dict[str, int]:
    """
    Count matches per team from an FBref 'Scores & Fixtures' export.

    Expects Home/Away columns (FBref labels them 'Home' and 'Away') and a Score
    column; unplayed rows have no score and are ignored.
    """
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    home_col = next((c for c in df.columns if c.lower() in ("home", "home team")), None)
    away_col = next((c for c in df.columns if c.lower() in ("away", "away team")), None)
    if not home_col or not away_col:
        raise ValueError(f"{path}: need Home and Away columns")
    score_col = next((c for c in df.columns if c.lower() == "score"), None)
    if score_col:
        df = df[df[score_col].notna() & (df[score_col].astype(str).str.strip() != "")]

    counts: dict[str, int] = {}
    for col in (home_col, away_col):
        for raw in df[col].dropna():
            team = canonical_team(_strip_country_code(raw))
            if team is None:
                raise ValueError(f"{path}: unrecognised team {raw!r}")
            counts[team] = counts.get(team, 0) + 1
    return counts
