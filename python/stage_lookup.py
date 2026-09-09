"""
Stage-reached classification for all 48 teams at the 2026 FIFA World Cup.

Grouping variable for the attacking-performance task:

    Knockout  = reached the Round of 32   (32 teams)
    Group     = eliminated at group stage (16 teams)

Format note (D-003): 12 groups of 4. The top two of each group advanced
automatically (24 teams), joined by the eight best third-placed teams, giving
32 knockout qualifiers. Six third-placed teams were therefore eliminated
despite finishing third, so "finished third" is NOT the classification rule --
"appeared in a Round of 32 fixture" is.

Naming note (D-004): FBref writes squad names with a country-code prefix
("br Brazil", "kr Korea Republic") and uses its own spellings, which differ
from FIFA's in several cases. Every plausible spelling is listed as an alias
and matched on a normalised key (lowercase, accents stripped, non-alphanumeric
characters removed), so the loader is robust to whichever export is used.
"""

from __future__ import annotations

import unicodedata

KNOCKOUT = "Knockout"
GROUP = "Group"

# Canonical team -> list of accepted spellings (canonical name included).
_TEAM_ALIASES: dict[str, list[str]] = {
    # --- Group A ---
    "Mexico": ["Mexico"],
    "South Africa": ["South Africa", "RSA"],
    "Korea Republic": ["Korea Republic", "South Korea", "Republic of Korea"],
    "Czechia": ["Czechia", "Czech Republic"],
    # --- Group B ---
    "Switzerland": ["Switzerland"],
    "Canada": ["Canada"],
    "Bosnia-Herzegovina": [
        "Bosnia-Herzegovina",
        "Bosnia-Herz",      # FBref's abbreviated squad-table label
        "Bosnia Herz",
        "Bosnia and Herzegovina",
        "Bosnia & Herzegovina",
        "Bosnia",
    ],
    "Qatar": ["Qatar"],
    # --- Group C ---
    "Brazil": ["Brazil"],
    "Morocco": ["Morocco"],
    "Scotland": ["Scotland"],
    "Haiti": ["Haiti"],
    # --- Group D ---
    "United States": ["United States", "USA", "US", "United States of America"],
    "Australia": ["Australia"],
    "Paraguay": ["Paraguay"],
    "Turkiye": ["Turkiye", "Türkiye", "Turkey"],
    # --- Group E ---
    "Germany": ["Germany"],
    "Cote d'Ivoire": [
        "Cote d'Ivoire",
        "Côte d'Ivoire",
        "Ivory Coast",
        "Cote dIvoire",
    ],
    "Ecuador": ["Ecuador"],
    "Curacao": ["Curacao", "Curaçao"],
    # --- Group F ---
    "Netherlands": ["Netherlands", "Holland"],
    "Japan": ["Japan"],
    "Sweden": ["Sweden"],
    "Tunisia": ["Tunisia"],
    # --- Group G ---
    "Belgium": ["Belgium"],
    "Egypt": ["Egypt"],
    "Iran": ["Iran", "IR Iran", "Islamic Republic of Iran"],
    "New Zealand": ["New Zealand"],
    # --- Group H ---
    "Spain": ["Spain"],
    "Cabo Verde": ["Cabo Verde", "Cape Verde"],
    "Uruguay": ["Uruguay"],
    "Saudi Arabia": ["Saudi Arabia", "KSA"],
    # --- Group I ---
    "France": ["France"],
    "Norway": ["Norway"],
    "Senegal": ["Senegal"],
    "Iraq": ["Iraq"],
    # --- Group J ---
    "Argentina": ["Argentina"],
    "Austria": ["Austria"],
    "Algeria": ["Algeria"],
    "Jordan": ["Jordan"],
    # --- Group K ---
    "Colombia": ["Colombia"],
    "Portugal": ["Portugal"],
    "Congo DR": ["Congo DR", "DR Congo", "Democratic Republic of the Congo", "Congo-Kinshasa"],
    "Uzbekistan": ["Uzbekistan"],
    # --- Group L ---
    "England": ["England"],
    "Croatia": ["Croatia"],
    "Ghana": ["Ghana"],
    "Panama": ["Panama"],
}

# Teams that appeared in a Round of 32 fixture.
_KNOCKOUT_TEAMS = {
    "Mexico", "South Africa",
    "Switzerland", "Canada", "Bosnia-Herzegovina",
    "Brazil", "Morocco",
    "United States", "Australia", "Paraguay",
    "Germany", "Cote d'Ivoire", "Ecuador",
    "Netherlands", "Japan", "Sweden",
    "Belgium", "Egypt",
    "Spain", "Cabo Verde",
    "France", "Norway", "Senegal",
    "Argentina", "Austria", "Algeria",
    "Colombia", "Portugal", "Congo DR",
    "England", "Croatia", "Ghana",
}

# Group of each team, kept for the descriptive tables and as a sanity check.
_GROUP_OF = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Switzerland", "Canada", "Bosnia-Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkiye"],
    "E": ["Germany", "Cote d'Ivoire", "Ecuador", "Curacao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Uruguay", "Saudi Arabia"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Colombia", "Portugal", "Congo DR", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def normalise(name: str) -> str:
    """Lowercase, strip accents, drop everything that is not a letter or digit."""
    if name is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in ascii_only.lower() if c.isalnum())


# normalised alias -> canonical team name
_ALIAS_INDEX: dict[str, str] = {}
for _canonical, _aliases in _TEAM_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_INDEX[normalise(_alias)] = _canonical

_TEAM_TO_GROUP: dict[str, str] = {
    team: letter for letter, teams in _GROUP_OF.items() for team in teams
}


def canonical_team(raw_name: str) -> str | None:
    """Map any accepted spelling to the canonical team name, or None."""
    return _ALIAS_INDEX.get(normalise(raw_name))


def stage_reached(canonical_name: str) -> str:
    """Return 'Knockout' or 'Group' for a canonical team name."""
    if canonical_name not in _TEAM_TO_GROUP:
        raise KeyError(f"Not a 2026 World Cup team: {canonical_name!r}")
    return KNOCKOUT if canonical_name in _KNOCKOUT_TEAMS else GROUP


def group_letter(canonical_name: str) -> str:
    return _TEAM_TO_GROUP[canonical_name]


def all_teams() -> list[str]:
    return sorted(_TEAM_TO_GROUP)


def _self_check() -> None:
    """Structural assertions -- these run on import and must never fail."""
    assert len(_TEAM_TO_GROUP) == 48, f"expected 48 teams, found {len(_TEAM_TO_GROUP)}"
    assert len(_TEAM_ALIASES) == 48, "alias table and group table disagree"
    assert set(_TEAM_ALIASES) == set(_TEAM_TO_GROUP), "alias/group name mismatch"
    assert len(_KNOCKOUT_TEAMS) == 32, f"expected 32 knockout teams, found {len(_KNOCKOUT_TEAMS)}"
    assert _KNOCKOUT_TEAMS <= set(_TEAM_TO_GROUP), "unknown team in knockout set"
    for letter, teams in _GROUP_OF.items():
        n_through = sum(1 for t in teams if t in _KNOCKOUT_TEAMS)
        assert 2 <= n_through <= 3, f"Group {letter} advanced {n_through} teams"
    n_thirds = sum(
        1 for letter, teams in _GROUP_OF.items()
        if sum(1 for t in teams if t in _KNOCKOUT_TEAMS) == 3
    )
    assert n_thirds == 8, f"expected 8 best third-placed teams, found {n_thirds}"


_self_check()


if __name__ == "__main__":
    for letter in sorted(_GROUP_OF):
        through = [t for t in _GROUP_OF[letter] if t in _KNOCKOUT_TEAMS]
        out = [t for t in _GROUP_OF[letter] if t not in _KNOCKOUT_TEAMS]
        print(f"Group {letter}: through={through}  out={out}")
    print(f"\nKnockout {len(_KNOCKOUT_TEAMS)} | Group {48 - len(_KNOCKOUT_TEAMS)} | Total 48")
