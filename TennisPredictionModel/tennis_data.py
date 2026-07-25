"""Tennis match spine: download, parse and normalise the historical archive.

Source: the tennis-data.co.uk historical archive — one workbook per tour-season
covering every ATP main-tour and WTA main-tour singles match with the final
scoreline, both players' rankings/points, and *closing* prices from up to ten
bookmakers (Bet365, Pinnacle, and the market Max/Avg aggregates). It is the
same archive the Weighted-Elo literature is built on (Angelini, Candila & De
Angelis, EJOR 297(1), 2022), which matters: it means our numbers are directly
comparable to published results.

Two physical formats hide behind the ``.xlsx`` URLs. Seasons through 2012 are
legacy BIFF ``.xls`` workbooks (parsed with ``xlrd``, an offline-only training
dependency); 2013 onward are real OOXML ``.xlsx``, which this module reads with
nothing but the standard library. That split is deliberate — the daily serving
job only ever needs the *current* season, so it never needs a third-party Excel
reader, while offline training can still reach back to 2000 for Elo burn-in.

Nothing here is committed to the repository: the raw workbooks are third-party
data pulled into a gitignored cache and regenerated on demand.
"""
from __future__ import annotations

import csv
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "tennis"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MATCHES_CSV = PROCESSED_DIR / "matches.csv"

ATP_URL = "http://tennis-data.co.uk/{year}/{year}.xlsx"
WTA_URL = "http://tennis-data.co.uk/{year}w/{year}.xlsx"
ATP_FIRST_SEASON = 2000
# The women's archive genuinely starts in 2007. The 2005w/2006w URLs exist but
# serve the *men's* workbook byte-for-byte, which silently poisons the WTA
# ratings with 5,818 mislabelled ATP matches; `parse_workbook` re-checks that
# defensively so the spine self-heals if the site's coverage ever shifts again.
WTA_FIRST_SEASON = 2007
# Seasons at or after this one ship as real OOXML and parse with the stdlib.
OOXML_FIRST_SEASON = 2013
CURRENT_SEASON_TTL_SECONDS = 6 * 60 * 60

USER_AGENT = "pickledger-tennis-model/1.0 (+https://harsh.bet)"

_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
# Excel's 1900 date system counts day 1 as 1900-01-01 but also believes 1900 was
# a leap year, so the usable epoch is 1899-12-30.
_EXCEL_EPOCH = date(1899, 12, 30)

# Round ordering doubles as the within-day sort key: two matches on the same
# date at the same event must be replayed in draw order or a quarter-final
# would update Elo before the second round that produced it.
ROUND_ORDER = {
    "0th round": 0,
    "qualifying": 0,
    "q1": 0,
    "q2": 0,
    "q3": 0,
    "round robin": 1,
    "1st round": 1,
    "2nd round": 2,
    "3rd round": 3,
    "4th round": 4,
    "quarterfinals": 5,
    "semifinals": 6,
    "the final": 7,
}

# Event tier, normalised across the ATP/WTA naming churn (the tours renamed
# their tiers several times inside the sample window).
SERIES_TIER = {
    "grand slam": 5,
    "masters cup": 4,
    "masters": 4,
    "masters 1000": 4,
    "atp500": 3,
    "international gold": 3,
    "atp250": 2,
    "international": 2,
    "premier mandatory": 4,
    "premier": 3,
    "wta250": 2,
    "wta500": 3,
    "wta1000": 4,
    "tier 1": 4,
    "tier 2": 3,
    "tier 3": 2,
    "tier 4": 2,
    "tier 5": 1,
    "international series": 2,
}

SURFACES = ("Hard", "Clay", "Grass", "Carpet")

# Odds columns, in the order we prefer them. Pinnacle (PS) is the sharpest book
# in the archive and the one the market-efficiency literature benchmarks
# against; B365 is the widest-covered; Max/Avg are market aggregates.
ODDS_COLUMNS = {
    "ps": ("PSW", "PSL"),
    "b365": ("B365W", "B365L"),
    "max": ("MaxW", "MaxL"),
    "avg": ("AvgW", "AvgL"),
}


@dataclass
class Match:
    """One normalised singles match, oriented winner-first as the source is."""

    date: str
    tour: str
    season: int
    tournament: str
    # The host city. Sponsor names churn yearly ("Internazionali Femminili di
    # Palermo" is ESPN's "Palermo Ladies Open"); the city does not, which makes
    # it the reliable join for classifying a live slate's surface.
    location: str
    series: str
    tier: int
    court: str
    surface: str
    round: str
    round_order: int
    best_of: int
    winner: str
    loser: str
    winner_key: str
    loser_key: str
    winner_rank: int | None
    loser_rank: int | None
    winner_points: int | None
    loser_points: int | None
    winner_games: int
    loser_games: int
    winner_sets: int
    loser_sets: int
    status: str
    odds: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def retired(self) -> bool:
        return self.status == "retired"

    @property
    def total_games(self) -> int:
        return self.winner_games + self.loser_games

    def sort_key(self) -> tuple[Any, ...]:
        return (self.date, self.tour, self.tournament, self.round_order)


# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z\s]")
_SPACE = re.compile(r"\s+")
# Suffixes that are part of neither the surname nor the initial.
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def player_key(name: str) -> str:
    """Canonical ``surname-i`` key for a player.

    The archive writes ``"Ugo Carabelli C."`` while ESPN writes
    ``"Camilo Ugo Carabelli"``; both have to land on ``ugo carabelli-c`` or the
    daily slate can never be joined to the ratings table. The rule is
    surname-tokens plus first initial, which is the same last-name + first-
    initial identity the rest of the repo already uses for people.
    """
    cleaned = _SPACE.sub(" ", _PUNCT.sub(" ", _strip_accents(name).lower())).strip()
    if not cleaned:
        return ""
    tokens = [tok for tok in cleaned.split(" ") if tok and tok not in _NAME_SUFFIXES]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    # Archive form: every trailing single letter is an initial ("Ugo Carabelli C.",
    # "Bautista Agut R.", "Auger-Aliassime F." -> "auger aliassime f").
    trailing_initials = []
    while len(tokens) > 1 and len(tokens[-1]) == 1:
        trailing_initials.insert(0, tokens.pop())
    if trailing_initials:
        return f"{' '.join(tokens)}-{trailing_initials[0]}"
    # Display form: first token is the given name, the rest is the surname.
    return f"{' '.join(tokens[1:])}-{tokens[0][0]}"


# --------------------------------------------------------------------------
# workbook readers
# --------------------------------------------------------------------------


def _column_index(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref or "")
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - 64)
    return index - 1


def read_xlsx(path: Path) -> list[list[str]]:
    """Read sheet 1 of an OOXML workbook using only the standard library."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root:
                shared.append("".join(node.text or "" for node in item.iter(_SHEET_NS + "t")))
        sheet_name = next(
            (name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")),
            "xl/worksheets/sheet1.xml",
        )
        rows: list[list[str]] = []
        root = ET.fromstring(archive.read(sheet_name))
        for row in root.iter(_SHEET_NS + "row"):
            cells: dict[int, str] = {}
            for position, cell in enumerate(row.iter(_SHEET_NS + "c")):
                ref = cell.get("r") or ""
                index = _column_index(ref) if ref else position
                kind = cell.get("t")
                value_node = cell.find(_SHEET_NS + "v")
                if kind == "s" and value_node is not None and value_node.text is not None:
                    value = shared[int(value_node.text)]
                elif kind == "inlineStr":
                    inline = cell.find(_SHEET_NS + "is")
                    value = "".join(n.text or "" for n in inline.iter(_SHEET_NS + "t")) if inline is not None else ""
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                cells[index] = value
            width = (max(cells) + 1) if cells else 0
            rows.append([cells.get(i, "") for i in range(width)])
        return rows


def read_xls(path: Path) -> list[list[str]]:
    """Read a legacy BIFF workbook. Offline-only: needs the ``xlrd`` package."""
    try:
        import xlrd  # noqa: PLC0415 - optional training-only dependency
    except ImportError as exc:  # pragma: no cover - exercised only without xlrd
        raise RuntimeError(
            f"{path.name} is a legacy .xls workbook; install xlrd to parse pre-2013 seasons"
        ) from exc
    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    rows: list[list[str]] = []
    for row_index in range(sheet.nrows):
        row: list[str] = []
        for col_index in range(sheet.ncols):
            value = sheet.cell_value(row_index, col_index)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            row.append(str(value))
        rows.append(row)
    return rows


def read_workbook(path: Path) -> list[list[str]]:
    with path.open("rb") as handle:
        signature = handle.read(4)
    return read_xlsx(path) if signature[:2] == b"PK" else read_xls(path)


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def _download(url: str, destination: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"[tennis] download failed {url}: {exc}", file=sys.stderr)
        return False
    if len(payload) < 5_000:
        print(f"[tennis] suspiciously small payload for {url} ({len(payload)}B)", file=sys.stderr)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.with_suffix(".stamp").write_text(str(time.time()), encoding="utf-8")
    return True


def workbook_path(tour: str, season: int) -> Path:
    return RAW_DIR / tour.lower() / f"{season}.xlsx"


def ensure_season(tour: str, season: int, *, refresh: bool = False) -> Path | None:
    """Return the local workbook for a tour-season, downloading when needed.

    In-season files change as results land, so the current season is refreshed
    on a TTL. Freshness rides on a sidecar stamp rather than the file mtime,
    because a CI checkout resets mtimes and would otherwise look permanently
    stale (the same trap the NFL spine documents).
    """
    path = workbook_path(tour, season)
    stamp = path.with_suffix(".stamp")
    if path.exists() and not refresh:
        if season < date.today().year:
            return path
        age = (time.time() - stamp.stat().st_mtime) if stamp.exists() else None
        if age is not None and age < CURRENT_SEASON_TTL_SECONDS:
            return path
    template = ATP_URL if tour.upper() == "ATP" else WTA_URL
    if _download(template.format(year=season), path):
        return path
    return path if path.exists() else None


def available_seasons(tour: str, last_season: int | None = None) -> range:
    first = ATP_FIRST_SEASON if tour.upper() == "ATP" else WTA_FIRST_SEASON
    return range(first, (last_season or date.today().year) + 1)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(round(number)) if number is not None else None


def _parse_date(value: Any) -> str:
    """Excel serial or ISO/EU text date -> ISO string."""
    text = str(value or "").strip()
    if not text:
        return ""
    number = _as_float(text)
    if number is not None and number > 20_000:
        return (_EXCEL_EPOCH + timedelta(days=int(number))).isoformat()
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            from datetime import datetime

            return datetime.strptime(text[: len(pattern) + 4], pattern).date().isoformat()
        except ValueError:
            continue
    return ""


def _repair_date(iso_date: str, season: int) -> str:
    """Fix source date typos that would otherwise reorder the whole spine.

    A season legitimately spills a few days either side of the calendar year —
    the tour opens in late December — so the guard is a wide window rather than
    an equality check. Anything outside it is a typo (the 2026 Iasi Open final
    is stamped 2029-07-20); re-stamping the season year lands it back in range,
    and a date that still fails is dropped rather than trusted. Left alone, one
    such row sorts to the end of the chronological replay and poisons the
    ratings snapshot's ``through_date``, which is what the daily job uses to
    decide how much history to catch up on.
    """
    if not iso_date:
        return ""
    try:
        parsed = date.fromisoformat(iso_date)
    except ValueError:
        return ""
    if date(season - 1, 11, 1) <= parsed <= date(season + 1, 3, 1):
        return iso_date
    try:
        repaired = parsed.replace(year=season)
    except ValueError:  # 29 February in a non-leap season
        return ""
    if date(season - 1, 11, 1) <= repaired <= date(season + 1, 3, 1):
        return repaired.isoformat()
    return ""


def _normalise_status(comment: str) -> str:
    text = str(comment or "").strip().lower()
    if not text or text.startswith("complet") or text.startswith("full"):
        return "completed"
    if "retired" in text or text == "ret.":
        return "retired"
    if "walkover" in text or text.startswith("w/o"):
        return "walkover"
    if "disqualified" in text or "default" in text:
        return "disqualified"
    return "other"


def _normalise_surface(value: str) -> str:
    text = str(value or "").strip().title()
    return text if text in SURFACES else "Hard"


def _normalise_series(value: str) -> str:
    """Fold the archive's tier typos back onto real tiers.

    The 2021 European Open rows are stamped WTA251…WTA263 (one per match) where
    the tier is plainly WTA250; left alone they fragment into 26 singleton
    tiers.
    """
    text = str(value or "").strip()
    match = re.fullmatch(r"(?i)(WTA|ATP)\s*(\d{3,4})", text)
    if not match:
        return text
    tour, number = match.group(1).upper(), int(match.group(2))
    for tier in (1000, 500, 250):
        if number >= tier:
            return f"{tour}{tier}"
    return text


def _tier(series: str) -> int:
    return SERIES_TIER.get(str(series or "").strip().lower(), 2)


def _round_order(value: str) -> int:
    return ROUND_ORDER.get(str(value or "").strip().lower(), 2)


def parse_workbook(path: Path, tour: str, season: int) -> list[Match]:
    rows = read_workbook(path)
    if not rows:
        return []
    if tour.upper() == "WTA":
        men = workbook_path("ATP", season)
        if men.exists() and men.read_bytes() == path.read_bytes():
            print(f"[tennis] {path.name} is the ATP workbook verbatim; skipping WTA {season}", file=sys.stderr)
            return []
    header = [str(cell or "").strip() for cell in rows[0]]
    index = {name: position for position, name in enumerate(header) if name}

    def cell(row: list[str], column: str) -> str:
        position = index.get(column)
        if position is None or position >= len(row):
            return ""
        return str(row[position] or "").strip()

    matches: list[Match] = []
    for row in rows[1:]:
        winner = cell(row, "Winner")
        loser = cell(row, "Loser")
        if not winner or not loser:
            continue
        match_date = _repair_date(_parse_date(cell(row, "Date")), season)
        if not match_date:
            continue
        winner_games = 0
        loser_games = 0
        for set_number in range(1, 6):
            won = _as_int(cell(row, f"W{set_number}"))
            lost = _as_int(cell(row, f"L{set_number}"))
            if won is None or lost is None:
                continue
            winner_games += max(0, won)
            loser_games += max(0, lost)
        odds: dict[str, tuple[float, float]] = {}
        for key, (winner_column, loser_column) in ODDS_COLUMNS.items():
            winner_price = _as_float(cell(row, winner_column))
            loser_price = _as_float(cell(row, loser_column))
            # A price at or below evens on both sides is a data error, and a
            # one-sided price cannot be de-vigged, so both must be sane.
            if winner_price and loser_price and winner_price > 1.0 and loser_price > 1.0:
                odds[key] = (winner_price, loser_price)
        series = _normalise_series(cell(row, "Series") or cell(row, "Tier"))
        round_name = cell(row, "Round")
        matches.append(
            Match(
                date=match_date,
                tour=tour.upper(),
                season=season,
                tournament=cell(row, "Tournament"),
                location=cell(row, "Location"),
                series=series,
                tier=_tier(series),
                court=cell(row, "Court") or "Outdoor",
                surface=_normalise_surface(cell(row, "Surface")),
                round=round_name,
                round_order=_round_order(round_name),
                best_of=_as_int(cell(row, "Best of")) or 3,
                winner=winner,
                loser=loser,
                winner_key=player_key(winner),
                loser_key=player_key(loser),
                winner_rank=_as_int(cell(row, "WRank")),
                loser_rank=_as_int(cell(row, "LRank")),
                winner_points=_as_int(cell(row, "WPts")),
                loser_points=_as_int(cell(row, "LPts")),
                winner_games=winner_games,
                loser_games=loser_games,
                winner_sets=_as_int(cell(row, "Wsets")) or 0,
                loser_sets=_as_int(cell(row, "Lsets")) or 0,
                status=_normalise_status(cell(row, "Comment")),
                odds=odds,
            )
        )
    return matches


def load_matches(
    tours: Iterable[str] = ("ATP", "WTA"),
    first_season: int | None = None,
    last_season: int | None = None,
    *,
    download: bool = True,
    refresh_current: bool = False,
    stdlib_only: bool = False,
) -> list[Match]:
    """Load every available match, sorted into strict chronological order.

    ``stdlib_only`` restricts the window to the OOXML seasons so the serving
    path never depends on ``xlrd``.
    """
    matches: list[Match] = []
    for tour in tours:
        seasons = available_seasons(tour, last_season)
        for season in seasons:
            if first_season and season < first_season:
                continue
            if stdlib_only and season < OOXML_FIRST_SEASON:
                continue
            path = ensure_season(tour, season, refresh=refresh_current and season == date.today().year) if download else workbook_path(tour, season)
            if path is None or not path.exists():
                continue
            try:
                matches.extend(parse_workbook(path, tour, season))
            except Exception as exc:  # a single unreadable season must not sink the spine
                print(f"[tennis] failed to parse {path}: {exc}", file=sys.stderr)
    matches.sort(key=Match.sort_key)
    return matches


# --------------------------------------------------------------------------
# processed spine
# --------------------------------------------------------------------------

# Scalar fields, in file order. The odds pairs are flattened separately.
SCALAR_COLUMNS = [
    "date", "tour", "season", "tournament", "location", "series", "tier", "court",
    "surface", "round", "round_order", "best_of", "winner", "loser", "winner_key",
    "loser_key", "winner_rank", "loser_rank", "winner_points", "loser_points",
    "winner_games", "loser_games", "winner_sets", "loser_sets", "status",
]
ODDS_BOOKS = ("ps", "b365", "max", "avg")
CSV_COLUMNS = SCALAR_COLUMNS + [f"{book}_{side}" for book in ODDS_BOOKS for side in ("w", "l")]


def write_spine(matches: list[Match], path: Path = MATCHES_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for match in matches:
            record = asdict(match)
            row = [record[name] for name in SCALAR_COLUMNS]
            for book in ODDS_BOOKS:
                prices = match.odds.get(book)
                row.extend(["", ""] if not prices else [prices[0], prices[1]])
            writer.writerow(row)
    return path


def read_spine(path: Path = MATCHES_CSV) -> list[Match]:
    if not path.exists():
        return []
    matches: list[Match] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            odds: dict[str, tuple[float, float]] = {}
            for book in ODDS_BOOKS:
                winner_price = _as_float(row.get(f"{book}_w"))
                loser_price = _as_float(row.get(f"{book}_l"))
                if winner_price and loser_price:
                    odds[book] = (winner_price, loser_price)
            matches.append(
                Match(
                    date=row["date"],
                    tour=row["tour"],
                    season=int(row["season"]),
                    tournament=row["tournament"],
                    location=row.get("location") or "",
                    series=row["series"],
                    tier=int(row["tier"]),
                    court=row["court"],
                    surface=row["surface"],
                    round=row["round"],
                    round_order=int(row["round_order"]),
                    best_of=int(row["best_of"]),
                    winner=row["winner"],
                    loser=row["loser"],
                    winner_key=row["winner_key"],
                    loser_key=row["loser_key"],
                    winner_rank=_as_int(row.get("winner_rank")),
                    loser_rank=_as_int(row.get("loser_rank")),
                    winner_points=_as_int(row.get("winner_points")),
                    loser_points=_as_int(row.get("loser_points")),
                    winner_games=int(row["winner_games"] or 0),
                    loser_games=int(row["loser_games"] or 0),
                    winner_sets=int(row["winner_sets"] or 0),
                    loser_sets=int(row["loser_sets"] or 0),
                    status=row["status"],
                    odds=odds,
                )
            )
    matches.sort(key=Match.sort_key)
    return matches


def iter_seasons(matches: list[Match]) -> Iterator[int]:
    seen: set[int] = set()
    for match in matches:
        if match.season not in seen:
            seen.add(match.season)
            yield match.season


def build_spine(*, download: bool = True, last_season: int | None = None) -> dict[str, Any]:
    matches = load_matches(download=download, last_season=last_season)
    write_spine(matches)
    seasons = sorted({match.season for match in matches})
    players = {match.winner_key for match in matches} | {match.loser_key for match in matches}
    priced = sum(1 for match in matches if match.odds.get("ps") or match.odds.get("b365"))
    return {
        "matches": len(matches),
        "seasons": [seasons[0], seasons[-1]] if seasons else [],
        "players": len(players),
        "atp": sum(1 for match in matches if match.tour == "ATP"),
        "wta": sum(1 for match in matches if match.tour == "WTA"),
        "completed": sum(1 for match in matches if match.completed),
        "retired": sum(1 for match in matches if match.retired),
        "with_closing_odds": priced,
        "path": str(MATCHES_CSV),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build_spine(download="--offline" not in sys.argv), indent=2))
