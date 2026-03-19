from __future__ import annotations

"""
EDA script (self-contained).

Generates:
 - Figures in results/figures/
 - Tables in results/tables/

Notes:
 - Code comments are in English (per project request).
 - Plot titles/axis labels are in Ukrainian (per thesis plan requirements).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_DIR = Path("./results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _parse_datetime_maybe(value: Any) -> datetime | None:
    """
    Parses matchStartingDateTime which usually looks like:
      {"$date": "2024-11-30T22:45:00Z"}
    """
    if value is None:
        return None
    if isinstance(value, dict) and "$date" in value:
        value = value["$date"]
    if not isinstance(value, str):
        return None

    # Handle common Z-suffix.
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except Exception:
        return None


def _convert_mongo_number(value: Any) -> Any:
    """
    Converts MongoDB Extended JSON number wrappers like:
      {"$numberDouble": "Infinity"}
    """
    if isinstance(value, dict):
        if "$numberDouble" in value:
            raw = value["$numberDouble"]
            if raw == "Infinity" or raw == "+Infinity":
                return np.nan
            if raw == "-Infinity":
                return np.nan
            try:
                return float(raw)
            except Exception:
                return np.nan
        if "$numberInt" in value:
            try:
                return int(value["$numberInt"])
            except Exception:
                return np.nan
    return value


def _flatten_dict(d: Any, prefix: str, out: dict[str, Any]) -> None:
    """
    Flattens dict into keys separated by '|'.
    For example:
      team1|allAverages|gameAverages|adr
    """
    d = _convert_mongo_number(d)

    if isinstance(d, dict):
        for k, v in d.items():
            new_prefix = f"{prefix}|{k}" if prefix else str(k)
            _flatten_dict(v, new_prefix, out)
        return

    if isinstance(d, list):
        # Lists can exist, but for summaryStats we mostly expect dicts.
        # We flatten list items by index.
        for i, item in enumerate(d):
            new_prefix = f"{prefix}|{i}"
            _flatten_dict(item, new_prefix, out)
        return

    # Scalar / None.
    out[prefix] = d


def parse_json_to_df(json_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses JSON into:
      1) raw_df - flattened per-match team1/team2 features (columns start with team1|... / team2|...)
      2) meta_df - match metadata and target y
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of matches.")

    raw_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for match in data:
        match_id = match.get("matchId")
        no_prediction_reason = match.get("noPredictionReason", "")
        actual_winner_id = match.get("actualWinnerId")

        # Parse match date.
        match_dt = _parse_datetime_maybe(match.get("matchStartingDateTime"))

        summary = match.get("summaryStats")
        if not isinstance(summary, list) or len(summary) < 2:
            continue

        team1_stats = summary[0]
        team2_stats = summary[1]

        if not isinstance(team1_stats, dict) or not isinstance(team2_stats, dict):
            continue

        team1_id = team1_stats.get("pandaScoreTeamId")
        team2_id = team2_stats.get("pandaScoreTeamId")

        # Target definition.
        y: int | None
        if actual_winner_id is None or team1_id is None or team2_id is None:
            y = None
        elif actual_winner_id == team1_id:
            y = 1
        elif actual_winner_id == team2_id:
            y = 0
        else:
            y = None

        # Skip matches with unknown winner.
        if y is None:
            continue

        # Flatten summaryStats for both teams.
        raw_team1: dict[str, Any] = {}
        raw_team2: dict[str, Any] = {}
        _flatten_dict(team1_stats, "team1", raw_team1)
        _flatten_dict(team2_stats, "team2", raw_team2)

        # Extract powerRank per team from powerRanks list.
        power_rank_map: dict[Any, Any] = {}
        power_ranks = match.get("powerRanks", [])
        if isinstance(power_ranks, list):
            for pr in power_ranks:
                if isinstance(pr, dict) and "pandaScoreTeamId" in pr and "powerRank" in pr:
                    power_rank_map[pr["pandaScoreTeamId"]] = _convert_mongo_number(pr["powerRank"])

        raw_team1["team1|powerRank"] = power_rank_map.get(team1_id, np.nan)
        raw_team2["team2|powerRank"] = power_rank_map.get(team2_id, np.nan)

        # Flatten headToHeadStats if present.
        head_to_head = match.get("headToHeadStats", None)
        if isinstance(head_to_head, list) and len(head_to_head) >= 2:
            t1_h2h = head_to_head[0]
            t2_h2h = head_to_head[1]
            if isinstance(t1_h2h, dict):
                h2h_t1: dict[str, Any] = {}
                _flatten_dict(t1_h2h, "team1|headToHeadStats", h2h_t1)
                raw_team1.update(h2h_t1)
            if isinstance(t2_h2h, dict):
                h2h_t2: dict[str, Any] = {}
                _flatten_dict(t2_h2h, "team2|headToHeadStats", h2h_t2)
                raw_team2.update(h2h_t2)

        row_features: dict[str, Any] = {"matchId": match_id}
        row_features.update(raw_team1)
        row_features.update(raw_team2)
        raw_rows.append(row_features)

        meta_rows.append(
            {
                "matchId": match_id,
                "date": match_dt,
                "noPredictionReason": no_prediction_reason,
                "actualWinnerId": actual_winner_id,
                "y": y,
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    meta_df = pd.DataFrame(meta_rows)
    return raw_df, meta_df


def count_total_matches(json_path: str | Path) -> int:
    """
    Counts JSON root matches without filtering by target y.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return 0
    return len(data)


def _extract_team_numeric_series(df: pd.DataFrame, col_team1: str, col_team2: str) -> pd.Series:
    """
    Collects both team series and concatenates them for distribution plots.
    """
    s1 = pd.to_numeric(df.get(col_team1), errors="coerce")
    s2 = pd.to_numeric(df.get(col_team2), errors="coerce")
    return pd.concat([s1, s2], ignore_index=True)


def _reason_to_category(reason: str) -> str:
    """
    Maps PandaScore noPredictionReason into the thesis categories.
    """
    r = (reason or "").strip().lower()
    if r == "":
        return "Прогнозований"

    # Match against common notebook strings.
    if "equal" in r or "unpredictable" in r:
        return "Рівні команди"
    if "too long ago" in r or "played too long" in r or "latest match" in r:
        return "Застарілі дані"
    if "power rank" in r or "can't be calculated" in r or "rank" in r:
        return "Не вдалося обчислити ранг"
    if "banned" in r or "team banned" in r or "player banned" in r:
        return "Бан гравця"

    return "Інші"


def _savefig(path: Path) -> None:
    """
    Saves and closes a plot with thesis-friendly settings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _generate_fig_2_1_target_distribution(meta_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.1 — target distribution ({suffix})...")
    y = meta_df["y"].astype(int)
    counts = y.value_counts().sort_index()
    total = len(y)
    team1_wins = counts.get(1, 0)
    team2_wins = counts.get(0, 0)

    labels = ["Перемога team 1", "Перемога team 2"]
    values = [team1_wins, team2_wins]
    perc = [v / total * 100 if total else 0 for v in values]
    bar_labels = [f"{v} ({p:.1f}%)" for v, p in zip(values, perc)]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    for bar, txt in zip(bars, bar_labels):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), txt, ha="center", va="bottom")
    plt.title("Розподіл цільової змінної (y)")
    plt.ylabel("Кількість матчів")

    out_path = FIGURES_DIR / f"fig_2_1_target_distribution_{suffix}.png"
    _savefig(out_path)
    return str(out_path)


def _generate_fig_2_2_matches_over_time(meta_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.2 — matches over time ({suffix})...")
    if "date" not in meta_df.columns:
        raise ValueError("meta_df must contain 'date' column for fig 2.2.")

    df = meta_df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["week"] = df["date"].dt.to_period("W").astype(str)
    counts = df["week"].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    plt.plot(counts.index, counts.values, marker="o", linewidth=1)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.title("Розподіл матчів за часом")
    plt.xlabel("Тиждень")
    plt.ylabel("Кількість матчів")

    out_path = FIGURES_DIR / f"fig_2_2_matches_over_time_{suffix}.png"
    _savefig(out_path)
    return str(out_path)


def _generate_fig_2_3_prediction_reasons(meta_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.3 — noPredictionReason distribution ({suffix})...")
    reasons = meta_df.get("noPredictionReason", pd.Series([""] * len(meta_df)))
    cats = reasons.apply(_reason_to_category)
    counts = cats.value_counts()

    # Ensure order.
    order = ["Прогнозований", "Рівні команди", "Застарілі дані", "Не вдалося обчислити ранг", "Бан гравця", "Інші"]
    counts = counts.reindex([c for c in order if c in counts.index] + [c for c in counts.index if c not in order])

    plt.figure(figsize=(8, 4))
    plt.barh(counts.index, counts.values, color="#7A5C92")
    plt.title("Розподіл noPredictionReason")
    plt.xlabel("Кількість матчів")
    plt.ylabel("Категорія")

    out_path = FIGURES_DIR / f"fig_2_3_prediction_reasons_{suffix}.png"
    _savefig(out_path)
    return str(out_path)


def _generate_fig_2_4_feature_distributions(raw_df: pd.DataFrame, meta_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.4 — feature distributions ({suffix})...")
    # Required features by thesis prompt.
    features = [
        ("winPercentage", "team1|winPercentage", "team2|winPercentage"),
        ("totalMatchesPlayed", "team1|totalMatchesPlayed", "team2|totalMatchesPlayed"),
        ("powerRank", "team1|powerRank", "team2|powerRank"),
        ("adr", "team1|allAverages|gameAverages|adr", "team2|allAverages|gameAverages|adr"),
    ]
    ua_labels = {
        "winPercentage": "winPercentage (відсоток перемог)",
        "totalMatchesPlayed": "totalMatchesPlayed (зіграні матчі)",
        "powerRank": "powerRank",
        "adr": "ADR",
    }

    plt.figure(figsize=(12, 8))
    for i, (pretty, c1, c2) in enumerate(features, start=1):
        ax = plt.subplot(2, 2, i)
        values = _extract_team_numeric_series(raw_df, c1, c2)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        sns.histplot(values, bins=30, kde=False, ax=ax, color="#4C78A8")
        ax.set_title(f"Розподіл: {ua_labels.get(pretty, pretty)}")
        ax.set_xlabel(ua_labels.get(pretty, pretty))
        ax.set_ylabel("Кількість")

    plt.suptitle("Розподіл ключових ознак (team1 + team2)", y=1.02)
    out_path = FIGURES_DIR / f"fig_2_4_feature_distributions_{suffix}.png"
    _savefig(out_path)
    return str(out_path)


def _get_col_if_exists(df: pd.DataFrame, col: str) -> str | None:
    return col if col in df.columns else None


def _generate_fig_2_5_winner_vs_loser(raw_df: pd.DataFrame, meta_df: pd.DataFrame, suffix: str) -> list[str]:
    print(f"Generating fig 2.5 — winner vs loser distributions ({suffix})...")
    merged = meta_df.merge(raw_df, on="matchId", how="left")

    y = merged["y"].astype(int)
    winners_mask = y == 1  # team1 winner

    metric_specs = [
        ("winPercentage", "team1|winPercentage", "team2|winPercentage"),
        ("adr", "team1|allAverages|gameAverages|adr", "team2|allAverages|gameAverages|adr"),
        ("killDeathDiff", "team1|allAverages|gameAverages|killDeathDiff", "team2|allAverages|gameAverages|killDeathDiff"),
        ("kast", "team1|allAverages|gameAverages|kast", "team2|allAverages|gameAverages|kast"),
    ]

    out_path = FIGURES_DIR / f"fig_2_5_winner_vs_loser_{suffix}.png"
    plt.figure(figsize=(12, 7))

    for idx, (name, t1_col, t2_col) in enumerate(metric_specs, start=1):
        ax = plt.subplot(2, 2, idx)
        if t1_col not in merged.columns or t2_col not in merged.columns:
            ax.set_title(f"{name} (missing columns)")
            continue

        t1_vals = pd.to_numeric(merged[t1_col], errors="coerce")
        t2_vals = pd.to_numeric(merged[t2_col], errors="coerce")

        winner_vals = pd.concat([t1_vals[winners_mask], t2_vals[~winners_mask]], ignore_index=True)
        loser_vals = pd.concat([t2_vals[winners_mask], t1_vals[~winners_mask]], ignore_index=True)

        data = pd.DataFrame({"Група": ["Переможець"] * len(winner_vals) + ["Переможений"] * len(loser_vals), "Значення": pd.concat([winner_vals, loser_vals], ignore_index=True)})
        sns.boxplot(x="Група", y="Значення", data=data, ax=ax, palette=["#4C78A8", "#F58518"])
        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("Значення")

    plt.suptitle("Статистики переможців vs переможених", y=1.02)
    _savefig(out_path)
    return [str(out_path)]


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c == "matchId":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _generate_fig_2_6_top_correlations_diff_features(raw_df: pd.DataFrame, meta_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.6 — top correlations of diff features ({suffix})...")
    merged = meta_df.merge(raw_df, on="matchId", how="left")
    y = merged["y"].astype(float)

    # Compute team1 - team2 for matched numeric features.
    diff_values: dict[str, pd.Series] = {}
    for col in raw_df.columns:
        if not col.startswith("team1|"):
            continue
        suffix = col[len("team1|") :]
        col2 = f"team2|{suffix}"
        if col2 not in raw_df.columns:
            continue
        a = pd.to_numeric(merged[col], errors="coerce")
        b = pd.to_numeric(merged[col2], errors="coerce")
        if a.notna().sum() == 0 or b.notna().sum() == 0:
            continue
        diff_values[suffix] = a - b

    corrs: list[tuple[str, float]] = []
    for name, s in diff_values.items():
        df_tmp = pd.DataFrame({"diff": s, "y": y})
        df_tmp = df_tmp.dropna()
        if len(df_tmp) < 3:
            continue
        corr = df_tmp["diff"].corr(df_tmp["y"], method="pearson")
        if pd.isna(corr):
            continue
        corrs.append((name, float(corr)))

    if not corrs:
        raise RuntimeError("No correlations could be computed for fig 2.6.")

    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)[:20]
    labels = [name for name, _ in corrs_sorted]
    vals = [corr for _, corr in corrs_sorted]

    plt.figure(figsize=(10, 7))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, vals, color="#4C78A8")
    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel("Кореляція (pearson) з y")
    plt.title("Кореляція distance-ознак з target (top-20)")

    out_path = FIGURES_DIR / f"fig_2_6_top_correlations_{suffix}.png"
    plt.gca().invert_yaxis()
    _savefig(out_path)
    return str(out_path)


def _generate_fig_2_7_power_rank_diff(raw_df: pd.DataFrame, meta_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.7 — powerRank diff distribution ({suffix})...")
    merged = meta_df.merge(raw_df, on="matchId", how="left")

    t1 = pd.to_numeric(merged.get("team1|powerRank"), errors="coerce")
    t2 = pd.to_numeric(merged.get("team2|powerRank"), errors="coerce")
    diff = t1 - t2

    winners = diff[merged["y"].astype(int) == 1].replace([np.inf, -np.inf], np.nan).dropna()
    losers = diff[merged["y"].astype(int) == 0].replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(8, 4))
    sns.histplot(winners, bins=30, color="#4C78A8", label="Перемога team 1", alpha=0.6, kde=False)
    sns.histplot(losers, bins=30, color="#F58518", label="Перемога team 2", alpha=0.6, kde=False)
    plt.title("Розподіл різниці powerRank")
    plt.xlabel("powerRank_team1 - powerRank_team2 (різниця)")
    plt.ylabel("Кількість")
    plt.legend()

    out_path = FIGURES_DIR / f"fig_2_7_powerrank_diff_{suffix}.png"
    _savefig(out_path)
    return str(out_path)


def _generate_fig_2_8_missing_values(raw_df: pd.DataFrame, suffix: str) -> str:
    print(f"Generating fig 2.8 — missing values ({suffix})...")
    # Group columns by type (summaryStats vs headToHeadStats vs powerRank).
    group_specs = {
        "summaryStats (team features)": [c for c in raw_df.columns if "|headToHeadStats|" not in c and c not in {"team1|powerRank", "team2|powerRank", "matchId"}],
        "powerRank": [c for c in raw_df.columns if c in {"team1|powerRank", "team2|powerRank"}],
        "headToHeadStats": [c for c in raw_df.columns if "|headToHeadStats|" in c],
    }

    missing_counts: list[tuple[str, int]] = []
    for group_name, cols in group_specs.items():
        if not cols:
            missing_counts.append((group_name, 0))
            continue
        missing = 0
        for c in cols:
            missing += int(pd.to_numeric(raw_df[c], errors="coerce").isna().sum())
        missing_counts.append((group_name, missing))

    labels = [x[0] for x in missing_counts]
    vals = [x[1] for x in missing_counts]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals, color="#7A5C92")
    plt.title("Пропущені значення (NaN/Infinity/None)")
    plt.ylabel("Кількість пропусків")
    plt.xticks(rotation=30, ha="right")

    out_path = FIGURES_DIR / f"fig_2_8_missing_values_{suffix}.png"
    _savefig(out_path)
    return str(out_path)


def _generate_table_2_1_dataset_comparison(
    meta_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    total_matches: int,
    suffix: str,
) -> pd.DataFrame:
    """
    Builds dataset comparison table rows for later concatenation.
    """
    # This function expects meta/raw already filtered to known winners.
    accepted = len(meta_df)
    wins_team1 = int((meta_df["y"].astype(int) == 1).sum())
    wins_team2 = int((meta_df["y"].astype(int) == 0).sum())
    team1_win_pct = wins_team1 / accepted * 100 if accepted else 0

    # Time range.
    dts = pd.to_datetime(meta_df["date"], utc=True, errors="coerce").dropna()
    date_min = dts.min()
    date_max = dts.max()
    time_range = f"{date_min.date()} — {date_max.date()}" if not pd.isna(date_min) and not pd.isna(date_max) else "N/A"

    # Feature count (summaryStats only): approximate as the count of unique prefixes from flattened team1 summary keys.
    # We assume raw_df contains flattened summary keys directly under team1|... excluding headToHeadStats and powerRank.
    summary_cols = [c for c in raw_df.columns if c.startswith("team1|") and "|headToHeadStats|" not in c and c not in {"team1|powerRank", "team2|powerRank"}]
    team_features_count = len(summary_cols)

    # noPredictionReason categories: from raw meta_df reasons (can include accepted and rejected matches in prompt).
    # Here meta_df already filters only known winners, so we interpret "accepted matches (reason=='')" within meta_df.
    reasons = meta_df.get("noPredictionReason", pd.Series([""] * len(meta_df))).astype(str)
    predicted_mask = reasons == ""
    predicted_count = int(predicted_mask.sum())

    cat_counts = reasons.apply(_reason_to_category).value_counts()
    even = int(cat_counts.get("Рівні команди", 0))
    outdated = int(cat_counts.get("Застарілі дані", 0))
    rank_fail = int(cat_counts.get("Не вдалося обчислити ранг", 0))
    banned = int(cat_counts.get("Бан гравця", 0))

    row = {
        "Характеристика": None,
        suffix: None,
    }
    # We'll return a rowset (DataFrame) with fixed keys, then merge later.
    table = pd.DataFrame(
        [
            ("Загальна кількість матчів", total_matches),
            ("Матчі з відомим переможцем", accepted),
            ("Перемога team 1", wins_team1),
            ("Перемога team 2", wins_team2),
            ("Баланс класів (team1 win %)", team1_win_pct),
            ("Часовий діапазон", time_range),
            ("Ознак на команду (summaryStats)", team_features_count),
            ("Прогнозованих матчів (reason=='')", predicted_count),
            ("Рівні команди", even),
            ("Застарілі дані", outdated),
            ("Не вдалося обчислити ранг", rank_fail),
            ("Бан гравця", banned),
        ],
        columns=["Характеристика", suffix],
    )
    return table


def _feature_stats_table(raw_df: pd.DataFrame, meta_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Builds table with mean/std/min/percentiles/max for key features.
    """
    merged = meta_df.merge(raw_df, on="matchId", how="left")

    feature_pairs = [
        ("winPercentage", "team1|winPercentage", "team2|winPercentage"),
        ("totalMatchesPlayed", "team1|totalMatchesPlayed", "team2|totalMatchesPlayed"),
        ("powerRank", "team1|powerRank", "team2|powerRank"),
        ("adr", "team1|allAverages|gameAverages|adr", "team2|allAverages|gameAverages|adr"),
        ("kast", "team1|allAverages|gameAverages|kast", "team2|allAverages|gameAverages|kast"),
        ("kills", "team1|allAverages|gameAverages|kills", "team2|allAverages|gameAverages|kills"),
        ("deaths", "team1|allAverages|gameAverages|deaths", "team2|allAverages|gameAverages|deaths"),
        ("killDeathDiff", "team1|allAverages|gameAverages|killDeathDiff", "team2|allAverages|gameAverages|killDeathDiff"),
        ("firstKillsDiff", "team1|allAverages|gameAverages|firstKillsDiff", "team2|allAverages|gameAverages|firstKillsDiff"),
        ("headshots", "team1|allAverages|gameAverages|headshots", "team2|allAverages|gameAverages|headshots"),
        ("assists", "team1|allAverages|gameAverages|assists", "team2|allAverages|gameAverages|assists"),
        ("flashAssists", "team1|allAverages|gameAverages|flashAssists", "team2|allAverages|gameAverages|flashAssists"),
    ]

    rows: list[dict[str, Any]] = []
    for name, c1, c2 in feature_pairs:
        values = _extract_team_numeric_series(merged, c1, c2)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            rows.append(
                {
                    "Feature": name,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "25%": np.nan,
                    "50%": np.nan,
                    "75%": np.nan,
                    "max": np.nan,
                }
            )
            continue

        rows.append(
            {
                "Feature": name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "25%": float(values.quantile(0.25)),
                "50%": float(values.quantile(0.5)),
                "75%": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def run_dataset_eda(json_path: str | Path, suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df, meta_df = parse_json_to_df(json_path)

    # Ensure date exists.
    if "date" in meta_df.columns:
        meta_df["date"] = pd.to_datetime(meta_df["date"], utc=True, errors="coerce")

    figs_generated = 0
    figs_generated += 1 if _generate_fig_2_1_target_distribution(meta_df, suffix) else 0
    figs_generated += 1 if _generate_fig_2_2_matches_over_time(meta_df, suffix) else 0
    figs_generated += 1 if _generate_fig_2_3_prediction_reasons(meta_df, suffix) else 0
    figs_generated += 1 if _generate_fig_2_4_feature_distributions(raw_df, meta_df, suffix) else 0
    figs_generated += len(_generate_fig_2_5_winner_vs_loser(raw_df, meta_df, suffix))
    figs_generated += 1 if _generate_fig_2_6_top_correlations_diff_features(raw_df, meta_df, suffix) else 0
    figs_generated += 1 if _generate_fig_2_7_power_rank_diff(raw_df, meta_df, suffix) else 0
    figs_generated += 1 if _generate_fig_2_8_missing_values(raw_df, suffix) else 0

    return raw_df, meta_df


def _generate_comparative_figs(raw60: pd.DataFrame, meta60: pd.DataFrame, raw90: pd.DataFrame, meta90: pd.DataFrame) -> None:
    print("Generating comparative figures (fig 2.9 — 2.11) — 60d vs 90d...")

    # fig 2.9 distributions
    print("Generating fig 2.9 — 60d vs 90d feature distributions...")
    plt.figure(figsize=(12, 8))
    feature_plots = [
        ("winPercentage", "team1|winPercentage", "team2|winPercentage"),
        ("totalMatchesPlayed", "team1|totalMatchesPlayed", "team2|totalMatchesPlayed"),
        ("adr", "team1|allAverages|gameAverages|adr", "team2|allAverages|gameAverages|adr"),
        ("killDeathDiff", "team1|allAverages|gameAverages|killDeathDiff", "team2|allAverages|gameAverages|killDeathDiff"),
    ]
    ua_labels = {
        "winPercentage": "winPercentage (відсоток перемог)",
        "totalMatchesPlayed": "totalMatchesPlayed (зіграні матчі)",
        "adr": "ADR",
        "killDeathDiff": "killDeathDiff",
    }

    for i, (name, c1, c2) in enumerate(feature_plots, start=1):
        ax = plt.subplot(2, 2, i)
        v60 = _extract_team_numeric_series(raw60.merge(meta60, on="matchId", how="left"), c1, c2).dropna()
        v90 = _extract_team_numeric_series(raw90.merge(meta90, on="matchId", how="left"), c1, c2).dropna()
        sns.histplot(v60, bins=30, color="#4C78A8", alpha=0.6, ax=ax, kde=False, label="60d")
        sns.histplot(v90, bins=30, color="#F58518", alpha=0.6, ax=ax, kde=False, label="90d")
        ax.set_title(ua_labels.get(name, name))
        ax.set_xlabel(ua_labels.get(name, name))
        ax.set_ylabel("Кількість")

    plt.suptitle("Порівняння розподілів ознак 60 vs 90 днів", y=1.02)
    _savefig(FIGURES_DIR / "fig_2_9_60_vs_90_distributions.png")

    # fig 2.10 correlation comparison (difference features)
    print("Generating fig 2.10 — correlation comparison...")
    def compute_corrs(raw: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
        merged = meta.merge(raw, on="matchId", how="left")
        y = merged["y"].astype(float)
        corrs: dict[str, float] = {}
        for col in raw.columns:
            if not col.startswith("team1|"):
                continue
            suffix = col[len("team1|") :]
            col2 = f"team2|{suffix}"
            if col2 not in raw.columns:
                continue
            a = pd.to_numeric(merged[col], errors="coerce")
            b = pd.to_numeric(merged[col2], errors="coerce")
            diff = a - b
            df_tmp = pd.DataFrame({"diff": diff, "y": y}).dropna()
            if len(df_tmp) < 3:
                continue
            corr = df_tmp["diff"].corr(df_tmp["y"], method="pearson")
            if pd.isna(corr):
                continue
            corrs[suffix] = float(corr)
        return pd.Series(corrs)

    corr60 = compute_corrs(raw60, meta60)
    corr90 = compute_corrs(raw90, meta90)

    common = sorted(set(corr60.index) & set(corr90.index))
    if not common:
        return

    x = corr60[common].values
    y = corr90[common].values
    diff = np.abs(x - y)
    top_idx = np.argsort(diff)[::-1][:5]
    top_names = [common[i] for i in top_idx]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=15, alpha=0.7)
    lim = max(np.max(np.abs(x)), np.max(np.abs(y)), 0.01)
    plt.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    plt.xlabel("Кореляція (diff-feature vs y) у 60d")
    plt.ylabel("Кореляція (diff-feature vs y) у 90d")
    plt.title("Порівняння кореляцій: 60d vs 90d")

    for i in top_idx:
        # `i` indexes into `common`, and `top_names` aligns with `top_idx`.
        label = common[i]
        plt.annotate(label, (x[i], y[i]), fontsize=8)

    _savefig(FIGURES_DIR / "fig_2_10_correlation_comparison.png")

    # fig 2.11 missing comparison
    print("Generating fig 2.11 — missing comparison...")
    def group_missing(raw: pd.DataFrame) -> dict[str, int]:
        group_specs = {
            "summaryStats": [c for c in raw.columns if "|headToHeadStats|" not in c and c not in {"team1|powerRank", "team2|powerRank", "matchId"}],
            "powerRank": [c for c in raw.columns if c in {"team1|powerRank", "team2|powerRank"}],
            "headToHeadStats": [c for c in raw.columns if "|headToHeadStats|" in c],
        }
        out: dict[str, int] = {}
        for g, cols in group_specs.items():
            miss = 0
            for c in cols:
                miss += int(pd.to_numeric(raw[c], errors="coerce").isna().sum())
            out[g] = miss
        return out

    miss60 = group_missing(raw60)
    miss90 = group_missing(raw90)
    groups = list(miss60.keys())
    v60 = [miss60[g] for g in groups]
    v90 = [miss90[g] for g in groups]

    plt.figure(figsize=(10, 4))
    x_pos = np.arange(len(groups))
    width = 0.35
    plt.bar(x_pos - width / 2, v60, width, label="60d", color="#4C78A8")
    plt.bar(x_pos + width / 2, v90, width, label="90d", color="#F58518")
    plt.xticks(x_pos, groups, rotation=25, ha="right")
    plt.ylabel("Кількість пропусків")
    plt.title("Порівняння пропусків 60 vs 90 днів")
    plt.legend()
    _savefig(FIGURES_DIR / "fig_2_11_missing_comparison.png")


def main() -> None:
    _ensure_dirs()

    parser = argparse.ArgumentParser(description="EDA for CS2 match predictor datasets (60d vs 90d).")
    parser.add_argument("--json-60", type=str, default=None, help="Path to match_prediction_60_days JSON.")
    parser.add_argument("--json-90", type=str, default=None, help="Path to match_prediction_90_days JSON.")
    args = parser.parse_args()

    if not args.json_60 and not args.json_90:
        raise ValueError("Provide at least --json-60 or --json-90.")

    raw60 = meta60 = None
    raw90 = meta90 = None

    json60_path = Path(args.json_60) if args.json_60 else None
    json90_path = Path(args.json_90) if args.json_90 else None

    if json60_path is not None:
        raw60, meta60 = run_dataset_eda(json60_path, suffix="60d")
        # Tables for 60d.
        print("Generating table 2.2 — feature stats for 60d...")
        table_2_2 = _feature_stats_table(raw60, meta60, suffix="60d")
        table_2_2.to_csv(TABLES_DIR / "table_2_2_feature_stats_60d.csv", index=False)

    if json90_path is not None:
        raw90, meta90 = run_dataset_eda(json90_path, suffix="90d")
        # Tables for 90d.
        print("Generating table 2.3 — feature stats for 90d...")
        table_2_3 = _feature_stats_table(raw90, meta90, suffix="90d")
        table_2_3.to_csv(TABLES_DIR / "table_2_3_feature_stats_90d.csv", index=False)

    # Comparative figures/tables if both provided.
    if json60_path is not None and json90_path is not None:
        _generate_comparative_figs(raw60, meta60, raw90, meta90)

        print("Generating table 2.1 — dataset comparison...")
        total60 = count_total_matches(json60_path)
        total90 = count_total_matches(json90_path)
        t60 = _generate_table_2_1_dataset_comparison(meta60, raw60, total60, suffix="60 днів")
        t90 = _generate_table_2_1_dataset_comparison(meta90, raw90, total90, suffix="90 днів")
        comp = t60.merge(t90, on="Характеристика", how="outer")
        comp.to_csv(TABLES_DIR / "table_2_1_dataset_comparison.csv", index=False)

    # Final summary.
    figs = sorted(FIGURES_DIR.glob("fig_2_*.png"))
    print(f"EDA summary: generated {len(figs)} figures.")
    print(f"Figures folder: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()

