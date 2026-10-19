from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.binary_filter import apply_filter, create_filter_features, create_filter_target, train_binary_filter
from src.data_loader import load_matches
from src.features import compute_team_distances_augmented, difference_distance

RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = RESULTS_DIR / "models"
TEAM_NAMES_CSV = DATA_DIR / "team_names.csv"

# Demo names for UI when raw JSON does not provide clean team names.
# We will map these names to a small subset of available team IDs.
DEMO_TEAM_NAMES = [
    "Natus Vincere",
    "G2 Esports",
    "FaZe Clan",
    "Team Vitality",
    "Team Spirit",
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _show_figure_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        # Render images in a narrower center column for better readability.
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.image(str(path), caption=caption, width=900)


def _dataset_json_path(dataset: str) -> Path:
    if dataset == "60d":
        return DATA_DIR / "match_prediction_60_days_31_july.json"
    return DATA_DIR / "match_prediction_90_days_31_july.json"


def _dataset_model_dir(dataset: str) -> Path:
    return MODELS_DIR / dataset


@st.cache_data(show_spinner=False)
def _load_team_name_map_from_csv() -> dict[int, str]:
    """
    Optional mapping from team_id to readable team name.
    Expected file: data/team_names.csv with columns: team_id, team_name
    """
    if not TEAM_NAMES_CSV.exists():
        return {}
    try:
        df = pd.read_csv(TEAM_NAMES_CSV)
    except Exception:
        return {}
    if "team_id" not in df.columns or "team_name" not in df.columns:
        return {}
    mapping: dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            tid = int(r["team_id"])
            tname = str(r["team_name"]).strip()
            if tname:
                mapping[tid] = tname
        except Exception:
            continue
    return mapping


def _extract_team_name_map_from_raw(df_raw: pd.DataFrame) -> dict[int, str]:
    """
    Try to infer team names directly from JSON-derived columns.
    """
    mapping: dict[int, str] = {}
    if df_raw.empty:
        return mapping

    t1_id_col = "match|summaryStats|team1|pandaScoreTeamId"
    t2_id_col = "match|summaryStats|team2|pandaScoreTeamId"
    if t1_id_col not in df_raw.columns or t2_id_col not in df_raw.columns:
        return mapping

    # Candidate name-like fields.
    cand_t1_cols = [
        c for c in df_raw.columns if "team1|" in c and any(x in c.lower() for x in ["name", "slug", "acronym"])
    ]
    cand_t2_cols = [
        c.replace("team1|", "team2|", 1) for c in cand_t1_cols if c.replace("team1|", "team2|", 1) in df_raw.columns
    ]

    if not cand_t1_cols:
        return mapping

    for _, row in df_raw.iterrows():
        t1_id = pd.to_numeric(row.get(t1_id_col), errors="coerce")
        t2_id = pd.to_numeric(row.get(t2_id_col), errors="coerce")

        def _first_name(cols: list[str], row: pd.Series = row) -> str | None:
            for col in cols:
                val = row.get(col)
                if isinstance(val, str):
                    v = val.strip()
                    if v:
                        return v
            return None

        n1 = _first_name(cand_t1_cols)
        n2 = _first_name(cand_t2_cols)

        if pd.notna(t1_id) and n1:
            mapping[int(t1_id)] = n1
        if pd.notna(t2_id) and n2:
            mapping[int(t2_id)] = n2

    return mapping


@st.cache_data(show_spinner=False)
def _load_team_name_map(dataset: str) -> dict[int, str]:
    """
    Build team name mapping with priority:
    1) inferred from dataset JSON columns
    2) optional overrides from data/team_names.csv
    """
    df_raw = _load_raw_df(dataset)
    inferred = _extract_team_name_map_from_raw(df_raw)
    csv_map = _load_team_name_map_from_csv()

    merged = dict(inferred)
    merged.update(csv_map)  # explicit CSV overrides inferred names
    return merged


def _build_demo_name_map(team_ids: list[int], name_map: dict[int, str]) -> dict[int, str]:
    """
    Build compact UI mapping:
    - keep existing named teams first
    - if not enough names, assign DEMO_TEAM_NAMES to first available IDs
    - return only named teams (hide raw ID-only entries)
    """
    out: dict[int, str] = {}

    # Keep already named teams.
    for tid in team_ids:
        if tid in name_map and str(name_map[tid]).strip():
            out[tid] = str(name_map[tid]).strip()

    # Fill missing with demo names.
    used_names = set(out.values())
    demo_iter = [n for n in DEMO_TEAM_NAMES if n not in used_names]
    unnamed_ids = [tid for tid in team_ids if tid not in out]
    for tid, demo_name in zip(unnamed_ids, demo_iter):
        out[tid] = demo_name

    # Only show a compact list (up to demo count).
    limited_ids = list(out.keys())[: len(DEMO_TEAM_NAMES)]
    return {tid: out[tid] for tid in limited_ids}


def _find_known_accepted_pair(df_raw: pd.DataFrame) -> tuple[int, int] | None:
    """
    Pick a real accepted match pair directly from dataset rows:
      noPredictionReason == ""
    """
    if df_raw.empty:
        return None

    reason_col = None
    if "match|noPredictionReason" in df_raw.columns:
        reason_col = "match|noPredictionReason"
    elif "noPredictionReason" in df_raw.columns:
        reason_col = "noPredictionReason"
    if reason_col is None:
        return None

    t1_col = "match|summaryStats|team1|pandaScoreTeamId"
    t2_col = "match|summaryStats|team2|pandaScoreTeamId"
    if t1_col not in df_raw.columns or t2_col not in df_raw.columns:
        return None

    accepted_rows = df_raw[df_raw[reason_col].fillna("").astype(str).str.strip() == ""]
    for _, row in accepted_rows.iterrows():
        t1 = pd.to_numeric(row.get(t1_col), errors="coerce")
        t2 = pd.to_numeric(row.get(t2_col), errors="coerce")
        if pd.notna(t1) and pd.notna(t2) and int(t1) != int(t2):
            return int(t1), int(t2)
    return None


def _find_filter_accepted_pair_and_ids(
    dataset: str,
    df_raw: pd.DataFrame,
) -> tuple[tuple[int, int] | None, set[int]]:
    """
    Run trained filter on all dataset rows and return:
    - one accepted pair (team1_id, team2_id)
    - all accepted team IDs encountered in accepted rows
    """
    filter_info = _train_filter_cached(dataset)
    if filter_info is None or df_raw.empty:
        return None, set()

    t1_col = "match|summaryStats|team1|pandaScoreTeamId"
    t2_col = "match|summaryStats|team2|pandaScoreTeamId"
    if t1_col not in df_raw.columns or t2_col not in df_raw.columns:
        return None, set()

    X_all = create_filter_features(df_raw)
    accepted_pred = apply_filter(X_all, filter_info["best_model"], filter_info["best_scaler"])
    accepted_mask = accepted_pred == 1
    if accepted_mask.sum() == 0:
        return None, set()

    accepted_rows = df_raw.loc[accepted_mask]
    accepted_ids: set[int] = set()
    accepted_pair: tuple[int, int] | None = None

    for _, row in accepted_rows.iterrows():
        t1 = pd.to_numeric(row.get(t1_col), errors="coerce")
        t2 = pd.to_numeric(row.get(t2_col), errors="coerce")
        if pd.notna(t1):
            accepted_ids.add(int(t1))
        if pd.notna(t2):
            accepted_ids.add(int(t2))
        if accepted_pair is None and pd.notna(t1) and pd.notna(t2) and int(t1) != int(t2):
            accepted_pair = (int(t1), int(t2))

    return accepted_pair, accepted_ids


def _find_filter_pair_by_label(
    dataset: str,
    df_raw: pd.DataFrame,
    label: int,
    exclude_ids: set[int] | None = None,
) -> tuple[tuple[int, int] | None, pd.DataFrame | None]:
    """
    Find one real row whose Binary-Filter prediction equals `label`.
    Returns ((team1_id, team2_id), row_df).
    """
    exclude_ids = exclude_ids or set()
    filter_info = _train_filter_cached(dataset)
    if filter_info is None or df_raw.empty:
        return None, None

    t1_col = "match|summaryStats|team1|pandaScoreTeamId"
    t2_col = "match|summaryStats|team2|pandaScoreTeamId"
    if t1_col not in df_raw.columns or t2_col not in df_raw.columns:
        return None, None

    X_all = create_filter_features(df_raw)
    pred = apply_filter(X_all, filter_info["best_model"], filter_info["best_scaler"])
    mask = pred == int(label)
    if mask.sum() == 0:
        return None, None

    for idx, row in df_raw.loc[mask].iterrows():
        t1 = pd.to_numeric(row.get(t1_col), errors="coerce")
        t2 = pd.to_numeric(row.get(t2_col), errors="coerce")
        if pd.isna(t1) or pd.isna(t2):
            continue
        t1i, t2i = int(t1), int(t2)
        if t1i == t2i:
            continue
        if t1i in exclude_ids or t2i in exclude_ids:
            continue
        return (t1i, t2i), df_raw.loc[[idx]].copy()

    # Fallback: return first match even if IDs overlap.
    first_idx = df_raw.loc[mask].index[0]
    row = df_raw.loc[first_idx]
    t1 = pd.to_numeric(row.get(t1_col), errors="coerce")
    t2 = pd.to_numeric(row.get(t2_col), errors="coerce")
    if pd.notna(t1) and pd.notna(t2) and int(t1) != int(t2):
        return (int(t1), int(t2)), df_raw.loc[[first_idx]].copy()
    return None, None


@st.cache_data(show_spinner=False)
def _load_raw_df(dataset: str) -> pd.DataFrame:
    json_path = _dataset_json_path(dataset)
    if not json_path.exists():
        return pd.DataFrame()
    return load_matches(json_path=json_path, header_path=None)


def _build_team_profiles(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build mean profile per team_id from team1/team2 columns.
    """
    if df_raw.empty:
        return pd.DataFrame(), []

    t1_id_col = "match|summaryStats|team1|pandaScoreTeamId"
    t2_id_col = "match|summaryStats|team2|pandaScoreTeamId"
    if t1_id_col not in df_raw.columns or t2_id_col not in df_raw.columns:
        return pd.DataFrame(), []

    team1_cols = [c for c in df_raw.columns if "team1|" in c]
    suffixes = []
    for c1 in team1_cols:
        suffix = c1.split("team1|", 1)[1]
        c2 = c1.replace("team1|", "team2|", 1)
        if c2 in df_raw.columns:
            suffixes.append(suffix)
    suffixes = sorted(set(suffixes))

    rows = []
    for _, row in df_raw.iterrows():
        t1_id = pd.to_numeric(row.get(t1_id_col), errors="coerce")
        t2_id = pd.to_numeric(row.get(t2_id_col), errors="coerce")

        if pd.notna(t1_id):
            r = {"team_id": int(t1_id)}
            for s in suffixes:
                r[s] = pd.to_numeric(row.get(f"match|summaryStats|team1|{s}", np.nan), errors="coerce")
            # add powerRank from powerRanks branch if exists
            if "match|powerRanks|team1|powerRank" in df_raw.columns:
                r["powerRank"] = pd.to_numeric(row.get("match|powerRanks|team1|powerRank"), errors="coerce")
            rows.append(r)
        if pd.notna(t2_id):
            r = {"team_id": int(t2_id)}
            for s in suffixes:
                r[s] = pd.to_numeric(row.get(f"match|summaryStats|team2|{s}", np.nan), errors="coerce")
            if "match|powerRanks|team2|powerRank" in df_raw.columns:
                r["powerRank"] = pd.to_numeric(row.get("match|powerRanks|team2|powerRank"), errors="coerce")
            rows.append(r)

    if not rows:
        return pd.DataFrame(), []

    team_df = pd.DataFrame(rows)
    num_cols = [c for c in team_df.columns if c != "team_id"]
    team_profiles = team_df.groupby("team_id", as_index=False)[num_cols].mean(numeric_only=True)
    return team_profiles, suffixes


@st.cache_resource(show_spinner=False)
def _train_filter_cached(dataset: str) -> dict[str, Any] | None:
    df_raw = _load_raw_df(dataset)
    if df_raw.empty:
        return None
    Xf = create_filter_features(df_raw)
    yf = create_filter_target(df_raw)
    if len(np.unique(yf)) < 2:
        return None
    return train_binary_filter(Xf, yf, dataset_name=dataset)


def _predict_match(
    dataset: str,
    team1_id: int,
    team2_id: int,
    model_name: str,
    pair_df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    df_raw = _load_raw_df(dataset)
    profiles, suffixes = _build_team_profiles(df_raw)
    if profiles.empty:
        return {"error": "No team profiles could be built from dataset."}

    row1 = profiles[profiles["team_id"] == team1_id]
    row2 = profiles[profiles["team_id"] == team2_id]
    if row1.empty or row2.empty:
        return {"error": "One of selected teams has no profile."}

    row1 = row1.iloc[0]
    row2 = row2.iloc[0]

    if pair_df_override is not None:
        pair_df = pair_df_override.copy()
    else:
        # Build one synthetic match row in team1/team2 format expected by features module.
        pair = {}
        for s in suffixes:
            pair[f"match|summaryStats|team1|{s}"] = row1.get(s, np.nan)
            pair[f"match|summaryStats|team2|{s}"] = row2.get(s, np.nan)
        pair["match|powerRanks|team1|powerRank"] = row1.get("powerRank", np.nan)
        pair["match|powerRanks|team2|powerRank"] = row2.get("powerRank", np.nan)
        pair["match|summaryStats|team1|pandaScoreTeamId"] = team1_id
        pair["match|summaryStats|team2|pandaScoreTeamId"] = team2_id
        pair["y"] = 1  # placeholder
        pair_df = pd.DataFrame([pair])

    # Filter decision.
    filter_info = _train_filter_cached(dataset)
    if filter_info is not None:
        Xf_one = create_filter_features(pair_df)
        accepted = int(apply_filter(Xf_one, filter_info["best_model"], filter_info["best_scaler"])[0])
        filter_model_name = filter_info["best_model_name"]
    else:
        accepted = 1
        filter_model_name = "N/A"

    # Predictor decision.
    ds_model_dir = _dataset_model_dir(dataset)
    scaler_path = ds_model_dir / "predictor_scaler.pkl"
    model_path = ds_model_dir / f"{model_name.replace(' ', '_')}_best_model.pkl"
    pca_path = ds_model_dir / f"{model_name.replace(' ', '_')}_best_pca.pkl"
    if not scaler_path.exists() or not model_path.exists() or not pca_path.exists():
        return {
            "error": f"Missing artifacts for {dataset}/{model_name}. Run run_experiments first.",
            "accepted": accepted,
            "filter_model_name": filter_model_name,
        }

    scaler = joblib.load(scaler_path)
    predictor = joblib.load(model_path)
    pca = joblib.load(pca_path)

    _, _, orig_distances, *_ = compute_team_distances_augmented(pair_df, difference_distance)
    X = orig_distances.drop(columns=["y"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
    if hasattr(scaler, "feature_names_in_"):
        X = X.reindex(columns=list(scaler.feature_names_in_)).fillna(0.0)

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    pred = int(predictor.predict(X_pca)[0])

    proba = None
    if hasattr(predictor, "predict_proba"):
        try:
            proba = float(predictor.predict_proba(X_pca)[0, 1])
        except Exception:
            proba = None

    return {
        "accepted": accepted,
        "filter_model_name": filter_model_name,
        "model_name": model_name,
        "predicted_winner": team1_id if pred == 1 else team2_id,
        "predicted_label": pred,
        "proba_team1_win": proba,
    }


def render_filter_tab() -> None:
    st.subheader("Binary-Filter")
    st.caption("Shows coverage, filter model metrics, and filter confusion matrices.")

    dataset = st.selectbox("Dataset", ["60d", "90d"], key="filter_dataset")

    table_path = TABLES_DIR / f"binary_filter_results_{dataset}.csv"
    df = _read_csv(table_path)
    if df is not None and not df.empty:
        st.markdown("#### Model metrics")
        st.dataframe(df, use_container_width=True)
        if "coverage" in df.columns:
            coverage = float(df["coverage"].iloc[0])
            st.metric("Coverage", f"{coverage:.1%}")
    else:
        st.warning(f"Table not found or empty: `{table_path.name}`")

    st.markdown("#### Visualizations")
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_1_filter_model_comparison_{dataset}.png",
        f"Filter model comparison ({dataset})",
    )
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_4_confusion_matrix_filter_{dataset}.png",
        f"Filter confusion matrix ({dataset})",
    )
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_7_feature_importance_filter_{dataset}.png",
        f"Filter feature importance ({dataset})",
    )


def render_predictor_tab() -> None:
    st.subheader("Match-Predictor")
    st.caption("Compares baseline vs proposed (with filter), plus predictor diagnostics.")

    dataset = st.selectbox("Dataset", ["60d", "90d"], key="pred_dataset")
    table_path = TABLES_DIR / f"baseline_vs_proposed_{dataset}.csv"
    df = _read_csv(table_path)

    if df is None or df.empty:
        st.warning(f"Table not found or empty: `{table_path.name}`")
    else:
        st.markdown("#### Baseline vs Proposed table")
        st.dataframe(df, use_container_width=True)

        # Show top model by proposed accuracy.
        df_sorted = df.sort_values("proposed_accuracy", ascending=False).reset_index(drop=True)
        best = df_sorted.iloc[0]
        st.markdown("#### Best model summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", str(best.get("model", "N/A")))
        col2.metric("Baseline accuracy", f"{float(best.get('baseline_accuracy', 0.0)):.4f}")
        prop = best.get("proposed_accuracy")
        col3.metric(
            "Proposed accuracy",
            "NaN" if pd.isna(prop) else f"{float(prop):.4f}",
        )

    st.markdown("#### Visualizations")
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_2_baseline_vs_proposed_{dataset}.png",
        f"Baseline vs Proposed ({dataset})",
    )
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_3_pca_accuracy_curves_{dataset}.png",
        f"PCA accuracy curves ({dataset})",
    )
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_5_confusion_matrix_best_model_{dataset}.png",
        f"Best predictor confusion matrix ({dataset})",
    )
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_6_roc_curves_{dataset}.png",
        f"ROC curves ({dataset})",
    )
    _show_figure_if_exists(
        FIGURES_DIR / f"fig_3_8_feature_importance_predictor_{dataset}.png",
        f"Predictor feature importance ({dataset})",
    )


def render_pipeline_tab() -> None:
    st.subheader("Pipeline & Artifacts")
    st.caption("High-level view of how the two-stage system works and where outputs are stored.")

    st.markdown(
        """
1. **Load + preprocess JSON**
2. **Binary-Filter** classifies matches as accepted/rejected
3. **Match-Predictor** runs baseline on all test matches
4. **Proposed mode** evaluates predictor only on accepted matches
5. Save tables and figures to `results/`
"""
    )

    st.markdown("#### Global figures")
    _show_figure_if_exists(
        FIGURES_DIR / "fig_3_9_coverage_vs_accuracy.png",
        "Coverage vs accuracy",
    )
    _show_figure_if_exists(
        FIGURES_DIR / "fig_3_10_60d_vs_90d_comparison.png",
        "60d vs 90d comparison",
    )

    st.markdown("#### Core tables")
    best_models = _read_csv(TABLES_DIR / "best_models_summary.csv")
    coverage = _read_csv(TABLES_DIR / "coverage_vs_accuracy.csv")
    if best_models is not None:
        st.markdown("`best_models_summary.csv`")
        st.dataframe(best_models, use_container_width=True)
    if coverage is not None:
        st.markdown("`coverage_vs_accuracy.csv`")
        st.dataframe(coverage, use_container_width=True)

    st.markdown("#### Artifacts browser")
    if RESULTS_DIR.exists():
        files = sorted([p for p in RESULTS_DIR.rglob("*") if p.is_file()])
        show_n = st.slider("Number of files to show", min_value=10, max_value=200, value=40, step=10)
        rows = [
            {"path": str(p.relative_to(ROOT_DIR)), "size_kb": round(p.stat().st_size / 1024, 2)} for p in files[:show_n]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("`results/` folder not found.")


def render_simulator_tab() -> None:
    st.subheader("Live Match Simulator")
    st.caption("Choose two teams and see how Binary-Filter + Predictor process the match.")

    dataset = st.selectbox("Dataset", ["60d", "90d"], key="sim_dataset")
    df_raw = _load_raw_df(dataset)
    if df_raw.empty:
        st.error(f"Dataset file is missing or empty: `{_dataset_json_path(dataset)}`")
        return

    profiles, _ = _build_team_profiles(df_raw)
    if profiles.empty:
        st.error("Could not build team profiles from dataset.")
        return

    # Build two demo cases: one ACCEPTED and one REJECTED.
    accepted_example, accepted_row_df = _find_filter_pair_by_label(dataset, df_raw, label=1)
    if accepted_example is None:
        # fallback via noPredictionReason if model-based search failed
        accepted_example = _find_known_accepted_pair(df_raw)
        accepted_row_df = None

    if accepted_example is None:
        st.error("Не знайдено accepted-кейс в поточному датасеті.")
        return

    rejected_example, rejected_row_df = _find_filter_pair_by_label(
        dataset,
        df_raw,
        label=0,
        exclude_ids=set(accepted_example),
    )
    if rejected_example is None:
        rejected_example, rejected_row_df = _find_filter_pair_by_label(dataset, df_raw, label=0)

    # STRICT DEMO MODE:
    # - show only 2 demo pairs
    # - fixed readable names
    default_t1, default_t2 = accepted_example
    name_map: dict[int, str] = {
        default_t1: "Natus Vincere",
        default_t2: "FaZe Clan",
    }
    if rejected_example is not None:
        r1, r2 = rejected_example
        if r1 not in name_map:
            name_map[r1] = "Team Spirit"
        if r2 not in name_map:
            name_map[r2] = "G2 Esports"
    team_ids = sorted(name_map.keys())

    def _fmt_team(tid: int) -> str:
        return str(name_map.get(tid, f"Team {tid}"))

    if "sim_team1" not in st.session_state:
        st.session_state["sim_team1"] = default_t1
    if "sim_team2" not in st.session_state:
        st.session_state["sim_team2"] = default_t2

    col1, col2 = st.columns(2)
    with col1:
        team1_id = st.selectbox(
            "Team 1",
            team_ids,
            key="sim_team1",
            format_func=_fmt_team,
        )
    with col2:
        team2_id = st.selectbox(
            "Team 2",
            team_ids,
            key="sim_team2",
            format_func=_fmt_team,
        )

    ds_model_dir = _dataset_model_dir(dataset)
    model_names = []
    for p in ds_model_dir.glob("*_best_model.pkl"):
        model_names.append(p.name.replace("_best_model.pkl", "").replace("_", " "))
    model_names = sorted(set(model_names))
    if not model_names:
        st.warning(f"No trained predictor models found in `{ds_model_dir}`.")
        return
    model_name = st.selectbox("Predictor model", model_names, key="sim_model")

    if st.button("Run pipeline for selected teams", type="primary"):
        if team1_id == team2_id:
            st.warning("Select two different teams.")
            return
        with st.spinner("Running filter and predictor..."):
            # Always use exact dataset rows for known demo pairs.
            pair_override = None
            if (
                accepted_row_df is not None
                and accepted_example is not None
                and team1_id == accepted_example[0]
                and team2_id == accepted_example[1]
            ):
                pair_override = accepted_row_df
            elif (
                rejected_example is not None
                and rejected_row_df is not None
                and team1_id == rejected_example[0]
                and team2_id == rejected_example[1]
            ):
                pair_override = rejected_row_df

            out = _predict_match(
                dataset,
                team1_id,
                team2_id,
                model_name,
                pair_df_override=pair_override,
            )

        if "error" in out:
            st.error(out["error"])
            return

        st.markdown("#### Step 1 — Binary-Filter")
        accepted = int(out["accepted"])
        st.write(f"Filter model: `{out['filter_model_name']}`")
        st.metric("Filter decision", "ACCEPTED ✅" if accepted == 1 else "REJECTED ❌")

        st.markdown("#### Step 2 — Match-Predictor")
        if accepted == 0:
            st.info("Match rejected by filter. Predictor result shown for inspection only.")
        st.write(f"Predictor model: `{out['model_name']}`")
        winner_id = int(out["predicted_winner"])
        winner_label = _fmt_team(winner_id)
        st.metric("Predicted winner", winner_label)
        if out.get("proba_team1_win") is not None:
            st.metric("Team1 win probability", f"{out['proba_team1_win']:.2%}")

    st.markdown("#### How it works")
    st.write(
        "- Select two teams and a predictor model.\n"
        "- Binary-Filter checks if the match is valid for prediction (ACCEPTED or REJECTED).\n"
        "- Match-Predictor returns the winner and Team 1 win probability."
    )


def main() -> None:
    st.set_page_config(page_title="CS2 Two-Stage Pipeline", layout="wide")
    st.title("CS2 Match Prediction UI")
    st.caption("Binary-Filter + Match-Predictor dashboard")

    if not RESULTS_DIR.exists():
        st.error("`results/` folder is missing. Run experiments first.")
        return

    # Keep UI focused on simulator only.
    render_simulator_tab()


if __name__ == "__main__":
    main()
