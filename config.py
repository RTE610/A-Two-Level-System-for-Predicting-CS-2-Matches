from __future__ import annotations

from pathlib import Path

# Paths (relative to repo root)
DATA_DIR = Path("./data")
MODELS_DIR = Path("./models")

# Training params
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_PCA_COMPONENTS = 181

# Preprocessing params
POWER_RANK_COLUMNS = [
    "match|powerRanks|team1|powerRank",
    "match|powerRanks|team2|powerRank",
]

COLUMNS_TO_DROP = [
    "match|actualWinnerId",
    "match|predictedWinnerId",
    "match|summaryStats|team1|pandaScoreTeamId",
    "match|summaryStats|team2|pandaScoreTeamId",
    "match|powerRanks|team1|powerRank|$numberDouble",
    "match|powerRanks|team2|powerRank|$numberDouble",
]

# Patterns to remove potentially leaky/technical features before distance creation.
EXCLUDE_PATTERNS = [
    "pandaScoreTeamId",         # Team IDs are leakage-prone.
    "gamesDraw",                # Near-constant feature, causes tree artifacts.
    "predictionBans",           # Internal PandaScore fields.
    "matchId",                  # Match identifier.
    "_id",                      # MongoDB identifier.
    "$oid",                     # MongoDB identifier wrapper.
    "$date",                    # Raw datetime wrappers.
    "createdAt",                # Technical metadata.
    "updatedAt",                # Technical metadata.
    "predictedWinnerId",        # External prediction leakage.
    "actualWinnerId",           # Target leakage.
    "noPredictionReason",       # Binary-filter target, not predictor feature.
    "powerRankingCalculator",   # Version/metadata field.
    "powerRankingCoefficients", # Version/metadata field.
]


def get_classifiers(
    random_state: int = RANDOM_STATE,
    fast: bool = False,
    use_gpu: bool = False,
):
    """
    Return the classifiers dict used for PCA sweep.
    Keys must match `get_best_pca_settings()` parsing.
    """

    # Import here to avoid importing heavy ML libs during simple tooling.
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    rf_estimators = 200 if not fast else 80
    gb_estimators = 100 if not fast else 60
    lgbm_estimators = 100 if not fast else 60
    xgb_estimators = 100 if not fast else 60
    mlp_max_iter = 5000 if not fast else 700

    xgb_params = {
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_estimators": xgb_estimators,
    }
    if use_gpu:
        # Works on recent xgboost builds.
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = "cuda"

    lgbm_params = {
        "random_state": random_state,
        "n_estimators": lgbm_estimators,
        "verbosity": -1,  # suppress training spam in PCA sweeps
    }
    if use_gpu:
        lgbm_params["device_type"] = "gpu"

    cat_params = {
        "verbose": 0,
        "random_state": random_state,
    }
    if use_gpu:
        cat_params["task_type"] = "GPU"

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000 if fast else 5000, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_estimators=rf_estimators, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state, n_estimators=gb_estimators),
        "Support Vector Machine": SVC(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(**xgb_params),
        "LightGBM": LGBMClassifier(**lgbm_params),
        "CatBoost": CatBoostClassifier(**cat_params),
        "k-NN": KNeighborsClassifier(n_jobs=-1),
        "Neural Network": MLPClassifier(max_iter=mlp_max_iter, random_state=random_state),
    }

