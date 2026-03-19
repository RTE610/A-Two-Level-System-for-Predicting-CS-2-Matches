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


def get_classifiers(random_state: int = RANDOM_STATE):
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

    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Support Vector Machine": SVC(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
        # NOTE: `use_label_encoder` is deprecated in recent xgboost versions.
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state),
        "LightGBM": LGBMClassifier(random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
        "k-NN": KNeighborsClassifier(),
        "Neural Network": MLPClassifier(max_iter=5000, random_state=random_state),
    }

