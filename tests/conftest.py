from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "smoke: fast end-to-end checks of the deployed service")
    config.addinivalue_line("markers", "nfr: non-functional requirement checks")


@pytest.fixture(scope="session")
def league(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Synthetic league: prepared data dir + a trained two-stage bundle (ultra mode)."""
    import json

    import joblib
    import run_temporal

    from src.splits import temporal_split
    from tests.fixtures import synthetic_league, write_league_data_dir

    records, strength = synthetic_league()
    root = tmp_path_factory.mktemp("league")
    data_dir = root / "data"
    write_league_data_dir(data_dir, records, strength)

    train_json = root / "train.json"
    train_json.write_text(json.dumps(json.loads((data_dir / "matches_v2.json").read_text())))
    frames = run_temporal.build_frames(train_json)
    result, _ = run_temporal.evaluate_two_stage(frames, temporal_split(frames.dates), "ultra", root / "out")
    bundle_path = root / "out" / "bundle.joblib"
    return {
        "data_dir": data_dir,
        "bundle_path": bundle_path,
        "bundle": joblib.load(bundle_path),
        "strength": strength,
        "result": result,
    }


@pytest.fixture(scope="session")
def client(league: dict[str, Any]) -> Any:
    from fastapi.testclient import TestClient

    from services.api.app import LoadedModel, create_app
    from src.data_source import SnapshotDataSource

    app = create_app(
        LoadedModel(league["bundle"], version="test/v1"),
        SnapshotDataSource(league["data_dir"]),
    )
    return TestClient(app)
