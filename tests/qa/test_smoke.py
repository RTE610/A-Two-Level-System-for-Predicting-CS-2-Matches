"""Smoke tests: the service starts and the main paths answer. Also run against a live URL with SMOKE_URL."""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def api(client: Any) -> Any:
    url = os.environ.get("SMOKE_URL")
    if not url:
        return client
    import httpx

    return httpx.Client(base_url=url, timeout=10)


def test_tc19_health(api: Any) -> None:
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_tc19_schedule_and_predict(api: Any) -> None:
    schedule = api.get("/schedule").json()
    assert isinstance(schedule, list) and schedule
    teams = api.get("/teams", params={"limit": 2}).json()
    assert len(teams) == 2
    response = api.post("/predict", json={"team1_id": teams[0]["id"], "team2_id": teams[1]["id"]})
    assert response.status_code == 200
