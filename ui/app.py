"""Streamlit UI v2.0: talks to the Prediction API only (ADR-0001); no model code here."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT_S = 10


def api_get(path: str, **params: Any) -> Any:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def fmt_dt(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%d.%m.%Y %H:%M")


def verdict(p: dict[str, Any]) -> str:
    if p["decision"] == "no_data":
        return "Немає даних"
    if p["decision"] == "rejected":
        return f"Відхилено: {p['reason']}"
    return f"{p['winner']['name']} · {p['win_probability']:.0%}"


def show_prediction(p: dict[str, Any]) -> None:
    if p["mode"] == "hypothetical":
        st.info("Гіпотетичний матч: цієї пари немає в розкладі")
    if p["incomplete_data"]:
        st.warning("Неповні дані: у профілі команди немає ігрової статистики (ADR, kills, KAST)")
    if p["decision"] == "no_data":
        st.info(f"Прогноз неможливий. {p['reason']}")
    elif p["decision"] == "rejected":
        st.error(f"Прогноз не надається. Причина: {p['reason']}")
    else:
        st.success(f"Переможець: **{p['winner']['name']}** · шанс перемоги {p['win_probability']:.0%}")
    st.caption(f"Дані станом на {fmt_dt(p['data_as_of'])} · модель {p['model_version']}")


def show_team(stats: dict[str, Any]) -> None:
    st.subheader(stats["name"])
    st.metric("Відсоток перемог", f"{stats['win_percentage']:.1f}%")
    st.metric("Рейтинг сили", "—" if stats["power_rank"] is None else f"{stats['power_rank']:.1f}")
    st.metric("Матчів у профілі", stats["total_matches"])
    note = f"Останній матч: {fmt_dt(stats['last_match_at'])} ({stats['profile_age_days']} дн. тому)"
    st.caption(note + (" · застарілий профіль" if stats["stale"] else ""))


def page_schedule() -> None:
    st.header("Розклад і автоматичні прогнози")
    days = st.slider("Горизонт, днів", 1, 30, 14)
    matches = api_get("/schedule", days=days)
    if not matches:
        st.info("У розкладі немає матчів на цей період")
        return
    rows = [
        {
            "Початок": fmt_dt(m["scheduled_at"]),
            "Команда 1": m["team1"]["name"],
            "Команда 2": m["team2"]["name"],
            "Прогноз": verdict(m),
            "Неповні дані": "так" if m["incomplete_data"] else "",
            "match_id": m["match_id"],
        }
        for m in matches
    ]
    st.dataframe(pd.DataFrame(rows).drop(columns=["match_id"]), use_container_width=True, hide_index=True)
    st.caption(f"Дані станом на {fmt_dt(matches[0]['data_as_of'])}")

    labels = {f"{r['Початок']} · {r['Команда 1']} vs {r['Команда 2']}": r["match_id"] for r in rows}
    choice = st.selectbox("Відкрити матч", list(labels))
    if choice:
        try:
            card = api_get(f"/matches/{labels[choice]}")
        except requests.HTTPError as exc:
            st.info(f"Картка матчу недоступна: {exc.response.json().get('detail', exc.response.text)}")
            return
        left, right = st.columns(2)
        with left:
            show_team(card["team1_stats"])
        with right:
            show_team(card["team2_stats"])
        st.caption(f"Особисті зустрічі в даних: {card['head_to_head_matches']}")
        show_prediction(card["prediction"])


def team_picker(label: str, key: str) -> int | None:
    query = st.text_input(f"{label}: пошук за назвою", key=f"{key}_q")
    options = api_get("/teams", query=query, limit=50)
    if not options:
        st.caption("Команд не знайдено")
        return None
    names = {f"{t['name']} ({t['total_matches']} матчів)": t["id"] for t in options}
    return names[st.selectbox(label, list(names), key=key)]


def page_hypothetical() -> None:
    st.header("Теоретичний прогноз для будь-яких двох команд")
    left, right = st.columns(2)
    with left:
        t1 = team_picker("Команда 1", "t1")
    with right:
        t2 = team_picker("Команда 2", "t2")
    if t1 is None or t2 is None:
        return
    if t1 == t2:
        st.warning("Оберіть дві різні команди")
        return
    if st.button("Прогноз"):
        try:
            show_prediction(api_post("/predict", {"team1_id": t1, "team2_id": t2}))
        except requests.HTTPError as exc:
            st.error(f"Сервіс не зміг зробити прогноз: {exc.response.text}")


def main() -> None:
    st.set_page_config(page_title="CS2 Match Predictor", layout="wide")
    st.title("CS2 Match Predictor v2.0")
    try:
        health = api_get("/health")
    except requests.RequestException:
        st.error(f"Сервіс прогнозів недоступний: {API_URL}")
        return
    st.caption(
        f"Модель {health['model_version']} · дані {health['data_snapshot']} · станом на {fmt_dt(health['data_as_of'])}"
    )
    page = st.sidebar.radio("Розділ", ["Розклад", "Теоретичний прогноз"])
    page_schedule() if page == "Розклад" else page_hypothetical()


main()
