from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _walk_json_leaves(prefix: str, node: Any, out: set[str]) -> None:
    """
    Build header strings like:
      match|summaryStats|team1|kills|$numberDouble
    for every scalar leaf in the JSON tree.
    """

    if isinstance(node, dict):
        for key, value in node.items():
            _walk_json_leaves(f"{prefix}|{key}", value, out)
        return

    if isinstance(node, list):
        for index, item in enumerate(node):
            # The notebook assumes lists represent teams and uses team{index+1}.
            _walk_json_leaves(f"{prefix}|team{index + 1}", item, out)
        return

    # Scalar leaf
    out.add(prefix)


def get_header_from_json(
    json_path: str | Path,
    output_path: str | Path | None = None,
) -> list[str]:
    """
    Create a list of header strings by walking JSON matches.

    If `output_path` is provided, also writes headers to a file (one per line).
    """

    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of matches.")

    header: set[str] = set()
    for match in data:
        _walk_json_leaves("match", match, header)

    headers = sorted(header)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(headers), encoding="utf-8")

    return headers


def _get_value_by_prefix(match: dict[str, Any], prefix: str) -> Any:
    """
    Resolve a value from `match` using the header-like path string.
    """

    # Notebook format: match|a|b|team1|c ...
    fields = prefix.split("|")[1:]
    val: Any = match

    for field in fields:
        if isinstance(val, list):
            matched_index = re.search(r"team(\d+)", field)
            if not matched_index:
                raise KeyError(f"Expected team index in field: {field}")
            index = int(matched_index.group(1)) - 1
            val = val[index]
        else:
            # When missing key happens, notebook used exception -> None.
            if not isinstance(val, dict):
                raise KeyError(f"Cannot access non-dict value by key: {field}")
            val = val[field]

    return val


def load_matches(
    json_path: str | Path,
    header_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Convert JSON matches into a flat DataFrame using `header_path` (or generated headers).

    Target `y`:
      - 1 if team1 won (actualWinnerId == team1 pandaScoreTeamId)
      - 0 if team2 won
      - None otherwise
    """

    json_path = Path(json_path)

    if header_path is not None:
        header_path = Path(header_path)
        header_lines = header_path.read_text(encoding="utf-8").splitlines()
        headers = [line.strip() for line in header_lines if line.strip()]
    else:
        headers = get_header_from_json(json_path)

    # Init columns as lists.
    all_dict: dict[str, list[Any]] = {prefix: [] for prefix in headers}
    all_dict["y"] = []

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of matches.")

    # Ensure target prefixes exist (if they don't, y will be computed as None).
    target_actual = "match|actualWinnerId"
    target_t1 = "match|summaryStats|team1|pandaScoreTeamId"
    target_t2 = "match|summaryStats|team2|pandaScoreTeamId"

    for match in data:
        for prefix in headers:
            try:
                val = _get_value_by_prefix(match, prefix)
            except Exception:
                val = None
            all_dict[prefix].append(val)

        try:
            actual = _get_value_by_prefix(match, target_actual)
            t1_id = _get_value_by_prefix(match, target_t1)
            t2_id = _get_value_by_prefix(match, target_t2)
        except Exception:
            actual, t1_id, t2_id = None, None, None

        if actual is None:
            all_dict["y"].append(None)
        else:
            if t1_id is not None and t1_id == actual:
                all_dict["y"].append(1)
            elif t2_id is not None and t2_id == actual:
                all_dict["y"].append(0)
            else:
                all_dict["y"].append(None)

    return pd.DataFrame(all_dict)
