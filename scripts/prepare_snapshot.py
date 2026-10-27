"""
Prepare the department snapshot for training and inference.

Outputs:
  <out>/matches_v2.json             feature set for training (up to the cutoff month)
  <out>/matches_v2_post_cutoff.json matches after the cutoff (degraded game stats)
  <out>/team_names.json             team id -> real team name (FR-8)
  <out>/schedule.json               scheduled matches after the snapshot date (FR-1)
  <reports>/data_quality_by_month.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.schema import validate_records  # noqa: E402
from src.snapshot import (  # noqa: E402
    MATCH_PREDICTION_FILE,
    MATCHES_FILE,
    TEAMS_FILE,
    FeatureSetVersion,
    iter_json_array,
    load_team_names,
    match_date,
    quality_by_month,
    scheduled_matches,
    select_feature_set,
    split_by_cutoff,
    suggest_cutoff,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=ROOT_DIR / "data", type=Path)
    parser.add_argument("--reports-dir", default=ROOT_DIR / "reports", type=Path)
    parser.add_argument("--calculator-version", default="2.4")
    parser.add_argument("--coefficients-version", default="1")
    parser.add_argument("--cutoff-month", default="auto", help="YYYY-MM, 'auto' or 'none'.")
    parser.add_argument("--max-missing-pct", type=float, default=5.0)
    parser.add_argument("--max-invalid-share", type=float, default=0.01, help="Abort if more records fail the schema.")
    parser.add_argument("--as-of", default=None, help="ISO date treated as 'now' for the schedule.")
    parser.add_argument("--horizon-days", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("prepare_snapshot")

    version = FeatureSetVersion(args.calculator_version, args.coefficients_version)
    log.info("Selecting feature set %s ...", version.label)
    records = select_feature_set(iter_json_array(args.snapshot_dir / MATCH_PREDICTION_FILE), version)
    log.info("Unique matches: %d (%s .. %s)", len(records), match_date(records[0]), match_date(records[-1]))

    records, validation = validate_records(records)
    log.info("Schema: %d invalid of %d (%.2f%%)", validation.invalid, validation.total, 100 * validation.invalid_share)
    for error, count in validation.errors.most_common():
        log.info("  %5d  %s", count, error)
    if validation.invalid_share > args.max_invalid_share:
        raise SystemExit(
            f"Invalid share {validation.invalid_share:.2%} exceeds {args.max_invalid_share:.2%}; ingest aborted"
        )

    report = quality_by_month(records)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.reports_dir / "data_quality_by_month.csv", index=False)

    if args.cutoff_month == "auto":
        cutoff = suggest_cutoff(report, args.max_missing_pct)
    elif args.cutoff_month == "none":
        cutoff = None
    else:
        cutoff = args.cutoff_month
    before, after = split_by_cutoff(records, cutoff)
    log.info("Cutoff month: %s -> train pool %d, post-cutoff %d", cutoff, len(before), len(after))

    write_json(before, args.out_dir / "matches_v2.json")
    write_json(after, args.out_dir / "matches_v2_post_cutoff.json")

    names = load_team_names(args.snapshot_dir / TEAMS_FILE)
    (args.out_dir / "team_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    log.info("Team names: %d", len(names))

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    schedule = scheduled_matches(args.snapshot_dir / MATCHES_FILE, as_of, args.horizon_days)
    write_json(schedule, args.out_dir / "schedule.json")
    log.info("Scheduled matches after %s: %d", as_of.date(), len(schedule))

    meta = {
        "feature_set": version.label,
        "cutoff_month": cutoff,
        "train_pool": len(before),
        "post_cutoff": len(after),
        "as_of": as_of.isoformat(),
        "scheduled": len(schedule),
        "snapshot_dir": args.snapshot_dir.name,
        "prepared_at": datetime.now(UTC).isoformat(),
        "validation": {
            "total": validation.total,
            "invalid": validation.invalid,
            "errors": dict(validation.errors),
        },
    }
    (args.out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
