"""Deploy or roll back a model: point the `production` alias in the MLflow registry to a version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.tracking import DEFAULT_TRACKING_URI, PREDICTOR_NAME, promote  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Registered model version to serve, e.g. 2")
    parser.add_argument("--name", default=PREDICTOR_NAME)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()
    previous = promote(args.name, args.version, tracking_uri=args.tracking_uri)
    print(f"{args.name}: production {previous or '—'} -> {args.version}. Restart the API to apply.")


if __name__ == "__main__":
    main()
