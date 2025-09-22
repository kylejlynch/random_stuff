"""Evaluate an exported rule portfolio against the sampled dataset."""

import argparse
import json
from pathlib import Path
from typing import Dict

from .data_loader import AMOUNT_COLUMN, LABEL_COLUMN, available_features, load_training_view
from .nsga2 import FitnessContext
from .rules import RuleSet, ruleset_from_dict
from .spark_utils import get_spark_session


def build_ruleset(path: Path) -> RuleSet:
    payload = json.loads(path.read_text())
    return ruleset_from_dict(payload)


def evaluate(portfolio: RuleSet, frame) -> Dict[str, float]:
    context = FitnessContext(
        frame=frame,
        features=tuple(available_features()),
        label_column=LABEL_COLUMN,
        amount_column=AMOUNT_COLUMN,
        min_coverage=0.0,
        max_coverage=1.0,
    )
    fit = context.evaluate(portfolio)
    return {
        "fraud_capture": fit.fraud_capture,
        "fraud_dollar_capture": fit.fraud_dollar_capture,
        "false_positive_rate": fit.false_positive_rate,
        "coverage": fit.coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path, help="Path to exported portfolio JSON")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to evaluate (defaults to full sample)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state when subsampling",
    )
    args = parser.parse_args()

    portfolio = build_ruleset(args.portfolio)

    spark = get_spark_session("EvaluatePortfolio")
    frame = load_training_view(
        spark,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state,
    )
    metrics = evaluate(portfolio, frame)

    print(f"Evaluated portfolio from {args.portfolio}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    frame.unpersist()


if __name__ == "__main__":
    main()
