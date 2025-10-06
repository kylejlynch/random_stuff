"""Entry point to execute NSGA-II rule search on the fraud dataset slice."""

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import AMOUNT_COLUMN, LABEL_COLUMN, available_features, load_training_view
from .nsga2 import (
    FitnessContext,
    FitnessResult,
    NSGAII,
    compute_crowding_distance,
    fast_non_dominated_sort,
)
from .rules import RuleSet, ruleset_from_dict
from .spark_utils import get_spark_session

CURRENT_ALERT_COLUMN = "alerted"


def summarise_population(population: Iterable, output_dir: Path) -> None:
    records = []
    for individual in population:
        fit = individual.fitness
        records.append(
            {
                "portfolio": individual.portfolio.describe(),
                "rule_count": len(individual.portfolio.rules),
                "fraud_capture": fit.fraud_capture,
                "fraud_dollar_capture": fit.fraud_dollar_capture,
                "false_positive_rate": fit.false_positive_rate,
                "coverage": fit.coverage,
                "pareto_rank": individual.rank,
                "crowding_distance": individual.crowding_distance,
            }
        )
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_dir / "population_metrics.csv", index=False)


def load_seed_portfolios(paths: Iterable[Path]) -> list[RuleSet]:
    seeds: list[RuleSet] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Seed file {path} contains a non-object entry."
                    )
                seeds.append(ruleset_from_dict(entry))
        elif isinstance(payload, dict):
            seeds.append(ruleset_from_dict(payload))
        else:
            raise ValueError(
                f"Seed file {path} must contain either a JSON object or array of objects."
            )
    return seeds


def plot_pareto(
    front, output_file: Path, baseline: FitnessResult | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not available; skipping plot generation")
        return

    xs = [ind.fitness.false_positive_rate for ind in front]
    ys = [ind.fitness.fraud_capture for ind in front]
    colors = [ind.fitness.fraud_dollar_capture for ind in front]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=colors, cmap="viridis", s=90, edgecolor="k")
    plt.colorbar(scatter, label="Fraud dollar capture")
    plt.xlabel("False positive rate")
    plt.ylabel("Fraud capture")
    plt.title("Pareto front on training subset")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if baseline is not None:
        plt.scatter(
            baseline.false_positive_rate,
            baseline.fraud_capture,
            marker="*",
            s=220,
            c="red",
            edgecolor="k",
            label="Current performance",
        )
        plt.legend(loc="lower left")

    plt.savefig(output_file)
    plt.close()


def evaluate_on_holdout(front, holdout_context: FitnessContext) -> pd.DataFrame:
    rows = []
    for individual in front:
        fit = holdout_context.evaluate(individual.portfolio)
        rows.append(
            {
                "portfolio": individual.portfolio.describe(),
                "rule_count": len(individual.portfolio.rules),
                "fraud_capture": fit.fraud_capture,
                "fraud_dollar_capture": fit.fraud_dollar_capture,
                "false_positive_rate": fit.false_positive_rate,
                "coverage": fit.coverage,
            }
        )
    return pd.DataFrame(rows)


def per_rule_contributions(
    portfolio,
    context: FitnessContext,
) -> pd.DataFrame:
    records = []
    prior = FitnessResult(0.0, 0.0, 0.0, 0.0)
    for idx, rule in enumerate(portfolio.rules, start=1):
        single_fit = context.evaluate(RuleSet([rule]))
        cumulative_portfolio = RuleSet(portfolio.rules[: idx])
        cumulative_fit = context.evaluate(cumulative_portfolio)
        delta_capture = cumulative_fit.fraud_capture - prior.fraud_capture
        delta_dollars = (
            cumulative_fit.fraud_dollar_capture - prior.fraud_dollar_capture
        )
        delta_fpr = cumulative_fit.false_positive_rate - prior.false_positive_rate
        delta_cov = cumulative_fit.coverage - prior.coverage
        records.append(
            {
                "rule_index": idx,
                "rule": rule.describe(),
                "single_fraud_capture": single_fit.fraud_capture,
                "single_fraud_dollar": single_fit.fraud_dollar_capture,
                "single_fpr": single_fit.false_positive_rate,
                "single_coverage": single_fit.coverage,
                "delta_fraud_capture": delta_capture,
                "delta_fraud_dollar": delta_dollars,
                "delta_fpr": delta_fpr,
                "delta_coverage": delta_cov,
                "cumulative_fraud_capture": cumulative_fit.fraud_capture,
                "cumulative_fraud_dollar": cumulative_fit.fraud_dollar_capture,
                "cumulative_fpr": cumulative_fit.false_positive_rate,
                "cumulative_coverage": cumulative_fit.coverage,
            }
        )
        prior = cumulative_fit
    return pd.DataFrame(records)


def run(
    population: int,
    generations: int,
    sample_fraction: float,
    output_dir: Path,
    train_fraction: float,
    split_seed: int,
    max_rules: int,
    min_coverage: float,
    max_coverage: float,
    seed_portfolios: list[RuleSet] | None = None,
) -> None:
    spark = get_spark_session()
    frame = load_training_view(
        spark,
        sample_fraction=sample_fraction,
        random_state=split_seed,
        extra_columns=(CURRENT_ALERT_COLUMN,),
    )
    features = tuple(available_features())

    splits = frame.randomSplit(
        [train_fraction, 1 - train_fraction], seed=split_seed
    )
    train = splits[0].cache()
    holdout = splits[1].cache()
    train.count()
    holdout.count()

    train_context = FitnessContext(
        frame=train,
        features=features,
        label_column=LABEL_COLUMN,
        amount_column=AMOUNT_COLUMN,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
    )
    holdout_context = FitnessContext(
        frame=holdout,
        features=features,
        label_column=LABEL_COLUMN,
        amount_column=AMOUNT_COLUMN,
        min_coverage=0.0,
        max_coverage=1.0,
    )

    nsga = NSGAII(
        context=train_context,
        population_size=population,
        generations=generations,
        max_rules=max_rules,
        seed_portfolios=seed_portfolios,
    )

    final_population = nsga.run()
    fronts = fast_non_dominated_sort(final_population)
    if not fronts:
        print("No feasible rules discovered.")
        return

    # Ensure crowding distance for reporting.
    for front in fronts:
        compute_crowding_distance(front)

    output_dir.mkdir(parents=True, exist_ok=True)
    summarise_population(final_population, output_dir)

    best_front = fronts[0]
    best_front_sorted = sorted(
        best_front,
        key=lambda ind: (
            ind.fitness.fraud_capture,
            ind.fitness.fraud_dollar_capture,
            -ind.fitness.false_positive_rate,
        ),
        reverse=True,
    )

    baseline_train: FitnessResult | None = None
    baseline_holdout: FitnessResult | None = None
    if CURRENT_ALERT_COLUMN in frame.columns:
        baseline_train = train_context.evaluate_flag_column(CURRENT_ALERT_COLUMN)
        baseline_holdout = holdout_context.evaluate_flag_column(CURRENT_ALERT_COLUMN)

    plot_pareto(
        best_front_sorted,
        output_dir / "pareto_front.png",
        baseline=baseline_train,
    )

    holdout_scores = evaluate_on_holdout(best_front_sorted, holdout_context)
    if baseline_holdout is not None:
        baseline_row = {
            "portfolio": "<current alerted policy>",
            "rule_count": 0,
            "fraud_capture": baseline_holdout.fraud_capture,
            "fraud_dollar_capture": baseline_holdout.fraud_dollar_capture,
            "false_positive_rate": baseline_holdout.false_positive_rate,
            "coverage": baseline_holdout.coverage,
        }
        holdout_scores = pd.concat(
            [holdout_scores, pd.DataFrame([baseline_row])], ignore_index=True
        )
    holdout_scores.to_csv(output_dir / "pareto_front_holdout.csv", index=False)

    unique_portfolios = []
    seen_signatures: set[str] = set()
    for individual in best_front_sorted:
        signature = individual.portfolio.describe()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_portfolios.append((signature, individual))
        if len(unique_portfolios) >= 10:
            break

    contribution_dir = output_dir / "per_rule_breakdown"
    contribution_dir.mkdir(parents=True, exist_ok=True)
    for idx, (signature, individual) in enumerate(unique_portfolios, start=1):
        train_contrib = per_rule_contributions(individual.portfolio, train_context)
        train_contrib.to_csv(
            contribution_dir / f"portfolio_{idx:02d}_train.csv", index=False
        )
        holdout_contrib = per_rule_contributions(
            individual.portfolio, holdout_context
        )
        holdout_contrib.to_csv(
            contribution_dir / f"portfolio_{idx:02d}_holdout.csv", index=False
        )

    print(
        f"Explored {len(final_population)} individuals across {generations} generations."
    )
    if baseline_train is not None and baseline_holdout is not None:
        print("Current alerted policy metrics:")
        print(
            f"train_capture={baseline_train.fraud_capture:.3f}, "
            f"train_dollars={baseline_train.fraud_dollar_capture:.3f}, "
            f"train_fpr={baseline_train.false_positive_rate:.3f}, "
            f"train_cov={baseline_train.coverage:.3f} | "
            f"holdout_capture={baseline_holdout.fraud_capture:.3f}, "
            f"holdout_dollars={baseline_holdout.fraud_dollar_capture:.3f}, "
            f"holdout_fpr={baseline_holdout.false_positive_rate:.3f}, "
            f"holdout_cov={baseline_holdout.coverage:.3f}"
        )

    print("Top Pareto-efficient portfolios (up to 10 unique shown):")
    for idx, (signature, individual) in enumerate(unique_portfolios, start=1):
        fit = individual.fitness
        hold_fit = holdout_context.evaluate(individual.portfolio)
        print(
            f"#{idx}: portfolio=({signature}) | "
            f"train_capture={fit.fraud_capture:.3f}, train_dollars={fit.fraud_dollar_capture:.3f}, "
            f"train_fpr={fit.false_positive_rate:.3f}, train_cov={fit.coverage:.3f} | "
            f"holdout_capture={hold_fit.fraud_capture:.3f}, holdout_dollars={hold_fit.fraud_dollar_capture:.3f}, "
            f"holdout_fpr={hold_fit.false_positive_rate:.3f}, holdout_cov={hold_fit.coverage:.3f}"
        )

    train.unpersist()
    holdout.unpersist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=90, help="Population size")
    parser.add_argument("--generations", type=int, default=12, help="Number of generations")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.10,
        help="Fraction of the dataset to use during the initial search",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/latest"),
        help="Directory where metrics and artefacts will be written",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of sampled data used for training (remainder is hold-out)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed controlling the train/hold-out split",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=4,
        help="Maximum number of rules allowed per portfolio",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.002,
        help="Minimum coverage threshold before penalties apply",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.6,
        help="Maximum coverage threshold before penalties apply",
    )
    parser.add_argument(
        "--seed-portfolio",
        type=Path,
        action="append",
        default=[],
        help="Path to a JSON file describing rule set(s) to seed the initial population",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_portfolios = load_seed_portfolios(args.seed_portfolio)
    run(
        population=args.population,
        generations=args.generations,
        sample_fraction=args.sample_fraction,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        max_rules=args.max_rules,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        seed_portfolios=seed_portfolios,
    )
