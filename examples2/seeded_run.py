"""Example script showing how to run NSGA-II with seeded rule portfolios."""

from pathlib import Path

from src.run_nsga2 import load_seed_portfolios, run


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    seed_file = project_root / "examples" / "seed_portfolios.json"
    output_dir = project_root / "artifacts" / "seeded_run"

    seed_portfolios = load_seed_portfolios([seed_file])

    run(
        population=90,
        generations=12,
        sample_fraction=0.10,
        output_dir=output_dir,
        train_fraction=0.8,
        split_seed=42,
        max_rules=4,
        min_coverage=0.002,
        max_coverage=0.6,
        seed_portfolios=seed_portfolios,
    )


if __name__ == "__main__":
    main()
