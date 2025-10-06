"""Rule representation utilities for the NSGA-II search."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
except ImportError:  # pragma: no cover - pyspark may not be present in some environments
    SparkDataFrame = None
    F = None


@dataclass
class Condition:
    """Axis-aligned interval constraint for a single feature."""

    feature: str
    lower: Optional[float]
    upper: Optional[float]

    def clamp(self, stats: Dict[str, Dict[str, float]]) -> None:
        """Ensure the interval stays within observed feature bounds."""
        bounds = stats[self.feature]
        if self.lower is not None:
            self.lower = max(self.lower, bounds["min"])
        if self.upper is not None:
            self.upper = min(self.upper, bounds["max"])
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            # Swap to maintain a valid interval.
            self.lower, self.upper = self.upper, self.lower

    def apply(self, series: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=series.index)
        if self.lower is not None:
            mask &= series >= self.lower
        if self.upper is not None:
            mask &= series <= self.upper
        return mask

    def to_sql(self) -> Optional[str]:
        clauses: List[str] = []
        if self.lower is not None:
            clauses.append(f"{self.feature} >= {self.lower}")
        if self.upper is not None:
            clauses.append(f"{self.feature} <= {self.upper}")
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return " AND ".join(clauses)


@dataclass
class Rule:
    """A conjunction of feature interval conditions."""

    conditions: List[Condition] = field(default_factory=list)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            return pd.Series(False, index=frame.index)

        mask = pd.Series(True, index=frame.index)
        for condition in self.conditions:
            mask &= condition.apply(frame[condition.feature])
        return mask

    def describe(self) -> str:
        parts: List[str] = []
        for cond in self.conditions:
            clause = cond.feature
            if cond.lower is not None and cond.upper is not None:
                clause += f" in [{cond.lower:.4f}, {cond.upper:.4f}]"
            elif cond.lower is not None:
                clause += f" >= {cond.lower:.4f}"
            elif cond.upper is not None:
                clause += f" <= {cond.upper:.4f}"
            parts.append(clause)
        return " AND ".join(parts) if parts else "<no conditions>"

    def copy(self) -> "Rule":
        return Rule([
            Condition(c.feature, c.lower, c.upper) for c in self.conditions
        ])

    def to_sql(self) -> str:
        if not self.conditions:
            return "1 = 0"
        parts = []
        for condition in self.conditions:
            clause = condition.to_sql()
            if clause is not None:
                parts.append(f"({clause})")
        if not parts:
            return "1 = 0"
        return " AND ".join(parts)


@dataclass
class RuleSet:
    """A disjunction (portfolio) of rules."""

    rules: List[Rule] = field(default_factory=list)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self.rules:
            return pd.Series(False, index=frame.index)

        mask = pd.Series(False, index=frame.index)
        for rule in self.rules:
            mask |= rule.predict(frame)
        return mask

    def describe(self) -> str:
        if not self.rules:
            return "<empty portfolio>"
        return " OR ".join(f"({rule.describe()})" for rule in self.rules)

    def copy(self) -> "RuleSet":
        return RuleSet([rule.copy() for rule in self.rules])

    def to_sql(self) -> str:
        clauses: List[str] = []
        for rule in self.rules:
            clause = rule.to_sql()
            if clause != "1 = 0":
                clauses.append(clause)
        if not clauses:
            return "1 = 0"
        return "(" + ") OR (".join(clauses) + ")"


def rule_from_dict(payload: Dict) -> Rule:
    conditions = []
    for cond in payload.get("conditions", []):
        conditions.append(
            Condition(
                feature=cond["feature"],
                lower=cond.get("lower"),
                upper=cond.get("upper"),
            )
        )
    return Rule(conditions)


def ruleset_from_dict(payload: Dict) -> RuleSet:
    rules = [rule_from_dict(rule_cfg) for rule_cfg in payload.get("clauses", [])]
    if not rules:
        rules = [Rule()]
    return RuleSet(rules)


def compute_feature_stats(frame, features: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Return min/max stats used to clamp conditions."""
    stats: Dict[str, Dict[str, float]] = {}
    if SparkDataFrame is not None and isinstance(frame, SparkDataFrame):
        for feature in features:
            row = frame.select(
                F.min(F.col(feature)).alias("min"),
                F.max(F.col(feature)).alias("max"),
            ).collect()[0]
            min_val = row["min"] if row["min"] is not None else 0.0
            max_val = row["max"] if row["max"] is not None else min_val
            stats[feature] = {"min": float(min_val), "max": float(max_val)}
        return stats

    # Fallback to pandas DataFrame for non-Spark usage.
    for feature in features:
        series = frame[feature]
        stats[feature] = {"min": float(series.min()), "max": float(series.max())}
    return stats
