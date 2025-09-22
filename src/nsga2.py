"""NSGA-II implementation for discovering rule portfolios using PySpark."""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from .rules import Condition, Rule, RuleSet, compute_feature_stats


@dataclass
class FitnessResult:
    fraud_capture: float
    fraud_dollar_capture: float
    false_positive_rate: float
    coverage: float

    def as_objectives(self) -> Tuple[float, float, float]:
        return (
            -self.fraud_capture,
            -self.fraud_dollar_capture,
            self.false_positive_rate,
        )


@dataclass
class Individual:
    portfolio: RuleSet
    fitness: FitnessResult
    objectives: Tuple[float, float, float]
    rank: int = 0
    crowding_distance: float = 0.0


class FitnessContext:
    def __init__(
        self,
        frame: SparkDataFrame,
        features: Sequence[str],
        label_column: str,
        amount_column: str,
        min_coverage: float = 0.002,
        max_coverage: float = 0.6,
    ) -> None:
        self.frame = frame.cache()
        self.features = features
        self.label_column = label_column
        self.amount_column = amount_column
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage

        self.stats = compute_feature_stats(frame, features)

        self.total_records = self.frame.count()
        self.total_fraud = (
            self.frame.filter(F.col(label_column) == 1).count()
        )
        self.total_non_fraud = self.total_records - self.total_fraud
        amount_col = F.col(amount_column)
        fraud_amount_sum = (
            self.frame.select(
                F.sum(
                    F.when(F.col(label_column) == 1, F.when(amount_col < 0, 0.0).otherwise(amount_col))
                ).alias("fraud_amount")
            )
            .collect()[0]
            .fraud_amount
        )
        self.total_fraud_amount = float(fraud_amount_sum or 1.0)

    def evaluate(self, portfolio: RuleSet) -> FitnessResult:
        sql_expr = portfolio.to_sql()
        if sql_expr == "1 = 0":
            return FitnessResult(0.0, 0.0, 0.0, 0.0)

        flagged = self.frame.where(sql_expr)
        positives = flagged.count()
        coverage = float(positives) / self.total_records if self.total_records else 0.0

        if positives == 0:
            return FitnessResult(0.0, 0.0, 0.0, coverage)

        agg = (
            flagged.select(
                F.sum(F.when(F.col(self.label_column) == 1, 1).otherwise(0)).alias("fraud_hits"),
                F.sum(
                    F.when(
                        F.col(self.label_column) == 1,
                        F.when(F.col(self.amount_column) < 0, 0.0).otherwise(F.col(self.amount_column)),
                    ).otherwise(0.0)
                ).alias("fraud_amount"),
                F.sum(F.when(F.col(self.label_column) == 0, 1).otherwise(0)).alias("non_fraud_hits"),
            ).collect()[0]
        )

        fraud_hits = float(agg.fraud_hits or 0.0)
        fraud_amount = float(agg.fraud_amount or 0.0)
        non_fraud_hits = float(agg.non_fraud_hits or 0.0)

        fraud_capture = (
            fraud_hits / self.total_fraud if self.total_fraud else 0.0
        )
        fraud_dollar_capture = (
            fraud_amount / self.total_fraud_amount if self.total_fraud_amount else 0.0
        )
        false_positive_rate = (
            non_fraud_hits / self.total_non_fraud if self.total_non_fraud else 0.0
        )

        if coverage < self.min_coverage or coverage > self.max_coverage:
            penalty = 0.5
            fraud_capture *= penalty
            fraud_dollar_capture *= penalty
            false_positive_rate = min(1.0, false_positive_rate + (1 - penalty))

        return FitnessResult(
            fraud_capture=fraud_capture,
            fraud_dollar_capture=fraud_dollar_capture,
            false_positive_rate=false_positive_rate,
            coverage=coverage,
        )


class PortfolioFactory:
    def __init__(
        self,
        features: Sequence[str],
        stats: Dict[str, Dict[str, float]],
        max_conditions: int = 4,
        max_rules: int = 4,
    ) -> None:
        self.features = list(features)
        self.stats = stats
        self.max_conditions = max_conditions
        self.max_rules = max_rules

    def random_rule(self) -> Rule:
        num_conditions = random.randint(1, self.max_conditions)
        selected = random.sample(self.features, k=num_conditions)
        conditions: List[Condition] = []
        for feature in selected:
            bounds = self.stats[feature]
            span = max(float(bounds["max"]) - float(bounds["min"]), 1e-6)
            base = random.uniform(bounds["min"], bounds["max"])
            width = random.uniform(0.05, 0.3) * span
            lower = max(bounds["min"], base - width / 2)
            upper = min(bounds["max"], base + width / 2)

            lower_bound = lower if random.random() < 0.7 else None
            upper_bound = upper if random.random() < 0.7 else None
            if lower_bound is None and upper_bound is None:
                if random.random() < 0.5:
                    lower_bound = lower
                else:
                    upper_bound = upper

            condition = Condition(feature, lower_bound, upper_bound)
            condition.clamp(self.stats)
            conditions.append(condition)
        return Rule(conditions)

    def _normalise(self, rules: List[Rule]) -> List[Rule]:
        unique: List[Rule] = []
        seen: set[str] = set()
        for rule in rules:
            signature = rule.describe()
            if signature not in seen:
                seen.add(signature)
                unique.append(rule)
            if len(unique) >= self.max_rules:
                break
        if not unique:
            unique.append(self.random_rule())
        return unique

    def random_portfolio(self) -> RuleSet:
        size = random.randint(1, self.max_rules)
        rules = [self.random_rule() for _ in range(size)]
        return RuleSet(self._normalise(rules))

    def crossover(self, parent_a: RuleSet, parent_b: RuleSet) -> RuleSet:
        child_rules: List[Rule] = []
        for rule in parent_a.rules:
            if random.random() < 0.5:
                child_rules.append(rule.copy())
        for rule in parent_b.rules:
            if random.random() < 0.5 or not child_rules:
                child_rules.append(rule.copy())
        if not child_rules:
            pool = parent_a.rules + parent_b.rules
            if pool:
                child_rules.append(random.choice(pool).copy())
        normalised = self._normalise(child_rules)
        return RuleSet([rule.copy() for rule in normalised])

    def mutate_rule(self, rule: Rule, mutation_rate: float = 0.2) -> Rule:
        mutated = rule.copy()
        if not mutated.conditions or random.random() < 0.1:
            return self.random_rule()

        for condition in mutated.conditions:
            if random.random() > mutation_rate:
                continue

            bounds = self.stats[condition.feature]
            span = max(float(bounds["max"]) - float(bounds["min"]), 1e-6)
            if condition.lower is not None:
                condition.lower += random.uniform(-0.1, 0.1) * span
            if condition.upper is not None:
                condition.upper += random.uniform(-0.1, 0.1) * span
            condition.clamp(self.stats)

        if random.random() < 0.1 and len(mutated.conditions) > 1:
            mutated.conditions.pop(random.randrange(len(mutated.conditions)))
        elif (
            random.random() < 0.1
            and len(mutated.conditions) < self.max_conditions
        ):
            mutated.conditions.append(self.random_rule().conditions[0])

        return mutated

    def mutate(self, portfolio: RuleSet, mutation_rate: float = 0.2) -> RuleSet:
        mutated = portfolio.copy()
        for idx, rule in enumerate(mutated.rules):
            if random.random() < mutation_rate:
                mutated.rules[idx] = self.mutate_rule(rule, mutation_rate)

        if mutated.rules and random.random() < 0.1:
            mutated.rules.pop(random.randrange(len(mutated.rules)))

        if len(mutated.rules) < self.max_rules and random.random() < 0.3:
            mutated.rules.append(self.random_rule())

        mutated.rules = [rule.copy() for rule in self._normalise(mutated.rules)]
        return mutated


def dominates(objectives_a: Tuple[float, ...], objectives_b: Tuple[float, ...]) -> bool:
    no_worse = all(a <= b for a, b in zip(objectives_a, objectives_b))
    strictly_better = any(a < b for a, b in zip(objectives_a, objectives_b))
    return no_worse and strictly_better


def fast_non_dominated_sort(individuals: Sequence[Individual]) -> List[List[Individual]]:
    population_size = len(individuals)
    domination_counts = [0] * population_size
    dominated_sets: List[List[int]] = [[] for _ in range(population_size)]
    fronts: List[List[int]] = [[]]

    for p_idx, p in enumerate(individuals):
        dominated: List[int] = []
        count = 0
        for q_idx, q in enumerate(individuals):
            if p_idx == q_idx:
                continue
            if dominates(p.objectives, q.objectives):
                dominated.append(q_idx)
            elif dominates(q.objectives, p.objectives):
                count += 1
        dominated_sets[p_idx] = dominated
        domination_counts[p_idx] = count
        if count == 0:
            p.rank = 0
            fronts[0].append(p_idx)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front: List[int] = []
        for p_idx in fronts[front_idx]:
            for q_idx in dominated_sets[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    individuals[q_idx].rank = front_idx + 1
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        front_idx += 1

    return [[individuals[idx] for idx in front] for front in fronts if front]


def compute_crowding_distance(front: Sequence[Individual]) -> None:
    if not front:
        return
    num_objectives = len(front[0].objectives)
    for individual in front:
        individual.crowding_distance = 0.0

    for m in range(num_objectives):
        front_sorted = sorted(front, key=lambda ind: ind.objectives[m])
        front_sorted[0].crowding_distance = float("inf")
        front_sorted[-1].crowding_distance = float("inf")
        min_obj = front_sorted[0].objectives[m]
        max_obj = front_sorted[-1].objectives[m]
        if max_obj == min_obj:
            continue
        for i in range(1, len(front_sorted) - 1):
            prev_obj = front_sorted[i - 1].objectives[m]
            next_obj = front_sorted[i + 1].objectives[m]
            distance = (next_obj - prev_obj) / (max_obj - min_obj)
            front_sorted[i].crowding_distance += distance


def tournament_selection(population: Sequence[Individual]) -> Individual:
    a, b = random.sample(population, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding_distance > b.crowding_distance else b


class NSGAII:
    def __init__(
        self,
        context: FitnessContext,
        population_size: int = 90,
        generations: int = 12,
        crossover_prob: float = 0.9,
        mutation_rate: float = 0.2,
        max_conditions: int = 4,
        max_rules: int = 4,
        random_seed: int = 42,
    ) -> None:
        self.context = context
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.factory = PortfolioFactory(
            context.features, context.stats, max_conditions, max_rules
        )

    def _evaluate_portfolio(self, portfolio: RuleSet) -> Individual:
        fitness = self.context.evaluate(portfolio)
        return Individual(
            portfolio=portfolio,
            fitness=fitness,
            objectives=fitness.as_objectives(),
        )

    def _initial_population(self) -> List[Individual]:
        return [
            self._evaluate_portfolio(self.factory.random_portfolio())
            for _ in range(self.population_size)
        ]

    def _breed(self, parents: Sequence[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        while len(offspring) < self.population_size:
            parent_a = tournament_selection(parents)
            parent_b = tournament_selection(parents)
            if random.random() < self.crossover_prob:
                child_portfolio = self.factory.crossover(
                    parent_a.portfolio, parent_b.portfolio
                )
            else:
                child_portfolio = parent_a.portfolio.copy()
            child_portfolio = self.factory.mutate(
                child_portfolio, self.mutation_rate
            )
            offspring.append(self._evaluate_portfolio(child_portfolio))
        return offspring

    def _combine_and_select(
        self, current: List[Individual], offspring: List[Individual]
    ) -> List[Individual]:
        population = current + offspring
        fronts = fast_non_dominated_sort(population)
        next_population: List[Individual] = []
        for front in fronts:
            compute_crowding_distance(front)
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
            else:
                front_sorted = sorted(
                    front, key=lambda ind: ind.crowding_distance, reverse=True
                )
                remaining_slots = self.population_size - len(next_population)
                next_population.extend(front_sorted[:remaining_slots])
                break
        return next_population

    def run(self) -> List[Individual]:
        population = self._initial_population()
        for _ in range(self.generations):
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                compute_crowding_distance(front)
            offspring = self._breed(population)
            population = self._combine_and_select(population, offspring)
        return population
