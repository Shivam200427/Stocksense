"""Genetic Algorithm optimizer for LSTM hyperparameters.

This module uses DEAP to evolve LSTM settings with a fitness of 1/validation_loss.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from deap import base, creator, tools

from data_pipeline import prepare_data
from lstm_model import train_and_evaluate


def _clip(individual: List[float]) -> List[float]:
    """Keep chromosome values inside legal numeric bounds."""
    individual[0] = int(max(32, min(128, round(individual[0]))))
    individual[1] = float(max(0.10, min(0.50, individual[1])))
    individual[2] = float(max(1e-4, min(5e-3, individual[2])))
    individual[3] = int(max(30, min(90, round(individual[3]))))
    return individual


def optimize_hyperparameters(
    ticker: str,
    generations: int = 15,
    population_size: int = 10,
) -> Tuple[Dict, List[Dict]]:
    """Run a DEAP GA search and return best params + generation history."""
    random.seed(42)
    np.random.seed(42)

    if not hasattr(creator, "FitnessMaxStockSense"):
        creator.create("FitnessMaxStockSense", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualStockSense"):
        creator.create("IndividualStockSense", list, fitness=creator.FitnessMaxStockSense)

    toolbox = base.Toolbox()
    toolbox.register("attr_units", random.randint, 32, 128)
    toolbox.register("attr_dropout", random.uniform, 0.10, 0.50)
    toolbox.register("attr_lr", random.uniform, 1e-4, 5e-3)
    toolbox.register("attr_lookback", random.randint, 30, 90)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.IndividualStockSense,
        (toolbox.attr_units, toolbox.attr_dropout, toolbox.attr_lr, toolbox.attr_lookback),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        units, dropout, lr, lookback = _clip(individual)

        try:
            prepared = prepare_data(ticker=ticker, lookback=lookback)
            result = train_and_evaluate(
                prepared,
                units=units,
                dropout=dropout,
                learning_rate=lr,
                epochs=12,
                batch_size=32,
                verbose=0,
            )
            val_loss = max(result.val_loss, 1e-8)
            fitness = 1.0 / float(val_loss)
        except Exception:
            # Penalize infeasible candidate but keep evolution moving.
            fitness = 1e-8

        return (fitness,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    history: List[Dict] = []

    # Initial fitness evaluation.
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(1, generations + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                _clip(child1)
                _clip(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                _clip(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring

        fitnesses = [ind.fitness.values[0] for ind in population]
        best_fit = float(np.max(fitnesses))
        avg_fit = float(np.mean(fitnesses))

        # Required progress logs so long optimization remains transparent.
        print(
            f"[GA] Generation {gen}/{generations} | "
            f"Best Fitness: {best_fit:.6f} | Avg Fitness: {avg_fit:.6f}"
        )

        history.append(
            {
                "generation": gen,
                "best_fitness": best_fit,
                "avg_fitness": avg_fit,
            }
        )

    best = tools.selBest(population, 1)[0]
    units, dropout, lr, lookback = _clip(best)

    best_params = {
        "lstm_units": int(units),
        "dropout_rate": float(dropout),
        "learning_rate": float(lr),
        "lookback_window": int(lookback),
        "fitness": float(best.fitness.values[0]),
    }

    return best_params, history
