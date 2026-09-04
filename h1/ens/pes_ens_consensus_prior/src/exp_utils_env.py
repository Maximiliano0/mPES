"""Severity-dynamics utilities, mirrored from ``ml/pes_dqn/src/exp_utils.py``.

Only the functions needed by :mod:`pandemic_env` are kept, decoupled from
``ml/`` so this ensemble evaluates without importing model-carrying modules.
"""

import numpy

from .env_constants import (AVAILABLE_RESOURCES_PER_SEQUENCE, MAX_ALLOCATABLE_RESOURCES,
                            MIN_ALLOCATABLE_RESOURCES, RESPONSE_MULTIPLIER, SEVERITY_MULTIPLIER)

# Per-sequence allocatable budget the agent actually controls (the env reserves
# 9 resources as "preassigned"; see Pandemic.__init__ in pandemic_env.py).
_FEASIBLE_BUDGET_PER_SEQUENCE = AVAILABLE_RESOURCES_PER_SEQUENCE - 9


def get_updated_severity(no_of_cities, resource_allocated, initial_severity) -> list[float]:
    """
    Update severity for existing cities given allocated resources.

    Parameters
    ----------
    no_of_cities : int
        Number of cities/trials to update severity for.
    resource_allocated : array-like
        Resources allocated to each city (0 to ``MAX_ALLOCATABLE_RESOURCES``).
    initial_severity : array-like
        Current severity values for each city.

    Returns
    -------
    list[float]
        Updated severity values, clipped to a minimum of 0.
    """
    updated_severity_list = []
    for c in range(no_of_cities):
        initial_severity_in_city = initial_severity[c]
        resources_allocated_to_city = resource_allocated[c]
        new_severity_in_city = (SEVERITY_MULTIPLIER * initial_severity_in_city
                                -RESPONSE_MULTIPLIER * resources_allocated_to_city)
        new_severity_in_city = max(new_severity_in_city, 0)
        updated_severity_list.append(new_severity_in_city)
    return updated_severity_list


def _evolve_single_city(initial_severity, allocation, num_evolutions):
    """Evolve one city under a constant per-trial allocation.

    Parameters
    ----------
    initial_severity : float
        Severity at the moment the city joins.
    allocation : int
        Constant resource allocation for this city.
    num_evolutions : int
        Number of evolution steps to apply.

    Returns
    -------
    float
        Final severity after all evolutions, clipped to >= 0.
    """
    severity = float(initial_severity)
    for _ in range(num_evolutions):
        severity = max(0.0, SEVERITY_MULTIPLIER * severity - RESPONSE_MULTIPLIER * allocation)
    return severity


def _best_feasible_sequence_severity(initial_sequence_severities,
                                     budget=_FEASIBLE_BUDGET_PER_SEQUENCE,
                                     max_alloc=MAX_ALLOCATABLE_RESOURCES):
    """Return the minimum total final severity achievable under a budget.

    Bounded-knapsack DP over per-city integer allocations in ``[0, max_alloc]``
    summing to at most ``budget``.

    Parameters
    ----------
    initial_sequence_severities : array-like
        Initial severity of each city in the sequence.
    budget : int, optional
        Total allocations available across the sequence.
    max_alloc : int, optional
        Per-trial allocation cap.

    Returns
    -------
    float
        Sum of per-city final severities under the optimal feasible allocation.
    """
    length = len(initial_sequence_severities)
    per_city = [
        [_evolve_single_city(float(initial_sequence_severities[c]), a, length - c)
         for a in range(max_alloc + 1)]
        for c in range(length)
    ]
    inf = float('inf')
    dp = [inf] * (budget + 1)
    dp[0] = 0.0
    for c in range(length):
        new_dp = [inf] * (budget + 1)
        for b in range(budget + 1):
            base = dp[b]
            if base == inf:
                continue
            for a in range(max_alloc + 1):
                nb = b + a
                if nb > budget:
                    break
                cost = base + per_city[c][a]
                if cost < new_dp[nb]:
                    new_dp[nb] = cost
        dp = new_dp
    return min(dp)


def get_sequence_severity_from_allocations(allocations, initial_severities):
    """Compute the total severity of a full sequence given allocations and initial severities.

    Parameters
    ----------
    allocations : array-like
        Resource allocations for each trial in the sequence.
    initial_severities : array-like
        Initial severity values for each trial in the sequence.

    Returns
    -------
    float
        Sum of all per-trial severities after applying the allocations.
    """
    return numpy.sum(get_array_of_sequence_severities_from_allocations(allocations, initial_severities))


def get_array_of_sequence_severities_from_allocations(allocations, initial_severities):
    """
    Calculate severity progression through a sequence given resource allocations.

    Parameters
    ----------
    allocations : array-like
        Resource allocation amounts for each trial in sequence (0-10).
    initial_severities : array-like
        Initial severity value for each trial.

    Returns
    -------
    list[float]
        Final severity values for each trial after resource allocation effects.
    """
    num_trials_in_sequence = len(initial_severities)
    severities: list[float] = []
    resources: list[float] = []
    for trial in range(num_trials_in_sequence):
        severities.append(initial_severities[trial])
        resources.append(allocations[trial])
        severities = get_updated_severity(len(severities), resources, severities)
    return severities.copy()


def calculate_normalised_final_severity_performance_metric(severities_from_sequence, initial_sequence_severities):
    """
    Calculate normalized performance metric comparing actual severity outcome to best/worst case scenarios.

    The metric ranges from 0 (worst case performance) to 1 (best case performance).

    Parameters
    ----------
    severities_from_sequence : array-like
        Final severity values achieved for each trial in the sequence.
    initial_sequence_severities : array-like
        Initial severity values for each trial in the sequence.

    Returns
    -------
    tuple
        - performance (float): Normalized performance metric (0-1).
        - worst_case_sequence_severity (float): Sum of severities if no resources allocated.
        - best_case_sequence_severity (float): Sum of severities if optimally allocated.
    """
    final_sequence_severity = numpy.sum(severities_from_sequence)
    worst_case_allocations = numpy.full_like(severities_from_sequence, MIN_ALLOCATABLE_RESOURCES)
    best_case_sequence_severity = _best_feasible_sequence_severity(initial_sequence_severities)
    worst_case_sequence_severity = get_sequence_severity_from_allocations(
        worst_case_allocations, initial_sequence_severities)
    performance = ((worst_case_sequence_severity - final_sequence_severity)
                  /(worst_case_sequence_severity - best_case_sequence_severity))
    return performance, worst_case_sequence_severity, best_case_sequence_severity
