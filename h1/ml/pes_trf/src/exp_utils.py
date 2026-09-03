"""
pes_trf — Pandemic Experiment Scenario (Transformer Deep Q-Network)

Utility functions module providing essential functionality for experiment execution,
severity calculations, resource allocation aggregation, and performance metrics.

Key Components
--------------
• Severity Calculations: Functions to compute and update pandemic severity based on
  resource allocations and time progression
• Performance Metrics: Normalized performance calculation comparing actual results
  against best/worst case allocations
• Confidence Aggregation: Weighted mean/median calculations for combining decisions
  from multiple participants
• Analysis Tools: Utilities for data transformation, sampling, and experiment control

Main Functions
---------------
• calculate_normalised_final_severity_performance_metric: Compute normalized performance
• get_updated_severity: Calculate new severity given resource allocations
• get_confidence_weighted_mean: Aggregate decisions using confidence-weighted mean
• get_confidence_weighted_median: Aggregate decisions using confidence-weighted median
• get_percent_deviation: Measure deviation from optimal outcomes
• random_severity_generator: Generate random initial conditions
• chain_ops: Pipe operations through sequential functions
• exit_experiment_gracefully: Clean shutdown with resource cleanup
"""

##########################
##  Imports externos    ##
##########################
import os
import numpy
import scipy.stats as ss
from statsmodels.stats.weightstats import DescrStatsW as WeightedStats

##########################
##  Imports internos    ##
##########################
from .. import (AVAILABLE_RESOURCES_PER_SEQUENCE, INPUTS_PATH, MAX_ALLOCATABLE_RESOURCES, MIN_ALLOCATABLE_RESOURCES,
                RESPONSE_MULTIPLIER, SEQ_LENGTHS_FILE, SEVERITY_MULTIPLIER)

# Per-sequence allocatable budget that the agent actually controls. The env
# reserves 9 resources as "preassigned" (Pandemic.__init__: max_resources =
# AVAILABLE_RESOURCES_PER_SEQUENCE - 9), so the maximum total allocation a
# feasible policy can issue across a sequence is this value.
_FEASIBLE_BUDGET_PER_SEQUENCE = AVAILABLE_RESOURCES_PER_SEQUENCE - 9


def _evolve_single_city(initial_severity, allocation, num_evolutions):
    """Evolve one city under a constant per-trial allocation.

    Mirrors the dynamics in ``get_array_of_sequence_severities_from_allocations``:
    after a city joins at trial ``c``, every subsequent trial applies
    ``s = max(0, SEVERITY_MULTIPLIER * s - RESPONSE_MULTIPLIER * allocation)``
    once. ``num_evolutions`` is ``L - c`` for a length-``L`` sequence.

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
    s = float(initial_severity)
    for _ in range(num_evolutions):
        s = max(0.0, SEVERITY_MULTIPLIER * s - RESPONSE_MULTIPLIER * allocation)
    return s


def _best_feasible_sequence_severity(InitialSequenceSeverities,
                                     budget=_FEASIBLE_BUDGET_PER_SEQUENCE,
                                     max_alloc=MAX_ALLOCATABLE_RESOURCES):
    """Return the minimum total final severity achievable under a budget.

    Bounded-knapsack DP over per-city integer allocations in ``[0, max_alloc]``
    summing to at most ``budget``. Replaces the previous "max_alloc to every
    trial" reference, which was infeasible for sequences longer than
    ``budget // max_alloc`` and capped the achievable normalised score below 1.

    Parameters
    ----------
    InitialSequenceSeverities : array-like
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
    L = len(InitialSequenceSeverities)
    per_city = [
        [_evolve_single_city(float(InitialSequenceSeverities[c]), a, L - c)
         for a in range(max_alloc + 1)]
        for c in range(L)
    ]
    INF = float('inf')
    dp = [INF] * (budget + 1)
    dp[0] = 0.0
    for c in range(L):
        new_dp = [INF] * (budget + 1)
        for b in range(budget + 1):
            base = dp[b]
            if base == INF:
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


def get_sequence_severity_from_allocations(Allocations, InitialSeverities):
    """Compute the total severity of a full sequence given allocations and initial severities.

    Parameters
    ----------
    Allocations : array-like
        Resource allocations for each trial in the sequence.
    InitialSeverities : array-like
        Initial severity values for each trial in the sequence.

    Returns
    -------
    float
        Sum of all per-trial severities after applying the allocations.
    """
    return numpy.sum(get_array_of_sequence_severities_from_allocations(Allocations, InitialSeverities))


def calculate_normalised_final_severity_performance_metric(SeveritiesFromSequence, InitialSequenceSeverities):
    """
    Calculate normalized performance metric comparing actual severity outcome to best/worst case scenarios.

    The metric ranges from 0 (worst case performance) to 1 (best case performance), representing
    how well the participant/agent performed relative to the theoretical bounds.

    Parameters
    ----------
    SeveritiesFromSequence : array-like
        Final severity values achieved for each trial in the sequence
    InitialSequenceSeverities : array-like
        Initial severity values for each trial in the sequence

    Returns
    -------
    tuple
        - Performance (float): Normalized performance metric (0-1)
        - WorstCaseSequenceSeverity (float): Sum of severities if no resources allocated
        - BestCaseSequenceSeverity (float): Sum of severities if max resources allocated
    """

    FinalSequenceSeverity = numpy.sum(SeveritiesFromSequence)
    WorstCaseAllocations = numpy.full_like(SeveritiesFromSequence, MIN_ALLOCATABLE_RESOURCES)
    # Best case is the truly optimal feasible allocation under the agent's
    # per-sequence budget, NOT "max_alloc to every trial" (infeasible for L > budget/max_alloc).
    BestCaseSequenceSeverity = _best_feasible_sequence_severity(InitialSequenceSeverities)
    WorstCaseSequenceSeverity = get_sequence_severity_from_allocations(WorstCaseAllocations, InitialSequenceSeverities)
    Performance = (WorstCaseSequenceSeverity - FinalSequenceSeverity) / \
        (WorstCaseSequenceSeverity - BestCaseSequenceSeverity)

    return Performance, WorstCaseSequenceSeverity, BestCaseSequenceSeverity


def get_array_of_sequence_severities_from_allocations(Allocations, InitialSeverities):
    """
    Calculate severity progression through a sequence given resource allocations.

    Simulates the pandemic scenario where severity evolves over time as resources
    are sequentially allocated to trials. Each trial's final severity depends on
    initial severity, resource allocation, and the combined effect of previous allocations.

    The severity update formula for each trial is:
        new_severity = max(0, SEVERITY_MULTIPLIER * initial - RESPONSE_MULTIPLIER * allocated)

    Parameters
    ----------
    Allocations : array-like
        Resource allocation amounts for each trial in sequence (0-10)
    InitialSeverities : array-like
        Initial severity value for each trial

    Returns
    -------
    list[float]
        Final severity values for each trial after resource allocation effects

    Notes
    -----
    • Severities are clipped to minimum of 0
    • The effect of allocations compounds across the sequence as each trial's outcome
      influences subsequent trials
    • Higher allocations reduce severity more effectively
    • If new_severity < 0, it is clipped to 0 (pandemic eliminated)
    """

    NumTrialsInSequence = len(InitialSeverities)
    severities = []
    resources = []

    for Trial in range(NumTrialsInSequence):

        severities . append(InitialSeverities[Trial])
        resources  . append(Allocations[Trial])

        severities = get_updated_severity(len(severities), resources, severities)

    return severities.copy()


def exit_experiment_gracefully(Message, Filehandles, MovementData, LogUtils, _PygameMediator):
    """
    Clean shutdown of experiment, closing all resources and logging final information.

    Gracefully terminates the experiment by closing files, pygame window, and
    performing final logging operations. Avoids circular import issues by
    accepting LogUtils and PygameMediator as arguments.

    Parameters
    ----------
    Message : str
        Final message to log before exit
    Filehandles : list of file
        Open file handles to close
    MovementData : tuple
        Movement tracking data to save
    LogUtils : module
        Log utilities module for final logging
    PygameMediator : module
        Pygame mediator module (currently unused for TRF-Agent mode)
    """

  # Output helpful message - use these values in config to 'resume' a subsequent experiment
    LogUtils.tee()
    LogUtils.tee(Message)

  # Tidy up remaining resources
    numpy.save(*MovementData)
    # PygameMediator.gracefully_quit_pygame()  # NOTE: Function not available for TRF-Agent mode
    for Filehandle in Filehandles:
        if Filehandle is not None:
            Filehandle.close()
    LogUtils.close_consolelog_filehandle()


def get_updated_severity(no_of_cities, resource_allocated, initial_severity) -> list[float]:
    """
    Update severity for existing cities given allocated resources.

    Updates the severity of each city based on the resources allocated to it,
    using the pandemic damage formula. This reflects how resource allocation
    reduces the growth/intensity of the pandemic in each location.

    Parameters
    ----------
    no_of_cities : int
        Number of cities/trials to update severity for
    resource_allocated : array-like
        Resources allocated to each city (0 to MAX_ALLOCATABLE_RESOURCES)
    initial_severity : array-like
        Current severity values for each city

    Returns
    -------
    list[float]
        Updated severity values, clipped to minimum of 0

    Notes
    -----
    Uses the formula: new_severity = max(0, SEVERITY_MULTIPLIER * initial - RESPONSE_MULTIPLIER * resources)
    """

    UpdatedSeverity_list = []

    for c in range(no_of_cities):

        InitialSeverityInCity = initial_severity[c]
        ResourcesAllocatedToCity = resource_allocated[c]
        NewSeverityInCity = SEVERITY_MULTIPLIER * InitialSeverityInCity - RESPONSE_MULTIPLIER * ResourcesAllocatedToCity
        NewSeverityInCity = max(NewSeverityInCity, 0)

        UpdatedSeverity_list.append(NewSeverityInCity)

    return UpdatedSeverity_list


def random_severity_generator(number_of_runs, lower_limit, upper_limit):
    """
    Generate random initial severity values following a custom probability distribution.

    Creates a distribution of severity values that can be used to randomly sample
    initial conditions for trials. Uses a normal distribution to weight the
    probability of selecting different severity levels.

    Parameters
    ----------
    number_of_runs : int
        Number of random severity values to generate
    lower_limit : int
        Minimum severity value to consider
    upper_limit : int
        Maximum severity value to consider

    Returns
    -------
    ndarray
        Array of random severity values
    """

    x = numpy.arange(lower_limit, upper_limit)
    xU, xL = x + 0.5, x - 0.5

    prob = ss.norm.cdf(xU, scale=100) - ss.norm.cdf(xL, scale=100)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1

    numpy.random.seed(3)

    nums = numpy.random.choice(x, size=number_of_runs, p=prob)

    return nums


def next_seq_length(index, seq_per_block):
    """
    Retrieve sequence lengths for the next block of sequences.

    Parameters
    ----------
    index : int
        Global sequence index to start from
    seq_per_block : int
        Number of sequences to retrieve

    Returns
    -------
    ndarray
        Array of sequence lengths for the next seq_per_block sequences
    """
    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    s = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sequence = s[index: index + seq_per_block]
    return sequence


def sampler(samples, sum_to, range_list, rn=100):
    """
    Distribute trials across sequences in a block with randomized sampling.

    Generates a random distribution of trial counts across multiple sequences
    such that the total number of trials in a block sums to a target value.
    This ensures that each sequence has a reasonable number of trials within
    specified bounds, with the overall block size remaining constant.

    Parameters
    ----------
    samples : int
        Number of sequences in the block (typically NUM_SEQUENCES = 8)
    sum_to : int
        Target total number of trials (typically TOTAL_NUM_TRIALS_IN_BLOCK = 45)
    range_list : list[int]
        [min_trials, max_trials] - Bounds on trials per sequence
        Example: [3, 10] means each sequence has 3-10 trials
    rn : int, optional
        Random seed for reproducibility. Default is 100.
        In practice, this is often set to the block number.

    Returns
    -------
    ndarray
        Array of trial counts for each sequence, summing to `sum_to`

    Raises
    ------
    ValueError
        If the specified range constraints make it impossible to reach `sum_to`
    """

    assert range_list[0] < range_list[1], "Range should be a list, the first element of which is smaller than the second"

    numpy.random.seed(rn)

    arr = numpy.random.rand(samples)

    sum_arr = sum(arr)

    new_arr = numpy.array([
                          int(item / sum_arr * sum_to)
                          if (range_list[0] < int(item / sum_arr * sum_to) < range_list[1])
                          else numpy.random.choice(range(range_list[0], range_list[1] + 1))

                          for item in arr
                          ])

    difference = sum(new_arr) - sum_to

    if len(samples) * range_list[1] < sum_to or len(samples) * range_list[0] > sum_to:
        raise ValueError(
            'The specified number of sequences is such that the desired TOTAL_NUM_TRIALS_IN_BLOCK value can never be reached')

    while difference != 0:

        if difference < 0:
            for idx in numpy.random.choice(range(len(new_arr)), abs(difference)):
                if new_arr[idx] != range_list[1]:
                    new_arr[idx] += 1

        if difference > 0:
            for idx in numpy.random.choice(range(len(new_arr)), abs(difference)):
                if new_arr[idx] != 0 and new_arr[idx] != range_list[0]:
                    new_arr[idx] -= 1

        difference = sum(new_arr) - sum_to

    return new_arr


def get_confidence_weighted_mean(all_messages, first_severity, _AbsoluteSequenceIndex, AbsoluteTrialCount):
    """
    Aggregate decisions from multiple participants using confidence-weighted mean.

    Combines resource allocation decisions from multiple participants, weighting
    each decision by the confidence reported by that participant. Handles missing
    or invalid responses (-1) gracefully.

    Note
    ----
    In the current single-agent RL pipelines (one participant), this aggregator
    reduces to ``numpy.round(response)``.

    Parameters
    ----------
    all_messages : array-like
        3D array of shape ``(num_participants, num_trials, 3)``.
        Columns: response, confidence, severity. Only columns 0 and 1 are used.
    first_severity : array-like
        Initial severity values for the sequence.
    _AbsoluteSequenceIndex : int
        Index of the current sequence (unused, kept for API consistency).
    AbsoluteTrialCount : int
        Total number of trials completed so far.

    Returns
    -------
    tuple
        - AggregatedAllocations (ndarray): Confidence-weighted mean allocations
        - SeverityFromAggregate (ndarray): Resulting severities from aggregated decisions
    """

    # First let's get the aggregated allocations
    NumTrials = numpy.shape(all_messages)[1]
    AggregatedAllocations = []

    for t in range(NumTrials):

        # Get only valid responses and confidences for this trial
        TrialResponses = numpy.array(all_messages)[:, t, 0]
        TrialConfidences = numpy.array(all_messages)[:, t, 1]

        OriginalTrialResponses = TrialResponses.copy()

        TrialResponses = TrialResponses[TrialConfidences != -1]
        TrialConfidences = TrialConfidences[TrialConfidences != -1]

        # TrialConfidences cannot be negative (-1 are already excluded).
        # If the sum is zero, all values are zero and cannot be used as weights
        # in numpy.average; set them all to 1.0 (Ticket:085).
        if numpy.sum(TrialConfidences) == 0:
            TrialConfidences[:] = 1.0

        # In the unlikely case of no valid confidences, set the allocations to
        # the plain mean of all participants.
        if numpy.size(TrialConfidences) == 0:
            ConfidenceWeightedMean = numpy.mean(OriginalTrialResponses)
        else:
            ConfidenceWeightedMean = numpy.average(TrialResponses, weights=TrialConfidences)

        AggregatedAllocations.append(ConfidenceWeightedMean)

    AggregatedAllocations = numpy.array(AggregatedAllocations)
    AggregatedAllocations = numpy.round(AggregatedAllocations)

    # Second, let's get the theoretical severity for that aggregate
    SeverityFromAggregate = get_array_of_sequence_severities_from_allocations(
        AggregatedAllocations,
        first_severity[AbsoluteTrialCount - NumTrials: AbsoluteTrialCount].copy(),
    )

    return AggregatedAllocations, SeverityFromAggregate


def get_confidence_weighted_mode(all_messages=None, first_severity=None,
                                 _AbsoluteSequenceIndex=None, AbsoluteTrialCount=None):
    """
    Aggregate decisions using confidence-weighted mode (NOT IMPLEMENTED).

    Placeholder for future implementation of confidence-weighted mode aggregation,
    which would select the most common allocation value, weighted by participant
    confidence.

    Parameters
    ----------
    all_messages : array-like, optional
        3D array of shape ``(num_participants, num_trials, 3)``.
    first_severity : array-like, optional
        Initial severity values for the sequence.
    _AbsoluteSequenceIndex : int, optional
        Index of the current sequence.
    AbsoluteTrialCount : int, optional
        Total number of trials completed so far.

    Raises
    ------
    NotImplementedError
        This method is not yet implemented.
    """
    raise NotImplementedError


def get_confidence_weighted_median(all_messages, first_severity, _AbsoluteSequenceIndex, AbsoluteTrialCount):
    """
    Aggregate decisions from multiple participants using confidence-weighted median.

    Combines resource allocation decisions from multiple participants using a
    weighted median, which is more robust to outliers than the mean. Weighting
    by confidence emphasizes decisions from more confident participants.

    Parameters
    ----------
    all_messages : array-like
        3D array of shape ``(num_participants, num_trials, 3)``.
        Columns: response, confidence, severity. Only columns 0 and 1 are used.
    first_severity : array-like
        Initial severity values for the sequence.
    _AbsoluteSequenceIndex : int
        Index of the current sequence (unused, kept for API consistency).
    AbsoluteTrialCount : int
        Total number of trials completed so far.

    Returns
    -------
    tuple
        - AggregatedAllocations (ndarray): Confidence-weighted median allocations
        - SeverityFromAggregate (ndarray): Resulting severities from aggregated decisions
    """

    # First let's get the aggregated allocations
    NumTrials = numpy.shape(all_messages)[1]
    AggregatedAllocations = []

    for t in range(NumTrials):

        # Get only valid responses and confidences for this trial
        TrialResponses = numpy.array(all_messages)[:, t, 0]
        TrialConfidences = numpy.array(all_messages)[:, t, 1]

        OriginalTrialResponses = TrialResponses.copy()

        TrialResponses = TrialResponses[TrialConfidences != -1]
        TrialConfidences = TrialConfidences[TrialConfidences != -1]

        # In the unlikely case of no valid confidences, set the allocations to
        # the plain median of all participants.
        if numpy.size(TrialConfidences) == 0:
            ConfidenceWeightedMedian = numpy.median(OriginalTrialResponses)
        else:

            # If only one valid response, duplicate for weightedmedian to work
            if numpy.size(TrialResponses) == 1:
                TrialResponses = numpy.repeat(TrialResponses, 2)
                TrialConfidences = numpy.repeat(TrialConfidences, 2)

            ConfidenceWeightedMedian = WeightedStats(
                data=TrialResponses,
                weights=TrialConfidences,
            ).quantile(
                probs=[0.5],
                return_pandas=False,
            )[0]

        AggregatedAllocations.append(ConfidenceWeightedMedian)

    AggregatedAllocations = numpy.array(AggregatedAllocations)
    AggregatedAllocations = numpy.round(AggregatedAllocations)

    # Second, let's get the theoretical severity for that aggregate
    SeverityFromAggregate = get_array_of_sequence_severities_from_allocations(
        AggregatedAllocations,
        first_severity[AbsoluteTrialCount - NumTrials: AbsoluteTrialCount].copy(),
    )

    return AggregatedAllocations, SeverityFromAggregate
