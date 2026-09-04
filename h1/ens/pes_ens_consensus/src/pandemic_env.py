"""Pandemic Gymnasium environment and experiment runner, mirrored from ``ml/pes_dqn/ext/pandemic.py``.

Only the environment (:class:`Pandemic`) and :func:`run_experiment` are kept —
the model-training classes from the original module are not needed for
ensemble evaluation, so this copy has no dependency on any ``ml/`` model file.
"""

import random
import numpy
from gymnasium import Env, spaces

from .env_constants import AVAILABLE_RESOURCES_PER_SEQUENCE, MAX_ALLOCATABLE_RESOURCES, MAX_SEVERITY, NUM_MAX_TRIALS
from .exp_utils_env import calculate_normalised_final_severity_performance_metric, get_updated_severity


class Pandemic(Env):
    """
    Pandemic environment implementing Gymnasium's Env interface.

    The Pandemic environment simulates a pandemic response scenario where an agent
    must allocate limited resources across multiple cities to minimize final severity.
    Each episode consists of multiple sequences, and each sequence contains multiple trials.

    Attributes
    ----------
    max_resources : int
        Maximum resources available per sequence (after 9 are pre-assigned)
    available_resources_states : int
        Number of possible resource states (max_resources + 1)
    max_seq_length : int
        Maximum number of trials per sequence
    trial_no_states : int
        Number of possible trial number states (max_seq_length + 1)
    max_severity : int
        Maximum initial severity value
    severity_states : int
        Number of possible severity states (max_severity + 1)
    max_allocation : int
        Maximum resources that can be allocated in a single action
    observation_space : spaces.Box
        3D observation space for [available_resources, trial_number, severity]
    action_space : spaces.Discrete
        Discrete action space representing resource allocations (0 to max_allocation)
    """

    def __init__(self):
        """
        Initialize the Pandemic environment.

        Sets up the state and action spaces, initializes internal variables,
        and configures the environment for simulation.
        """
        super(Pandemic, self).__init__()

        # Number of available resources at the beginning (9 are preassigned)
        self.max_resources = AVAILABLE_RESOURCES_PER_SEQUENCE - 9
        self.available_resources_states = self.max_resources + 1

        # Ten trials per sequence, from 3 to 10
        self.max_seq_length = NUM_MAX_TRIALS
        self.trial_no_states = self.max_seq_length + 1

        # Ten severities, from 0 to 10
        self.max_severity = MAX_SEVERITY
        self.severity_states = self.max_severity + 1

        # Ten is the max alloc, Eleven choices, from 0 to 10
        self.max_allocation = MAX_ALLOCATABLE_RESOURCES

        self.observation_shape = (self.available_resources_states,
                                  self.trial_no_states,
                                  self.severity_states)

        self.observation_space = spaces.Box(low=numpy.zeros(self.observation_shape, dtype=numpy.float16),
                                            high=numpy.ones(self.observation_shape, dtype=numpy.float16),
                                            dtype=numpy.float16)

        self.action_space = spaces.Discrete(self.max_allocation + 1,)

        self.canvas = numpy.ones(self.observation_shape)

        self.elements = []
        self.verbose = True
        self.number_cities_prob = numpy.asarray([], dtype=numpy.float64)
        self.severity_prob = numpy.asarray([], dtype=numpy.float64)

    def random_sequence(self):
        """
        Generate a random sequence with severities and allocations.

        Sets
        ----
        self.seq_length : int
            Length of the randomly generated sequence
        self.initial_severities : list
            Initial severity values for each trial in the sequence
        self.allocations : list
            Resource allocations for each trial in the sequence
        """
        if self.number_cities_prob.shape[0] == 0:
            self.seq_length = random.randrange(int(3), int(self.max_seq_length))
            self.allocations = [self.action_space.sample() for _s in range(self.seq_length)]
            self.initial_severities = [random.randrange(int(0), int(self.max_severity))
                                       for _s in range(self.seq_length)]
        else:
            self.seq_length = int(numpy.random.choice(self.number_cities_prob[:, 0], p=(self.number_cities_prob[:, 1])))
            self.initial_severities = numpy.random.choice(
                self.severity_prob[:, 0], size=(self.seq_length,), p=self.severity_prob[:, 1])

    def set_fixed_sequence(self, length, init_severities, allocs=None):
        """
        Set a fixed sequence with specified parameters.

        Parameters
        ----------
        length : int
            Number of trials in the sequence
        init_severities : array-like
            Initial severity values for each trial
        allocs : array-like, optional
            Resource allocations for each trial. If None, allocations are randomly
            generated. Default: None
        """
        self.seq_length = int(length)
        self.set_initial_severities(init_severities)
        if allocs is None:
            self.allocations = [0] * self.seq_length
        else:
            self.set_fixed_allocations(allocs)

    def set_fixed_allocations(self, allocs):
        """
        Set fixed resource allocations for the current sequence.

        Parameters
        ----------
        allocs : array-like
            Resource allocations for each trial in the sequence
        """
        self.allocations = allocs

    def set_initial_severities(self, init_severities):
        """
        Set the initial severity values for the current sequence.

        Parameters
        ----------
        init_severities : array-like
            Initial severity value for each trial in the sequence
        """
        self.initial_severities = init_severities

    def new_city(self):
        """
        Get the initial severity for the next city/trial.

        Returns
        -------
        float
            The initial severity value of the current iteration
        """
        return self.initial_severities[self.iteration]

    def sample(self):
        """
        Get the allocated resources for the current trial.

        Returns
        -------
        int
            Resource allocation for the current iteration
        """
        return self.allocations[self.iteration]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed : int or None, optional
            Random seed (unused, kept for Gym API compatibility).
        options : dict or None, optional
            Extra reset options (unused, kept for Gym API compatibility).

        Returns
        -------
        tuple
            - observation (list): Initial observation
              ``[available_resources, trial_number, initial_severity]``
            - info (dict): Empty info dict (Gymnasium API)
        """
        self.available_resources = self.max_resources
        self.ep_return = 0
        self.iteration = 0
        self.severities = []
        self.resources = []
        self.severity_evolution = numpy.zeros((len(self.initial_severities) + 1, len(self.initial_severities)))
        self.severity_city_counter = 0
        self.done = False
        new_severity = self.new_city()
        self.severities.append(new_severity)
        return [self.available_resources, self.iteration, int(new_severity)], {}

    def render(self):
        """
        Render the current state of the environment.

        Returns
        -------
        ndarray
            The canvas/observation array
        """
        if self.done:
            print("--", ':',
                  ":".join([" {:5.2f}".format(sev) for sev in self.severities]), '->', ' Done!')
        elif len(self.resources) > 0:
            print("{:02d}".format(self.iteration + 1), ':',
                  ":".join(["{:5.2f}".format(sev) for sev in self.severities]), '->', self.resources[-1])
        return self.canvas

    def close(self):
        """
        Close the environment and clean up resources.

        Placeholder method for environment cleanup (currently does nothing).
        """

    def get_action_meanings(self):
        """
        Get the mapping between action indices and their meanings.

        Returns
        -------
        dict
            Dictionary mapping action indices (0-10) to resource allocation amounts
        """
        return {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10"}

    def damage(self):
        """
        Calculate the updated severity based on current allocations.

        Returns
        -------
        ndarray
            Updated severity values for all trials based on resource allocations
        """
        return get_updated_severity(len(self.severities), self.resources, self.severities)

    def step(self, action):
        """
        Execute one step of the environment.

        Parameters
        ----------
        action : int
            The action to take (resource allocation amount, 0-10)

        Returns
        -------
        tuple
            - observation (list): New state [available_resources, trial_number, severity]
            - reward (float): Reward for this step (negative sum of severities)
            - done (bool): Whether the episode is finished
            - truncated (bool): Always ``False`` (no time-limit truncation)
            - info (dict): Additional information (empty dict)
        """
        done = False
        assert self.action_space.contains(action), f'Invalid Action {action}'
        reward = 0

        if (self.available_resources - action) <= 0:
            action = self.available_resources

        self.available_resources -= action
        self.resources.append(action)

        if self.verbose:
            self.render()

        self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities
        self.severities = get_updated_severity(len(self.severities), self.resources, self.severities)
        self.severity_city_counter = self.severity_city_counter + 1
        self.ep_return += 1
        self.iteration += 1
        reward = (-1) * numpy.sum(self.severities)

        if self.iteration == self.seq_length:
            done = True
            new_severity = 0
            self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities
        else:
            new_severity = self.new_city()
            self.severities.append(new_severity)

        return [self.available_resources, self.iteration, int(new_severity)], reward, done, False, {}


def run_experiment(env, actionfunction, RandomSequences=True,
                   trials_per_sequence=None, sevs=None,
                   AssignAllocations=False, allocs=None,
                   NumberOfIterations=64):
    """
    Execute a pandemic simulation experiment over multiple sequences.

    Runs an experiment in the Pandemic environment, executing a specified action function
    at each step and collecting performance metrics across multiple sequences. Supports both
    random and fixed sequence generation with optional pre-defined severities and allocations.

    Parameters
    ----------
    env : Pandemic
        The Pandemic environment instance to run the experiment on.
    actionfunction : callable
        Function that takes (env, state, sequence_id) and returns an action (int).
    RandomSequences : bool, optional
        If True, generates random sequences. If False, uses fixed sequences from parameters.
        Default: True
    trials_per_sequence : array-like, optional
        Number of trials in each sequence. Required if RandomSequences=False or
        AssignAllocations=False. Shape: (NumberOfIterations,)
    sevs : array-like, optional
        Initial severity values for each trial in each sequence. Required if
        RandomSequences=False. Shape: (NumberOfIterations, variable_length)
    AssignAllocations : bool, optional
        If True, uses pre-defined allocations from the 'allocs' parameter.
        Default: False
    allocs : array-like, optional
        Pre-defined resource allocations for each trial. Required if
        AssignAllocations=True. Shape: (NumberOfIterations, variable_length)
    NumberOfIterations : int, optional
        Number of sequences to simulate. Default: 64

    Returns
    -------
    seqs : list
        Total severity sum for each completed sequence. Shape: (NumberOfIterations,)
    perfs : list
        Normalized performance metric (final severity / initial severity) for each sequence.
        Shape: (NumberOfIterations,)
    seq_ev : list
        Severity evolution over time for each sequence. Each element contains the
        evolution matrix for that sequence.
    """
    seqid = 0
    if RandomSequences:
        env.random_sequence()
    elif AssignAllocations:
        assert trials_per_sequence is not None and sevs is not None and allocs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid], allocs[seqid])
    else:
        assert trials_per_sequence is not None and sevs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
    state, _ = env.reset()
    seqs = []
    perfs = []
    seq_ev = []
    iterations = NumberOfIterations
    while seqid < iterations:
        print(f'State: {state}')
        action = actionfunction(env, state, seqid)
        state2, _reward, done, _truncated, _info = env.step(action)

        if done:
            env.done = True
            env.render()
            seqs.append(numpy.sum(env.severities))
            perf = calculate_normalised_final_severity_performance_metric(env.severities,
                                                                          env.initial_severities)
            perfs.append(perf[0])
            seq_ev.append(env.severity_evolution)
            seqid = seqid + 1

            if seqid < iterations:
                if RandomSequences:
                    env.random_sequence()
                elif AssignAllocations:
                    assert trials_per_sequence is not None and sevs is not None and allocs is not None
                    env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid], allocs[seqid])
                else:
                    assert trials_per_sequence is not None and sevs is not None
                    env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
            state2, _ = env.reset()

        state = state2

    print(numpy.array(seqs))
    env.close()

    return seqs, perfs, seq_ev
