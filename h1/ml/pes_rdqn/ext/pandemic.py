'''
pes_rdqn - Pandemic Experiment Scenario: Gymnasium Environment and RDQN Algorithm

Provides the core simulation components:

- **Pandemic** (gymnasium.Env):  Gymnasium environment that models a pandemic
  resource-allocation problem.  State = (resources_left, trial_no, severity);
  action = resources to allocate (0-10).
- **rdqn_agent_meta_cognitive**:  Entropy-based meta-cognitive function that
  computes confidence and simulated response times from Q-network outputs.
- **run_experiment**:  Runs multiple sequences through the environment using
  any action-selection function and collects performance metrics.
- **RDQNTraining**:  Recurrent Deep Q-Network training loop with experience replay,
  target network, epsilon-greedy exploration, linear epsilon decay, and
  optional seed for reproducibility.

Network architecture:  Input(3) → Dense(hidden, ReLU) × N → Dense(11, linear)
  → uses normalised states in [0, 1] and Huber loss with gradient clipping.
'''

##########################
##  Imports externos    ##
##########################
import numpy
import random
import tensorflow as tf
from gymnasium import Env, spaces

##########################
##  Imports internos    ##
##########################
from .. import AVAILABLE_RESOURCES_PER_SEQUENCE
from .. import MAX_SEVERITY
from .. import MAX_ALLOCATABLE_RESOURCES
from .. import NUM_MAX_TRIALS

from .tools import entropy_from_pdf
from .rdqn_model import (build_q_network, normalize_state, ReplayBuffer,
                         train_step_rdqn, sync_target_network,
                         HistoryDeque)
from ..src.exp_utils import get_updated_severity
from ..src.exp_utils import calculate_normalised_final_severity_performance_metric


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
        # Construct the parent class
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

        # Define a 3-D observation space
        self.observation_shape = (self.available_resources_states,
                                  self.trial_no_states,
                                  self.severity_states)

        self.observation_space = spaces.Box(low=numpy.zeros(self.observation_shape, dtype=numpy.float16),
                                            high=numpy.ones(self.observation_shape, dtype=numpy.float16),
                                            dtype=numpy.float16)

        # Define an action space
        self.action_space = spaces.Discrete(self.max_allocation + 1,)

        # Create a canvas to render the environment images upon
        self.canvas = numpy.ones(self.observation_shape)

        # Define elements present inside the environment
        self.elements = []
        self.verbose = True
        self.number_cities_prob = numpy.asarray([], dtype=numpy.float64)
        self.severity_prob = numpy.asarray([], dtype=numpy.float64)

    def random_sequence(self):
        """
        Generate a random sequence with severities and allocations.

        Generates a sequence for simulation with random trial count, severities,
        and allocations. Uses uniform random values if no probability distributions
        are set, otherwise samples from the configured distributions.

        Sets
        ----
        self.seq_length : int
            Length of the randomly generated sequence
        self.initial_severities : list
            Initial severity values for each trial in the sequence
        self.allocations : list
            Resource allocations for each trial in the sequence
        """
        if (self.number_cities_prob.shape[0] == 0):
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

        Configures the environment with a predefined sequence length, initial
        severities, and optionally allocations. If allocations are not provided,
        they are randomly generated.

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

        Resets all tracking variables, initializes resources and severities,
        and returns an initial observation of the new sequence.

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
        # Reload the available resources
        self.available_resources = self.max_resources

        # Reset the reward
        self.ep_return = 0

        # City number
        self.iteration = 0

        self.severities = []
        self.resources = []

        self.severity_evolution = numpy.zeros((len(self.initial_severities) + 1, len(self.initial_severities)))
        self.severity_city_counter = 0

        self.done = False

        # Get a new city with its own severity, and keep going....
        new_severity = self.new_city()
        self.severities.append(new_severity)

        # return the observation
        return [self.available_resources, self.iteration, int(new_severity)], {}

    def render(self):
        """
        Render the current state of the environment.

        Prints human-readable information about the current episode state,
        including trial number, severities, and actions taken.

        Returns
        -------
        ndarray
            The canvas/observation array
        """
        if (self.done):
            print("--", ':',
                  ":".join([" {:5.2f}".format(sev) for sev in self.severities]), '->', ' Done!')
        elif (len(self.resources) > 0):
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

        Applies the specified action, updates the environment state, calculates
        rewards, and determines if the episode is complete.

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
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), f'Invalid Action {action}'

        # Reward for executing a step.
        reward = 0

        if ((self.available_resources - action) <= 0):
            action = self.available_resources

        self.available_resources -= action
        self.resources.append(action)

        if (self.verbose):
            self.render()

        self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities

        self.severities = get_updated_severity(len(self.severities), self.resources, self.severities)

        self.severity_city_counter = self.severity_city_counter + 1

        # Increment the episodic return
        self.ep_return += 1
        self.iteration += 1

        # Get a new city with its own severity, and keep going....
        reward = (-1) * numpy.sum(self.severities)

        # If the length of the sequence was achieved, stop
        if (self.iteration) == self.seq_length:
            done = True
            new_severity = 0

            # Update the evolution of the severity one more time for the final severity of all the cities.
            self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities
        else:
            new_severity = self.new_city()
            self.severities.append(new_severity)

        return [self.available_resources, self.iteration, int(new_severity)], reward, done, False, {}


def rdqn_agent_meta_cognitive(options, resources_left, response_timeout):
    """
    Compute meta-cognitive confidence and response time estimates from Q-network outputs.

    This function evaluates the entropy of the Q-value distribution to determine
    agent confidence and maps that confidence to human-like response times
    (reaction hold and release times).

    Parameters
    ----------
    options : array-like
        Q-values produced by the Q-network for the current state.  Shape: ``(n_actions,)``.
    resources_left : int
        Number of resources remaining.
    response_timeout : float
        Maximum response time allowed in milliseconds.

    Returns
    -------
    response : int
        The selected action (argmax of feasible options).
    confidence : float
        Normalised confidence score based on entropy (range: typically 0-1).
        Lower entropy → higher confidence.
    rt_hold : float
        Response time for button hold phase (in seconds).
    rt_release : float
        Response time for button release phase (in seconds).

    Notes
    -----
    - Confidence is calculated as:
      ``(entropy - min_entropy) / (max_entropy - min_entropy)``
    - Response times are sampled from normal distributions parameterised by confidence.
    - Both ``rt_hold`` and ``rt_release`` are clipped to ``[0, response_timeout/1000]``.
    """

    # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((len(options),),)
    m_entropy[0] = 1

    # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((len(options),),)

    # Calculate the entropy of the options distribution
    _entrp1 = entropy_from_pdf(options)

    o = numpy.arange(len(options), dtype=numpy.float32)

    # Mask infeasible actions (consistent with optimize_rdqn / RDQNTraining)
    options[o > resources_left] = -1e9

    # available resources, trial, severity
    dec_entropy = entropy_from_pdf(options)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    # Calculate confidence as a normalized inverse of entropy
    confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)

    # Select the action with the highest Q-value as the response
    response = numpy.argmax(options)

    # Ensure response never exceeds available resources
    response = int(numpy.clip(response, 0, int(resources_left)))

    # Map confidence to response times using a linear transformation
    def map_to_response_time(x):
        """Map confidence to a response-time scale via linear transform."""
        return x * (-2) + 1
    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    _meta_rng = numpy.random.RandomState(abs(int(confidence * 1e6)) % (2**31))
    rt_hold = _meta_rng.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + _meta_rng.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip(rt_hold, 0, response_timeout / 1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout / 1000.0)

    return response, confidence, rt_hold, rt_release


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
    if (RandomSequences):
        env.random_sequence()
    elif (AssignAllocations):
        assert trials_per_sequence is not None and sevs is not None and allocs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid], allocs[seqid])
    else:
        assert trials_per_sequence is not None and sevs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
    state, _ = env.reset()
    seqs = []
    perfs = []
    seq_ev = []
    ITERATIONS = NumberOfIterations
    while seqid < ITERATIONS:
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

            if seqid < ITERATIONS:
                if (RandomSequences):
                    env.random_sequence()
                elif (AssignAllocations):
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


def RDQNTraining(env, learning_rate, discount, epsilon, min_eps, episodes,
                 hidden_units=None, batch_size=64, buffer_size=50_000,
                 target_sync_freq=1_000, max_grad_norm=1.0, seed=None,
                 penalty_coeff=0.0, compute_confidence=True,
                 pruning_callback=None,
                 warmup_ratio=0.05, target_ratio=0.60,
                 learning_starts=None,
                 history_len=6, lstm_units=64):
    """
    Train a Recurrent Deep Q-Network agent on the Pandemic environment.

    Replaces the tabular Q-table from ``pes_ql`` with a neural-network
    function approximator.  Key RDQN components:

    - **Experience replay**: transitions are stored in a circular buffer
      and uniformly sampled in mini-batches to break temporal correlation.
    - **Target network**: a frozen copy of the online Q-network provides
      stable TD targets; it is hard-synced every ``target_sync_freq`` steps.
    - **Epsilon-greedy** exploration with exponential decay and warm-up.
    - **PBRS** (optional): Potential-Based Reward Shaping (Ng et al., 1999)
      augments the reward with ``β·(γ·Φ(s') − Φ(s))`` where
      ``Φ(s) = −Σ max(0, sᵢ)``.

    Parameters
    ----------
    env : Pandemic
        The Pandemic environment instance to train on.
    learning_rate : float
        Adam optimiser learning rate.
    discount : float
        Discount factor (γ) for TD targets.
    epsilon : float
        Initial exploration rate.
    min_eps : float
        Minimum exploration rate after decay.
    episodes : int
        Number of training episodes.
    hidden_units : list of int or None, optional
        Hidden-layer widths for the Q-network.  Default: ``[64, 64]``.
    batch_size : int, optional
        Mini-batch size for replay sampling.  Default: 64.
    buffer_size : int, optional
        Replay buffer capacity.  Default: 50 000.
    target_sync_freq : int, optional
        Steps between target network hard syncs.  Default: 1 000.
    max_grad_norm : float, optional
        Global gradient norm clipping threshold.  Default: 1.0.
    seed : int or None, optional
        Random seed for reproducibility.  Default: ``None``.
    penalty_coeff : float, optional
        PBRS reward shaping coefficient (β).  When > 0, the reward is
        augmented: ``r' = r + β·(γ·Φ(s') − Φ(s))`` where
        ``Φ(s) = −Σ max(0, sᵢ)``.  Set to 0 to disable.  Default: ``0.0``.
    compute_confidence : bool, optional
        If ``True``, compute meta-cognitive confidence at every step
        (requires an extra forward pass).  Set to ``False`` during
        optimisation to save ~33 % of forward-pass time.  Default: ``True``.
    pruning_callback : callable or None, optional
        Called every 10 000 episodes with ``(episode_index, avg_reward)``.
        If it returns ``True``, training stops early (used by Optuna
        MedianPruner).  Default: ``None``.
    warmup_ratio : float, optional
        Fraction of episodes during which ε is held constant at its initial
        value (pure exploration phase).  Default: ``0.05``.
    target_ratio : float, optional
        Fraction of total episodes at which ε reaches ``min_eps`` via
        exponential decay.  Default: ``0.60``.
    learning_starts : int or None, optional
        Minimum number of transitions stored in the replay buffer before
        gradient updates begin (RDQN warm-up).  When ``None`` it defaults
        to ``max(10 * batch_size, buffer_size // 10)``.  Standard RDQN
        practice (Mnih et al., 2015) delays learning to decorrelate the
        buffer.
    history_len : int, optional
        Length of the sliding window of past normalised states fed to
        the recurrent Q-network.  Default: 6.
    lstm_units : int, optional
        Hidden-state width of the LSTM trunk.  Default: 64.

    Returns
    -------
    ave_reward_list : list of float
        Average reward computed every 10 000 episodes.
    online_net : tf.keras.Model
        Trained Q-network.
    conf_list : list of float
        Meta-cognitive confidence values recorded during training.
    """

    if hidden_units is None:
        hidden_units = [64, 64]

    # Seed ALL RNGs for reproducibility (numpy, python, TF, hash)
    if seed is not None:
        import os as _os
        _os.environ['PYTHONHASHSEED'] = str(seed)
        tf.keras.utils.set_random_seed(seed)   # seeds numpy + random + tf in one call
        env.action_space.seed(seed)

    # Dedicated RNG for ε-greedy decisions: decouples action-selection
    # randomness from the global numpy RNG used by env.random_sequence(),
    # so the sequence stream is identical across hyperparameter trials.
    eps_rng = numpy.random.default_rng(seed)

    state_dim = 3
    action_dim = env.action_space.n

    # Build online and target networks (deterministic init when seed is set)
    online_net = build_q_network(state_dim, action_dim, hidden_units,
                                 history_len=history_len,
                                 lstm_units=lstm_units, seed=seed)
    target_net = build_q_network(state_dim, action_dim, hidden_units,
                                 history_len=history_len,
                                 lstm_units=lstm_units, seed=seed)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Build the optimiser so that variable slots exist before the first call
    online_net(tf.zeros((1, history_len, state_dim)))
    target_net(tf.zeros((1, history_len, state_dim)))
    optimizer.build(online_net.trainable_variables)
    sync_target_network(online_net, target_net)

    replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    # Wrap train_step_rdqn with tf.function for this training session.
    # ``reduce_retracing=True`` keeps the trace cache small across Optuna
    # trials, matching the pattern used in pes_a2c's A2CTraining.
    # NOTE: jit_compile=True (XLA) is intentionally omitted. The cuDNN LSTM
    # kernel (CudnnRNNV3) is not supported by XLA_GPU_JIT, so combining
    # use_cudnn=True with jit_compile=True raises InvalidArgumentError at
    # trace time. The cuDNN kernel already provides the GPU speed-up.
    compiled_train_step = tf.function(train_step_rdqn, reduce_retracing=True)

    discount_t = tf.constant(discount, dtype=tf.float32)
    max_grad_norm_t = tf.constant(max_grad_norm, dtype=tf.float32)
    max_resources_t = tf.constant(env.max_resources, dtype=tf.float32)

    # Replay-buffer warm-up: don't train until we have enough samples
    if learning_starts is None:
        learning_starts = max(10 * int(batch_size), int(buffer_size) // 10)
    learning_starts = max(int(learning_starts), int(batch_size))

    reward_list = []
    ave_reward_list = []
    conf_list = []

    # Exponential ε-decay with warm-up (adapted from pes_a2c)
    epsilon_initial = epsilon
    warmup_episodes = int(warmup_ratio * episodes)
    resolved_decay_rate = (min_eps / max(epsilon, 1e-8)) ** (
        1.0 / max(1, int((target_ratio - warmup_ratio) * episodes))
    )
    global_step = 0

    history = HistoryDeque(history_len, state_dim)

    for i in range(episodes):
        done = False
        tot_reward = 0.0
        env.random_sequence()
        state, _ = env.reset()
        history.reset()

        while not done:
            norm_state = normalize_state(state, env.max_resources,
                                         env.max_seq_length, env.max_severity)
            history.append_step(norm_state)
            state_window = history.current_window()

            # Single forward pass; reused by greedy branch and confidence.
            q_vals = online_net(state_window[numpy.newaxis], training=False).numpy()[0]

            # Epsilon-greedy action selection (random branch is also masked
            # to feasible actions so the replay buffer never stores infeasible
            # transitions; otherwise env.step clamps the action and the stored
            # (action, reward) pair becomes inconsistent).
            feasible = numpy.arange(action_dim) <= state[0]
            if eps_rng.random() < epsilon:
                feasible_actions = numpy.flatnonzero(feasible)
                action = int(eps_rng.choice(feasible_actions))
            else:
                q_masked = numpy.where(feasible, q_vals, -1e9)
                action = int(numpy.argmax(q_masked))

            # Compute confidence (meta-cognitive tracking) — skip if disabled
            if compute_confidence:
                _, confidence, _, _ = rdqn_agent_meta_cognitive(
                    q_vals.copy(), state[0], 10000)
                conf_list.append(confidence)

            # PBRS: compute potential Φ(s) BEFORE the step
            phi_s = 0.0
            if penalty_coeff > 0.0:
                phi_s = -sum(max(0.0, sv) for sv in env.severities)

            state2, reward, done, _truncated, _info = env.step(action)

            # Potential-Based Reward Shaping (Ng et al., 1999)
            # F(s, s') = β · (γ · Φ(s') − Φ(s)),  Φ(s) = −Σ max(0, sᵢ)
            if penalty_coeff > 0.0:
                phi_s_prime = 0.0 if done else -sum(max(0.0, sv) for sv in env.severities)
                reward += penalty_coeff * (discount * phi_s_prime - phi_s)

            norm_state2 = normalize_state(state2, env.max_resources,
                                           env.max_seq_length, env.max_severity)
            # Build the next-state window WITHOUT mutating the live history
            # deque so the unaltered window is what the next iteration sees.
            next_window = numpy.concatenate(
                [state_window[1:], norm_state2[numpy.newaxis]], axis=0)

            replay_buffer.push(state_window, action, reward, next_window, done)

            # Train only after the buffer warm-up
            if len(replay_buffer) >= learning_starts:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(batch_size)
                compiled_train_step(
                    online_net, target_net, optimizer,
                    tf.constant(s_b), tf.constant(a_b),
                    tf.constant(r_b), tf.constant(ns_b),
                    tf.constant(d_b), discount_t, max_grad_norm_t,
                    max_resources_t,
                )

            global_step += 1

            # Periodically sync target network
            if global_step % target_sync_freq == 0:
                sync_target_network(online_net, target_net)

            tot_reward += reward
            state = state2

        # Exponential ε-decay with warm-up
        if i < warmup_episodes:
            epsilon = epsilon_initial                # Phase 1: pure exploration
        else:
            epsilon = max(min_eps,                   # Phase 2: exponential decay
                          epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))

        reward_list.append(tot_reward)

        if (i + 1) % 10_000 == 0:
            # Rolling mean over the last 10 000 episodes (window kept, not
            # reset, so the pruner sees a stable signal across reports).
            ave_reward = float(numpy.mean(reward_list[-10_000:]))
            ave_reward_list.append(ave_reward)
            print(f'Episode {i + 1} Average Reward: {ave_reward}')

            # Pruning callback for early stopping (Optuna MedianPruner)
            if pruning_callback is not None:
                if pruning_callback(i, ave_reward):
                    break

    env.close()

    return ave_reward_list, online_net, conf_list
