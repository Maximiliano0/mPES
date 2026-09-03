'''
pes_a2c - Pandemic Experiment Scenario: Gymnasium Environment and A2C Training

Provides the core simulation components:

- **Pandemic** (gymnasium.Env):  Gymnasium environment that models a pandemic
  resource-allocation problem.  State = (resources_left, trial_no, severity);
  action = resources to allocate (0-10).
- **ac_agent_meta_cognitive**:  Actor-policy-based meta-cognitive function that
  computes confidence and simulated response times from the Actor's π(a|s)
  probability distribution (a theoretically grounded entropy measure).
- **run_experiment**:  Runs multiple sequences through the environment using
  any action-selection function and collects performance metrics.
- **A2CTraining**:  Advantage Actor-Critic training loop with on-policy
  softmax sampling over feasible actions, entropy regularisation,
  GAE(\u03bb) advantage estimation, gradient clipping, cosine-annealed
  learning rates, optional PBRS reward shaping, training-only spending
  cost, and optional seed for reproducibility.

Network architecture
--------------------
::

    Actor:   Input(3)  →  Dense(128, ReLU)  →  Dense(11, softmax)
    Critic:  Input(3)  →  Dense(128, ReLU)  →  Dense(1, linear)

The hidden-layer sizes are configurable via ``config/CONFIG.py``
(``AC_ACTOR_HIDDEN_UNITS = [128]``, ``AC_CRITIC_HIDDEN_UNITS = [128]``).

State normalisation: [resources/30, trial/10, severity/9] → [0, 1]³.
'''

##########################
##  Imports externos    ##
##########################
import numpy
import random
from gymnasium import Env, spaces
import tensorflow as tf

##########################
##  Imports internos    ##
##########################
from .. import AVAILABLE_RESOURCES_PER_SEQUENCE
from .. import MAX_SEVERITY
from .. import MAX_ALLOCATABLE_RESOURCES
from .. import NUM_MAX_TRIALS

from .tools import entropy_from_pdf
from .ac_model import (build_actor, build_critic, normalize_state,
                       train_step_actor_critic)
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
            Resource allocations for each trial in the sequence.
            **Only set when no probability distributions are configured**
            (i.e. the ``if`` branch).  When distributions are configured
            (the ``else`` branch), only *seq_length* and
            *initial_severities* are set.
        """
        if (self.number_cities_prob.shape[0] == 0):
            self.seq_length = random.randrange(int(3), int(self.max_seq_length))
            self.allocations = [self.action_space.sample() for s in range(self.seq_length)]
            self.initial_severities = [random.randrange(int(0), int(self.max_severity)) for s in range(self.seq_length)]
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
            self.allocations = [self.action_space.sample() for s in range(self.seq_length)]
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


def ac_agent_meta_cognitive(policy_probs, resources_left, response_timeout):
    """
    Compute meta-cognitive confidence and response time estimates from Actor policy output.

    Unlike Q-Learning or DQN (which compute entropy over Q-values — a heuristic),
    this function uses the Actor's π(a|s) output **directly**.  Since π(a|s) is
    already a proper probability distribution, its entropy is a theoretically
    grounded measure of decision uncertainty.

    Confidence formula::

        confidence = (H(π_feasible) - H_max) / (H_min - H_max)

    where H_max = log₂(n_actions) (uniform), H_min ≈ 0 (peaked), and
    H(π_feasible) is the Shannon entropy of the policy restricted to feasible
    actions (allocation ≤ resources_left).

    Parameters
    ----------
    policy_probs : array-like
        Action probabilities from the Actor's softmax output.  Shape: ``(n_actions,)``.
    resources_left : int
        Number of resources remaining.
    response_timeout : float
        Maximum response time allowed in milliseconds.

    Returns
    -------
    response : int
        The selected action (argmax of feasible probabilities).
    confidence : float
        Normalised confidence score based on entropy (range: typically 0-1).
        Lower entropy → higher confidence.
    rt_hold : float
        Response time for button hold phase (in seconds).
    rt_release : float
        Response time for button release phase (in seconds).

    Notes
    -----
    - Infeasible actions (allocation > resources_left) are set to a tiny value
      and the distribution is **renormalised** before entropy computation and
      action selection.
    - Confidence is computed as ``(H − H_max) / (H_min − H_max)`` using the
      min/max entropy reference distributions.
    - Response times are sampled from normal distributions parameterised by
      confidence.
    - Both *rt_hold* and *rt_release* are clipped to ``[0, response_timeout / 1000]``.
    """

    # Ensure we work with a numpy copy
    probs = numpy.array(policy_probs, dtype=numpy.float64).flatten()

    # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((len(probs),),)
    m_entropy[0] = 1

    # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((len(probs),),)

    # Calculate the entropy of the raw policy distribution
    _entrp1 = entropy_from_pdf(probs)

    o = numpy.arange(len(probs), dtype=numpy.float32)

    # Zero-out infeasible actions (allocation > resources left) and
    # renormalise so that argmax and entropy are computed only over
    # feasible actions — consistent with the optimizer's evaluation.
    probs[o > resources_left] = 0.00001  # Tiny value to avoid zero probabilities
    total = numpy.sum(probs)
    if total > 0:
        probs = probs / total

    # available resources, trial, severity
    dec_entropy = entropy_from_pdf(probs)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    # Calculate confidence as a normalized inverse of entropy
    confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)

    # Select the action with the highest probability as the response
    response = numpy.argmax(probs)

    # Map confidence to response times using a linear transformation
    def map_to_response_time(x): return x * (-2) + 1
    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip(rt_hold, 0, response_timeout / 1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout / 1000.0)

    return response, confidence, rt_hold, rt_release


def run_experiment(env, actionfunction, RandomSequences=True,
                   trials_per_sequence=None, sevs=None,
                   NumberOfIterations=64, verbose=True):
    """
    Execute a pandemic simulation experiment over multiple sequences.

    Runs the Pandemic environment with a specified action function at each step,
    collecting performance metrics across multiple sequences.  Supports both
    random and fixed sequence generation with optional pre-defined severities.

    Parameters
    ----------
    env : Pandemic
        The Pandemic environment instance to run the experiment on.
    actionfunction : callable
        Function that takes ``(env, state, sequence_id)`` and returns an action (int).
    RandomSequences : bool, optional
        If ``True``, generates random sequences.  If ``False``, uses fixed sequences
        from *trials_per_sequence* and *sevs*.  Default: ``True``.
    trials_per_sequence : array-like, optional
        Number of trials in each sequence.  Required when ``RandomSequences=False``.
        Shape: ``(NumberOfIterations,)``.
    sevs : array-like, optional
        Initial severity values for each trial in each sequence.  Required when
        ``RandomSequences=False``.  Shape: ``(NumberOfIterations, variable_length)``.
    NumberOfIterations : int, optional
        Number of sequences to simulate.  Default: 64.
    verbose : bool, optional
        When ``True``, print state and sequence information during the
        experiment.  Set to ``False`` to suppress output (useful during
        Bayesian optimisation).  Default: ``True``.

    Returns
    -------
    seqs : list of float
        Total severity sum for each completed sequence.
    perfs : list of float
        Normalised performance metric for each sequence.
    seq_ev : list
        Severity evolution matrix for each sequence.
    """

    seqid = 0
    if RandomSequences:
        env.random_sequence()
    else:
        assert trials_per_sequence is not None and sevs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
    state, _ = env.reset()
    seqs = []
    perfs = []
    seq_ev = []
    ITERATIONS = NumberOfIterations
    while seqid < ITERATIONS:
        if verbose:
            print(f'State: {state}')
        action = actionfunction(env, state, seqid)
        state2, _reward, done, _truncated, _info = env.step(action)

        if done:
            env.done = True
            if verbose:
                env.render()
            seqs.append(numpy.sum(env.severities))
            perf = calculate_normalised_final_severity_performance_metric(env.severities,
                                                                          env.initial_severities)
            perfs.append(perf[0])
            seq_ev.append(env.severity_evolution)
            seqid = seqid + 1

            if seqid < ITERATIONS:
                if RandomSequences:
                    env.random_sequence()
                else:
                    assert trials_per_sequence is not None and sevs is not None
                    env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
            state2, _ = env.reset()

        state = state2

    if verbose:
        print(numpy.array(seqs))
    env.close()

    return seqs, perfs, seq_ev


def A2CTraining(env, actor_lr, critic_lr, discount, entropy_coeff,
                epsilon, min_eps, episodes,
                actor_hidden, critic_hidden,
                seed=None, compute_confidence=False,
                verbose=True,
                warmup_ratio=0.05, target_ratio=0.60,
                penalty_coeff=0.0, gae_lambda=0.95,
                max_grad_norm=0.5, lr_min_ratio=0.1,
                spend_cost_coeff=0.0, last_action_bias=0.0,
                progress_callback=None):
    """
    Train an Advantage Actor-Critic (A2C) agent on the Pandemic environment.

    Uses on-policy updates — transitions collected during each episode are
    used to update both the Actor (policy network) and the Critic (value
    network) in a **single batched gradient step at the end of the episode**.
    Action selection is **pure on-policy softmax sampling** from the
    Actor distribution renormalised over feasible actions; the legacy
    ε-greedy schedule is still computed for API compatibility but does
    not influence action selection.

    Parameters
    ----------
    env : Pandemic
        Gym environment instance.
    actor_lr : float
        Adam optimiser learning rate for the Actor.
    critic_lr : float
        Adam optimiser learning rate for the Critic.
    discount : float
        Discount factor γ ∈ (0, 1].
    entropy_coeff : float
        Weight for the entropy bonus in the Actor loss.  When 0, the
        bonus is disabled and exploration relies solely on the softmax
        randomness of π_θ.
    epsilon : float
        Legacy initial exploration rate ε₀.  Unused under on-policy
        sampling but retained for API compatibility.
    min_eps : float
        Legacy minimum exploration rate ε_min.  Unused (see above).
    episodes : int
        Total number of training episodes.
    actor_hidden : list of int
        Widths of the hidden dense layers for the Actor (e.g. ``[64, 64]``).
    critic_hidden : list of int
        Widths of the hidden dense layers for the Critic (e.g. ``[64, 64]``).
    seed : int or None, optional
        Random seed for full reproducibility.  Propagated to ``numpy``,
        ``random``, ``tf.random`` and per-layer ``GlorotUniform``
        initialisers; also triggers
        ``tf.config.experimental.enable_op_determinism()``.
        Default: ``None``.
    compute_confidence : bool, optional
        When ``True``, run ``ac_agent_meta_cognitive`` every step to
        record meta-cognitive confidence values.  **Disabling this
        (default) reduces computation during training.**
        Default: ``False``.
    verbose : bool, optional
        When ``True``, print average reward every 10 000 episodes.
        Set to ``False`` to suppress training output (useful during
        Bayesian optimisation).  Default: ``True``.
    warmup_ratio : float, optional
        Legacy: fraction of episodes during which ε stays at ε₀.
        Currently has no effect on action selection.  Default: ``0.05``.
    target_ratio : float, optional
        Legacy: fraction of episodes at which ε reaches *min_eps*.
        Currently has no effect on action selection.  Default: ``0.60``.
    penalty_coeff : float, optional
        PBRS reward shaping coefficient β.  When > 0, the reward is
        augmented: r' = r + β·(γ·Φ(s') − Φ(s)) where Φ(s) = −Σ sᵢ.
        Set to 0 to disable.  Default: ``0.0``.
    gae_lambda : float, optional
        GAE(λ) parameter controlling the bias-variance trade-off of
        the advantage estimator.  λ=0 → TD(0); λ=1 → MC.
        Default: ``0.95``.
    max_grad_norm : float, optional
        Global gradient norm clipping threshold applied to both Actor
        and Critic gradients.  Default: ``0.5``.
    lr_min_ratio : float, optional
        Minimum learning rate as a fraction of the initial learning
        rate, used by cosine annealing.  Default: ``0.1``.
    spend_cost_coeff : float, optional
        Per-action spending cost added to the training reward as
        ``r' = r − spend_cost_coeff · action``.  Discourages the
        degenerate "spend the maximum feasible amount on every
        trial" policy that otherwise saturates the normalised
        performance metric (≈0.858) regardless of training quality.
        Set to 0 to disable.  **Applied only during training**, so
        the evaluation metric remains unbiased.  Default: ``0.0``.
    last_action_bias : float, optional
        Initial bias logit applied to the last action of the Actor
        policy head.  Negative values reduce the initial probability
        of the "spend max feasible" action, breaking the symmetry
        that drives ``argmax`` toward the trivial policy.  Forwarded
        to :func:`build_actor`.  Default: ``0.0``.
    progress_callback : callable or None, optional
        Optional callable invoked every 10 000 episodes with
        ``(avg_reward: float, episode: int)``.  Returning ``True`` aborts
        training early (used by Optuna's ``MedianPruner`` via
        ``trial.report()`` + ``trial.should_prune()``).  Default: ``None``.

    Returns
    -------
    ave_reward_list : list of float
        Average reward computed every 10 000 episodes.
    actor_model : tf.keras.Model
        Trained Actor (policy) network.
    conf_list : list of float
        Meta-cognitive confidence values (one per environment step).
        Empty list when *compute_confidence* is ``False``.

    Notes
    -----
    - Action selection is on-policy: ``a ~ Multinomial(masked_probs)``
      where ``masked_probs`` is π_θ(·|s) restricted to feasible actions
      and renormalised.  A uniform fallback over feasible actions is
      used if the masked sum is numerically zero.
    - **PBRS** (Ng et al., 1999): when *penalty_coeff* > 0, the reward
      is augmented with a potential-based shaping term that provides
      denser gradient signal without altering the optimal policy.
    - **Cosine annealing**: both Actor and Critic learning rates follow
      a cosine decay schedule from their initial value down to
      *lr_min_ratio* × initial over *episodes* steps.
    - The state vector is normalised to [0, 1]³ before being fed to the
      networks (see :func:`ac_model.normalize_state`).
    - The function prints average reward every 10 000 episodes to track
      convergence.
    - A2C performs on-policy updates: all transitions from one episode
      are batched and used in a **single gradient step** at the end of
      the episode via :func:`ac_model.train_step_actor_critic`.
    """

    # ----- Reproducibility -----
    if seed is not None:
        numpy.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        # Force deterministic TF ops (CPU & GPU).  Best-effort: some
        # TF/CUDA combinations may still report non-determinism on certain
        # kernels — we swallow the error so training never aborts here.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    # ----- Dimensions -----
    state_dim = 3
    action_dim = env.action_space.n
    max_res = env.available_resources_states - 1       # 30
    max_tri = env.trial_no_states - 1                  # 10
    max_sev = env.severity_states - 1                  # 9

    # ----- Networks -----
    actor_model = build_actor(state_dim, action_dim, actor_hidden, seed=seed,
                              last_action_bias=last_action_bias)
    critic_model = build_critic(state_dim, critic_hidden, seed=seed)

    # Build both networks with a dummy forward pass
    _dummy = tf.zeros((1, state_dim))
    actor_model(_dummy)
    critic_model(_dummy)

    # Cosine annealing learning rate scheduling
    actor_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=actor_lr,
        decay_steps=episodes,
        alpha=lr_min_ratio,
    )
    critic_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=critic_lr,
        decay_steps=episodes,
        alpha=lr_min_ratio,
    )
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_schedule)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_schedule)

    # Pre-build optimisers so their internal tf.Variables are created
    # *outside* `@tf.function`.  Without this, the second Optuna trial
    # would attempt to create new Variables inside the traced graph,
    # raising "tf.function only supports singleton tf.Variables …".
    actor_optimizer.build(actor_model.trainable_variables)
    critic_optimizer.build(critic_model.trainable_variables)

    # Per-trial JIT-compiled training step.
    # A fresh tf.function wrapper is created for each call to A2CTraining
    # so that the traced graph (and optimiser tf.Variables) do not leak
    # between Optuna trials.  reduce_retracing=True avoids warnings
    # when Optuna creates multiple wrappers across trials.
    compiled_train_step = tf.function(train_step_actor_critic, reduce_retracing=True)

    # Per-trial JIT-compiled inference function.  Replaces the eager
    # ``actor_model(s, training=False)`` call inside the per-step loop:
    # eager Keras incurs ~1 ms of dispatch overhead per call, and with
    # ~1.5 M calls per Optuna trial that adds 25-30 minutes of pure
    # Python/TF overhead on Colab CPU.  A tf.function trace runs in
    # graph mode and reduces this to <0.05 ms/call (~5x trial speed-up).
    @tf.function(reduce_retracing=True)
    def actor_predict(state_batch):
        return actor_model(state_batch, training=False)

    # Wrap scalar hyper-parameters as tf.constant so that tf.function
    # treats them as symbolic tensor inputs instead of Python literals.
    # Without this, each unique float value triggers a costly retrace.
    discount_t = tf.constant(discount, dtype=tf.float32)
    entropy_coeff_t = tf.constant(entropy_coeff, dtype=tf.float32)
    max_grad_norm_t = tf.constant(max_grad_norm, dtype=tf.float32)
    gae_lambda_t = tf.constant(gae_lambda, dtype=tf.float32)

    # ----- Tracking -----
    reward_list: list[float] = []
    ave_reward_list: list[float] = []
    conf_list: list[float] = []

    # Exponential ε-decay with warm-up (replaces linear decay)
    epsilon_initial = epsilon
    warmup_episodes = int(warmup_ratio * episodes)
    resolved_decay_rate = (min_eps / max(epsilon, 1e-8)) ** (
        1.0 / max(1, int((target_ratio - warmup_ratio) * episodes))
    )

    # ----- Training loop -----
    for i in range(episodes):
        done = False
        tot_reward: float = 0.0
        env.random_sequence()
        state, _ = env.reset()

        # Collect episode transitions for batched update
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_next_states = []
        ep_dones = []
        ep_masks = []

        while not done:
            state_norm = normalize_state(state, max_res, max_tri, max_sev)

            # On-policy action selection: sample from the masked Actor
            # distribution.  Replaces the previous ε-greedy branch, which
            # introduced an off-policy bias in the policy gradient (the
            # actor loss assumes a ~ π(a|s) but ε-greedy sampled uniformly
            # over feasible actions).
            probs = actor_predict(
                state_norm[numpy.newaxis, :]
            )[0].numpy()
            feasibility_mask = numpy.zeros(action_dim, dtype=numpy.float32)
            feasibility_mask[:min(int(state[0]), action_dim - 1) + 1] = 1.0
            masked_probs = probs * feasibility_mask
            masked_sum = masked_probs.sum()
            if masked_sum > 1e-8:
                masked_probs = masked_probs / masked_sum
            else:
                # Degenerate softmax (all feasible probs ~0) → uniform fallback.
                masked_probs = feasibility_mask / feasibility_mask.sum()
            action = int(numpy.random.choice(action_dim, p=masked_probs))

            # Meta-cognitive confidence (observational only)
            if compute_confidence:
                _, confidence, _, _ = ac_agent_meta_cognitive(
                    probs.copy(), state[0], 10000
                )
                conf_list.append(confidence)

            # PBRS: compute potential Φ(s) BEFORE the step
            phi_s = 0.0
            if penalty_coeff > 0.0:
                phi_s = -sum(max(0.0, sv) for sv in env.severities)

            # Step
            state2, reward, done, _trunc, _info = env.step(action)

            # Spending-cost shaping (training-only): discourage "spend
            # the maximum feasible amount" by subtracting a small cost
            # proportional to the action.  Eval reward is untouched.
            if spend_cost_coeff > 0.0:
                reward -= spend_cost_coeff * float(action)

            # Potential-Based Reward Shaping (Ng et al., 1999)
            # F(s, s') = β · (γ · Φ(s') − Φ(s)),  Φ(s) = −Σ max(0, sᵢ)
            if penalty_coeff > 0.0:
                phi_s_prime = 0.0 if done else -sum(max(0.0, sv) for sv in env.severities)
                reward += penalty_coeff * (discount * phi_s_prime - phi_s)

            state2_norm = normalize_state(state2, max_res, max_tri, max_sev)

            ep_states.append(state_norm)
            ep_actions.append(action)
            ep_rewards.append(float(reward))
            ep_next_states.append(state2_norm)
            ep_dones.append(float(done))
            ep_masks.append(feasibility_mask)

            tot_reward += reward
            state = state2

        # ---------- End-of-episode batch update ----------
        if len(ep_states) > 0:
            compiled_train_step(
                actor_model, critic_model,
                actor_optimizer, critic_optimizer,
                tf.constant(numpy.array(ep_states, dtype=numpy.float32)),
                tf.constant(numpy.array(ep_actions, dtype=numpy.int32)),
                tf.constant(numpy.array(ep_rewards, dtype=numpy.float32)),
                tf.constant(numpy.array(ep_next_states, dtype=numpy.float32)),
                tf.constant(numpy.array(ep_dones, dtype=numpy.float32)),
                discount_t,
                entropy_coeff_t,
                max_grad_norm_t,
                gae_lambda_t,
                tf.constant(numpy.array(ep_masks, dtype=numpy.float32)),
            )

        # Exponential ε-decay with warm-up (kept for API symmetry; unused
        # by the on-policy action-selection branch above).
        if i < warmup_episodes:
            epsilon = epsilon_initial
        else:
            epsilon = max(min_eps, epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))

        reward_list.append(tot_reward)

        if (i + 1) % 10000 == 0:
            ave_reward = float(numpy.mean(reward_list))
            ave_reward_list.append(ave_reward)
            reward_list = []
            if verbose:
                print(f"Episode {i + 1} Average Reward: {ave_reward:.4f}")
            if progress_callback is not None:
                # Returning True signals early termination (e.g. Optuna prune)
                if progress_callback(ave_reward, i + 1):
                    break

    env.close()
    return ave_reward_list, actor_model, conf_list
