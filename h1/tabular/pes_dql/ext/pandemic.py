'''
pes_dql — Pandemic Gymnasium Environment and Enhanced Q-Learning.

Implements the Pandemic environment as a custom Gymnasium ``Env`` and provides
four public entry points:

1. **Pandemic** — Gymnasium environment (state space 31×11×10, action space 11).
2. **QLearning()** — Tabular Q-Learning with three optional improvements:
   - *Double Q-Learning* (van Hasselt, 2010): two independent Q-tables to
     eliminate maximisation bias.
   - *Exponential ε-decay with warm-up*: configurable warm-up phase followed
     by exponential schedule calibrated via ``warmup_ratio`` / ``target_ratio``.
   - *Potential-Based Reward Shaping — PBRS* (Ng et al., 1999):
     Φ(s) = −Σ sᵢ  →  F(s, s') = β·(γ·Φ(s') − Φ(s)), controlled by
     ``penalty_coeff``.
3. **run_experiment()** — Greedy evaluation of a trained Q-table over the
   64 experimental sequences (8 blocks × 8 sequences).
4. **rl_agent_meta_cognitive()** — Entropy-based confidence estimation.

Q-table shape: (31, 11, 10, 11) = 37 510 cells.

Dependencies:
    - ``exp_utils.get_updated_severity``
    - ``exp_utils.calculate_normalised_final_severity_performance_metric``
    - ``tools.entropy_from_pdf``
'''

##########################
##  Imports externos    ##
##########################
import numpy
import random
from gymnasium import Env, spaces

##########################
##  Imports internos    ##
##########################
from .. import AVAILABLE_RESOURCES_PER_SEQUENCE
from .. import MAX_SEVERITY
from .. import MAX_ALLOCATABLE_RESOURCES
from .. import NUM_MAX_TRIALS

from .tools import entropy_from_pdf
from ..src.exp_utils import get_updated_severity
from ..src.exp_utils import calculate_normalised_final_severity_performance_metric

class Pandemic(Env):
    """
    Pandemic environment implementing Gymnasium's Env interface.

    The Pandemic environment simulates a pandemic response scenario where an agent
    must allocate limited resources across multiple cities to minimize final severity.
    Each episode consists of multiple sequences, and each sequence contains multiple trials.

    The environment follows the standard MDP formulation (Puterman, 1994):
    - State: s = (available_resources, trial_number, current_severity)
    - Action: a ∈ {0, 1, ..., 10} (resource allocation)
    - Transition: deterministic via get_updated_severity() with SEVERITY_MULTIPLIER (1.4)
      and RESPONSE_MULTIPLIER (0.4)
    - Reward: r(s, a) = −Σ s_i (negative sum of post-update severities)

    References
    ----------
    Brockman, G. et al. (2016). "OpenAI Gym / Gymnasium." arXiv:1606.01540.

    Puterman, M. L. (1994). "Markov Decision Processes: Discrete Stochastic
    Dynamic Programming." Wiley.

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
        self.max_resources  =   AVAILABLE_RESOURCES_PER_SEQUENCE - 9
        self.available_resources_states = self.max_resources + 1

        # Ten trials per sequence, from 3 to 10
        self.max_seq_length =   NUM_MAX_TRIALS
        self.trial_no_states = self.max_seq_length + 1

        # Ten severities, from 0 to 10
        self.max_severity   =   MAX_SEVERITY
        self.severity_states = self.max_severity + 1

        # Ten is the max alloc, Eleven choices, from 0 to 10
        self.max_allocation =   MAX_ALLOCATABLE_RESOURCES

        # Define a 3-D observation space
        self.observation_shape = (self.available_resources_states,
                                  self.trial_no_states,
                                  self.severity_states)

        self.observation_space = spaces.Box(low = numpy.zeros(self.observation_shape, dtype=numpy.float16),
                                            high = numpy.ones(self.observation_shape, dtype=numpy.float16),
                                            dtype = numpy.float16)

        # Define an action space
        self.action_space = spaces.Discrete(self.max_allocation+1,)

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
            self.allocations = [self.action_space.sample() for s in range(self.seq_length)]
            self.initial_severities = [random.randrange(int(0), int(self.max_severity)) for s in range(self.seq_length)]
        else:
            self.seq_length = int(numpy.random.choice(self.number_cities_prob[:,0], p=(self.number_cities_prob[:,1])))
            self.initial_severities = numpy.random.choice(self.severity_prob[:,0], size=(self.seq_length,), p=self.severity_prob[:,1])

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
            Random seed for reproducibility. Default: None
        options : dict or None, optional
            Additional reset options. Default: None

        Returns
        -------
        tuple
            (observation, info) where observation is
            [available_resources, trial_number, initial_severity]
            and info is an empty dict.
        """
        _ = seed, options  # accepted for Gym API compliance
        # Reload the available resources
        self.available_resources = self.max_resources

        # Reset the reward
        self.ep_return  = 0

        # City number
        self.iteration = 0

        self.severities = []
        self.resources = []

        self.severity_evolution = numpy.zeros((len(self.initial_severities)+1, len(self.initial_severities)))
        self.severity_city_counter = 0

        self.done = False

        # Get a new city with its own severity, and keep going....
        new_severity = self.new_city()
        self.severities.append( new_severity )

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
            print("--" , ':' , ":".join([" {:5.2f}".format(sev) for sev in self.severities]), '->', ' Done!')
        elif (len(self.resources)>0):
            print("{:02d}".format(self.iteration+1) , ':' , ":".join(["{:5.2f}".format(sev) for sev in self.severities]), '->', self.resources[-1])
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
        return {0: "0", 1:"1", 2:"2", 3:"3", 4:"4",5:"5", 6:"6", 7:"7",8:"8",9:"9", 10:"10"}

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

        Implements the MDP transition function:
          s' = T(s, a),  r = R(s, a) = −Σ severity_i

        Severity update follows: new_sev = max(0, 1.4 · old_sev − 0.4 · resources)
        (see get_updated_severity in exp_utils.py).

        Parameters
        ----------
        action : int
            The action to take (resource allocation amount, 0-10)

        Returns
        -------
        tuple of (list, float, bool, bool, dict)
            - observation (list): New state [available_resources, trial_number, severity]
            - reward (float): Reward for this step (negative sum of severities)
            - done (bool): Whether the episode is finished (``True`` when all trials exhausted)
            - truncated (bool): Always ``False`` (no time-based truncation)
            - info (dict): Empty auxiliary information dict
        """
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), f'Invalid Action {action}'

        # Reward for executing a step.
        reward = 0

        if ( (self.available_resources-action)<= 0):
            action = self.available_resources

        self.available_resources -= action
        self.resources.append( action )

        if (self.verbose): self.render()

        self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities

        # if self.verbose:
        #     print(f"\n[DEBUG] pandemic.step() - Trial {self.iteration}")
        #     print(f"[DEBUG]   Action (resources allocated): {action}")
        #     print(f"[DEBUG]   Severities before update: {['%.2f' % s for s in self.severities]}")

        self.severities = get_updated_severity(len(self.severities), self.resources, self.severities)

        # if self.verbose:
        #     print(f"[DEBUG]   Severities after update:  {['%.2f' % s for s in self.severities]}")

        self.severity_city_counter=self.severity_city_counter+1

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
            self.severities.append( new_severity )

        return [self.available_resources, self.iteration, int(new_severity)], reward, done, False, {}

def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    """
    Computes meta-cognitive confidence and response time estimates from Q-learning options.

    This function evaluates the entropy of action options to determine agent confidence
    and maps that confidence to human-like response times (reaction hold and release times).

    Confidence is derived from the Shannon entropy (Shannon, 1948) of the Q-value
    distribution, normalized between the minimum entropy (deterministic distribution)
    and maximum entropy (uniform distribution over all actions).

    Parameters:
    -----------
    options : array-like
        Q-values for available actions (from Q-table). Shape: (n_actions,)
    resources_left : int
        Number of resources remaining 
    response_timeout : float
        Maximum response time allowed in milliseconds

    Returns:
    --------
    response : int
        The selected action (argmax of options)
    confidence : float
        Normalized confidence score based on entropy (range: typically 0-1)
        Lower entropy → higher confidence
    rt_hold : float
        Response time for button hold phase (in seconds)
    rt_release : float
        Response time for button release phase (in seconds)

    Notes:
    ------
    - Confidence is calculated as: (entropy - min_entropy) / (max_entropy - min_entropy)
    - Response times are sampled from normal distributions parameterized by confidence
    - Both rt_hold and rt_release are clipped to [0, response_timeout/1000]
    - Infeasible actions (index > resources_left) are masked with a very negative
      sentinel (-1e9) before argmax. A small positive sentinel would dominate the
      typically negative learned Q-values and force the policy to pick an infeasible
      index that the env would silently clamp to "spend all remaining resources".

    References
    ----------
    Shannon, C. E. (1948). "A Mathematical Theory of Communication." Bell System
    Technical Journal, 27(3), 379–423.
    """

    # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((len(options),),)
    m_entropy[0] = 1

    # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((len(options),),)

    # Calculate the entropy of the options distribution
    _entrp1 = entropy_from_pdf(options)

    o = [i for i in range(len(options))]
    o = numpy.asarray(o, dtype=numpy.float32)

    # Set options that are not feasible (greater than resources left)
    # to a very negative sentinel so they are never selected by argmax.
    # A small positive value (e.g. 0.00001) would dominate the typically
    # negative learned Q-values and force the policy to pick the first
    # infeasible index (silently clamped by the env to "spend all").
    options[o>resources_left] = -1e9

    # available resources, trial, severity
    dec_entropy = entropy_from_pdf(options)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    # Calculate confidence as a normalized inverse of entropy
    confidence = (1./(m_entropy-M_entropy)) * (dec_entropy - M_entropy)

    # Select the action with the highest Q-value as the response
    response = numpy.argmax(options)

    # Map confidence to response times using a linear transformation
    map_to_response_time = lambda x: x * (-2) + 1
    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip( rt_hold, 0, response_timeout/1000.0)
    rt_release = numpy.clip( rt_release, 0, response_timeout/1000.0)

    return response, confidence, rt_hold, rt_release

def run_experiment(env, actionfunction, RandomSequences=True,
                   trials_per_sequence=None, sevs=None,
                   AssignAllocations=False, allocs=None,
                   NumberOfIterations=64):
    """
    Executes a pandemic simulation experiment over multiple sequences.

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

    Notes
    ------
    - Each sequence runs until completion (iteration reaches seq_length).
    - The environment is reset between sequences.
    - Results are printed to console during execution.
    - The environment is closed at the end of the experiment.
    """

    seqid = 0
    if (RandomSequences):
        env.random_sequence()
    elif (AssignAllocations):
        assert trials_per_sequence is not None and sevs is not None and allocs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid],sevs[seqid], allocs[seqid])
    else:
        assert trials_per_sequence is not None and sevs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid],sevs[seqid])
    state, _   = env.reset()
    seqs    = []
    perfs   = []
    seq_ev  = []
    ITERATIONS = NumberOfIterations
    while seqid<ITERATIONS:
        print(f'State: {state}')
        action = actionfunction(env,state, seqid)
        state2, _reward, done, _truncated, _info = env.step(action)

        if done==True:
            env.done = True
            env.render()
            seqs.append(  numpy.sum(env.severities) )
            perf = calculate_normalised_final_severity_performance_metric( env.severities,
                                                                           env.initial_severities)
            perfs.append( perf[0] )
            seq_ev.append( env.severity_evolution )
            seqid = seqid + 1

            if seqid<ITERATIONS:
                if (RandomSequences):
                    env.random_sequence()
                elif (AssignAllocations):
                    assert trials_per_sequence is not None and sevs is not None and allocs is not None
                    env.set_fixed_sequence(trials_per_sequence[seqid],sevs[seqid], allocs[seqid])
                else:
                    assert trials_per_sequence is not None and sevs is not None
                    env.set_fixed_sequence(trials_per_sequence[seqid],sevs[seqid])
            state2, _ = env.reset()

        state = state2

    print( numpy.array(seqs) )
    env.close()

    return seqs, perfs, seq_ev

def QLearning(env, learning, discount, epsilon, min_eps, episodes,
              decay_rate=None, warmup_ratio=0.05, target_ratio=0.66,
              double_q=True, penalty_coeff=0.0,
              UsePreloadedReward=False, R=None, seed=None,
              track_confidence=True, progress_callback=None):
    """
    Implements Q-Learning (standard or Double) to train an agent in the Pandemic environment.

    Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking
    actions in states. The agent learns by exploring the environment randomly (epsilon-greedy) 
    and updating a Q-table based on observed rewards and future state values.

    Supports two Q-Learning variants:
    - **Double Q-Learning (default):** Maintains two independent Q-tables (Q_A, Q_B).
      At each step, one table selects the best action (argmax) and the other evaluates it,
      eliminating the maximization bias of standard Q-Learning. The final policy uses the
      average (Q_A + Q_B) / 2.
    - **Standard Q-Learning:** Single Q-table with standard update rule.
      Set double_q=False to use this variant.

    Supports two epsilon decay strategies:
    - **Exponential with warm-up (default):** Maintains full exploration (ε₀) during a
      warm-up phase, then decays exponentially: ε_t = max(ε_min, ε₀ · λ^(t - W)).
      λ is auto-computed from N (episodes) so that ε reaches ε_min at target_ratio·N:
        λ = (ε_min / ε₀)^(1 / ((target_ratio - warmup_ratio) · N))
      This concentrates exploration at the start and allows longer exploitation at the end.
    - **Linear (legacy):** Set decay_rate='linear' to use linear decay:
      ε_t = max(ε_min, ε₀ - t · (ε₀ - ε_min) / N)

    Parameters
    ----------
    env : Pandemic
        The Pandemic environment instance to train on.
    learning : float
        Learning rate (alpha). Controls how much new information overrides old Q-values.
        Typical range: 0.1 - 0.9
    discount : float
        Discount factor (gamma). Controls the importance of future rewards.
        Typical range: 0.9 - 0.99. Higher values emphasize long-term rewards.
    epsilon : float
        Initial exploration rate. Probability of taking random actions.
        Typical range: 0.1 - 1.0. Higher values = more exploration.
    min_eps : float
        Minimum epsilon value. Exploration rate will decay until reaching this value.
        Typical range: 0.0 - 0.1
    episodes : int
        Number of training episodes to run.
    decay_rate : float or None, optional
        Exponential decay rate (λ) for epsilon. Controls how fast exploration decreases
        after the warm-up phase. Values closer to 1.0 produce slower decay.
        If None (default), it is auto-computed from the formula:
          λ = (ε_min / ε₀)^(1 / ((target_ratio - warmup_ratio) · N))
        so that ε reaches ε_min at (target_ratio · N) episodes.
        Typical range (when set manually): 0.9990 - 0.9999.
        Set to None to auto-compute (recommended).
    warmup_ratio : float, optional
        Fraction of total episodes dedicated to the warm-up phase (pure exploration
        at ε₀). Only used when decay_rate is not None or is auto-computed.
        Default: 0.05. Typical range: 0.0 - 0.1.
    target_ratio : float, optional
        Fraction of total episodes at which ε should reach ε_min.
        Only used when decay_rate is auto-computed (decay_rate=None).
        Default: 0.66 (i.e., ε reaches ε_min at 66% of episodes,
        leaving 34% for pure exploitation).
    double_q : bool, optional
        If True, uses Double Q-Learning with two independent Q-tables to reduce
        maximization bias. If False, uses standard single-table Q-Learning.
        Default: True
    penalty_coeff : float, optional
        Potential-Based Reward Shaping (PBRS) coefficient (β). Uses the
        potential function Φ(s) = −Σ s_i to compute a shaping reward:
          F(s, s') = β · (γ · Φ(s') − Φ(s))
        This gives a bonus for transitions that reduce total severity and a
        penalty for transitions that let severities grow. By Ng et al. (1999),
        this form guarantees that the optimal policy is invariant to β.
        Set to 0.0 to disable (default). Typical range: 0.01 - 0.5.
    UsePreloadedReward : bool, optional
        If True, uses preloaded reward values from matrix R instead of environment rewards.
        Default: False
    R : array-like, optional
        Preloaded reward matrix. Shape: (available_resources, trials, severities, actions).
        Required if UsePreloadedReward=True. Default: None
    seed : int or None, optional
        Random seed for reproducibility.  When set, both ``numpy.random`` and
        ``random`` are seeded before Q-table initialisation so that identical
        hyperparameters produce identical training runs.  Default: None (non-deterministic).
    track_confidence : bool, optional
        If True, computes the per-step meta-cognitive confidence via
        ``rl_agent_meta_cognitive`` and accumulates it in ``conf_list``.
        Disable (set False) to skip the entropy bookkeeping during Bayesian
        optimisation, where confidences are not needed and only add overhead.
        Default: True.
    progress_callback : callable or None, optional
        Optional callable invoked every 10 000 episodes with
        ``(avg_reward: float, episode: int)``.  Returning ``True`` aborts
        training early (used by Optuna's ``MedianPruner`` via
        ``trial.report()`` + ``trial.should_prune()``).  Default: ``None``.

    Returns
    -------
    ave_reward_list : list
        Average rewards computed every 10,000 episodes. Used to track learning progress.
    Q : ndarray
        Trained Q-table with shape (available_resources, trials, severities, actions).
        Contains the learned action values for each state.
        For Double Q-Learning, this is the average (Q_A + Q_B) / 2.
    conf_list : list
        Confidence values (meta-cognitive scores) calculated by rl_agent_meta_cognitive 
        for each step across all episodes.

    Algorithm Details
    -----------------
    For each episode:
    1. Initialize environment with random sequence
    2. Use epsilon-greedy policy: random action with probability epsilon, 
       otherwise take best action (argmax of Q, or argmax of (Q_A+Q_B)/2 for Double)
    3. Execute action and observe next state + reward
    4. Update Q-value:
       - Standard:  Q[s,a] += α * (r + γ * max(Q[s',a']) - Q[s,a])
       - Double:    With prob 0.5, update Q_A using Q_B for evaluation (or vice versa):
                    Q_A[s,a] += α * (r + γ * Q_B[s', argmax_a' Q_A[s',a']] - Q_A[s,a])
    5. Move to next state
    6. Decay epsilon:
       - Exponential (default): ε = max(ε_min, ε₀ · λ^(t - W))  if t ≥ W, else ε = ε₀
         where λ is auto-computed from N so that ε reaches ε_min at target_ratio·N
       - Linear (legacy):       ε -= (ε₀ - ε_min) / N

    Notes
    ------
    - Q-values are initialized with random uniform values between -1 and 1
    - Average rewards are computed and printed every 10,000 episodes
    - Meta-cognitive confidence is tracked but doesn't affect learning
    - The environment is closed after training completes
    - State representation: [available_resources, trial_number, current_severity]
    - When *seed* is provided, results are fully reproducible across runs
      with the same hyperparameters.
    - Exponential decay concentrates ~80% of exploration in the first third of training,
      leaving more episodes for pure exploitation compared to linear decay.
      Decay rate (λ) is auto-computed unless explicitly provided, ensuring
      consistent behavior regardless of the number of episodes.
    - Double Q-Learning uses ~2× memory during training (~586 KB vs ~293 KB) but the saved
      q.npy file remains the same size (single averaged table)
    - Reward Shaping (penalty_coeff > 0) uses Potential-Based Reward Shaping (PBRS)
      with Φ(s) = −Σ s_i.  The shaping reward F = β·(γ·Φ(s') − Φ(s)) gives a bonus
      for transitions that reduce total severity and a penalty when severities grow.
      By the Ng et al. (1999) invariance theorem, the optimal policy is preserved
      for any β ≥ 0.  Shaping is applied only during training.

    References
    ----------
    Watkins, C. J. C. H., & Dayan, P. (1992). "Q-learning." Machine Learning,
    8(3-4), 279–292.

    Hado van Hasselt (2010). "Double Q-learning." Advances in Neural Information
    Processing Systems 23 (NeurIPS 2010).

    Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward
    transformations: Theory and application to reward shaping." Proceedings of the
    16th International Conference on Machine Learning (ICML 1999), pp. 278–287.

    Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
    (2nd ed.). MIT Press. Chapters 6.5 (ε-greedy), 6.7 (Q-learning).
    """

    # Seed RNGs for reproducibility
    if seed is not None:
        numpy.random.seed(seed)
        random.seed(seed)

    # Q-table shape: (resources, trial, severity, action) → e.g. (31, 11, 10, 11) = 37 510 celdas.
    # Q[r, t, s, a] = "¿Qué tan bueno es asignar 'a' recursos cuando tengo 'r'
    #                   disponibles, estoy en el trial 't' y la severidad es 's'?"
    # Valores aleatorios en [-1, 1] (no ceros) para que argmax no sea determinista
    # al inicio y favorezca la exploración antes de que los Q-values se aprendan.
    q_shape = (env.available_resources_states,
               env.trial_no_states,
               env.severity_states,
               env.action_space.n)

    # Initialise Q-tables and decay variables before the branch to avoid
    # pyright "possibly unbound" warnings.  The branch immediately overrides
    # the values used by the chosen variant.
    Q: numpy.ndarray = numpy.empty(0)
    Q_A: numpy.ndarray = numpy.empty(0)
    Q_B: numpy.ndarray = numpy.empty(0)
    warmup_episodes: int = 0
    reduction: float = 0.0
    ave_reward: float = 0.0

    if double_q:
        # Double Q-Learning: initialize two independent Q-tables
        Q_A = numpy.random.uniform(low=-1, high=1, size=q_shape)
        Q_B = numpy.random.uniform(low=-1, high=1, size=q_shape)
        print(f'Algoritmo: Double Q-Learning (2 tablas, {Q_A.nbytes * 2:,} bytes)')
    else:
        # Standard Q-Learning: single Q-table
        Q = numpy.random.uniform(low=-1, high=1, size=q_shape)
        print(f'Algoritmo: Standard Q-Learning (1 tabla, {Q.nbytes:,} bytes)')

    if penalty_coeff > 0.0:
        print(f'Reward Shaping: PBRS β={penalty_coeff} — Φ(s) = −Σ sᵢ')
    else:
        print(f'Reward Shaping: Desactivado (β=0)')

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    conf_list = []

    # Store initial epsilon for decay calculations
    epsilon_initial = epsilon

    # Configure epsilon decay strategy
    use_exponential_decay = (decay_rate != 'linear')

    # Resolved decay rate — guaranteed to be float when exponential decay is used
    resolved_decay_rate: float = 0.0

    if use_exponential_decay:
        # Auto-compute decay_rate from the formula if not provided
        if decay_rate is None:
            resolved_decay_rate = (min_eps / epsilon) ** (1.0 / ((target_ratio - warmup_ratio) * episodes))
        else:
            resolved_decay_rate = float(decay_rate)
        # Exponential decay with warm-up phase
        warmup_episodes = int(warmup_ratio * episodes)
        target_episodes = int(target_ratio * episodes)
        print(f'Cantidad de episodios: {episodes:,}')
        print(f'Epsilon decay: Exponencial (λ={resolved_decay_rate:.10f})')
        print(f'  ├─ Warm-up:      {warmup_ratio*100:.0f}% ({warmup_episodes:,} episodios)')
        print(f'  ├─ Target:       {target_ratio*100:.0f}% ({target_episodes:,} episodios para alcanzar ε_min)')
        print(f'  └─ Explotación:  {100 - target_ratio*100:.0f}% ({episodes - target_episodes:,} episodios restantes)')
    else:
        # Legacy linear decay
        reduction = (epsilon - min_eps) / episodes
        print(f'Cantidad de episodios: {episodes:,}')
        print(f'Epsilon decay: Lineal ({epsilon_initial} → {min_eps} en {episodes:,} episodios)')

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        env.random_sequence()
        state, _ = env.reset()

        while done != True:

            # State → Q-table indices. By construction (env bounds: state[0] ∈
            # [0, max_resources], state[1] ∈ [0, seq_length ≤ NUM_MAX_TRIALS],
            # state[2] integer in [0, MAX_SEVERITY]) every component is already
            # within Q-table dimensions, so we can index directly. The previous
            # silent min(...) clip was dead defensive code that would have
            # masked any future env regression — if a bound is ever violated,
            # let it fail loudly instead of collapsing onto the boundary cell.
            s = (int(state[0]), int(state[1]), int(state[2]))

            # Determine next action - epsilon greedy strategy with action masking.
            # The env clamps infeasible actions silently, so without masking the
            # Q-cell of the *requested* action would be updated from a transition
            # that actually used a *clamped* (smaller) allocation, polluting the
            # Q-table with off-policy data and wasting capacity on unreachable
            # cells. Restrict both the greedy and random branches to feasible
            # actions a in [0, min(state[0], action_space.n - 1)].
            max_feasible_action = min(int(state[0]), env.action_space.n - 1)
            if numpy.random.random() < 1 - epsilon and state[0] is not None:
                if double_q:
                    # Use the sum of both tables for action selection (argmax is
                    # invariant to the /2 factor used elsewhere for confidence).
                    q_sel = Q_A[s] + Q_B[s]
                else:
                    q_sel = Q[s]
                # Mask infeasible actions before argmax
                q_masked = q_sel.copy()
                if max_feasible_action + 1 < q_masked.shape[0]:
                    q_masked[max_feasible_action + 1:] = -1e9
                action = int(numpy.argmax(q_masked))
            else:
                action = numpy.random.randint(0, max_feasible_action + 1)

            # Meta-cognitive confidence tracking (optional — expensive at
            # ~10⁶ episodes × several steps; the value never feeds the learner,
            # so it is gated by ``track_confidence`` and skipped during
            # Optuna optimisation, where it can dominate per-trial wall time).
            if track_confidence:
                if double_q:
                    q_values_for_confidence = (Q_A[s] + Q_B[s]) / 2.0
                else:
                    q_values_for_confidence = Q[s]
                _, confidence, _, _ = rl_agent_meta_cognitive(q_values_for_confidence, state[0], 10000)
                conf_list.append(confidence)

            # PBRS: compute potential Φ(s) BEFORE the step.
            # The env's severity update already clamps to [0, +inf), so the
            # explicit max(0.0, ...) was redundant.
            phi_s = 0.0
            if penalty_coeff > 0.0:
                phi_s = -float(numpy.sum(env.severities))

            # Get next state and reward
            state2, reward, done, _truncated, _step_info = env.step(action)

            if (UsePreloadedReward):
                assert R is not None
                reward = R[state[0], state[1], state[2], action]

            # Potential-Based Reward Shaping (Ng et al., 1999)
            # F(s, s') = β · (γ · Φ(s') − Φ(s)),  where Φ(s) = −Σ sᵢ
            # Terminal state: Φ(s_terminal) = 0
            if penalty_coeff > 0.0:
                if done:
                    phi_s_prime = 0.0
                else:
                    phi_s_prime = -float(numpy.sum(env.severities))
                reward += penalty_coeff * (discount * phi_s_prime - phi_s)

            # Next-state → Q-table indices (see comment above for s).
            s2 = (int(state2[0]), int(state2[1]), int(state2[2]))

            # Mask infeasible actions in s2 BEFORE the bootstrap argmax / max.
            # The env clamps infeasible actions silently, so an unmasked argmax
            # could pick an action index a' > state2[0] whose Q-cell is
            # unreachable on-policy (the env would never let the agent execute
            # it from s2).  Bootstrapping from such a Q-cell pollutes the TD
            # target with off-policy noise and contradicts the selection-time
            # masking applied to the behaviour policy a few lines above.
            max_feasible_action_s2 = min(int(state2[0]), env.action_space.n - 1)

            if double_q:
                # ── Double Q-Learning update ──
                # With probability 0.5, update Q_A (select with Q_A, evaluate with Q_B)
                # otherwise update Q_B (select with Q_B, evaluate with Q_A).
                # Terminal target is just the reward (Φ(s')=0 already handled in
                # PBRS above and there is no future bootstrap). Use the standard
                # incremental TD form so the terminal sample is blended in with
                # the learning rate (a hard assignment causes high variance and
                # wipes out previous experience for that state-action).
                if numpy.random.random() < 0.5:
                    if done:
                        target = reward
                    else:
                        q_a_s2 = Q_A[s2].copy()
                        if max_feasible_action_s2 + 1 < q_a_s2.shape[0]:
                            q_a_s2[max_feasible_action_s2 + 1:] = -1e9
                        best_action = int(numpy.argmax(q_a_s2))
                        target = reward + discount * Q_B[s2 + (best_action,)]
                    Q_A[s + (action,)] += learning * (target - Q_A[s + (action,)])
                else:
                    if done:
                        target = reward
                    else:
                        q_b_s2 = Q_B[s2].copy()
                        if max_feasible_action_s2 + 1 < q_b_s2.shape[0]:
                            q_b_s2[max_feasible_action_s2 + 1:] = -1e9
                        best_action = int(numpy.argmax(q_b_s2))
                        target = reward + discount * Q_A[s2 + (best_action,)]
                    Q_B[s + (action,)] += learning * (target - Q_B[s + (action,)])
            else:
                # ── Standard Q-Learning update ──
                if done:
                    target = reward
                else:
                    q_s2 = Q[s2].copy()
                    if max_feasible_action_s2 + 1 < q_s2.shape[0]:
                        q_s2[max_feasible_action_s2 + 1:] = -1e9
                    target = reward + discount * float(numpy.max(q_s2))
                Q[s + (action,)] += learning * (target - Q[s + (action,)])

            # Update variables
            tot_reward += reward
            state = state2

        # Decay epsilon
        if use_exponential_decay:
            if i < warmup_episodes:
                epsilon = epsilon_initial                                    # warm-up: pure exploration
            else:
                epsilon = max(min_eps, epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))  # exponential decay
        else:
            if epsilon > min_eps:
                epsilon -= reduction                                        # legacy linear decay

        # Track rewards
        reward_list.append(tot_reward)

        if (i+1) % 10000 == 0:
            ave_reward = float(numpy.mean(reward_list))
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (i+1) % 10000 == 0:
            print(f'Episodio {i+1:,}/{episodes:,} — Recompensa promedio: {ave_reward:.4f} — ε: {epsilon:.6f}')
            if progress_callback is not None:
                # Returning True signals early termination (e.g. Optuna prune)
                if progress_callback(ave_reward, i + 1):
                    break

    env.close()

    # For Double Q-Learning, return the averaged table
    if double_q:
        Q = (Q_A + Q_B) / 2.0

    return ave_reward_list, Q, conf_list
