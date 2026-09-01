'''
pes_base - Pandemic Experiment Scenario: Gymnasium Environment and RL Algorithms

Provides the core simulation components:

- **Pandemic** (gymnasium.Env):  Gymnasium environment that models a pandemic
  resource-allocation problem.  State = (resources_left, trial_no, severity);
  action = resources to allocate (0-10).
- **rl_agent_meta_cognitive**:  Entropy-based meta-cognitive function that
  computes confidence and simulated response times from Q-table values.
- **run_experiment**:  Runs multiple sequences through the environment using
  any action-selection function and collects performance metrics.
- **QLearning**:  Tabular Q-Learning training loop with epsilon-greedy
  exploration and linear epsilon decay.

Dependencies
------------
- exp_utils.get_updated_severity, calculate_normalised_final_severity_performance_metric
- tools.entropy_from_pdf
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
    # to a very negative sentinel so they never win argmax. A small positive
    # value would dominate the typically-negative learned Q-values and force
    # the policy to pick an infeasible index (silently clamped by the env).
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
    state, _ = env.reset()
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

def QLearning(env, learning, discount, epsilon, min_eps, episodes, UsePreloadedReward=False, R=None, seed=None):
    """
    Implements the Q-Learning algorithm to train an agent in the Pandemic environment.

    Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking
    actions in states. The agent learns by exploring the environment randomly (epsilon-greedy) 
    and updating a Q-table based on observed rewards and future state values.

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

    Returns
    -------
    ave_reward_list : list
        Average rewards computed every 10,000 episodes. Used to track learning progress.
    Q : ndarray
        Trained Q-table with shape (available_resources, trials, severities, actions).
        Contains the learned action values for each state.
    conf_list : list
        Confidence values (meta-cognitive scores) calculated by rl_agent_meta_cognitive 
        for each step across all episodes.

    Algorithm Details
    -----------------
    For each episode:
    1. Initialize environment with random sequence
    2. Use epsilon-greedy policy: random action with probability epsilon, 
       otherwise take best action (argmax Q)
    3. Execute action and observe next state + reward
    4. Update Q-value using: Q[s,a] += α * (r + γ * max(Q[s',a']) - Q[s,a])
    5. Move to next state
    6. Decay epsilon after each episode: ε -= (ε_initial - ε_min) / episodes

    Notes
    ------
    - Epsilon decay is linear across episodes
    - Q-values are initialized with random uniform values between -1 and 1
    - Average rewards are computed and printed every 10,000 episodes
    - Meta-cognitive confidence is tracked but doesn't affect learning
    - The environment is closed after training completes
    - State representation: [available_resources, trial_number, current_severity]
    - When *seed* is provided, results are fully reproducible across runs
      with the same hyperparameters.
    """

    # Seed RNGs for reproducibility
    if seed is not None:
        numpy.random.seed(seed)
        random.seed(seed)

    # Initialize Q table
    Q = numpy.random.uniform(low = -1, high = 1,
                        size = (  env.available_resources_states,
                                    env.trial_no_states,
                                    env.severity_states,
                                    env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    conf_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        env.random_sequence()
        state, _ = env.reset()

        while done != True:

            # Clip state indices to ensure they stay within Q-table bounds
            state_idx = [min(int(state[0]), env.available_resources_states - 1),
                        min(int(state[1]), env.trial_no_states - 1),
                        min(int(state[2]), env.severity_states - 1)]

            # Determine next action - epsilon greedy strategy
            if numpy.random.random() < 1 - epsilon and state[0] is not None:
                action = numpy.argmax(Q[state_idx[0], state_idx[1], state_idx[2]])
            else:
                action = numpy.random.randint(0, env.action_space.n)

            _, confidence, _,_ = rl_agent_meta_cognitive( Q[state_idx[0], state_idx[1], state_idx[2]], state[0], 10000)
            conf_list.append( confidence )

            # Get next state and reward
            state2, reward, done, _truncated, _info = env.step(action)

            if (UsePreloadedReward):
                assert R is not None
                reward = R[state[0], state[1], state[2], action]

            # Clip next state indices as well
            state2_idx = [min(int(state2[0]), env.available_resources_states - 1),
                         min(int(state2[1]), env.trial_no_states - 1),
                         min(int(state2[2]), env.severity_states - 1)]

            #Allow for terminal states
            if done:
                Q[state_idx[0], state_idx[1], state_idx[2], action] = reward

            # Adjust Q value for current state
            else:
                delta = learning*(reward +
                                discount*numpy.max(Q[state2_idx[0],
                                                state2_idx[1],
                                                state2_idx[2]]) -
                                Q[state_idx[0], state_idx[1], state_idx[2], action])
                Q[state_idx[0], state_idx[1], state_idx[2], action] += delta

            # Update variables
            tot_reward += reward
            state = state2

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Track rewards
        reward_list.append(tot_reward)

        if (i+1) % 10000 == 0:
            ave_reward = numpy.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))

    env.close()

    return ave_reward_list, Q, conf_list
