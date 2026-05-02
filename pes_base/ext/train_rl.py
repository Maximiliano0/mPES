'''
pes_base - Pandemic Experiment Scenario: RL-Agent Training Pipeline

Trains a Q-Learning agent on the Pandemic environment and evaluates it
against a random-player baseline.

Pipeline stages
---------------
1. Load training data (initial_severity.csv, sequence_lengths.csv)
2. Run random-player baseline and save performance plots
3. Train Q-Learning agent (configurable episodes via CLI or default 1 000 000)
4. Save Q-table, rewards history, and training config to a dated directory
5. Evaluate trained agent on the same sequences and generate
   performance/confidence visualisations

Outputs (saved to ``inputs/<date>_RL_TRAIN/``)
----------------------------------------------
- q_<date>.npy              — Trained Q-table  (resources × trials × severity × actions)
- rewards_<date>.npy        — Average rewards every 10 000 episodes
- training_config_<date>.txt — Hyperparameters and training metadata
- \\*.png                     — 8 plots (random baseline, agent performance, confidences)
- confsrl_<date>.npy        — Raw confidence scores from evaluation

Usage
-----
::

    python3 -m pes_base.ext.train_rl [num_episodes]

Parameters are read from config/CONFIG.py via the package ``__init__``.
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import numpy
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

# Force TensorFlow to use CPU by default before any TF import happens.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

##########################
##  Imports internos    ##
##########################
from .. import INPUTS_PATH

from .tools import plot_confidences
from ..src.pygameMediator import convert_globalseq_to_seqs
from ..src.terminal_utils import header, section, success, info, list_item
from ..config.CONFIG import SEED
from .pandemic import Pandemic, rl_agent_meta_cognitive, run_experiment, QLearning

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

###################################
##             Main             ###
###################################
def main():
    """Execute the full RL-Agent training pipeline.

    Stages: load data → random baseline → Q-Learning training →
    save artefacts → evaluate trained agent → generate plots.

    Reads ``num_episodes`` from ``sys.argv[1]`` (default 1 000 000).
    All output is written to ``inputs/<date>_RL_TRAIN/``.
    """
    header("RL-AGENT TRAINING PIPELINE", width=80)

    # Configure matplotlib for better aesthetics
    try:
        plt.style.use('ggplot')
    except OSError:
        pass  # Use default if style is not available

    matplotlib_config = {
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6
    }
    plt.rcParams.update(matplotlib_config)

    # Create training output directory with date stamp
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_dir = os.path.join(INPUTS_PATH, f'{train_date}_RL_TRAIN')
    os.makedirs(train_dir, exist_ok=True)
    info(f"Output directory: {train_dir}")

    # Load initial severity and sequence lengths data
    section("Loading Training Data", width=80)
    trials_per_sequence = numpy.loadtxt(os.path.join( INPUTS_PATH,'sequence_lengths.csv'), delimiter=',')
    all_severities = numpy.loadtxt(os.path.join( INPUTS_PATH, 'initial_severity.csv'), delimiter=',')

    list_item(f"Sequence lengths shape: {trials_per_sequence.shape}")
    list_item(f"Initial severities shape: {all_severities.shape}")
    list_item(f"Total trials: {int(sum(trials_per_sequence))}")
    print()

    # Convert global sequences to per-sequence format
    # Reorganizes flat severity array into nested lists grouped by sequence
    # Input:  trials_per_sequence = [3, 2] (2 sequences with 3 and 2 trials respectively)
    #         all_severities = [0.5, 0.6, 0.7, 0.8, 0.9] (flat array of all trials)
    # Output: sevs = [[0.5, 0.6, 0.7], [0.8, 0.9]] (severities grouped by sequence)
    sevs = convert_globalseq_to_seqs(trials_per_sequence, all_severities)

    # Calculate probability distributions for number of cities (trials per sequence)
    val_cities, count_cities = numpy.unique(trials_per_sequence, return_counts=True)
    number_cities_prob = numpy.asarray((val_cities, count_cities/len(trials_per_sequence))).T

    # Calculate probability distributions for initial severities
    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    severity_prob = numpy.asarray((val_severity, count_severity/len(all_severities))).T

    env = Pandemic()

    def random_qf(env, _state, _seqid):
        """Return a uniformly random action (baseline policy)."""
        return env.sample()

    section("Random Player Baseline", width=80)
    info("Training random agent for comparison...")
    seqs1, perfs1, _ = run_experiment(env, random_qf, False, trials_per_sequence,sevs)
    success("Random player experiment completed")

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seqs1, color='#1f77b4', linewidth=2.5, marker='o', markersize=5, label='Random Player')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'random_player_sequence_performance_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item("Saved: random_player_sequence_performance.png")

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(perfs1, color='#ff7f0e', linewidth=2.5, marker='s', markersize=5, label='Random Player')
    ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Normalised Performance per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'random_player_normalised_performance_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item("Saved: random_player_normalised_performance.png")
    print()

    # Initialize pandemic environment with the calculated probability distributions
    env = Pandemic()

    env.number_cities_prob = number_cities_prob
    env.severity_prob = severity_prob
    env.verbose = False

    # Run Q-learning algorithm (always trains and overwrites previous files)
    section("Q-Learning Training", width=80)

    # Q-Learning hyperparameters
    learning_rate = 0.2
    discount_factor = 0.9
    epsilon_initial = 0.8
    epsilon_min = 0
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000

    info(f"Starting Q-Table training ({num_episodes:,} episodes)...")
    info("(This may take several minutes)")
    print()

    rewards, Q, confsrl = QLearning(env, learning_rate, discount_factor, epsilon_initial, epsilon_min, num_episodes,
                                    seed=SEED)
    print()
    success(f"Training completed")
    list_item(f"Q-Table shape: {Q.shape}")
    list_item(f"Rewards history length: {len(rewards)}")

    info("Saving trained models...")

    # Save Q-table and rewards with date stamp
    q_file = os.path.join(train_dir, f'q_{train_date}.npy')
    rewards_file = os.path.join(train_dir, f'rewards_{train_date}.npy')
    config_file = os.path.join(train_dir, f'training_config_{train_date}.txt')

    numpy.save(q_file, Q)
    numpy.save(rewards_file, rewards)

    # Create configuration file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RL-AGENT TRAINING CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {train_date}\n")
        f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Q-LEARNING HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning Rate (α):           {learning_rate}\n")
        f.write(f"Discount Factor (γ):         {discount_factor}\n")
        f.write(f"Initial Epsilon (ε):         {epsilon_initial}\n")
        f.write(f"Minimum Epsilon (ε_min):     {epsilon_min}\n")
        f.write(f"Number of Episodes:          {num_episodes:,}\n")
        f.write(f"Epsilon Decay:               Linear ({epsilon_initial} → {epsilon_min})\n\n")

        f.write("TRAINING RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Q-Table Shape:               {Q.shape}\n")
        f.write(f"State Space:\n")
        f.write(f"  - Available Resources:     {Q.shape[0]}\n")
        f.write(f"  - Trial Numbers:           {Q.shape[1]}\n")
        f.write(f"  - Severity Levels:         {Q.shape[2]}\n")
        f.write(f"  - Action Space:            {Q.shape[3]}\n")
        f.write(f"Rewards History Length:      {len(rewards)}\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Q-Table File:                q_{train_date}.npy\n")
        f.write(f"Rewards File:                rewards_{train_date}.npy\n")
        f.write(f"Configuration File:          training_config_{train_date}.txt\n\n")

        f.write("DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write("Files saved from Q-Learning training on the Pandemic Scenario.\n")
        f.write("The Q-table maps (resources, trial, severity) states to action values.\n")
        f.write("The rewards file contains average reward progression every 10,000 episodes.\n")

    success(f"✓ Q-Table saved to q_{train_date}.npy")
    success(f"✓ Rewards saved to rewards_{train_date}.npy")
    success(f"✓ Configuration saved to training_config_{train_date}.txt")
    list_item(f"Training Directory: {train_dir}")
    print()


    section("Training Performance Analysis", width=80)
    info("Generating reward history visualization...")

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(100*(numpy.arange(len(rewards)) + 1), rewards, color='#2ca02c', linewidth=2.5, label='Average Reward')
    ax.fill_between(100*(numpy.arange(len(rewards)) + 1), rewards, alpha=0.2, color='#2ca02c')
    ax.set_xlabel('Episodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Q-Learning Training: Average Reward Progression', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'rl_agent_rewards_vs_episodes_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: rl_agent_rewards_vs_episodes_{train_date}.png")
    print()


    section("RL-Agent Evaluation", width=80)
    info("Running evaluation experiment with trained agent...")
    confsrl = []

    def eval_qf(_env, state, _seqid):
        """Select an action from the trained Q-table with confidence tracking."""
        response, confidence, _rt_hold, _rt_release = rl_agent_meta_cognitive(
            Q[state[0], state[1], int(state[2])], state[0], 10000
        )

        if (state[0] == 0):
            confidence = -1.0

        confsrl.append(confidence)
        return response

    seqs, perfs, _ = run_experiment(env, eval_qf, False, trials_per_sequence, sevs)
    success("Evaluation experiment completed")

    info("Generating performance visualizations...")

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seqs, color='#d62728', linewidth=2.5, marker='o', markersize=5, label='RL-Agent')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
    ax.set_title('RL-Agent Evaluation: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'rl_agent_sequence_performance_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: rl_agent_sequence_performance_{train_date}.png")

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(perfs, color='#9467bd', linewidth=2.5, marker='s', markersize=5, label='RL-Agent')
    ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_title('RL-Agent Evaluation: Normalised Performance per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 64)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'rl_agent_normalised_performance_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: rl_agent_normalised_performance_{train_date}.png")

    cumperfs = numpy.cumsum(perfs)
    Domain = numpy.arange(1, 1 + 64)
    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumperfs / Domain, color='#8c564b', linewidth=2.5, marker='^', markersize=5, label='RL-Agent')
    ax.set_ylabel('Cumulative Normalised Performance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_title('RL-Agent Evaluation: Cumulative Performance Trend', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlim(0, 64)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'rl_agent_cumulative_performance_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: rl_agent_cumulative_performance_{train_date}.png")

    _fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(numpy.asarray(range(len(confsrl))), confsrl, color='#1f77b4', s=40, alpha=0.6, edgecolors='navy', linewidth=0.5)
    ax.set_title('RL-Agent: Decision Confidence Scores During Evaluation', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-10, 360)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'rl_agent_confidences_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: rl_agent_confidences_{train_date}.png")

    confsrl_arr = numpy.asarray( confsrl, dtype=numpy.float32)

    val_confidences = numpy.arange(11, dtype=numpy.float32) / 10.0
    _confsrl_hist = numpy.histogram( confsrl_arr, bins = val_confidences)

    plot_confidences(confsrl_arr, 'Confidences', Show=False)

    numpy.save(os.path.join(train_dir, f'confsrl_{train_date}.npy'), confsrl_arr)

    confsrl_arr = confsrl_arr [ confsrl_arr != -1 ]


    print ( confsrl_arr)

    I = confsrl_arr
    rescaled = (I - numpy.min(I) )* (  (1.0 - 0.0) / ( numpy.max(I) - numpy.min(I)) ) + 0.0
    remapconfrl= numpy.clip( rescaled, 0.0, 1.0)

    print (remapconfrl.shape )


    _fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(numpy.asarray(range(remapconfrl.shape[0])), remapconfrl, color='#2ca02c', s=40, alpha=0.6, edgecolors='darkgreen', linewidth=0.5)
    ax.set_ylabel('Remapped Confidence (0-1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
    ax.set_title('RL-Agent: Normalised Confidence Scores', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-10, 360)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'rl_agent_remapped_confidences_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plot_confidences(remapconfrl, 'Remapped Confidences', Show=False)

    section("Training Complete", width=80)
    success("RL-Agent training pipeline finished successfully!")
    info(f"Output directory: {train_dir}")
    print()
#
### END OF 'main()

if __name__ == '__main__':  main()
