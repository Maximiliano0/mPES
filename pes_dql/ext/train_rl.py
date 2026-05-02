'''
pes_dql — Standalone RL-Agent training script.

Trains a Q-Learning agent on the Pandemic environment using Bayesian-optimised
hyperparameters and three algorithmic improvements:

1. **Double Q-Learning** (``double_q=True``) — two independent Q-tables.
2. **Exponential ε-decay with warm-up** — ``warmup_ratio`` / ``target_ratio``.
3. **PBRS** — Potential-Based Reward Shaping with ``penalty_coeff``.

Pipeline stages:
    1. Instantiate Pandemic environment with CONFIG parameters.
    2. Train via ``QLearning()`` (default 860 000 episodes, SEED=42).
    3. Evaluate the Q-table over 64 sequences with infeasible-action masking.
    4. Save Q-table (``q.npy``), rewards (``rewards.npy``), and confidence
       plots to ``INPUTS_PATH/<date>_RL_TRAIN/``.

Optimised hyperparameters (from Bayesian optimisation)::

    learning_rate    = 0.2593
    discount_factor  = 0.9806
    epsilon_initial  = 0.8392
    epsilon_min      = 0.0799
    warmup_ratio     = 0.0240
    target_ratio     = 0.5174
    penalty_coeff    = 0.2177  (PBRS)
    num_episodes     = 860 000

Usage::

    python3 -m pes_dql.ext.train_rl
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import json
import glob
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
from ..config.CONFIG import SEED

from .tools import plot_confidences, convert_globalseq_to_seqs
from ..src.terminal_utils import header, section, success, info, list_item
from .pandemic import Pandemic, rl_agent_meta_cognitive, run_experiment, QLearning
from .repro import (
    capture_fingerprint,
    diff_fingerprints,
    find_fingerprint_for,
    load_fingerprint,
)
from .. import ANSI

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

###################################
##     Best-params loader        ##
###################################
# Hard-coded fallback (overridden by best_params.json when present)
_DEFAULT_HYPERPARAMS = {
    'learning_rate':   0.2592852466099094,
    'discount_factor': 0.9806357182178841,
    'epsilon_initial': 0.839196365086843,
    'epsilon_min':     0.07993292420985183,
    'num_episodes':    860000,
    'warmup_ratio':    0.02403950683025824,
    'target_ratio':    0.5174250836504598,
    'penalty_coeff':   0.21766241123453672,
}


def _load_best_params():
    """Resolve the best Bayesian-optimisation hyperparameters.

    Resolution order:
      1. ``inputs/best_params.json``  (mirrored by ``optimize_rl`` after every
         optimisation run).
      2. Newest ``inputs/<date>_BAYESIAN_OPT/best_params_*.json``.
      3. ``_DEFAULT_HYPERPARAMS`` with ``seed=SEED`` (legacy fallback).

    Returns
    -------
    dict
        Payload with keys ``hyperparameters`` (dict), ``seed`` (int),
        ``track_confidence`` (bool), ``double_q`` (bool), ``mean_perf``
        (float or None), and ``source`` (str describing the origin).
    """
    candidates = []
    std_path = os.path.join(INPUTS_PATH, 'best_params.json')
    if os.path.isfile(std_path):
        candidates.append(std_path)
    candidates.extend(sorted(
        glob.glob(os.path.join(INPUTS_PATH, '*_BAYESIAN_OPT', 'best_params_*.json')),
        reverse=True,
    ))

    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8') as _f:
                payload = json.load(_f)
            payload['source'] = path
            payload.setdefault('track_confidence', False)
            payload.setdefault('double_q', True)
            # Accept canonical 'trial_seed' (matches pes_dqn/a2c/ql) first,
            # then legacy 'seed', then derive from trial_number formula:
            # ``SEED + trial.number + 1`` (matches optimize_rl.objective).
            if 'trial_seed' in payload:
                payload['seed'] = int(payload['trial_seed'])
            elif 'seed' not in payload:
                _tn = payload.get('best_trial_number', payload.get('trial_number'))
                if _tn is not None:
                    payload['seed'] = int(SEED + int(_tn) + 1)
                else:
                    payload['seed'] = SEED
            payload.setdefault('mean_perf', None)
            return payload
        except Exception:
            continue

    return {
        'hyperparameters':  dict(_DEFAULT_HYPERPARAMS),
        'seed':             SEED,
        'track_confidence': False,
        'double_q':         True,
        'mean_perf':        None,
        'source':           '<hardcoded defaults>',
    }

###################################
##             Main             ###
###################################
def main():
    """Train a Q-Learning agent on the Pandemic environment and evaluate it."""

    header("RL-AGENT TRAINING PIPELINE", width=80)

    # Configure matplotlib for better aesthetics
    try:
        plt.style.use('ggplot')
    except Exception:  # pylint: disable=broad-exception-caught
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

    # Load initial serverity and sequence lengths data
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

    def qf_random(env, _state, _seqid):
        """Random-player action function: delegates to env.sample()."""
        return env.sample()

    section("Random Player Baseline", width=80)
    info("Training random agent for comparison...")
    seqs1, perfs1, _ = run_experiment(env, qf_random, False, trials_per_sequence,sevs)
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

    # --- Load best hyperparameters from Bayesian optimisation ---
    bp_payload = _load_best_params()
    hp = bp_payload['hyperparameters']
    info(f"Hyperparameters source: {bp_payload['source']}")
    if bp_payload.get('mean_perf') is not None:
        info(f"Reference mean_perf (from optimisation): {bp_payload['mean_perf']:.6f}")

    # --- Reproducibility check (matches pes_ql workflow) ---
    # If the params came from a *_BAYESIAN_OPT directory, verify the
    # sibling repro_fingerprint_<date>.json matches the local env.
    fp_path = find_fingerprint_for(bp_payload['source'])
    if fp_path is not None:
        diffs = diff_fingerprints(load_fingerprint(fp_path), capture_fingerprint())
        if diffs:
            print(f"{ANSI.ORANGE}{ANSI.BOLD}\u26a0  Environment differs from optimisation run:"
                  f"{ANSI.RESET}")
            for d in diffs:
                list_item(d)
            print(f"{ANSI.ORANGE}   mean_perf may NOT match the Bayesian-opt value.{ANSI.RESET}")
        else:
            success("Environment matches optimisation run \u2014 fully reproducible.")

    # Q-Learning hyperparameters (from best Bayesian trial)
    learning_rate             = float(hp['learning_rate'])
    discount_factor           = float(hp['discount_factor'])
    epsilon_initial           = float(hp['epsilon_initial'])
    epsilon_min               = float(hp['epsilon_min'])

    # CLI override: `python -m pes_dql.ext.train_rl <num_episodes>`.
    # Defaults to the value from best_params.json so the resulting Q-table
    # matches the one produced by the best trial.
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else int(hp['num_episodes'])

    # Epsilon decay hyperparameters (exponential with warm-up)
    # λ is computed dynamically based on num_episodes (N):
    #   Formula: λ = (ε_min / ε₀)^(1 / ((target_ratio - warmup_ratio) · N))
    #   where  T = target_ratio · N  →  episode where ε reaches ε_min
    #          W = warmup_ratio · N  →  warm-up episodes with pure exploration
    warmup_ratio = float(hp['warmup_ratio'])     # W: fraction of N dedicated to pure exploration
    target_ratio = float(hp['target_ratio'])     # T: fraction of N at which ε reaches ε_min
    decay_rate   = (epsilon_min / epsilon_initial) ** (1.0 / ((target_ratio - warmup_ratio) * num_episodes))

    # Double Q-Learning: reduces maximization bias for more stable convergence
    double_q = bool(bp_payload.get('double_q', True))

    # Reward Shaping (PBRS): β = penalty_coeff
    penalty_coeff = float(hp['penalty_coeff'])

    # Seed: must equal the per-trial seed used by the best Bayesian trial
    # (SEED + best_trial_number + 1) so the resulting Q-table reproduces the
    # mean_perf reported by Optuna for that trial.
    qlearning_seed = int(bp_payload.get('seed', SEED))
    track_confidence = bool(bp_payload.get('track_confidence', False))

    _warmup_episodes = int(warmup_ratio * num_episodes)
    _target_episodes = int(target_ratio * num_episodes)

    info(f"Training seed:    {qlearning_seed}")
    info(f"track_confidence: {track_confidence}  (must match optimisation to reproduce Q-table)")
    info("Iniciando entrenamiento... (esto puede tardar varios minutos)")
    print()

    rewards, Q, confsrl = QLearning(env, learning_rate, discount_factor, epsilon_initial, epsilon_min, num_episodes,
                                    decay_rate=decay_rate, warmup_ratio=warmup_ratio,
                                    target_ratio=target_ratio, double_q=double_q,
                                    penalty_coeff=penalty_coeff, seed=qlearning_seed,
                                    track_confidence=track_confidence)
    print()
    success(f"Entrenamiento completado ({num_episodes:,} episodios)")
    list_item(f"Q-Table shape: {Q.shape}")
    list_item(f"Cantidad de registros de recompensas: {len(rewards)}")

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
        f.write(f"Algorithm:                   {'Double Q-Learning' if double_q else 'Standard Q-Learning'}\n")
        f.write(f"Learning Rate (α):           {learning_rate}\n")
        f.write(f"Discount Factor (γ):         {discount_factor}\n")
        f.write(f"Initial Epsilon (ε):         {epsilon_initial}\n")
        f.write(f"Minimum Epsilon (ε_min):     {epsilon_min}\n")
        f.write(f"Number of Episodes:          {num_episodes:,}\n")
        f.write(f"Decay Rate (λ):              {decay_rate:.10f} (computed from N={num_episodes:,})\n")
        f.write(f"Warm-up Ratio:               {warmup_ratio} ({int(warmup_ratio * num_episodes):,} episodes)\n")
        f.write(f"Target Ratio:                {target_ratio} ({int(target_ratio * num_episodes):,} episodes)\n")
        f.write(f"Exploitation Phase:          {num_episodes - int(target_ratio * num_episodes):,} episodes\n")
        f.write(f"Epsilon Decay:               Exponential with warm-up ({epsilon_initial} → {epsilon_min})\n")
        f.write(f"Decay Formula:               λ = (ε_min / ε₀)^(1 / ((target - warmup) · N))\n")
        f.write(f"Reward Shaping (β):          {penalty_coeff}{'  (desactivado)' if penalty_coeff == 0.0 else ''}\n")
        f.write(f"Random Seed:                 {qlearning_seed}  (= SEED + best_trial_number + 1)\n")
        f.write(f"Track Confidence:            {track_confidence}\n")
        f.write(f"Hyperparameters Source:      {bp_payload['source']}\n")
        if bp_payload.get('mean_perf') is not None:
            f.write(f"Reference mean_perf (opt):   {bp_payload['mean_perf']:.6f}\n")
        f.write("\n")

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

    # --- Mirror to standard input paths so __main__.py picks up the new Q-table ---
    std_q       = os.path.join(INPUTS_PATH, 'q.npy')
    std_rewards = os.path.join(INPUTS_PATH, 'rewards.npy')
    std_bp      = os.path.join(INPUTS_PATH, 'best_params.json')
    numpy.save(std_q, Q)
    numpy.save(std_rewards, rewards)

    # Refresh the standard best_params.json so __main__.py and any future
    # train_rl run see the exact hyperparameters/seed/track_confidence that
    # produced the mirrored q.npy (the upstream JSON may have come from a
    # different study, or the user may have overridden num_episodes via CLI).
    mirrored_bp = {
        'hyperparameters': {
            'learning_rate':   learning_rate,
            'discount_factor': discount_factor,
            'epsilon_initial': epsilon_initial,
            'epsilon_min':     epsilon_min,
            'num_episodes':    num_episodes,
            'warmup_ratio':    warmup_ratio,
            'target_ratio':    target_ratio,
            'penalty_coeff':   penalty_coeff,
        },
        'seed':              qlearning_seed,
        'track_confidence':  track_confidence,
        'double_q':          double_q,
        'mean_perf':         bp_payload.get('mean_perf'),
        'source':            f"train_rl mirror @ {train_date} (upstream: {bp_payload['source']})",
    }
    with open(std_bp, 'w', encoding='utf-8') as _f:
        json.dump(mirrored_bp, _f, indent=2)

    success(f"✓ Mirrored to {std_q}")
    success(f"✓ Mirrored to {std_rewards}")
    success(f"✓ Mirrored to {std_bp}")
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
    confsrl: list = []  # populated by qf() during the (stochastic) evaluation pass below

    def qf(_env, state, _seqid):
        """RL-agent action function with meta-cognitive confidence tracking."""
        response, confidence, _rt_hold, _rt_release = rl_agent_meta_cognitive(
            Q[state[0], state[1], int(state[2])], state[0], 10000
        )

        if (state[0] == 0):
            confidence = -1.0

        confsrl.append(confidence)
        return response

    # --- Deterministic evaluation that matches optimize_rl.objective() exactly ---
    # Same masking (-1e9), same indexing clamp, no random draws (skips
    # rl_agent_meta_cognitive's normal-distribution sampling). This is what
    # produces the mean_perf that must match the best Bayesian trial.
    def qf_eval(_env, state, _seqid):
        s0 = min(int(state[0]), Q.shape[0] - 1)
        s1 = min(int(state[1]), Q.shape[1] - 1)
        s2 = min(int(state[2]), Q.shape[2] - 1)
        options = Q[s0, s1, s2].copy()
        o = numpy.arange(len(options), dtype=numpy.float32)
        options[o > state[0]] = -1e9
        return int(numpy.argmax(options))

    env_eval = Pandemic()
    env_eval.verbose = False
    _, perfs_det, _ = run_experiment(env_eval, qf_eval, False, trials_per_sequence, sevs)
    mean_perf_local = float(numpy.mean(perfs_det))
    info(f"Local mean_perf (deterministic eval, matches optimisation): {mean_perf_local:.6f}")
    if bp_payload.get('mean_perf') is not None:
        delta = mean_perf_local - float(bp_payload['mean_perf'])
        info(f"Reference mean_perf (best trial):                            {bp_payload['mean_perf']:.6f}")
        info(f"Delta:                                                       {delta:+.6f}")
        if abs(delta) < 1e-9:
            success("mean_perf reproduced exactly — pipeline is deterministic.")
        else:
            warnings.warn(
                f"Local mean_perf differs from optimisation reference by {delta:+.6f}. "
                "Check seed, track_confidence, and CSV inputs.",
                stacklevel=1,
            )

    seqs, perfs, _ = run_experiment(env, qf, False, trials_per_sequence, sevs)
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
