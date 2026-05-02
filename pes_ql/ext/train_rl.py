'''
pes_ql - Pandemic Experiment Scenario: RL-Agent Training Pipeline

Trains a Q-Learning agent on the Pandemic environment using hyperparameters
obtained from Bayesian optimisation (ext/optimize_rl.py, trial #40) and
evaluates it against a random-player baseline.

Pipeline stages
---------------
1. Load training data (initial_severity.csv, sequence_lengths.csv)
2. Run random-player baseline and save performance plots
3. Train Q-Learning agent (default 900 000 episodes, configurable via CLI)
4. Save Q-table, rewards history, and training config to a dated directory
5. Evaluate trained agent on the same sequences and generate
   performance/confidence visualisations

Key differences from pes_base/ext/train_rl.py
----------------------------------------
- Uses Bayesian-optimised hyperparameters (lr ≈ 0.360, γ ≈ 0.865, ε₀ ≈ 0.679,
  ε_min ≈ 0.085) instead of pes_base defaults
- Default episodes: 900 000 (pes_base uses 20 000)
- Uses SEED from CONFIG.py for reproducible training
- Q-table shape: (31, 11, 10, 11) since MAX_SEVERITY = 9

Usage
-----
::

    python3 -m pes_ql.ext.train_rl [num_episodes]
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import warnings
from datetime import datetime
import numpy
import matplotlib.pyplot as plt

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, rl_agent_meta_cognitive, run_experiment, QLearning
from .repro import (
    capture_fingerprint,
    diff_fingerprints,
    find_latest_artifacts,
    load_fingerprint,
    load_params,
)
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import plot_confidences, convert_globalseq_to_seqs
from ..config.CONFIG import SEED
from .. import ANSI, INPUTS_PATH

# Force TensorFlow to use CPU by default before any TF import happens.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box.*precision lowered.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

###################################
##             Main             ###
###################################


def main():
    """Run the full RL-Agent training and evaluation pipeline.

    Stages: (1) load ``initial_severity.csv`` and
    ``sequence_lengths.csv``; (2) run a random-player baseline and
    save its plots; (3) auto-load Bayesian-optimised hyperparameters
    via ``repro.find_latest_artifacts`` (falling back to
    ``_FALLBACK_PARAMS`` if no artefact is present) and verify the
    repro fingerprint; (4) train the Q-table with
    ``track_confidence=False`` using the per-trial seed
    ``SEED + trial_number + 1``; (5) evaluate the greedy policy on the
    fixed 64 sequences, compare ``mean_perf`` against the value stored
    in ``best_params.json`` and emit confidence/performance plots; (6)
    write all artefacts to ``inputs/<date>_RL_TRAIN/`` and overwrite
    ``inputs/q.npy`` / ``inputs/rewards.npy``.

    Accepts an optional positional ``num_episodes`` argument that
    overrides the value loaded from the sidecar.

    Returns
    -------
    None
        All artefacts are written to disk; nothing is returned.
    """

    header("RL-AGENT TRAINING PIPELINE", width=80)

    # Configure matplotlib for better aesthetics
    try:
        plt.style.use('ggplot')
    except BaseException:
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
    trials_per_sequence = numpy.loadtxt(os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=',')
    all_severities = numpy.loadtxt(os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=',')

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
    number_cities_prob = numpy.asarray((val_cities, count_cities / len(trials_per_sequence))).T

    # Calculate probability distributions for initial severities
    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    severity_prob = numpy.asarray((val_severity, count_severity / len(all_severities))).T

    env = Pandemic()

    def random_qf(_env, _state, _seqid):
        """Return a random action from the environment's action space."""
        return env.sample()

    section("Random Player Baseline", width=80)
    info("Training random agent for comparison...")
    seqs1, perfs1, _ = run_experiment(env, random_qf, False, trials_per_sequence, sevs)
    success("Random player experiment completed")

    _fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seqs1, color='#1f77b4', linewidth=2.5, marker='o', markersize=5, label='Random Player')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            train_dir,
            f'random_player_sequence_performance_{train_date}.png'),
        dpi=150,
        bbox_inches='tight')
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
    plt.savefig(
        os.path.join(
            train_dir,
            f'random_player_normalised_performance_{train_date}.png'),
        dpi=150,
        bbox_inches='tight')
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

    # --- Hyperparameters: auto-load from latest Bayesian opt run ---
    # Layout written by ext/optimize_rl.py + ext/repro.py:
    #     inputs/<date>_BAYESIAN_OPT/best_params_<date>.json
    #     inputs/<date>_BAYESIAN_OPT/repro_fingerprint_<date>.json
    #
    # The JSON also stores ``trial_number`` (0-based). Optuna trained that
    # trial with ``seed = SEED + trial_number + 1``. We MUST use the same
    # seed here so the resulting Q-table — and therefore mean_perf — is
    # bit-for-bit identical to what Optuna reported.
    #
    # Falls back to the historical trial-#40 values if no optimisation
    # artifacts are present locally.
    _FALLBACK_PARAMS = {
        'hyperparameters': {
            'learning_rate':    0.35965545888114453,
            'discount_factor':  0.8650520580454709,
            'epsilon_initial':  0.6791201210873763,
            'epsilon_min':      0.08483331103075126,
            'num_episodes':     900_000,
        },
        'trial_number': 39,                 # 0-based (1-based id = 40)
        'expected_mean_perf': None,
        'training_seed': SEED + 39 + 1,
    }

    artifacts = find_latest_artifacts()
    if artifacts is not None:
        params_path, fp_path = artifacts
        params = load_params(params_path)
        info(f"Loaded hyperparameters from {os.path.basename(params_path)}")

        # --- Reproducibility check ---
        ref_fp = load_fingerprint(fp_path)
        diffs = diff_fingerprints(ref_fp, capture_fingerprint())
        if diffs:
            print(f"{ANSI.ORANGE}{ANSI.BOLD}⚠  Environment differs from optimisation run:"
                  f"{ANSI.RESET}")
            for d in diffs:
                list_item(d)
            print(f"{ANSI.ORANGE}   mean_perf may NOT match the Bayesian-opt value.{ANSI.RESET}")
        else:
            success("Environment matches optimisation run — fully reproducible.")
    else:
        info("No Bayesian optimisation artifacts found, using fallback hyperparameters.")
        params = _FALLBACK_PARAMS

    hp              = params['hyperparameters']
    learning_rate   = float(hp['learning_rate'])
    discount_factor = float(hp['discount_factor'])
    epsilon_initial = float(hp['epsilon_initial'])
    epsilon_min     = float(hp['epsilon_min'])
    num_episodes    = int(sys.argv[1]) if len(sys.argv) > 1 else int(hp['num_episodes'])
    # Accept both canonical (best_trial_number / trial_seed / mean_perf) and
    # legacy (trial_number / training_seed / expected_mean_perf) field names
    # so old sidecars keep working after the 2026-04-21 schema unification.
    # Guard against missing keys: legacy sidecars may omit either name; default to -1
    # so downstream messages flag the absence instead of crashing on ``int(None)``.
    trial_number    = int(params.get('best_trial_number', params.get('trial_number', -1)) or -1)
    training_seed   = int(params.get('trial_seed',
                          params.get('training_seed', SEED + trial_number + 1)) or 0)
    expected_perf   = params.get('mean_perf', params.get('expected_mean_perf'))

    list_item(f"learning_rate    = {learning_rate}")
    list_item(f"discount_factor  = {discount_factor}")
    list_item(f"epsilon_initial  = {epsilon_initial}")
    list_item(f"epsilon_min      = {epsilon_min}")
    list_item(f"num_episodes     = {num_episodes:,}")
    list_item(f"trial_number     = {trial_number} (0-based)")
    list_item(f"training_seed    = {training_seed} (= SEED {SEED} + {trial_number} + 1)")
    if expected_perf is not None:
        list_item(f"expected_mean_perf = {expected_perf:.6f}")

    info(f"Starting Q-Table training ({num_episodes:,} episodes)...")
    info("(This may take several minutes)")
    print()

    # ``track_confidence=False`` is REQUIRED for bit-for-bit reproducibility of
    # the Optuna trial. ``optimize_rl.py`` runs every trial with the flag off,
    # which skips the two ``numpy.random.normal`` draws inside
    # ``rl_agent_meta_cognitive`` on every step. Leaving the default ``True``
    # here would consume different samples from the global RNG and produce a
    # divergent Q-table even with the same seed (=> mean_perf would not match
    # the value reported by Bayesian optimisation). Confidence values for the
    # plots in this script are recomputed post-hoc from the trained Q-table on
    # the evaluation sequences below — they don't need to be tracked during
    # training.
    rewards, Q, confsrl = QLearning(env, learning_rate, discount_factor,
                                    epsilon_initial, epsilon_min, num_episodes,
                                    seed=training_seed,
                                    track_confidence=False)
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
    ax.plot(100 * (numpy.arange(len(rewards)) + 1), rewards, color='#2ca02c', linewidth=2.5, label='Average Reward')
    ax.fill_between(100 * (numpy.arange(len(rewards)) + 1), rewards, alpha=0.2, color='#2ca02c')
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

    if True:  # pylint: disable=using-constant-test
        section("RL-Agent Evaluation", width=80)
        info("Running evaluation experiment with trained agent...")
        confsrl = []

        def eval_qf(_env, state, _seqid):
            """Select the best action using the trained Q-table.

            Mirrors the evaluation function used by ext/optimize_rl.py exactly:
            copies the Q-slice (so the table is not mutated), masks infeasible
            actions with the same -1e9 sentinel, and returns ``argmax``. Also
            collects the meta-cognitive confidence for the visualisation, but
            on a SEPARATE copy so the action choice is not affected.
            """
            s0 = min(int(state[0]), Q.shape[0] - 1)
            s1 = min(int(state[1]), Q.shape[1] - 1)
            s2 = min(int(state[2]), Q.shape[2] - 1)

            # Action choice — bit-identical to optimize_rl.qf
            options = Q[s0, s1, s2].copy()
            o = numpy.arange(len(options), dtype=numpy.float32)
            options[o > state[0]] = -1e9
            response = int(numpy.argmax(options))

            # Confidence (read-only side-effect for visualisation)
            _r, confidence, _h, _l = rl_agent_meta_cognitive(
                Q[s0, s1, s2].copy(), state[0], 10000
            )
            if state[0] == 0:
                confidence = -1.0
            confsrl.append(confidence)

            return response

        seqs, perfs, _ = run_experiment(env, eval_qf, False, trials_per_sequence, sevs)
        success("Evaluation experiment completed")

        # --- Reproducibility verification: compare local mean_perf vs Optuna's ---
        local_mean_perf = float(numpy.clip(numpy.mean(perfs), 0.0, 1.0))
        list_item(f"Local mean_perf:    {local_mean_perf:.6f}")
        if expected_perf is not None:
            list_item(f"Expected mean_perf: {expected_perf:.6f}  (from Bayesian opt)")
            delta = abs(local_mean_perf - float(expected_perf))
            if delta < 1e-6:
                success(f"✓ Reproduced exactly (Δ = {delta:.2e})")
            elif delta < 1e-3:
                info(f"≈ Match within tolerance (Δ = {delta:.2e})")
            else:
                print(f"{ANSI.ORANGE}{ANSI.BOLD}⚠  mean_perf diverges from Optuna by "
                      f"{delta:.4f}{ANSI.RESET}")
                print(f"{ANSI.ORANGE}   Check the env-fingerprint warnings printed earlier.{ANSI.RESET}")

        info("Generating performance visualizations...")

        _fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(seqs, color='#d62728', linewidth=2.5, marker='o', markersize=5, label='RL-Agent')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
        ax.set_title('RL-Agent Evaluation: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'rl_agent_sequence_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
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
        plt.savefig(
            os.path.join(
                train_dir,
                f'rl_agent_normalised_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
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
        plt.savefig(
            os.path.join(
                train_dir,
                f'rl_agent_cumulative_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: rl_agent_cumulative_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(
            numpy.asarray(
                range(
                    len(confsrl))),
            confsrl,
            color='#1f77b4',
            s=40,
            alpha=0.6,
            edgecolors='navy',
            linewidth=0.5)
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

        confsrl_arr = numpy.asarray(confsrl, dtype=numpy.float32)

        val_confidences = numpy.arange(11, dtype=numpy.float32) / 10.0
        _confsrl_hist = numpy.histogram(confsrl_arr, bins=val_confidences)

        plot_confidences(confsrl_arr, 'Confidences', Show=False)

        numpy.save(os.path.join(train_dir, f'confsrl_{train_date}.npy'), confsrl_arr)

        confsrl_arr = confsrl_arr[confsrl_arr != -1]

        print(confsrl_arr)

        I = confsrl_arr
        rescaled = (I - numpy.min(I)) * ((1.0 - 0.0) / (numpy.max(I) - numpy.min(I))) + 0.0
        remapconfrl = numpy.clip(rescaled, 0.0, 1.0)

        print(remapconfrl.shape)

        _fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(
            numpy.asarray(
                range(
                    remapconfrl.shape[0])),
            remapconfrl,
            color='#2ca02c',
            s=40,
            alpha=0.6,
            edgecolors='darkgreen',
            linewidth=0.5)
        ax.set_ylabel('Remapped Confidence (0-1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_title('RL-Agent: Normalised Confidence Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'rl_agent_remapped_confidences_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()

        plot_confidences(remapconfrl, 'Remapped Confidences', Show=False)

    section("Training Complete", width=80)
    success("RL-Agent training pipeline finished successfully!")
    info(f"Output directory: {train_dir}")
    print()
#
# END OF 'main()


if __name__ == '__main__':
    main()
