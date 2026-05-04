"""
pes_rdqn — standalone evaluator for an already-trained RDQN model.

Loads ``rdqn_model.keras`` (or any ``.keras`` file passed on the CLI) and
runs the **same 64-fixed-sequence evaluation protocol** used by
``optimize_rdqn.objective`` so the resulting ``mean_perf`` is directly
comparable to the value reported by Optuna / by ``train_rdqn``'s own
evaluation block.

Useful when:

* a model trained on Colab is downloaded locally and you want to verify
  the metric matches the Colab-reported ``mean_perf`` (rules out
  silent corruption / version drift);
* the full ``__main__.py`` experiment reports a different number and
  you want to disentangle "metric/protocol mismatch" from "model is
  broken".

Usage
-----
::

    python -m ml.pes_rdqn.ext.eval_model
    python -m ml.pes_rdqn.ext.eval_model path/to/rdqn_model.keras

The script does **not** train, does **not** touch any Optuna study, and
prints `mean / std / min / max` over the 64 evaluation sequences plus the
per-sequence vector for spot-checking.
"""

##########################
##  Imports externos    ##
##########################
import json
import os
import sys

import numpy
import tensorflow as tf

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, run_experiment, rdqn_agent_meta_cognitive
from .rdqn_model import normalize_state, HistoryDeque
from .tools import convert_globalseq_to_seqs
from ..config.CONFIG import RDQN_MODEL_FILE, SEED
from ..src.terminal_utils import header, info, list_item, success
from .. import INPUTS_PATH


def _resolve_model_path() -> str:
    """Pick the model path: CLI arg if given, otherwise the canonical file."""
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if not os.path.isfile(candidate):
            raise SystemExit(f'Model file not found: {candidate}')
        return candidate
    canonical = os.path.join(INPUTS_PATH, RDQN_MODEL_FILE)
    if not os.path.isfile(canonical):
        raise SystemExit(
            f'Canonical model not found: {canonical}\n'
            'Train one first (python -m ml.pes_rdqn.ext.train_rdqn) '
            'or pass an explicit .keras path.'
        )
    return canonical


def _load_eval_sequences():
    """Load the 64 fixed evaluation sequences from inputs/."""
    trials_per_sequence = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=',')
    all_severities = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=',')
    sevs = convert_globalseq_to_seqs(trials_per_sequence, all_severities)
    return trials_per_sequence, sevs


def evaluate(model: tf.keras.Model, history_len: int):
    """Run the 64-sequence eval used by optimize_rdqn.objective.

    Parameters
    ----------
    model : tf.keras.Model
        Trained recurrent Q-network.
    history_len : int
        Length of the LSTM input window. Read from the loaded model's
        input shape.

    Returns
    -------
    perfs : numpy.ndarray
        Per-sequence normalised performance (final / initial severity).
    """
    # Re-seed RNGs to match train_rdqn's parity block, so this script
    # produces the same numbers as that block when pointed at the same
    # .keras file.  The eval loop itself is deterministic given fixed
    # severity sequences and a deterministic policy, but the meta-cognitive
    # response-time draws and any TF stateful ops still consume RNG state.
    tf.keras.utils.set_random_seed(SEED)
    numpy.random.seed(SEED)

    trials_per_sequence, sevs = _load_eval_sequences()

    env_eval = Pandemic()
    env_eval.verbose = False
    env_eval.action_space.seed(SEED)
    max_res = env_eval.max_resources
    max_seq = env_eval.max_seq_length
    max_sev = env_eval.max_severity

    history = HistoryDeque(history_len, 3)
    state = {'seqid': -1}

    def qf(_env, raw_state, seqid):
        if seqid != state['seqid']:
            history.reset()
            state['seqid'] = seqid
        norm_s = normalize_state(raw_state, max_res, max_seq, max_sev)
        history.append_step(norm_s)
        window = history.current_window()
        q_vals = model(window[numpy.newaxis], training=False).numpy()[0].copy()
        response, _conf, _rt_h, _rt_r = rdqn_agent_meta_cognitive(
            q_vals, raw_state[0], 10000)
        return response

    _, perfs, _ = run_experiment(env_eval, qf, False, trials_per_sequence, sevs)
    return numpy.asarray(perfs, dtype=float)


def main():
    """Entry point."""
    header('RDQN MODEL EVALUATOR (64 fixed sequences)', width=80)

    model_path = _resolve_model_path()
    info(f'Loading model: {model_path}')

    model = tf.keras.models.load_model(model_path)

    # Infer history_len from the model's input shape: (None, history_len, state_dim).
    input_shape = model.input_shape
    if len(input_shape) != 3:
        raise SystemExit(
            f'Unexpected model input_shape={input_shape}; '
            'expected (None, history_len, state_dim).'
        )
    history_len = int(input_shape[1])
    list_item(f'history_len = {history_len}')
    list_item(f'parameters  = {model.count_params():,}')
    print()

    perfs = evaluate(model, history_len=history_len)

    info('Per-sequence performance (final/initial severity):')
    print('   ' + ', '.join(f'{p:.3f}' for p in perfs))
    print()

    success(f'mean_perf = {perfs.mean():.6f}')
    list_item(f'std_perf  = {perfs.std():.6f}')
    list_item(f'min_perf  = {perfs.min():.6f}')
    list_item(f'max_perf  = {perfs.max():.6f}')
    list_item(f'n_seqs    = {len(perfs)}')

    # Compare with the value Optuna reported in best_params.json (if present).
    bp_path = os.path.join(INPUTS_PATH, 'best_params.json')
    if os.path.isfile(bp_path):
        with open(bp_path, 'r', encoding='utf-8') as _f:
            _bp = json.load(_f)
        expected = float(_bp.get('mean_perf', float('nan')))
        if not numpy.isnan(expected):
            delta = abs(float(perfs.mean()) - expected)
            print()
            info(f'best_params.json mean_perf : {expected:.6f}')
            info(f'local mean_perf            : {perfs.mean():.6f}')
            if delta < 1e-6:
                verdict = 'OK (bit-exact)'
            elif delta < 1e-3:
                verdict = 'within float tolerance'
            elif delta < 5e-2:
                verdict = 'GPU↔CPU LSTM drift (expected)'
            else:
                verdict = 'MISMATCH — check TF version / hardware / weights'
            info(f'|Δ| = {delta:.6f}  ({verdict})')


if __name__ == '__main__':
    main()
