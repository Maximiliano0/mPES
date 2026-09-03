"""
Main entry point for the pes_dqn experiment.

Defines the main() function that orchestrates the full experiment lifecycle:
validation of DQN-Agent training files, session creation with logging,
block/sequence/trial assignment, DQN-Agent decision collection via
pygameMediator, severity updates, performance calculation, and
result report generation (JSON + PNG).

This module is structurally identical to pes_ql/__main__.py.  The DQN-specific
behaviour (neural-network training) is performed offline via ext/train_dqn.py;
the resulting model is consumed here at experiment time.

Experiment Structure
--------------------
::

  Experimento (1)
  ├─ Bloque (8)
  │  ├─ Secuencia / Mapa (8)
  │  │  ├─ Trial / Ciudad (3~10)
  │  │  │  └─ Decisión de Recursos (0-10)

Summary
-------
- 1 Experimento
- 8 Bloques (NUM_BLOCKS)
- 8 Secuencias por Bloque (NUM_SEQUENCES)
- 3-10 Trials por Secuencia (NUM_MIN_TRIALS - NUM_MAX_TRIALS)
- ~360 trials totales, ~45 por bloque (TOTAL_NUM_TRIALS_IN_BLOCK)

Usage
-----
::

    python3 -m ml.pes_dqn

Configuration
-------------
All experiment parameters come from config/CONFIG.py.
"""

##############################################################
## Ensure we're running in package mode before proceeding!  ##
##############################################################

assert __package__ is not None and len(__package__) > 0, """
The '__main__' module does not seem to have been run in the context of a runnable package.
Did you forget to add the '-m' flag?

Usage: python3 -m <packagename>
"""

######################
## External Imports ##
######################
import os

# Default: pin TF to CPU.  Set MPES_USE_GPU=1 (e.g. on Colab) to use GPU.
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy
import datetime
import tensorflow as tf

######################
## Internal Imports ##
######################
from . import (AGGREGATION_METHOD, ANSI, AVAILABLE_RESOURCES_PER_SEQUENCE,
               INITIAL_SEVERITY_FILE, INIT_NO_OF_CITIES, INPUTS_PATH,
               MAX_ALLOCATABLE_RESOURCES, MAX_INIT_RESOURCES, MAX_INIT_SEVERITY,
               MIN_ALLOCATABLE_RESOURCES, MIN_INIT_RESOURCES, MIN_INIT_SEVERITY,
               NUM_ATTEMPTS_TO_ASSIGN_SEQ, NUM_BLOCKS, NUM_MAX_TRIALS, NUM_MIN_TRIALS,
               NUM_SEQUENCES, OUTPUT_FILE_PREFIX, OUTPUTS_PATH, PLAYER_TYPE,
               RANDOM_INITIAL_SEVERITY, SAVE_INITIAL_SEVERITY_TO_FILE, SAVE_RESULTS,
               STARTING_BLOCK_INDEX, STARTING_SEQ_INDEX, TOTAL_NUM_TRIALS_IN_BLOCK,
               TRUST_MAX, USE_FIXED_BLOCK_SEQUENCES)
from . src import exp_utils
from . src import log_utils
from . src import pygameMediator
from . src import result_formatter
from . src import terminal_utils

#################################################
##  Initialization of Module-global variables  ##
#################################################
Responses_filehandle = None
Starting_sequence_cumulative_index = STARTING_BLOCK_INDEX * NUM_SEQUENCES + STARTING_SEQ_INDEX

call_nominated_aggregator = {
    'confidence_weighted_median': exp_utils.get_confidence_weighted_median,
    'confidence_weighted_mean':   exp_utils.get_confidence_weighted_mean,
    'confidence_weighted_mode':   exp_utils.get_confidence_weighted_mode,
}[AGGREGATION_METHOD]

###################################
##             Main             ###
###################################


def main():
    """Orchestrate the full pes_dqn experiment lifecycle."""

    global Responses_filehandle

    # Validate DQN-Agent training files if needed
    if PLAYER_TYPE == 'DQN_AGENT':
        from .config.CONFIG import DQN_MODEL_FILE
        model_file = os.path.join(INPUTS_PATH, DQN_MODEL_FILE)
        rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')

        terminal_utils.header("DQN-AGENT VALIDATION", width=80)
        terminal_utils.info("Checking training files...")

        if not os.path.isfile(model_file):
            terminal_utils.error("DQN model file not found!")
            terminal_utils.list_item(f"Expected path: {model_file}", level=2)
            print("\nTo train the DQN-Agent, run:")
            terminal_utils.list_item("python3 -m ml.pes_dqn.ext.train_dqn")
            print()
            return

        if not os.path.isfile(rewards_file):
            terminal_utils.error("Rewards history file not found!")
            terminal_utils.list_item(f"Expected path: {rewards_file}", level=2)
            print("\nTo train the DQN-Agent, run:")
            terminal_utils.list_item("python3 -m ml.pes_dqn.ext.train_dqn")
            print()
            return

        # Validate that files can be loaded
        try:
            dqn_model = tf.keras.models.load_model(model_file)
            rewards = numpy.load(rewards_file)
            terminal_utils.success("DQN model loaded successfully")
            terminal_utils.list_item(f"Parameters: {dqn_model.count_params():,}", level=2)

            terminal_utils.success("Rewards history loaded successfully")
            terminal_utils.list_item(f"Episodes: {len(rewards)}", level=2)
            terminal_utils.list_item(f"Data type: {rewards.dtype}", level=2)
            print()
        except Exception as e:
            terminal_utils.error("Failed to load training files!")
            terminal_utils.list_item(f"Error: {str(e)}", level=2)
            print("\nPlease retrain the model by running:")
            terminal_utils.list_item("python3 -m ml.pes_dqn.ext.train_dqn")
            print()
            return

        terminal_utils.success("All training files validated successfully!")
        print()

    # Defaults for variables populated inside the SAVE_RESULTS block
    MyPerformances: list = []
    MySubjectId = ""
    session_outputs_path = ""
    total_movement: dict = {}
    Responses_filehandle = None

    # --------------------------------- #
    #       Files to be written         #
    # --------------------------------- #
    if SAVE_RESULTS:

        # MySubjectId: a unique identifier for the current experiment session
        experiment_date = datetime.date.today().strftime("%Y-%m-%d")
        MySubjectId = f"{experiment_date}_{PLAYER_TYPE}"

        # Create experiment-specific folder
        session_outputs_path = os.path.join(OUTPUTS_PATH, MySubjectId)
        os.makedirs(session_outputs_path, exist_ok=True)
        log_utils.create_ConsoleLog_filehandle_singleton(MySubjectId)

        terminal_utils.header(f"EXPERIMENT: {PLAYER_TYPE.upper()}", width=80)
        terminal_utils.info(f"Session ID: {MySubjectId}")
        terminal_utils.info(f"Output directory: {session_outputs_path}")
        print()

        # Save experiment configuration parameters to file
        SubjectInfo_filename = os.path.join(
            session_outputs_path,
            f'{OUTPUT_FILE_PREFIX}{MySubjectId}.txt')

        SubjectInfo_filehandle = open(SubjectInfo_filename, 'w')

        SubjectInfo_filehandle.write("=" * 80 + "\n")
        SubjectInfo_filehandle.write("EXPERIMENT CONFIGURATION PARAMETERS\n")
        SubjectInfo_filehandle.write("=" * 80 + "\n\n")
        SubjectInfo_filehandle.write(f"{'Variable Name':<45} {'Value':<35}\n")
        SubjectInfo_filehandle.write("-" * 80 + "\n")

        config_params = [
            ('AVAILABLE_RESOURCES_PER_SEQUENCE', AVAILABLE_RESOURCES_PER_SEQUENCE),
            ('INIT_NO_OF_CITIES', INIT_NO_OF_CITIES),
            ('INITIAL_SEVERITY_FILE', INITIAL_SEVERITY_FILE),
            ('MAX_ALLOCATABLE_RESOURCES', MAX_ALLOCATABLE_RESOURCES),
            ('MAX_INIT_RESOURCES', MAX_INIT_RESOURCES),
            ('MAX_INIT_SEVERITY', MAX_INIT_SEVERITY),
            ('MIN_ALLOCATABLE_RESOURCES', MIN_ALLOCATABLE_RESOURCES),
            ('MIN_INIT_RESOURCES', MIN_INIT_RESOURCES),
            ('MIN_INIT_SEVERITY', MIN_INIT_SEVERITY),
            ('NUM_ATTEMPTS_TO_ASSIGN_SEQ', NUM_ATTEMPTS_TO_ASSIGN_SEQ),
            ('NUM_BLOCKS', NUM_BLOCKS),
            ('NUM_MIN_TRIALS', NUM_MIN_TRIALS),
            ('NUM_MAX_TRIALS', NUM_MAX_TRIALS),
            ('NUM_SEQUENCES', NUM_SEQUENCES),
            ('PLAYER_TYPE', PLAYER_TYPE),
            ('STARTING_BLOCK_INDEX', STARTING_BLOCK_INDEX),
            ('STARTING_SEQ_INDEX', STARTING_SEQ_INDEX),
            ('TOTAL_NUM_TRIALS_IN_BLOCK', TOTAL_NUM_TRIALS_IN_BLOCK),
            ('TRUST_MAX', TRUST_MAX),
            ('USE_FIXED_BLOCK_SEQUENCES', USE_FIXED_BLOCK_SEQUENCES),
        ]

        for param_name, param_value in config_params:
            SubjectInfo_filehandle.write(f"{param_name:<45} {param_value}\n")

        SubjectInfo_filehandle.write("\n" + "=" * 80 + "\n")

        SubjectInfo_filehandle.close()

        # Save responses to file
        Responses_filename = os.path.join(
            session_outputs_path,
            f'{OUTPUT_FILE_PREFIX}responses_{MySubjectId}.txt'
        )

        Responses_filehandle = open(Responses_filename, 'w')

        # ---------------------------------#
        # MapIndex and sequences assigment #
        # ---------------------------------#
        response = {}
        hold_response_times = {}
        release_response_times = {}
        confidence = {}
        total_movement = {}

        NumTrials__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))

        MapIndices__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))

        NumTrialsPerSequence_list = []

        pygameMediator.number_of_trials = NumTrials__blocks_x_sequences__2darray

        # Block (blk) into the Experiment (MySubjectId)
        for blk in range(NUM_BLOCKS):

            numpy.random.seed(100 + blk)

            # Sequence (seq) into the Block (blk)
            for seq in range(NUM_SEQUENCES):

                counter_seq = NUM_ATTEMPTS_TO_ASSIGN_SEQ

                while counter_seq > 0:

                    b = numpy.random.randint(0, 9)

                    if b not in MapIndices__blocks_x_sequences__2darray[blk, :]:
                        MapIndices__blocks_x_sequences__2darray[blk, seq] = b
                        break

                    counter_seq -= 1

                if (USE_FIXED_BLOCK_SEQUENCES):
                    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.next_seq_length(
                        blk * NUM_SEQUENCES,
                        NUM_SEQUENCES
                    )
                else:
                    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.sampler(
                        NUM_SEQUENCES,
                        TOTAL_NUM_TRIALS_IN_BLOCK,
                        [NUM_MIN_TRIALS, NUM_MAX_TRIALS],
                        rn=blk
                    )

                _counter_trial = INIT_NO_OF_CITIES + NumTrials__blocks_x_sequences__2darray[blk, seq]

                NumTrialsPerSequence_list.append(NumTrials__blocks_x_sequences__2darray[blk, seq])

        # ----------------------------- #
        # Initial severity assignation  #
        # ----------------------------- #
        InitialSeverityCsv = os.path.join(INPUTS_PATH, INITIAL_SEVERITY_FILE)

        if RANDOM_INITIAL_SEVERITY:
            first_severity = exp_utils.random_severity_generator(
                int(numpy.sum(NumTrials__blocks_x_sequences__2darray)), 2, 9)
            if SAVE_INITIAL_SEVERITY_TO_FILE:
                numpy.savetxt(InitialSeverityCsv, first_severity, fmt='%d', delimiter=',')
        else:
            first_severity = numpy.loadtxt(InitialSeverityCsv)
            first_severity = first_severity[0: int(numpy.sum(NumTrials__blocks_x_sequences__2darray))]

        pygameMediator.first_severity = first_severity

        # Initialise output (i.e. responses file) by creating a header
        Responses_filehandle.write("#InitialSeverity, Response, Confidence, PressEvent_seconds, ReleaseEvent_seconds\n")
        Responses_filehandle.flush()

        # ----------------------- #
        # Experimental procedure  #
        # ----------------------- #
        print("--- Starting experiment --- \n")
        total_number_of_sequences = NUM_BLOCKS * NUM_SEQUENCES

        AbsoluteTrialCount = 0
        AbsoluteSequenceIndex = 0
        ExperimentStartTime = datetime.datetime.now(tz=datetime.timezone.utc)

        MyPerformances = []
        TrustRatings = numpy.array([1, TRUST_MAX])

        log_utils.tee(
            f"\n{'=' * 100}"
        )
        log_utils.tee(
            f"{ANSI.BOLD}{ANSI.GREEN}  STARTING EXPERIMENT: {PLAYER_TYPE}{ANSI.RESET}"
        )
        log_utils.tee(
            f"  Total: {NUM_BLOCKS} Blocks × {NUM_SEQUENCES} Sequences = {NUM_BLOCKS * NUM_SEQUENCES} Sessions"
        )
        log_utils.tee(
            f"{'=' * 100}\n"
        )

        for CurrentBlockIndex in range(NUM_BLOCKS):

            if CurrentBlockIndex < STARTING_BLOCK_INDEX:
                AbsoluteTrialCount += int(NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, :].sum())
                AbsoluteSequenceIndex += NUM_SEQUENCES
                continue

            log_utils.tee(
                f"\n{ANSI.BOLD}{ANSI.BLUE}▶ BLOQUE {CurrentBlockIndex + 1} / {NUM_BLOCKS}{ANSI.RESET}"
            )

            TimeElapsed = (datetime.datetime.now(tz=datetime.timezone.utc) - ExperimentStartTime).total_seconds()
            HoursElapsed = max(0, TimeElapsed / 60 / 60 // 1)
            MinutesElapsed = max(0, (TimeElapsed - HoursElapsed * 60 * 60) / 60 // 1)
            HoursElapsed_str = (
                f'{HoursElapsed:.0f} hours, and ' if HoursElapsed > 1
                else '1 hour, and ' if HoursElapsed == 1 else ''
            )
            MinutesElapsed_str = (
                f'{MinutesElapsed:.0f} minutes' if MinutesElapsed != 1
                else '1 minute' if MinutesElapsed == 1 else ''
            )
            TimeElapsed_str = f'Time elapsed: {HoursElapsed_str}{MinutesElapsed_str}'

            if AbsoluteSequenceIndex > Starting_sequence_cumulative_index:
                TimePerSeq = TimeElapsed / (AbsoluteSequenceIndex - Starting_sequence_cumulative_index)
                SeqsLeft = total_number_of_sequences - AbsoluteSequenceIndex
                TimeLeft = SeqsLeft * TimePerSeq
                HoursLeft = max(0, TimeLeft / 60 / 60 // 1)
                MinutesLeft = max(0, (TimeLeft - HoursLeft * 60 * 60) / 60 // 1)
                HoursLeft_str = (
                    f'{HoursLeft:.0f} hours, and ' if HoursLeft > 1
                    else '1 hour, and ' if HoursLeft == 1 else ''
                )
                MinutesLeft_str = (
                    f'{MinutesLeft:.0f} minutes' if MinutesLeft != 1
                    else '1 minute' if MinutesLeft == 1 else ''
                )
                TimeLeft_str = f'Estimated time left: {HoursLeft_str}{MinutesLeft_str}'
                _EndOfBlock_str = (
                    f'End of block {CurrentBlockIndex}   '
                    f'({NUM_BLOCKS - CurrentBlockIndex} blocks to go)\nTake rest if needed.'
                )
            else:
                _EndOfBlock_str = ' '
                TimeLeft_str = ' '

            log_utils.tee(f'{TimeElapsed_str} -- {TimeLeft_str}')

            CurrentBlockMapIndices = MapIndices__blocks_x_sequences__2darray[CurrentBlockIndex, :]

            response[CurrentBlockIndex] = {}
            hold_response_times[CurrentBlockIndex] = {}
            release_response_times[CurrentBlockIndex] = {}
            confidence[CurrentBlockIndex] = {}
            total_movement[CurrentBlockIndex] = {}

            for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate(CurrentBlockMapIndices):

                if CurrentBlockIndex == STARTING_BLOCK_INDEX and CurrentSequenceIndex < STARTING_SEQ_INDEX:
                    AbsoluteTrialCount += int(
                        NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex])
                    AbsoluteSequenceIndex += 1
                    continue

                resources_to_allocate = AVAILABLE_RESOURCES_PER_SEQUENCE

                log_utils.tee(
                    f"  └─ Secuencia {AbsoluteSequenceIndex + 1}/{total_number_of_sequences}"
                    f" (Mapa #{int(CurrentSequenceMapIndex)})"
                    f" - Progreso: {CurrentSequenceIndex + 1}/{NUM_SEQUENCES}"
                )

                _ScreenMessage = (f"Map Number #{AbsoluteSequenceIndex + 1} / {total_number_of_sequences}\n\n\n"
                                 f"Press the LEFT MOUSE BUTTON to start.\n\n\n"
                                 f"(Note: If you need to terminate the\n"
                                 f"experiment prematurely, press ESC now.)"
                                 )

                response[CurrentBlockIndex][CurrentSequenceIndex] = []
                hold_response_times[CurrentBlockIndex][CurrentSequenceIndex] = []
                release_response_times[CurrentBlockIndex][CurrentSequenceIndex] = []
                confidence[CurrentBlockIndex][CurrentSequenceIndex] = []
                total_movement[CurrentBlockIndex][CurrentSequenceIndex] = []

                init_severity = []
                ResourceAllocationsAtCurrentlyVisibleCities = []

                numpy.random.seed(3)

                for c in range(INIT_NO_OF_CITIES):

                    init_severity.append(numpy.random.randint(MIN_INIT_SEVERITY, 1 + MAX_INIT_SEVERITY))
                    ResourceAllocationsAtCurrentlyVisibleCities.append(
                        numpy.random.randint(MIN_INIT_RESOURCES, 1 + MAX_INIT_RESOURCES))

                resources_left = resources_to_allocate - numpy.sum(ResourceAllocationsAtCurrentlyVisibleCities)

                SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(INIT_NO_OF_CITIES,
                                                                                    ResourceAllocationsAtCurrentlyVisibleCities,
                                                                                    init_severity)

                direction = []
                for i in range(len(SeveritiesOfCurrentlyVisibleCities)):
                    if SeveritiesOfCurrentlyVisibleCities[i] < init_severity[i]:
                        direction.append(2)
                    else:
                        direction.append(1)

                _ShowEndOfTrialFeedback = True

                MyMessage = numpy.zeros((0, 3))

                for trial_no in range(
                        int(NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex])):

                    AbsoluteTrialCount += 1
                    AbsoluteTrialIndex = AbsoluteTrialCount - 1

                    init_severity = SeveritiesOfCurrentlyVisibleCities

                    log_utils.tee()

                    log_utils.tee(
                        f'Current trial: {trial_no + 1} out of'
                        f' {NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex]:.0f}'
                        f' in sequence',
                        f'({AbsoluteTrialCount} out of {TOTAL_NUM_TRIALS_IN_BLOCK * NUM_BLOCKS} overall)'
                    )

                    log_utils.tee(
                        'Initial severity values before annotation: ',
                        ", ".join([f'{i:.2f}' for i in init_severity])
                    )

                    log_utils.tee(
                        'Resources remaining: ', int(resources_left) if hasattr(
                            resources_left, 'numpy') else resources_left
                    )

                    new_locations = INIT_NO_OF_CITIES + trial_no + 1
                    severity_new_location = first_severity[AbsoluteTrialIndex]

                    log_utils.tee(f"Current trial's initial severity: {severity_new_location}")

                    init_severity.append(severity_new_location)

                    ResourceAllocationsAtCurrentlyVisibleCities.append(-1)

                    # Get response from DQN Agent
                    if resources_left > 0 and PLAYER_TYPE == 'DQN_AGENT':
                        (pc,
                         r,
                         rt_h,
                         rt_rel,
                         mov
                         ) = pygameMediator.provide_dqn_agent_response(
                            ResourceAllocationsAtCurrentlyVisibleCities,
                            resources_left,
                            CurrentBlockIndex,
                            CurrentSequenceIndex,
                            trial_no
                        )

                        if r == 0.0:
                            c = -1
                            mov = [0]
                        else:
                            c = pc

                        confidence[CurrentBlockIndex][CurrentSequenceIndex].append(c)

                        response[CurrentBlockIndex][CurrentSequenceIndex].append(r)
                        hold_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(rt_h)
                        release_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(rt_rel)
                        total_movement[CurrentBlockIndex][CurrentSequenceIndex].append(mov)

                        log_utils.tee(f'Response was: {r}')
                        log_utils.tee(f'Confidence was: {c}')
                        log_utils.tee(f'PressEvent_seconds was: {rt_h}')
                        log_utils.tee(f'ReleaseEvent_seconds was: {rt_rel}')

                    else:
                        confidence[CurrentBlockIndex][CurrentSequenceIndex].append(-1)
                        response[CurrentBlockIndex][CurrentSequenceIndex].append(0)
                        hold_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(0)
                        release_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(0)
                        total_movement[CurrentBlockIndex][CurrentSequenceIndex].append([0])

                    if SAVE_RESULTS:

                        Responses_filehandle.write(
                            ", ".join([
                                f"{first_severity[AbsoluteTrialIndex]: >{len('#InitialSeverity')}}",
                                f"{response[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('Response')}}",
                                f"{confidence[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('Confidence')}}",
                                f"{hold_response_times[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('PressEvent_seconds')}}",
                                f"{release_response_times[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('ReleaseEvent_seconds')}}"
                            ]) + "\n"
                        )

                        Responses_filehandle.flush()

                    _NumTrialsInSequence = int(NumTrialsPerSequence_list[AbsoluteSequenceIndex])
                    StartingIndex = int(sum(NumTrialsPerSequence_list[: AbsoluteSequenceIndex]))
                    InitialSeveritiesInSequence = first_severity[StartingIndex: AbsoluteTrialCount].copy()

                    MyMessage = numpy.c_[
                        response[CurrentBlockIndex][CurrentSequenceIndex],
                        confidence[CurrentBlockIndex][CurrentSequenceIndex],
                        exp_utils.get_array_of_sequence_severities_from_allocations(
                            response[CurrentBlockIndex][CurrentSequenceIndex],
                            InitialSeveritiesInSequence,
                        ),
                    ]

                    if resources_left > 0:
                        ResourceAllocationsAtCurrentlyVisibleCities[-1] = int(MyMessage[-1, 0])
                        SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(
                            new_locations, ResourceAllocationsAtCurrentlyVisibleCities, init_severity)

                        if SeveritiesOfCurrentlyVisibleCities[-1] < init_severity[-1]:
                            direction.append(2)
                        else:
                            direction.append(1)

                        resources_left = int(resources_left - MyMessage[-1, 0])

                    else:
                        ResourceAllocationsAtCurrentlyVisibleCities[-1] = 0
                        SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(
                            new_locations, ResourceAllocationsAtCurrentlyVisibleCities, init_severity)

                        if SeveritiesOfCurrentlyVisibleCities[-1] < init_severity[-1]:
                            direction.append(2)
                        else:
                            direction.append(1)

                _NumTrialsInSequence = int(NumTrialsPerSequence_list[AbsoluteSequenceIndex])
                StartingIndex = int(sum(NumTrialsPerSequence_list[: AbsoluteSequenceIndex]))
                InitialSeveritiesInSequence = first_severity[StartingIndex: AbsoluteTrialCount].copy()
                (MyPerformance,
                 _WorstCaseSequenceSeverity,
                 _BestCaseSequenceSeverity) = exp_utils.calculate_normalised_final_severity_performance_metric(
                    MyMessage[:, 2],
                    InitialSeveritiesInSequence
                )
                MyPerformances.append(MyPerformance)
                log_utils.tee(
                    "My Sequence Performances: ",
                    ", ".join([f'{i:.2f}' for i in MyPerformances])
                )
                log_utils.tee()

                AllMessages = [MyMessage]
                _aggregated_allocations, aggregated_final_severity = call_nominated_aggregator(
                    AllMessages, first_severity, AbsoluteSequenceIndex, AbsoluteTrialCount)
                AggregatedPerformance, *_ = exp_utils.calculate_normalised_final_severity_performance_metric(
                    aggregated_final_severity,
                    InitialSeveritiesInSequence
                )

                _StartingAbsoluteSequence = STARTING_BLOCK_INDEX * NUM_SEQUENCES + STARTING_SEQ_INDEX

                log_utils.tee(f"Sequence {AbsoluteSequenceIndex}: Performance = {AggregatedPerformance:.4f}")

                AbsoluteSequenceIndex += 1

            log_utils.tee("Trust Ratings:\n", TrustRatings)

    # Generate result summary JSON and visualization PNG
    if PLAYER_TYPE == 'DQN_AGENT' and len(MyPerformances) > 0:
        try:
            performances_by_block = []
            perf_idx = 0
            for _block_idx in range(NUM_BLOCKS):
                block_performances = []
                for _seq_idx in range(NUM_SEQUENCES):
                    if perf_idx < len(MyPerformances):
                        block_performances.append(MyPerformances[perf_idx])
                        perf_idx += 1
                if block_performances:
                    performances_by_block.append(block_performances)

            resource_data = {
                'total_resources_per_sequence': AVAILABLE_RESOURCES_PER_SEQUENCE,
                'agent_type': PLAYER_TYPE,
                'num_blocks': NUM_BLOCKS,
                'num_sequences': NUM_SEQUENCES,
                'total_trials': len(MyPerformances)
            }

            json_path, png_path = result_formatter.generate_results_report(
                MySubjectId,
                session_outputs_path,
                MyPerformances,
                performances_by_block,
                resource_data
            )
            log_utils.tee(f"\n✓ Results summary JSON: {json_path}")
            log_utils.tee(f"✓ Results visualization PNG: {png_path}\n")

            # Print the raw 64-sequence mean explicitly so it can be compared
            # one-glance against ``best_params.json['mean_perf']`` (the metric
            # Optuna reports).  Note that on Windows CPU the value can drift
            # ~3% from a Colab-GPU Optuna run because of cuDNN kernel
            # differences — see pes_dqn/ext/eval_model.py (if present) for a
            # stand-alone reproducer.
            try:
                _raw_mean = float(numpy.mean(MyPerformances))
                _raw_std = float(numpy.std(MyPerformances))
                log_utils.tee(
                    f"raw_mean_perf = {_raw_mean:.6f}  "
                    f"(std={_raw_std:.6f}, n={len(MyPerformances)})"
                )
                _bp_path = os.path.join(INPUTS_PATH, 'best_params.json')
                if os.path.isfile(_bp_path):
                    import json as _json
                    with open(_bp_path, 'r', encoding='utf-8') as _f:
                        _bp = _json.load(_f)
                    _expected = float(_bp.get('mean_perf', float('nan')))
                    if not numpy.isnan(_expected):
                        _delta = abs(_raw_mean - _expected)
                        log_utils.tee(
                            f"best_params.json mean_perf = {_expected:.6f}  "
                            f"|Δ| = {_delta:.6f}"
                        )
            except Exception as _cmp_err:  # pylint: disable=broad-except
                log_utils.tee(
                    f"Warning: could not compute raw_mean_perf comparison: {_cmp_err}"
                )
        except Exception as e:
            log_utils.tee(f"Warning: Could not generate result reports: {str(e)}")

    # Insert 'successful completion' marker in logfile.
    if SAVE_RESULTS:
        exp_utils.exit_experiment_gracefully(
            Message="",
            Filehandles=[Responses_filehandle],
            MovementData=(
                os.path.join(
                    session_outputs_path,
                    f'{OUTPUT_FILE_PREFIX}movement_log_{MySubjectId}.npy'),
                total_movement),
            LogUtils=log_utils,
            _PygameMediator=pygameMediator
        )

    return
#
# END OF 'main()


if __name__ == '__main__':
    main()
